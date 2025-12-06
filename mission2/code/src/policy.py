import pytorch_lightning as pl
import ml_networks as ml
import torch
import torch.nn as nn
import torch.nn.functional as F
from .config import (PolicyConfig,  TransformerConfig, DiTConfig,
                     DiffuserConfig, FlowMatchingConfig, PosEmbConfig,
                     StableFlowMatchingConfig, StreamingFlowMatchingConfig,
                    )
from typing import Optional, Tuple, Union
from .backbone import (TransformerPolicy, SinusoidalPosEmb, DiT)
from .fm import StableFlowMatcher, StreamingFlowMatcher
from diffusers import DDIMScheduler, DDPMScheduler
from torchcfm import ConditionalFlowMatcher
from torchdyn.core import NeuralODE
from torch_ema import ExponentialMovingAverage
from einops import rearrange


def get_framework(
        cfg: Union[
        DiffuserConfig, 
        FlowMatchingConfig, 
        StableFlowMatchingConfig, 
        StreamingFlowMatchingConfig,
    ]):
    if isinstance(cfg, DiffuserConfig):
        if cfg.diff_type == "ddpm":
            framework = DDPMScheduler(
                num_train_timesteps=cfg.num_timesteps,
                beta_start=cfg.beta_start,
                beta_end=cfg.beta_end,
                beta_schedule=cfg.beta_schedule,
                prediction_type=cfg.prediction_type,
                clip_sample=cfg.clip_sample,
            )


        elif cfg.diff_type == "ddim":
            framework = DDIMScheduler(
                num_train_timesteps=cfg.num_timesteps,
                beta_start=cfg.beta_start,
                beta_end=cfg.beta_end,
                beta_schedule=cfg.beta_schedule,
                prediction_type=cfg.prediction_type,
                clip_sample=cfg.clip_sample,
            )
            framework.set_timesteps(cfg.num_inference_steps)
        is_fm = False

    elif isinstance(cfg, StreamingFlowMatchingConfig):
        framework = StreamingFlowMatcher(
            cfg.sigma_0, 
            cfg.stabilization
        )
        is_fm = True

    elif isinstance(cfg, StableFlowMatchingConfig):
        framework = StableFlowMatcher(
            cfg.sigma, 
            cfg.lambda_x,
            cfg.eps
        )
        is_fm = True
    elif isinstance(cfg, FlowMatchingConfig):
        framework = ConditionalFlowMatcher(cfg.sigma)

        is_fm = True
    else:
        raise ValueError("Unsupported framework type")
    return framework, is_fm

class CondWrapper(nn.Module):
    def __init__(
            self, 
            model: Union[ml.ConditionalUnet1d, TransformerPolicy, DiT],
            condition: Optional[torch.Tensor] = None
        ):
        super().__init__()
        self.model = model
        self.condition = condition

    def forward(
            self,
            t: torch.Tensor,
            x : torch.Tensor,
            *args,
            **kwargs
        ):
        
        return self.model.forward(t=t, x=x, cond=self.condition)

class PolicyBase(pl.LightningModule):
    def __init__(self, cfg: PolicyConfig):
        super().__init__()
        self.register_buffer("noise_mean", torch.zeros(1, cfg.policy_length, cfg.action_dim))
        self.register_buffer("noise_std", torch.ones(1, cfg.policy_length, cfg.action_dim))


        self.time_emb = SinusoidalPosEmb(
                cfg.time_emb.vector, cfg.time_emb.scale
            ) if isinstance(
                cfg.time_emb, PosEmbConfig
            ) else ml.SoftmaxTransformation(
                cfg.time_emb
            )
        self.time_encoder = ml.MLPLayer(
            cfg.time_emb.vector, cfg.time_emb.vector, cfg.cond_cfg
        )

        if not cfg.pos_only:
            self.encoder = ml.Encoder(
                cfg.obs_dim, 
                cfg.obs_shape,
                cfg.encoder.backbone,
                cfg.encoder.full_connection
            )

            if cfg.freeze_encoder:
                self.encoder.freeze()

        if not cfg.obs_only:
            self.pos_encoder = ml.MLPLayer(
                cfg.action_dim, 
                cfg.obs_dim,
                cfg.pos_encoder
            )
        self.framework_cfg = cfg.framework_cfg
        self.framework, self.is_fm = get_framework(cfg.framework_cfg)

        self.cfg = cfg

    def configure_optimizers(self):
        name = self.cfg.optimizer_cfg.pop("name")
        self.ema = ExponentialMovingAverage(self.parameters(), **self.cfg.ema)
        return ml.get_optimizer(self.parameters(), name, **self.cfg.optimizer_cfg)

    def optimizer_step(self, *args, **kwargs):
        super().optimizer_step(*args, **kwargs)
        self.ema.update(self.parameters())

    def training_step(self, batch, batch_idx):
        ret = self._shared_step(batch)
        loss_dict = {}
        for key, value in ret.items():
            if "loss" == key:
                loss = value
                loss_dict["loss/train"] = value
            loss_dict[f"loss/{key}/train"] = ret[key]
        self.log_dict(loss_dict, prog_bar=True, sync_dist=True)
        return dict(loss=loss)
    
    def validation_step(self, batch, batch_idx):
        obs, action, pos, goal = batch
        ret = self._shared_step(batch)

        if not self.cfg.pos_only:
            obs_emb = self.encoder(obs[:4])
        else:
            obs_emb = None
        if not self.cfg.obs_only:
            pos_emb = self.pos_encoder(pos[:4])
        else:
            pos_emb = None
        if self.cfg.goal_conditioned and not self.cfg.pos_only:
            goal_emb = self.encoder(goal[:4])
        else:
            goal_emb = None
        samples = []
        for i in range(4):
            sample = self.inference(
                batch_size=4,
                obs=obs_emb[i].unsqueeze(0) if obs_emb is not None else None,
                pos=pos_emb[i].unsqueeze(0) if pos_emb is not None else None,
                goal=goal_emb[i].unsqueeze(0) if goal_emb is not None else None,
                nfe=self.cfg.framework_cfg.num_inference_steps,
                initial_action=pos[i, -1].unsqueeze(0) 
            )
            samples.append(sample)
        samples = torch.stack(samples, dim=0)
        pred_dict = {
            "prediction/samples": samples,
            "prediction/original": action[:4],
            "prediction/obs": obs,
            "prediction/spatial": obs_emb,
        }
        loss_dict = {
            "loss": ret["loss"],
            "loss/val": ret["loss"],
        }
        for key, value in ret.items():
            if "loss" == key:
                pred_dict["loss"] = value
                pred_dict["loss/val"] = value
            else:
                pred_dict[key] = ret[key]
                loss_dict[f"loss/{key}/val"] = ret[key]

        self.log_dict(loss_dict, prog_bar=True, sync_dist=True)

        return pred_dict

    def inference(
        self,
        *args,
        **kwargs
        ) -> torch.Tensor:
        raise NotImplementedError("Inference method must be implemented in subclasses")

    def _shared_step(self, batch):
        raise NotImplementedError("Shared step method must be implemented in subclasses")

class Policy(PolicyBase):
    def __init__(self, cfg: PolicyConfig):
        super().__init__(cfg)

        cond_step = cfg.n_obs_steps

        if not cfg.pos_only and not cfg.obs_only:
            cond_step *= 2
        cond_step += cfg.goal_conditioned

        cond_step += 1 # time cond
        if isinstance(cfg.policy, ml.UNetConfig):
            self.policy = ml.ConditionalUnet1d(cfg.obs_dim*cond_step, (cfg.action_dim, cfg.policy_length), cfg.policy)
        elif isinstance(cfg.policy, TransformerConfig):
            cfg.policy.cond_step = cond_step
            self.policy = TransformerPolicy(
                cfg.pred_obs_action,
                cfg.policy)
        elif isinstance(cfg.policy, DiTConfig):
            cfg.policy.cond_step = cond_step
            self.policy = DiT(cfg.pred_obs_action, False, cfg.policy)
        else:
            raise ValueError("Unsupported policy type")

    @torch.no_grad()
    def inference(
        self,
        batch_size: int,
        obs: torch.Tensor,
        pos: torch.Tensor,
        goal: Optional[torch.Tensor] = None,
        nfe: Optional[int] = None,
        **kwargs
        ) -> torch.Tensor:
        cond = None
        if self.cfg.obs_only:
            cond = obs
        elif self.cfg.pos_only:
            cond = pos
        else:
            cond = torch.cat([obs, pos], dim=-2)
        if self.cfg.goal_conditioned:
            cond = torch.cat([cond, goal], dim=-2)

        if cond is None or cond.shape[1] == 0:
            cond = None
        else:
            cond = cond.expand(
                batch_size, -1, -1
            )


        trajectory = torch.normal(
            mean=self.noise_mean.expand(
                batch_size, -1, -1
            ),
            std=self.noise_std.expand(
                batch_size, -1, -1
            )
        ).to(self.device).float()
        if self.cfg.use_uniform:
            trajectory = trajectory.uniform_(-1, 1)


        if self.is_fm and isinstance(self.framework, StableFlowMatcher):
            t = torch.zeros(batch_size, device=self.device)

            for i in range(self.framework_cfg.num_inference_steps):
                # 1. predict model output
                tau = self.framework.compute_tau(t)
                time_emb = self.time_emb(tau).reshape(batch_size, -1).unsqueeze(1)
                time_emb = self.time_encoder(time_emb)
                cond_t = torch.cat([cond, time_emb], dim=-2) if cond is not None else time_emb
                if not isinstance(self.policy, TransformerPolicy):
                    cond_t = rearrange(cond_t, 'b t d -> b (t d)')

                if isinstance(self.policy, ml.ConditionalUnet1d):
                    trajectory = trajectory.transpose(-1,-2)
                vector = self.policy.forward(
                    trajectory, cond=cond_t 
                )
                if isinstance(self.policy, ml.ConditionalUnet1d):
                    vector = vector.transpose(-1,-2)
                    trajectory = trajectory.transpose(-1,-2)

                # 2. compute previous image: x_t -> x_t-1
                if i == 0:
                    trajectory += vector / self.framework.lambda_x
                    t += 1.0 / self.framework.lambda_x
                else:
                    trajectory += vector * self.framework.eps
                    t += self.framework.eps


        elif self.is_fm:
            cond_policy = CondWrapper(self.policy, cond)
            node = NeuralODE(
                cond_policy,
                solver=self.framework_cfg.solver,
                sensitivity=self.framework_cfg.sensitivity,
                atol=self.framework_cfg.atol,
                rtol=self.framework_cfg.atol,
            )
            if nfe is None:
                nfe = self.framework_cfg.num_inference_steps
            trajectory = node.trajectory(
                    torch.randn(
                        (batch_size, self.cfg.policy_length, self.cfg.action_dim), device=self.device
                    ),
                    t_span=torch.linspace(
                        0, 1, nfe + 1
                ))[-1]

        else:
            for t in self.framework.timesteps:
                # 1. predict model output
                t = torch.tensor([t], device=self.device).reshape(-1, 1)
                time_emb = self.time_emb(t).expand(batch_size, -1, -1)
                time_emb = self.time_encoder(time_emb)
                cond_t = torch.cat([cond, time_emb], dim=-2) if cond is not None else time_emb
                if not isinstance(self.policy, TransformerPolicy):
                    cond_t = rearrange(cond_t, 'b t d -> b (t d)')
                if isinstance(self.policy, ml.ConditionalUnet1d):
                    trajectory = trajectory.transpose(-1,-2)
                pred_noise = self.policy.forward(
                    trajectory, cond=cond_t
                )
                if isinstance(self.policy, ml.ConditionalUnet1d):
                    pred_noise = pred_noise.transpose(-1,-2)
                    trajectory = trajectory.transpose(-1,-2)

                # 2. compute previous image: x_t -> x_t-1
                trajectory = self.framework.step(
                    pred_noise, t, trajectory, 
                    ).prev_sample


        if self.cfg.pred_obs_action:
            trajectory = trajectory[:, self.cfg.n_obs_steps:, :]
        return trajectory

    def _shared_step(self, batch):
        obs, action, pos, goal = batch
        cond = None

        if not self.cfg.obs_only:
            pos_emb = self.pos_encoder(pos)
        if not self.cfg.pos_only:
            obs_emb = self.encoder(obs)

        if self.cfg.obs_only:
            cond = obs_emb
        elif self.cfg.pos_only:
            cond = pos_emb
        else:
            cond = torch.cat([obs_emb, pos_emb], dim=-2)
        if self.cfg.goal_conditioned:
            goal_emb = self.encoder(goal)
            cond = torch.cat([cond, goal_emb], dim=-2)

        if cond.shape[1] == 0:
            cond = None

        noise = torch.normal(
            mean=self.noise_mean.expand_as(action),
            std=self.noise_std.expand_as(action)
        ).to(self.device).float()
        if self.cfg.use_uniform:
            noise = noise.uniform_(-1, 1)
        if self.is_fm:
            t = torch.rand(
                (len(action),), 
                device=self.device
            )
            t, x_t, u_t = self.framework.sample_location_and_conditional_flow(
                noise, action, t 
            )
        else:
            t = torch.randint(
            0, 
            self.cfg.framework_cfg.num_timesteps, 
            (len(action),), 
            device=self.device
            ).long()
            
            x_t = self.framework.add_noise(
                action, noise, t
            )

            u_t = noise

        timesteps = t.expand(noise.shape[0])
        time_emb = self.time_emb(timesteps).reshape(len(noise), -1).unsqueeze(1)
        time_emb = self.time_encoder(time_emb)
        cond = torch.cat([cond, time_emb], dim=-2) if cond is not None else time_emb
        if not isinstance(self.policy, TransformerPolicy):
            cond = rearrange(cond, 'b t d -> b (t d)')
        if isinstance(self.policy, ml.ConditionalUnet1d):
            x_t = x_t.transpose(-1,-2)
        v_t = self.policy.forward(
            x_t, cond=cond
        )
        if isinstance(self.policy, ml.ConditionalUnet1d):
            v_t = v_t.transpose(-1,-2)

        loss = F.mse_loss(v_t, u_t, reduction="none").sum(dim=[-2, -1]).mean()

        loss_dict = {
            "loss": loss,
        }

        return loss_dict


class StreamingPolicy(PolicyBase):
    def __init__(self, cfg: PolicyConfig):
        super().__init__(cfg)

        cond_step = cfg.n_obs_steps

        if not cfg.pos_only and not cfg.obs_only:
            cond_step *= 2
        cond_step += cfg.goal_conditioned
        cond_step += 1 # time cond
        if isinstance(cfg.policy, ml.MLPConfig):
            self.policy = ml.MLPLayer(cfg.action_dim+ cond_step*cfg.obs_dim, cfg.action_dim, cfg.policy)
        else:
            raise ValueError("Unsupported policy type")
        assert isinstance(self.framework, StreamingFlowMatcher), "Streaming Policy requires StreamingFlowMatcher framework"

    @torch.no_grad()
    def inference(
        self,
        batch_size: int,
        obs: torch.Tensor,
        pos: torch.Tensor,
        goal: Optional[torch.Tensor] = None,
        nfe: Optional[int] = None,
        initial_action: Optional[torch.Tensor] = None,
        ) -> torch.Tensor:
        if nfe is None:
            nfe = self.cfg.framework_cfg.num_inference_steps


        if self.cfg.obs_only:
            cond = obs
        elif self.cfg.pos_only:
            cond = pos
        else:
            obs_emb = obs
            pos_emb = pos
            cond = torch.cat([obs_emb, pos_emb], dim=-2)
        if self.cfg.goal_conditioned:
            goal_emb = goal
            cond = torch.cat([cond, goal_emb], dim=-2)

        if cond is None or cond.shape[1] == 0:
            cond = None
        else:
            cond = cond.expand(
                batch_size, -1, -1
            )
        if initial_action is not None:
            a_0 = initial_action.expand(
                batch_size, -1, 
            )
        else:
            a_0 = pos[:, -1].reshape(pos.shape[0],  -1).expand(
                batch_size, -1
            )
        trajectory = []
        a_t = torch.normal(a_0, self.framework.sigma_0)
        for t in range(nfe):
            t = torch.tensor([t], device=self.device).reshape(-1, 1)
            time_emb = self.time_emb(t)
            time_emb = self.time_encoder(time_emb).expand(batch_size, -1, -1)
            cond_t = torch.cat([time_emb, cond], dim=-2) if cond is not None else time_emb
            if isinstance(self.policy, ml.MLPLayer):
                cond_t= rearrange(cond_t, 'b t d -> b (t d)')
                v_t = self.policy.forward(
                    torch.cat([a_t, cond_t], dim=-1)
                )
            elif isinstance(self.policy, TransformerPolicy):
                cond_t = rearrange(cond_t, 'b t d -> b t d')
                v_t = self.policy.forward(
                    a_t, cond=cond_t
                )
            else:
                raise ValueError("Unsupported policy type")
            a_t = a_t + v_t
            trajectory.append(a_t)
        trajectory = torch.stack(trajectory, dim=1)

        return trajectory

    def _shared_step(self, batch):
        obs, action, pos, goal = batch
        cond = None
        batch, length, _ = action.shape

        if self.cfg.obs_only:
            obs_emb = self.encoder(obs)
            cond = obs_emb
        elif self.cfg.pos_only:
            pos_emb = self.pos_encoder(pos)
            cond = pos_emb
        else:
            obs_emb = self.encoder(obs)
            pos_emb = self.pos_encoder(pos)
            cond = torch.cat([obs_emb, pos_emb], dim=-2)
        if self.cfg.goal_conditioned:
            goal_emb = self.encoder(goal)
            cond = torch.cat([cond, goal_emb], dim=-2)

        if cond.shape[1] == 0:
            cond = None

        t, a_t, u_t = self.framework.get_streaming_tav(
                action
        )
        a_t = rearrange(a_t, 'b l d -> (b l) d')

        cond = cond.unsqueeze(1).expand(
            batch, length-1, -1, -1
        ) if cond is not None else None

        time_emb = self.time_emb(t)
        time_emb = self.time_encoder(time_emb).squeeze(1).unsqueeze(2)
        cond = torch.cat([time_emb, cond], dim=-2) if cond is not None else time_emb
        if isinstance(self.policy, ml.MLPLayer):

            cond = rearrange(cond, 'b l t d -> (b l) (t d)')
            v_t = self.policy.forward(
                torch.cat([a_t, cond], dim=-1)
            )
            v_t = rearrange(v_t, '(b l) d -> b l d', b=batch)
        elif isinstance(self.policy, TransformerPolicy):
            cond = rearrange(cond, 'b l t d -> (b l) t d')
            a_t = a_t.unsqueeze(-1)
            v_t = self.policy.forward(
                a_t, cond=cond
            )
            v_t = rearrange(v_t, '(b l) t d -> b l d', b=batch)

        loss = F.mse_loss(v_t, u_t, reduction="none").sum(dim=-1).mean()

        loss_dict = {
            "loss": loss,
        }

        return loss_dict
