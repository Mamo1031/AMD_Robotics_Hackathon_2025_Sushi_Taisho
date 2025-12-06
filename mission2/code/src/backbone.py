import einops
import torch
import torch.nn as nn
import numpy as np
import ml_networks as ml

from .config import TransformerConfig, PosEmbConfig, DiTConfig
from timm.models.vision_transformer import PatchEmbed, Attention, Mlp


class TransformerPolicy(nn.Module):
    def __init__(
        self,
        pred_obs_action: bool,
        cfg: TransformerConfig,
        use_nested: bool = False,
    ) -> None:
        super().__init__()
        self.cfg = cfg


        T = cfg.horizon + pred_obs_action * cfg.cond_step
        T_cond = 1 + cfg.cond_step

        # input embedding stem
        self.input_emb = nn.Linear(cfg.input_dim, cfg.d_model)
        self.pos_emb = nn.Parameter(torch.zeros(1, T, cfg.d_model))
        self.drop = nn.Dropout(cfg.dropout)


        if cfg.cond_step > 0:
            self.cond_obs_emb = nn.Linear(cfg.cond_dim, cfg.d_model)

        encoder_only = cfg.only_encoder
        if not encoder_only:
            self.cond_pos_emb = nn.Parameter(torch.zeros(1, T_cond, cfg.d_model))
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=cfg.d_model,
                nhead=cfg.nhead,
                dim_feedforward=cfg.dim_feedforward,
                dropout=cfg.dropout,
                activation=cfg.hidden_activation,
                batch_first=True,
                norm_first=True
            )
            self.encoder = nn.TransformerEncoder(
                encoder_layer=encoder_layer,
                num_layers=cfg.n_enc_layers,
                enable_nested_tensor=use_nested
            )
            # decoder
            decoder_layer = nn.TransformerDecoderLayer(
                d_model=cfg.d_model,
                nhead=cfg.nhead,
                dim_feedforward=cfg.dim_feedforward,
                dropout=cfg.dropout,
                activation=cfg.hidden_activation,
                batch_first=True,
                norm_first=True # important for stability
            )
            self.decoder = nn.TransformerDecoder(
                decoder_layer=decoder_layer,
                num_layers=cfg.n_layers,
            )
        else:
            # encoder only BERT
            encoder_only = True
            self.cond_pos_emb = nn.Parameter(torch.zeros(1, T_cond, cfg.d_model))

            encoder_layer = nn.TransformerEncoderLayer(
                d_model=cfg.d_model,
                nhead=cfg.nhead,
                dim_feedforward=cfg.dim_feedforward,
                dropout=cfg.dropout,
                activation=cfg.hidden_activation,
                batch_first=True,
                norm_first=True
            )
            self.encoder = nn.TransformerEncoder(
                encoder_layer=encoder_layer,
                num_layers=cfg.n_layers,
                enable_nested_tensor=use_nested
            )

        # attention mask
        if cfg.is_causal:
            # causal mask to ensure that attention is only applied to the left in the input sequence
            # torch.nn.Transformer uses additive mask as opposed to multiplicative mask in minGPT
            # therefore, the upper triangle should be -inf and others (including diag) should be 0.
            sz = T
            mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
            mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
            self.register_buffer("mask", mask)
            
        else:
            self.mask = None

        # decoder head
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, input_dim)
            
        # constants
        self.T = T
        self.T_cond = T_cond
        self.encoder_only = encoder_only

        # init
        self.apply(self._init_weights)
        

    def _init_weights(self, module):
        ignore_types = (nn.Dropout, 
            ml.SoftmaxTransformation,
            nn.TransformerEncoderLayer, 
            nn.TransformerDecoderLayer,
            nn.TransformerEncoder,
            nn.TransformerDecoder,
            nn.ModuleList,
            nn.Mish,
            nn.Sequential,
            ml.MLPLayer,
            nn.Identity,
            ml.Activation,
            ml.LinearNormActivation,

            )
        if isinstance(module, (nn.Linear, nn.Embedding)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.MultiheadAttention):
            weight_names = [
                'in_proj_weight', 'q_proj_weight', 'k_proj_weight', 'v_proj_weight']
            for name in weight_names:
                weight = getattr(module, name)
                if weight is not None:
                    torch.nn.init.normal_(weight, mean=0.0, std=0.02)
            
            bias_names = ['in_proj_bias', 'bias_k', 'bias_v']
            for name in bias_names:
                bias = getattr(module, name)
                if bias is not None:
                    torch.nn.init.zeros_(bias)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
        elif isinstance(module, TransformerPolicy):
            torch.nn.init.normal_(module.pos_emb, mean=0.0, std=0.02)
            if module.cond_obs_emb is not None:
                torch.nn.init.normal_(module.cond_pos_emb, mean=0.0, std=0.02)
        elif isinstance(module, ignore_types):
            # no param
            pass
        else:
            raise RuntimeError("Unaccounted module {}".format(module))



    def forward(
        self, 
        x: torch.Tensor, 
        cond: Optional[torch.Tensor]=None, 
        ):
        """
        x: (B,T,input_dim)
        t: (B,) or int, diffusion or flow step
        cond: (B,T',cond_dim)
        plan: (B, plan_dim)
        output: (B,T,input_dim)
        """

        # process input
        input_emb = self.input_emb(x)

        if self.encoder_only:
            # BERT
            token_embeddings = input_emb
            t = token_embeddings.shape[1]
            position_embeddings = self.pos_emb[
                :, :t, :
            ]  # each position maps to a (learnable) vector
            x = self.drop(token_embeddings + position_embeddings)
            # (B,T+1,cfg.d_model)
            x = self.encoder(src=x, mask=self.mask)
            # (B,T+1,cfg.d_model)
            x = x[:,1:,:]
            # (B,T,cfg.d_model)
        else:
            # encoder

            # (B,2*To,cfg.d_model)
            cond_embeddings = self.cond_obs_emb(cond)
                
            tc = cond_embeddings.shape[1]
            position_embeddings = self.cond_pos_emb[
                :, :tc, :
            ]  # each position maps to a (learnable) vector
            x = self.drop(cond_embeddings + position_embeddings)
            x = self.encoder(x)
            memory = x
            # (B,T_cond,cfg.d_model)
            
            # decoder
            token_embeddings = input_emb
            t = token_embeddings.shape[1]
            position_embeddings = self.pos_emb[
                :, :t, :
            ]  # each position maps to a (learnable) vector
            x = self.drop(token_embeddings + position_embeddings)
            # (B,T,cfg.d_model)
            x = self.decoder(
                tgt=x,
                memory=memory,
                tgt_mask=self.mask,
            )
            # (B,T,cfg.d_model)
        
        # head
        x = self.ln_f(x)
        x = self.head(x)
        # (B,T,n_out)
        return x

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int, scale: float):
        super().__init__()
        self.dim = dim
        self.scale = scale

    def forward(self, x: torch.Tensor):
        x = x * self.scale
        device = x.device
        half_dim = self.dim // 2
        emb = np.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

#################################################################################
#                                 Core DiT Model                                #
#################################################################################

def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

class DiTBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    def __init__(self, hidden_size: int, n_cond_step: int, num_heads: int, mlp_ratio: float=4.0, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size*n_cond_step, 6 * hidden_size, bias=True)
        )

    def forward(self, x: torch.Tensor, c: torch.Tensor):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """
    def __init__(self, hidden_size: int, n_cond_step: int,  out_dim: int):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, out_dim, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size * n_cond_step, 2 * hidden_size, bias=True)
        )

    def forward(self, x: torch.Tensor, c: torch.Tensor):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class DiT(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    """
    def __init__(
        self,
        pred_obs_action: bool,
        pred_variance: bool,
        cfg: DiTConfig,
    ):
        super().__init__()
        self.cfg = cfg
        self.pred_variance = pred_variance


        T = cfg.horizon + pred_obs_action * cfg.cond_step 
        T_cond = cfg.cond_step

        # input embedding stem
        self.input_emb = nn.Linear(cfg.input_dim, cfg.d_model)


        # cond encoder
        self.time_emb = SinusoidalPosEmb(
            cfg.time_emb.vector, cfg.time_emb.scale
        ) if isinstance(
            cfg.time_emb, PosEmbConfig
        ) else ml.SoftmaxTransformation(
            cfg.time_emb
        )
        self.time_encoder = ml.MLPLayer(
            cfg.time_emb.vector, 
            cfg.d_model, 
            cfg.cond_cfg
        )
        self.learn_sigma = cfg.learn_sigma
        self.num_heads = cfg.nhead

        # Will use fixed sin-cos embedding:
        self.pos_embed = nn.Parameter(torch.zeros(1, T, cfg.d_model), requires_grad=False)

        self.blocks = nn.ModuleList([
            DiTBlock(cfg.d_model, T_cond, self.num_heads, mlp_ratio=cfg.mlp_ratio) for _ in range(cfg.n_layers)
        ])
        self.final_layer = FinalLayer(cfg.d_model, T_cond, cfg.input_dim*2 if pred_variance else cfg.input_dim)
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize (and freeze) pos_embed by sin-cos embedding:
        pos_embed = sinusoidal_positional_encoding(self.pos_embed.shape[-2], self.pos_embed.shape[-1])
        self.pos_embed.data.copy_(pos_embed)

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.input_emb.weight
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.input_emb.bias, 0)


        # Initialize timestep embedding MLP:
        nn.init.normal_(self.time_encoder.dense[0].linear.weight,std=0.02)
        nn.init.normal_(self.time_encoder.dense[1].linear.weight,std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def forward(
        self, 
        x: torch.Tensor, 
        cond: torch.Tensor
    ):
        """
        Forward pass of DiT.
        x: (B, T, input_dim) tensor of input patches
        t: (B,) tensor of diffusion timesteps
        cond: (B, *, cond_dim) tensor of conditioning information
        """
        x = self.input_emb(x) + self.pos_embed  # (N, T, D), 
        c = cond
        for block in self.blocks:
            x = block(x, c)                      # (N, T, D)
        x = self.final_layer(x, c)                # (N, T, patch_size ** 2 * out_channels)
        if self.pred_variance:
            x, var = x.chunk(2, dim=-1)
            var = torch.tanh(var)
            var = (var + 1) / 2  # scale to [0, 1]
            x = torch.cat([x, var], dim=-1)  # (N, T, input_dim * 2)
        return x
