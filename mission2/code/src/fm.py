import torch
import torch.nn as nn
from torchcfm import ConditionalFlowMatcher
from typing import Optional, Tuple, Union

class StableFlowMatcher(ConditionalFlowMatcher):
    def __init__(self, sigma: float, lambda_x: float, eps: float = 0.1):
        super().__init__(sigma)

        self.lambda_x = lambda_x
        self.eps = min(1 / self.lambda_x, eps)

        self.tau_0 = 0.0
        self.tau_1 = 1.0

    def sample_xt(
        self, 
        x0: torch.Tensor, 
        x1: torch.Tensor, 
        t: torch.Tensor,
        epsilon: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Sample xt from x0 and x1."""

        t = t.view(-1, 1, 1)

        return torch.exp(-self.lambda_x * t) * (x0 - x1) + x1

    def compute_tau(
        self, 
        t: torch.Tensor,
    ) -> torch.Tensor:
        """Compute tau."""
        
        return self.tau_1 + torch.exp(-self.lambda_x * t) * (self.tau_0 - self.tau_1)

    def compute_conditional_flow(
        self, 
        x0: torch.Tensor, 
        x1: torch.Tensor, 
        t: torch.Tensor,
        xt: torch.Tensor,
    ) -> torch.Tensor:
        """Compute conditional flow."""
        
        return - self.lambda_x * (xt - x1)

    def sample_location_and_conditional_flow(
        self, 
        x0: torch.Tensor, 
        x1: torch.Tensor, 
        t: Optional[torch.Tensor] =None, 
        return_noise=False
    ):
        if t is None:
            t = torch.rand(x0.shape[0]).type_as(x0)
        assert len(t) == x0.shape[0], "t has to have batch size dimension"

        x_t = self.sample_xt(x0, x1, t)
        u_t = self.compute_conditional_flow(x0, x1, t, x_t)

        tau = self.compute_tau(t)

        if return_noise:
            return tau, x_t, u_t, torch.randn_like(x_t)

        return tau, x_t, u_t

class StreamingFlowMatcher:
    def __init__(self, sigma_0: float, stabilization: float):
        self.sigma_0 = sigma_0
        self.stabilization_coeff = stabilization


    def get_streaming_tav(
        self, 
        data: torch.Tensor, 
        t: Optional[torch.Tensor] = None,
        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample xt from data and compute the conditional flow.

        Args:
            data (torch.Tensor): shape (B, T, A) where B is batch size, T is time steps, and A is action dimension.
            t (torch.Tensor, optional): shape (B, T) or None. If None, random t will be sampled from uniform distribution.
        """
        batch, length, _ = data.shape
        if t is None:
            t = torch.rand(
                (batch, length-1, 1),
                device=data.device
            )

        # trajectoryの速度を計算
        tgt_velocity = data[:, 1:] - data[:, :-1]  # shape: (B, T-1, A)

        data_t_mean = data[:, 1:] * t + data[:, :-1] * (1 - t)  # shape: (B, T-1, A)

        timesteps = t + torch.arange(0, length-1, device=data.device).reshape(1, -1, 1) 

        data_t = torch.normal(
            mean=data_t_mean,
            std=self.sigma_0*torch.exp(-timesteps/length*self.stabilization_coeff),
        )
        stabilization_term = (data_t - data_t_mean) * self.stabilization_coeff  # shape: (B, T-1, A)

        u_t = tgt_velocity - stabilization_term
        return timesteps, data_t, u_t

