"""
This file contains the VarianceSchedule class,
which is used to schedule the variance of the Diffusion process.
Currently, the following schedules are implemented:
1. linear_beta_schedule
2. cosine_beta_schedule
3. quadratic_beta_schedule
4. sigmoid_beta_schedule
"""
import math
import torch
from torch import nn
from torch import FloatTensor


def cosine_beta_schedule(timesteps: int, s: float = 0.008, **kwargs) -> FloatTensor:
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(
        ((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)


def linear_beta_schedule(timesteps: int, beta_start: float = 0.0001,
                         beta_end: float = 0.02) -> FloatTensor:
    return torch.linspace(beta_start, beta_end, timesteps)


def quadratic_beta_schedule(timesteps: int, beta_start: float = 0.0001,
                            beta_end: float = 0.02) -> FloatTensor:
    return torch.linspace(beta_start**0.5, beta_end**0.5, timesteps) ** 2


def sigmoid_beta_schedule(timesteps: int, beta_start: float = 0.0001,
                          beta_end: float = 0.02) -> FloatTensor:
    betas = torch.linspace(-6, 6, timesteps)
    return torch.sigmoid(betas) * (beta_end - beta_start) + beta_start


class VarianceSchedule(nn.Module):
    def __init__(self, schedule_name: str = "linear_beta_schedule",
                 beta_start: float = None, beta_end: float = None) -> None:
        super().__init__()
        self.schedule_name = schedule_name
        beta_schedule_dict = {'linear_beta_schedule': linear_beta_schedule,
                              'cosine_beta_schedule': cosine_beta_schedule,
                              'quadratic_beta_schedule': quadratic_beta_schedule,
                              'sigmoid_beta_schedule': sigmoid_beta_schedule}

        if schedule_name in beta_schedule_dict:
            self.selected_schedule = beta_schedule_dict[schedule_name]
        else:
            raise ValueError('Function not found in dictionary')

        if beta_end and beta_start is None and schedule_name != "cosine_beta_schedule":
            self.beta_start = 0.0001
            self.beta_end = 0.02
        else:
            self.beta_start = beta_start
            self.beta_end = beta_end

    def forward(self, timesteps):
        return self.selected_schedule(timesteps=timesteps)\
            if self.schedule_name == "cosine_beta_schedule" \
            else self.selected_schedule(timesteps=timesteps, beta_start=self.beta_start,
                                        beta_end=self.beta_end)
