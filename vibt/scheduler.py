from diffusers.schedulers import UniPCMultistepScheduler
import torch
import time


class ViBTScheduler(UniPCMultistepScheduler):
    def __init__(self, **kwargs):
        super().__init__(**{**kwargs, "use_flow_sigmas": True})
        self.set_parameters()

    def set_parameters(self, noise_scale=1.0, shift_gamma=5.0, seed=None):
        self.noise_scale = noise_scale
        self.config.flow_shift = shift_gamma
        self.generator = (
            None if seed is None else torch.Generator("cuda").manual_seed(seed)
        )

    def step(self, model_output, timestep, sample, ode=False, **kwargs):
        t_val = float(timestep)
        
        if not hasattr(self, "_t_list") or len(self._t_list) != len(self.timesteps):
            self._t_list = self.timesteps.cpu().tolist()
            
        idx = -1
        for i, val in enumerate(self._t_list):
            if abs(val - t_val) < 1e-4:
                idx = i
                break
                
        if idx != -1 and idx + 1 < len(self._t_list):
            next_t = self._t_list[idx + 1]
        else:
            next_t = -1.0

        delta_t = (next_t - t_val) / 1000.0
        current_t = (t_val + 1.0) / 1000.0
        
        eta_radicand = -delta_t * (current_t + delta_t) / current_t
        eta = (max(0.0, eta_radicand)) ** 0.5

        noise = torch.randn(
            sample.shape,
            generator=self.generator,
            device=sample.device,
            dtype=sample.dtype,
        )

        latents = sample + delta_t * model_output + eta * self.noise_scale * noise

        return (latents,)

    @classmethod
    def from_scheduler(
        cls, scheduler: UniPCMultistepScheduler, noise_scale=1.0, shift_gamma=5.0
    ):
        obj = cls.__new__(cls)
        obj.__dict__ = scheduler.__dict__.copy()
        obj.set_parameters(noise_scale, shift_gamma)
        return obj
