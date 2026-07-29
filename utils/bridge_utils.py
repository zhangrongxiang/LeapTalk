"""
Utility functions for Bridge DMD2 training.

This module provides helper functions for model initialization,
EMA updates, checkpointing, and validation.
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any
import os
import json
import math
from pathlib import Path


class EMAModel:
    """
    Exponential Moving Average for model parameters.
    """

    def __init__(
        self,
        model: nn.Module,
        decay: float = 0.9999,
        update_interval: int = 1,
    ):
        """
        Initialize EMA model.

        Args:
            model: Model to track with EMA
            decay: EMA decay rate
            update_interval: Update EMA every N steps
        """
        self.decay = decay
        self.update_interval = update_interval
        self.step_count = 0

        # Create shadow parameters
        self.shadow_params = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow_params[name] = param.data.clone()

    @torch.no_grad()
    def update(self, model: nn.Module):
        """
        Update EMA parameters.

        Args:
            model: Current model with updated parameters
        """
        self.step_count += 1

        if self.step_count % self.update_interval != 0:
            return

        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow_params:
                self.shadow_params[name].mul_(self.decay).add_(
                    param.data, alpha=1 - self.decay
                )

    @torch.no_grad()
    def copy_to(self, model: nn.Module):
        """
        Copy EMA parameters to model.

        Args:
            model: Model to copy parameters to
        """
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow_params:
                param.data.copy_(self.shadow_params[name])

    def state_dict(self) -> Dict[str, Any]:
        """Get EMA state dict."""
        return {
            "decay": self.decay,
            "update_interval": self.update_interval,
            "step_count": self.step_count,
            "shadow_params": self.shadow_params,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Load EMA state dict."""
        self.decay = state_dict["decay"]
        self.update_interval = state_dict["update_interval"]
        self.step_count = state_dict["step_count"]
        self.shadow_params = state_dict["shadow_params"]


def save_checkpoint(
    output_dir: str,
    step: int,
    student_model: nn.Module,
    fake_velocity_model: nn.Module,
    student_optimizer: torch.optim.Optimizer,
    fake_optimizer: torch.optim.Optimizer,
    ema_model: Optional[EMAModel] = None,
    lr_scheduler: Optional[Any] = None,
    keep_last_n: int = 3,
):
    """
    Save training checkpoint.

    Args:
        output_dir: Directory to save checkpoint
        step: Current training step
        student_model: Student model
        fake_velocity_model: Fake velocity model
        student_optimizer: Student optimizer
        fake_optimizer: Fake velocity optimizer
        ema_model: Optional EMA model
        lr_scheduler: Optional learning rate scheduler
        keep_last_n: Keep only last N checkpoints
    """
    os.makedirs(output_dir, exist_ok=True)

    checkpoint_path = os.path.join(output_dir, f"checkpoint-{step}")
    os.makedirs(checkpoint_path, exist_ok=True)

    # Save models
    torch.save(
        student_model.state_dict(),
        os.path.join(checkpoint_path, "student_model.pt")
    )
    torch.save(
        fake_velocity_model.state_dict(),
        os.path.join(checkpoint_path, "fake_velocity_model.pt")
    )

    # Save optimizers
    torch.save(
        student_optimizer.state_dict(),
        os.path.join(checkpoint_path, "student_optimizer.pt")
    )
    torch.save(
        fake_optimizer.state_dict(),
        os.path.join(checkpoint_path, "fake_optimizer.pt")
    )

    # Save EMA if available
    if ema_model is not None:
        torch.save(
            ema_model.state_dict(),
            os.path.join(checkpoint_path, "ema_model.pt")
        )

    # Save LR scheduler if available
    if lr_scheduler is not None:
        torch.save(
            lr_scheduler.state_dict(),
            os.path.join(checkpoint_path, "lr_scheduler.pt")
        )

    # Save metadata
    metadata = {
        "step": step,
    }
    with open(os.path.join(checkpoint_path, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    # Clean up old checkpoints
    cleanup_old_checkpoints(output_dir, keep_last_n)

    print(f"Checkpoint saved at step {step}")


def cleanup_old_checkpoints(output_dir: str, keep_last_n: int):
    """
    Remove old checkpoints, keeping only the last N.

    Args:
        output_dir: Directory containing checkpoints
        keep_last_n: Number of checkpoints to keep
    """
    checkpoint_dirs = []
    for item in os.listdir(output_dir):
        if item.startswith("checkpoint-"):
            checkpoint_dirs.append(item)

    # Sort by step number
    checkpoint_dirs.sort(key=lambda x: int(x.split("-")[1]))

    # Remove old checkpoints
    if len(checkpoint_dirs) > keep_last_n:
        for old_checkpoint in checkpoint_dirs[:-keep_last_n]:
            old_path = os.path.join(output_dir, old_checkpoint)
            import shutil
            shutil.rmtree(old_path)
            print(f"Removed old checkpoint: {old_checkpoint}")


def load_checkpoint(
    checkpoint_path: str,
    student_model: nn.Module,
    fake_velocity_model: nn.Module,
    student_optimizer: Optional[torch.optim.Optimizer] = None,
    fake_optimizer: Optional[torch.optim.Optimizer] = None,
    ema_model: Optional[EMAModel] = None,
    lr_scheduler: Optional[Any] = None,
    device: str = "cuda",
) -> int:
    """
    Load training checkpoint.

    Args:
        checkpoint_path: Path to checkpoint directory
        student_model: Student model to load weights into
        fake_velocity_model: Fake velocity model to load weights into
        student_optimizer: Optional student optimizer
        fake_optimizer: Optional fake velocity optimizer
        ema_model: Optional EMA model
        lr_scheduler: Optional learning rate scheduler
        device: Device to load tensors to

    Returns:
        step: Training step from checkpoint
    """
    print(f"Loading checkpoint from {checkpoint_path}")

    # Load models
    student_model.load_state_dict(
        torch.load(
            os.path.join(checkpoint_path, "student_model.pt"),
            map_location=device
        )
    )
    fake_velocity_model.load_state_dict(
        torch.load(
            os.path.join(checkpoint_path, "fake_velocity_model.pt"),
            map_location=device
        )
    )

    # Load optimizers if provided
    if student_optimizer is not None:
        student_optimizer.load_state_dict(
            torch.load(
                os.path.join(checkpoint_path, "student_optimizer.pt"),
                map_location=device
            )
        )
    if fake_optimizer is not None:
        fake_optimizer.load_state_dict(
            torch.load(
                os.path.join(checkpoint_path, "fake_optimizer.pt"),
                map_location=device
            )
        )

    # Load EMA if available
    ema_path = os.path.join(checkpoint_path, "ema_model.pt")
    if ema_model is not None and os.path.exists(ema_path):
        ema_model.load_state_dict(
            torch.load(ema_path, map_location=device)
        )

    # Load LR scheduler if available
    lr_scheduler_path = os.path.join(checkpoint_path, "lr_scheduler.pt")
    if lr_scheduler is not None and os.path.exists(lr_scheduler_path):
        lr_scheduler.load_state_dict(
            torch.load(lr_scheduler_path, map_location=device)
        )

    # Load metadata
    with open(os.path.join(checkpoint_path, "metadata.json"), "r") as f:
        metadata = json.load(f)

    step = metadata["step"]
    print(f"Checkpoint loaded from step {step}")

    return step


def compute_snr(timesteps: torch.Tensor) -> torch.Tensor:
    """
    Compute signal-to-noise ratio for Bridge process.

    For Brownian Bridge: SNR(t) = (1-t)² / (t(1-t)) = (1-t) / t

    Args:
        timesteps: Timesteps in [0, 1]

    Returns:
        SNR values
    """
    eps = 1e-8
    snr = (1 - timesteps) / (timesteps + eps)
    return snr


def get_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    scheduler_type: str,
    num_training_steps: int,
    num_warmup_steps: int = 0,
):
    """
    Create learning rate scheduler.

    Args:
        optimizer: Optimizer to schedule
        scheduler_type: Type of scheduler (constant, linear, cosine)
        num_training_steps: Total training steps
        num_warmup_steps: Warmup steps

    Returns:
        LR scheduler
    """
    from torch.optim.lr_scheduler import LambdaLR

    if scheduler_type == "constant":
        def lr_lambda(step):
            if step < num_warmup_steps:
                return step / max(1, num_warmup_steps)
            return 1.0

    elif scheduler_type == "linear":
        def lr_lambda(step):
            if step < num_warmup_steps:
                return step / max(1, num_warmup_steps)
            return max(0.0, (num_training_steps - step) / (num_training_steps - num_warmup_steps))

    elif scheduler_type == "cosine":
        def lr_lambda(step):
            if step < num_warmup_steps:
                return step / max(1, num_warmup_steps)
            progress = (step - num_warmup_steps) / (num_training_steps - num_warmup_steps)
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")

    return LambdaLR(optimizer, lr_lambda)
