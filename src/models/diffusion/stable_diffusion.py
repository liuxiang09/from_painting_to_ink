from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn


class ImageConditionProjector(nn.Module):
    def __init__(self, cross_attention_dim: int = 768, tokens_per_image: int = 32, style_weight: float = 0.75):
        super().__init__()
        self.tokens_per_image = tokens_per_image
        self.style_weight = style_weight
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1),
            nn.SiLU(inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.GroupNorm(16, 128),
            nn.SiLU(inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.GroupNorm(32, 256),
            nn.SiLU(inplace=True),
            nn.Conv2d(256, 512, 4, 2, 1),
            nn.GroupNorm(32, 512),
            nn.SiLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.to_tokens = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, tokens_per_image * cross_attention_dim),
            nn.LayerNorm(tokens_per_image * cross_attention_dim),
        )
        self.content_type = nn.Parameter(torch.zeros(1, tokens_per_image, cross_attention_dim))
        self.style_type = nn.Parameter(torch.zeros(1, tokens_per_image, cross_attention_dim))
        self.null_token = nn.Parameter(torch.zeros(1, 1, cross_attention_dim))

    def _encode(self, image: torch.Tensor) -> torch.Tensor:
        tokens = self.to_tokens(self.encoder(image))
        return tokens.view(image.shape[0], self.tokens_per_image, -1)

    def forward(self, content: torch.Tensor, style: torch.Tensor, style_weight: float | None = None) -> torch.Tensor:
        weight = self.style_weight if style_weight is None else style_weight
        content_tokens = self._encode(content) + self.content_type
        style_tokens = self._encode(style) * weight + self.style_type
        null_tokens = self.null_token.expand(content.shape[0], -1, -1)
        return torch.cat([content_tokens, style_tokens, null_tokens], dim=1)


class StableDiffusionStyleTransfer(nn.Module):
    def __init__(
        self,
        pretrained_model: str,
        style_weight: float = 0.75,
        train_unet: bool = True,
        train_vae: bool = False,
        tokens_per_image: int = 32,
        unet_train_mode: str = "full",
        unet_train_patterns: list[str] | None = None,
        gradient_checkpointing: bool = False,
        torch_dtype: torch.dtype | None = None,
    ):
        super().__init__()
        try:
            from diffusers import DDPMScheduler, StableDiffusionPipeline
        except ImportError as exc:
            raise ImportError(
                "Stable Diffusion training requires diffusers. Install dependencies with: "
                "pip install torch torchvision diffusers transformers safetensors accelerate"
            ) from exc

        dtype = torch_dtype or torch.float32
        pretrained_path = Path(pretrained_model)
        if pretrained_path.suffix.lower() in {".ckpt", ".safetensors"}:
            pipe = StableDiffusionPipeline.from_single_file(pretrained_model, torch_dtype=dtype, safety_checker=None)
        else:
            pipe = StableDiffusionPipeline.from_pretrained(pretrained_model, torch_dtype=dtype, safety_checker=None)

        self.vae = pipe.vae
        self.unet = pipe.unet
        self.noise_scheduler = DDPMScheduler.from_config(pipe.scheduler.config)
        self.latent_scaling_factor = getattr(self.vae.config, "scaling_factor", 0.18215)

        cross_attention_dim = int(getattr(self.unet.config, "cross_attention_dim", 768))
        self.condition_projector = ImageConditionProjector(
            cross_attention_dim=cross_attention_dim,
            tokens_per_image=tokens_per_image,
            style_weight=style_weight,
        )

        self.vae.requires_grad_(train_vae)

        if train_unet:
            mode = unet_train_mode.strip().lower()
            if mode == "full":
                self.unet.requires_grad_(True)
                self.unet_mode = "full"
            elif mode == "partial":
                self.unet.requires_grad_(False)
                selected_patterns = [p.strip() for p in (unet_train_patterns or []) if p and p.strip()]
                if not selected_patterns:
                    selected_patterns = [
                        "mid_block",
                        "up_blocks.2",
                        "up_blocks.3",
                        "to_q",
                        "to_k",
                        "to_v",
                        "to_out",
                    ]

                enabled = 0
                for name, param in self.unet.named_parameters():
                    if any(pattern in name for pattern in selected_patterns):
                        param.requires_grad = True
                        enabled += 1
                self.unet_mode = f"partial({enabled}_tensors)"
            else:
                raise ValueError(f"invalid unet_train_mode: {unet_train_mode}, expected one of ['full', 'partial']")
        else:
            self.unet.requires_grad_(False)
            self.unet_mode = "frozen"

        if gradient_checkpointing and any(param.requires_grad for param in self.unet.parameters()):
            self.unet.enable_gradient_checkpointing()

        if not train_vae:
            self.vae.eval()

        if getattr(pipe, "text_encoder", None) is not None:
            pipe.text_encoder.cpu()
        del pipe

    def train(self, mode: bool = True):
        super().train(mode)
        if not any(param.requires_grad for param in self.vae.parameters()):
            self.vae.eval()
        return self

    @property
    def num_train_timesteps(self) -> int:
        return int(self.noise_scheduler.config.num_train_timesteps)

    def trainable_parameters(self):
        for param in self.condition_projector.parameters():
            if param.requires_grad:
                yield param

        for param in self.unet.parameters():
            if param.requires_grad:
                yield param

    def encode_images(self, images: torch.Tensor) -> torch.Tensor:
        images = images.to(dtype=next(self.vae.parameters()).dtype)
        posterior = self.vae.encode(images).latent_dist
        return posterior.sample() * self.latent_scaling_factor

    def decode_latents(self, latents: torch.Tensor) -> torch.Tensor:
        latents = latents.to(dtype=next(self.vae.parameters()).dtype)
        images = self.vae.decode(latents / self.latent_scaling_factor).sample
        return images.float().clamp(-1, 1)

    def add_noise(self, latents: torch.Tensor, noise: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        return self.noise_scheduler.add_noise(latents, noise, timesteps)

    def predict_noise(
        self,
        noisy_latents: torch.Tensor,
        timesteps: torch.Tensor,
        content: torch.Tensor,
        style: torch.Tensor,
        style_weight: float | None = None,
    ) -> torch.Tensor:
        cond = self.condition_projector(content.float(), style.float(), style_weight=style_weight)
        cond = cond.to(device=noisy_latents.device, dtype=noisy_latents.dtype)
        return self.unet(noisy_latents, timesteps, encoder_hidden_states=cond).sample

    def predict_x0(self, noisy_latents: torch.Tensor, pred_noise: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        alpha_prod = self.noise_scheduler.alphas_cumprod.to(device=noisy_latents.device, dtype=torch.float32)
        alpha_t = alpha_prod[timesteps].view(-1, 1, 1, 1)
        sqrt_alpha_t = torch.sqrt(alpha_t.clamp_min(1e-6))
        sqrt_one_minus_alpha_t = torch.sqrt((1 - alpha_t).clamp_min(1e-6))

        noisy_latents_f = noisy_latents.float()
        pred_noise_f = pred_noise.float()
        pred_x0 = (noisy_latents_f - sqrt_one_minus_alpha_t * pred_noise_f) / sqrt_alpha_t
        return pred_x0.to(dtype=noisy_latents.dtype)

    @torch.no_grad()
    def generate(
        self,
        content: torch.Tensor,
        style: torch.Tensor,
        num_inference_steps: int = 50,
        strength: float = 0.75,
        style_weight: float = 0.75,
    ) -> torch.Tensor:
        self.eval()
        self.noise_scheduler.set_timesteps(num_inference_steps, device=content.device)
        content_latents = self.encode_images(content)
        noise = torch.randn_like(content_latents)
        start_idx = max(num_inference_steps - int(num_inference_steps * strength), 0)
        start_idx = min(start_idx, num_inference_steps - 1)
        start_timestep = self.noise_scheduler.timesteps[start_idx].repeat(content.shape[0])
        latents = self.add_noise(content_latents, noise, start_timestep)

        for timestep in self.noise_scheduler.timesteps[start_idx:]:
            t = timestep.expand(content.shape[0])
            pred_noise = self.predict_noise(latents, t, content, style, style_weight=style_weight)
            latents = self.noise_scheduler.step(pred_noise, timestep, latents).prev_sample

        return self.decode_latents(latents)

    def save_training_checkpoint(self, path: str | Path, extra: dict[str, Any] | None = None) -> None:
        checkpoint: dict[str, Any] = {
            "condition_projector": self.condition_projector.state_dict(),
            "noise_scheduler": self.noise_scheduler.config,
            "latent_scaling_factor": self.latent_scaling_factor,
            "unet_mode": self.unet_mode,
            "unet": self.unet.state_dict(),
        }

        if extra:
            checkpoint.update(extra)
        torch.save(checkpoint, path)

    def load_training_checkpoint(self, path: str | Path, map_location: str | torch.device = "cpu") -> dict[str, Any]:
        checkpoint = torch.load(path, map_location=map_location)

        if "condition_projector" in checkpoint:
            self.condition_projector.load_state_dict(checkpoint["condition_projector"])

        if "unet" in checkpoint:
            self.unet.load_state_dict(checkpoint["unet"], strict=False)
            self.unet_mode = str(checkpoint.get("unet_mode", self.unet_mode))
        else:
            print("warning: checkpoint has no UNet weights; keeping current UNet parameters")

        return checkpoint
