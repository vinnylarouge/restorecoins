"""Fine-tune a coin-domain LoRA on SDXL-inpaint (PROJECT_SPEC §4.4).

Forward model: `weathered + mask → pristine` (target).
Inverse model: SDXL-inpaint with our LoRA learns the inverse.

Wall-clock: ~6-12h on a single A100 for ~2000 steps at 1024² (per spec §4.4).
This script is not designed to be run by Claude Code in-session — it expects a
real GPU and a fully-pulled SDXL-inpaint checkpoint. Run on a Lambda/RunPod box.

Usage:
    accelerate launch -m training.train_lora \\
        --data_root data/filtered \\
        --output_dir runs/lora-v0.1 \\
        --rank 32 --steps 2000 --batch_size 1
"""

from __future__ import annotations

import argparse
import math
import os
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

# Diffusers/PEFT imports are local to keep the rest of the module light to import.

from training.datasets import WeatheredCoinPairs


SDXL_INPAINT_ID = "diffusers/stable-diffusion-xl-1.0-inpainting-0.1"


@dataclass
class TrainConfig:
    data_root: Path
    output_dir: Path
    model_id: str = SDXL_INPAINT_ID
    rank: int = 32
    steps: int = 2000
    batch_size: int = 1
    resolution: int = 1024
    lr: float = 1e-4
    grad_accum: int = 4
    mixed_precision: str = "fp16"
    save_every: int = 500
    seed: int = 0
    push_to_hub: str | None = None  # e.g. "vinnylarouge/restorecoins-lora-v0.1"


def train(cfg: TrainConfig) -> None:
    from accelerate import Accelerator
    from accelerate.utils import set_seed
    from diffusers import (
        AutoencoderKL,
        StableDiffusionXLInpaintPipeline,
        UNet2DConditionModel,
        DDPMScheduler,
    )
    from peft import LoraConfig, get_peft_model_state_dict, inject_adapter_in_model
    from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer
    from safetensors.torch import save_file

    set_seed(cfg.seed)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    accelerator = Accelerator(
        gradient_accumulation_steps=cfg.grad_accum,
        mixed_precision=cfg.mixed_precision,
        log_with="tensorboard",
        project_dir=str(cfg.output_dir / "tb"),
    )

    # Load the full pipeline once to grab its sub-modules, then drop the pipeline.
    pipe = StableDiffusionXLInpaintPipeline.from_pretrained(
        cfg.model_id, torch_dtype=torch.float32
    )
    vae: AutoencoderKL = pipe.vae
    unet: UNet2DConditionModel = pipe.unet
    text1: CLIPTextModel = pipe.text_encoder
    text2: CLIPTextModelWithProjection = pipe.text_encoder_2
    tok1: CLIPTokenizer = pipe.tokenizer
    tok2: CLIPTokenizer = pipe.tokenizer_2
    sched: DDPMScheduler = DDPMScheduler.from_config(pipe.scheduler.config)
    del pipe

    # Freeze everything except the LoRA we inject into the UNet's attention blocks.
    for m in (vae, text1, text2):
        m.requires_grad_(False)
    unet.requires_grad_(False)

    lora_cfg = LoraConfig(
        r=cfg.rank,
        lora_alpha=cfg.rank,
        init_lora_weights="gaussian",
        # Inject into attention projections only — matches the diffusers LoRA recipe
        # and keeps trainable params <1% of the full UNet.
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
    )
    unet = inject_adapter_in_model(lora_cfg, unet)
    trainable = [p for p in unet.parameters() if p.requires_grad]

    optimizer = torch.optim.AdamW(trainable, lr=cfg.lr, weight_decay=1e-2)
    dataset = WeatheredCoinPairs(cfg.data_root, resolution=cfg.resolution, seed=cfg.seed)
    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True,
                        num_workers=2, drop_last=True, persistent_workers=True)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.steps)

    unet, optimizer, loader, scheduler = accelerator.prepare(unet, optimizer, loader, scheduler)
    vae.to(accelerator.device, dtype=torch.float32)
    text1.to(accelerator.device); text2.to(accelerator.device)

    # SDXL needs a prompt for both text encoders. Coins don't have a natural
    # caption, so we use a neutral domain-anchoring prompt and let the LoRA do
    # the visual specialisation. This is a deliberate choice: training with
    # synthetic captions risks the LoRA learning a captioning artifact.
    domain_prompt = "an ancient coin, museum photograph, sharp detail"
    prompt_embeds, pooled = _encode_prompt(domain_prompt, text1, text2, tok1, tok2, accelerator.device)

    global_step = 0
    pbar = tqdm(total=cfg.steps, disable=not accelerator.is_main_process, desc="LoRA")
    while global_step < cfg.steps:
        for batch in loader:
            with accelerator.accumulate(unet):
                pristine = batch["pristine"].to(accelerator.device) * 2 - 1   # [-1, 1] for VAE
                weathered = batch["weathered"].to(accelerator.device) * 2 - 1
                mask = batch["mask"].to(accelerator.device)

                with torch.no_grad():
                    latents_p = vae.encode(pristine).latent_dist.sample() * vae.config.scaling_factor
                    latents_w = vae.encode(weathered).latent_dist.sample() * vae.config.scaling_factor
                    # SDXL-inpaint concatenates: noisy_latents | mask (downsampled) | masked_image_latents
                    mask_small = F.interpolate(mask, size=latents_p.shape[-2:], mode="nearest")
                    masked_input = latents_w  # weathered = the "masked image" in our setup

                noise = torch.randn_like(latents_p)
                timesteps = torch.randint(0, sched.config.num_train_timesteps,
                                          (latents_p.shape[0],), device=latents_p.device).long()
                noisy = sched.add_noise(latents_p, noise, timesteps)

                inp = torch.cat([noisy, mask_small, masked_input], dim=1)
                # SDXL needs `added_cond_kwargs` with text_embeds + time_ids.
                add_time_ids = _build_time_ids(cfg.resolution, latents_p.shape[0], accelerator.device)
                pred = unet(
                    inp, timesteps,
                    encoder_hidden_states=prompt_embeds.expand(latents_p.shape[0], -1, -1),
                    added_cond_kwargs={
                        "text_embeds": pooled.expand(latents_p.shape[0], -1),
                        "time_ids": add_time_ids,
                    },
                ).sample

                if sched.config.prediction_type == "v_prediction":
                    target = sched.get_velocity(latents_p, noise, timesteps)
                else:
                    target = noise

                loss = F.mse_loss(pred.float(), target.float())
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(trainable, 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            if accelerator.sync_gradients:
                global_step += 1
                pbar.update(1)
                if accelerator.is_main_process and global_step % 50 == 0:
                    accelerator.log({"loss": loss.item(), "lr": scheduler.get_last_lr()[0]},
                                    step=global_step)
                if accelerator.is_main_process and global_step % cfg.save_every == 0:
                    _save_lora(unet, cfg.output_dir / f"step_{global_step}")
                if global_step >= cfg.steps:
                    break

    if accelerator.is_main_process:
        _save_lora(unet, cfg.output_dir / "final")
        if cfg.push_to_hub:
            from huggingface_hub import HfApi
            HfApi().upload_folder(
                folder_path=str(cfg.output_dir / "final"),
                repo_id=cfg.push_to_hub,
                repo_type="model",
            )


def _encode_prompt(prompt, text1, text2, tok1, tok2, device):
    """SDXL prompt encoding: concat of CLIP-L hidden + CLIP-G hidden, plus a pooled vector."""
    with torch.no_grad():
        ids1 = tok1(prompt, padding="max_length", max_length=tok1.model_max_length,
                    truncation=True, return_tensors="pt").input_ids.to(device)
        ids2 = tok2(prompt, padding="max_length", max_length=tok2.model_max_length,
                    truncation=True, return_tensors="pt").input_ids.to(device)
        e1 = text1(ids1, output_hidden_states=True).hidden_states[-2]
        out2 = text2(ids2, output_hidden_states=True)
        e2 = out2.hidden_states[-2]
        pooled = out2[0]
    return torch.cat([e1, e2], dim=-1), pooled


def _build_time_ids(resolution: int, bsz: int, device) -> torch.Tensor:
    """SDXL's `time_ids`: (orig_h, orig_w, crop_top, crop_left, target_h, target_w)."""
    ids = torch.tensor([[resolution, resolution, 0, 0, resolution, resolution]],
                       dtype=torch.float32, device=device)
    return ids.expand(bsz, -1)


def _save_lora(unet, out_dir: Path) -> None:
    from peft import get_peft_model_state_dict
    from safetensors.torch import save_file

    out_dir.mkdir(parents=True, exist_ok=True)
    state = get_peft_model_state_dict(unet)
    save_file(state, str(out_dir / "lora.safetensors"))
    (out_dir / "metadata.json").write_text(
        '{"base_model": "%s", "format": "diffusers-lora"}' % SDXL_INPAINT_ID
    )


def _cli() -> None:
    p = argparse.ArgumentParser(description="LoRA fine-tune SDXL-inpaint on coins.")
    p.add_argument("--data_root", type=Path, required=True, help="Dir of clean coin images.")
    p.add_argument("--output_dir", type=Path, required=True)
    p.add_argument("--model_id", default=SDXL_INPAINT_ID)
    p.add_argument("--rank", type=int, default=32)
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--resolution", type=int, default=1024)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--grad_accum", type=int, default=4)
    p.add_argument("--mixed_precision", default="fp16",
                   choices=["no", "fp16", "bf16"])
    p.add_argument("--save_every", type=int, default=500)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--push_to_hub", default=None,
                   help="Optional HF repo to push the final LoRA to.")
    args = p.parse_args()
    train(TrainConfig(**vars(args)))


if __name__ == "__main__":
    _cli()
