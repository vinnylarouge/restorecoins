"""Train the weathered-region mask proposer (PROJECT_SPEC §4.3, M6).

A small UNet predicts P(weathered | pixel). Training data is free: the
`synthetic_weathering` module emits ground-truth masks alongside each
weathered image, so we get supervised labels without any human annotation.

At inference time the frontend converts the probability map to a binary mask at
a user-adjustable threshold; users can paint over it to refine.

Wall-clock: ~1-2h on an M-series Mac, ~20 min on an A100, for 5k steps at 512².
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from training.datasets import WeatheredCoinPairs


# --------------------------------------------------------------------------- #
# Small UNet — ~2M params, plenty for binary segmentation at 512².            #
# --------------------------------------------------------------------------- #


class _DoubleConv(nn.Module):
    def __init__(self, in_c: int, out_c: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1, bias=False),
            nn.GroupNorm(8, out_c),
            nn.SiLU(),
            nn.Conv2d(out_c, out_c, 3, padding=1, bias=False),
            nn.GroupNorm(8, out_c),
            nn.SiLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MaskProposer(nn.Module):
    """3-channel input (RGB), 1-channel logits output."""

    def __init__(self, base: int = 32) -> None:
        super().__init__()
        self.e1 = _DoubleConv(3, base)
        self.e2 = _DoubleConv(base, base * 2)
        self.e3 = _DoubleConv(base * 2, base * 4)
        self.e4 = _DoubleConv(base * 4, base * 8)
        self.bottleneck = _DoubleConv(base * 8, base * 8)
        self.up3 = nn.ConvTranspose2d(base * 8, base * 4, 2, stride=2)
        self.d3 = _DoubleConv(base * 8, base * 4)
        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.d2 = _DoubleConv(base * 4, base * 2)
        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.d1 = _DoubleConv(base * 2, base)
        self.head = nn.Conv2d(base, 1, 1)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.e1(x)
        e2 = self.e2(self.pool(e1))
        e3 = self.e3(self.pool(e2))
        e4 = self.e4(self.pool(e3))
        b = self.bottleneck(self.pool(e4))
        # Upsample back to e4's resolution before concat — fixes any odd-input drift.
        b = F.interpolate(b, size=e4.shape[-2:], mode="nearest")
        d3 = self.d3(torch.cat([self.up3(b), e3], dim=1))
        d2 = self.d2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.d1(torch.cat([self.up1(d2), e1], dim=1))
        return self.head(d1)


# --------------------------------------------------------------------------- #
# Training loop                                                               #
# --------------------------------------------------------------------------- #


@dataclass
class ProposerConfig:
    data_root: Path
    output_dir: Path
    resolution: int = 512
    steps: int = 5000
    batch_size: int = 8
    lr: float = 2e-4
    seed: int = 0
    device: str = "auto"


def train(cfg: ProposerConfig) -> None:
    device = _pick_device(cfg.device)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    dataset = WeatheredCoinPairs(cfg.data_root, resolution=cfg.resolution, seed=cfg.seed)
    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True,
                        num_workers=2, drop_last=True, persistent_workers=True)

    model = MaskProposer().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.steps)

    # Mask is heavily imbalanced (most pixels unweathered) — pos_weight matters.
    # The factor is calibrated from the synthetic weathering's mean coverage.
    pos_weight = torch.tensor([3.0], device=device)
    bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    pbar = tqdm(total=cfg.steps, desc="MaskProposer")
    step = 0
    while step < cfg.steps:
        for batch in loader:
            x = batch["weathered"].to(device)
            y = batch["mask"].to(device)
            logits = model(x)
            loss = bce(logits, y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            step += 1
            pbar.update(1)
            pbar.set_postfix(loss=f"{loss.item():.4f}")
            if step >= cfg.steps:
                break

    out_path = cfg.output_dir / "mask_proposer.pt"
    torch.save({"state_dict": model.state_dict(), "config": vars(cfg)}, out_path)
    print(f"Saved {out_path}")


def _pick_device(name: str) -> str:
    if name != "auto":
        return name
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _cli() -> None:
    p = argparse.ArgumentParser(description="Train the weathered-region mask proposer.")
    p.add_argument("--data_root", type=Path, required=True)
    p.add_argument("--output_dir", type=Path, required=True)
    p.add_argument("--resolution", type=int, default=512)
    p.add_argument("--steps", type=int, default=5000)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"])
    args = p.parse_args()
    train(ProposerConfig(**vars(args)))


if __name__ == "__main__":
    _cli()
