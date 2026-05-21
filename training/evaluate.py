"""Three-layer evaluation per PROJECT_SPEC §4.5.

Layer 1: pixel/perceptual metrics on a synthetic test set (PSNR, SSIM, LPIPS).
    Diagnostic only — not the goal.

Layer 2: type-classification recovery rate.
    Take well-preserved OCRE coins with known type, synthetically weather them,
    restore via the backend, run CLIP retrieval against per-type exemplars on
    both (weathered) and (restored). Restoration is useful iff top-k accuracy
    improves. This is the operational metric for Phase A.

Layer 3: numismatist evaluation.
    Out of scope for this script — runs as a designed pilot study post-launch.
    See PROJECT_SPEC.md §4.5(3) and the protocol in `notebooks/curator_pilot.md`
    once it exists.

Usage:
    python -m training.evaluate \\
        --backend http://localhost:7860 \\
        --eval_set data/filtered_eval \\
        --output runs/eval-v0.1
"""

from __future__ import annotations

import argparse
import base64
import csv
import io
import json
from collections import defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import requests
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from tqdm.auto import tqdm

from training.synthetic_weathering import WeatherParams, weather


@dataclass
class EvalResult:
    n: int
    psnr_weathered_vs_pristine: float
    psnr_restored_vs_pristine: float
    ssim_weathered_vs_pristine: float
    ssim_restored_vs_pristine: float
    lpips_weathered_vs_pristine: float | None
    lpips_restored_vs_pristine: float | None
    top1_weathered: float
    top1_restored: float
    top5_weathered: float
    top5_restored: float
    # The operational metric: relative improvement in top-1 recovery.
    top1_lift: float


# --------------------------------------------------------------------------- #
# Backend client                                                              #
# --------------------------------------------------------------------------- #


def call_backend(backend_url: str, image: Image.Image, mask: np.ndarray) -> Image.Image:
    """POST to /restore, return the first candidate as a PIL image."""
    req = {
        "image": _to_b64(image),
        "mask": _to_b64(Image.fromarray(mask, mode="L")),
        "n_candidates": 1,
        "seed": 0,
    }
    resp = requests.post(f"{backend_url}/restore", json=req, timeout=180)
    resp.raise_for_status()
    cand = resp.json()["candidates"][0]
    return _from_b64(cand["image"])


def _to_b64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _from_b64(s: str) -> Image.Image:
    return Image.open(io.BytesIO(base64.b64decode(s))).convert("RGB")


# --------------------------------------------------------------------------- #
# Layer 1: pixel/perceptual                                                   #
# --------------------------------------------------------------------------- #


def pixel_metrics(a: Image.Image, b: Image.Image) -> tuple[float, float]:
    a_arr = np.asarray(a.resize(b.size)).astype(np.float32)
    b_arr = np.asarray(b).astype(np.float32)
    psnr = peak_signal_noise_ratio(b_arr, a_arr, data_range=255)
    ssim = structural_similarity(b_arr, a_arr, channel_axis=-1, data_range=255)
    return float(psnr), float(ssim)


def lpips_distance(a: Image.Image, b: Image.Image, model=None) -> float | None:
    """Lazy-imports lpips so eval can run without it (it pulls torch)."""
    try:
        import lpips
        import torch
    except ImportError:
        return None
    if model is None:
        model = lpips.LPIPS(net="alex", verbose=False)
    def _t(img: Image.Image) -> "torch.Tensor":
        arr = np.asarray(img.resize((256, 256))).astype(np.float32) / 127.5 - 1
        return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).float()
    with torch.no_grad():
        d = model(_t(a), _t(b)).item()
    return float(d)


# --------------------------------------------------------------------------- #
# Layer 2: CLIP retrieval as the operational metric                           #
# --------------------------------------------------------------------------- #


class TypeRetriever:
    """CLIP-based nearest-exemplar retriever.

    For Phase A we use OpenCLIP off-the-shelf with no fine-tuning. This is
    deliberately weak: if restoration helps top-k accuracy here, it'll help
    more with a fine-tuned classifier. If it doesn't help here, fine-tuning
    won't save it either.
    """

    def __init__(self, exemplars: dict[str, list[Image.Image]], device: str = "cpu") -> None:
        import open_clip
        import torch
        self.torch = torch
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained="openai"
        )
        self.model.to(device).eval()
        self.device = device

        # Per-type prototype = mean of exemplar embeddings.
        self.type_ids: list[str] = []
        proto_list = []
        for type_id, imgs in exemplars.items():
            embs = self._embed(imgs)
            proto = embs.mean(dim=0, keepdim=True)
            proto = proto / proto.norm(dim=-1, keepdim=True)
            proto_list.append(proto)
            self.type_ids.append(type_id)
        self.prototypes = torch.cat(proto_list, dim=0)  # [n_types, dim]

    def _embed(self, imgs: list[Image.Image]):
        torch = self.torch
        batch = torch.stack([self.preprocess(i) for i in imgs]).to(self.device)
        with torch.no_grad():
            emb = self.model.encode_image(batch)
        return emb / emb.norm(dim=-1, keepdim=True)

    def topk(self, img: Image.Image, k: int = 5) -> list[str]:
        emb = self._embed([img])
        sims = (emb @ self.prototypes.T).squeeze(0)
        idx = sims.topk(k).indices.tolist()
        return [self.type_ids[i] for i in idx]


# --------------------------------------------------------------------------- #
# Driver                                                                      #
# --------------------------------------------------------------------------- #


def run_eval(
    eval_set: Path,
    metadata_csv: Path,
    backend_url: str,
    output: Path,
    n_per_type: int = 5,
    severity: float = 0.6,
    seed: int = 0,
) -> EvalResult:
    output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    # Build (type → image paths) index.
    types: dict[str, list[Path]] = defaultdict(list)
    with metadata_csv.open() as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            for side in ("obverse_path", "reverse_path"):
                rel = row.get(side, "")
                if not rel:
                    continue
                p = eval_set / rel
                if p.exists():
                    types[row["type_uri"]].append(p)

    types = {t: ps for t, ps in types.items() if len(ps) >= 2}
    if len(types) < 5:
        raise SystemExit(f"Only {len(types)} types have ≥2 specimens — need a larger eval set.")

    # Hold out one specimen per type as the eval target; rest become exemplars.
    exemplars: dict[str, list[Image.Image]] = {}
    targets: list[tuple[str, Path]] = []
    for t, paths in types.items():
        rng.shuffle(paths)
        targets.append((t, paths[0]))
        exemplars[t] = [Image.open(p).convert("RGB") for p in paths[1:1 + n_per_type]]

    retriever = TypeRetriever(exemplars)

    rows = []
    psnr_w, psnr_r = [], []
    ssim_w, ssim_r = [], []
    lpips_w, lpips_r = [], []
    top1_w = top1_r = top5_w = top5_r = 0
    for t, path in tqdm(targets, desc="eval"):
        pristine = Image.open(path).convert("RGB").resize((512, 512))
        weathered_arr, mask = weather(pristine, WeatherParams(
            mechanical=severity, corrosion=severity, patination=severity, seed=seed,
        ))
        weathered = Image.fromarray(weathered_arr)
        try:
            restored = call_backend(backend_url, weathered, mask)
        except requests.RequestException as e:
            print(f"backend failed for {path.name}: {e}")
            continue

        pw, sw = pixel_metrics(weathered, pristine)
        pr, sr = pixel_metrics(restored, pristine)
        lw = lpips_distance(weathered, pristine)
        lr = lpips_distance(restored, pristine)
        topw = retriever.topk(weathered, k=5)
        topr = retriever.topk(restored, k=5)
        rows.append({
            "path": str(path), "type": t,
            "psnr_w": pw, "psnr_r": pr, "ssim_w": sw, "ssim_r": sr,
            "lpips_w": lw, "lpips_r": lr,
            "top1_w_hit": int(topw[0] == t), "top1_r_hit": int(topr[0] == t),
            "top5_w_hit": int(t in topw), "top5_r_hit": int(t in topr),
        })
        psnr_w.append(pw); psnr_r.append(pr)
        ssim_w.append(sw); ssim_r.append(sr)
        if lw is not None: lpips_w.append(lw)
        if lr is not None: lpips_r.append(lr)
        top1_w += rows[-1]["top1_w_hit"]; top1_r += rows[-1]["top1_r_hit"]
        top5_w += rows[-1]["top5_w_hit"]; top5_r += rows[-1]["top5_r_hit"]

    n = len(rows)
    if n == 0:
        raise SystemExit("No eval rows produced — every backend call failed.")
    res = EvalResult(
        n=n,
        psnr_weathered_vs_pristine=float(np.mean(psnr_w)),
        psnr_restored_vs_pristine=float(np.mean(psnr_r)),
        ssim_weathered_vs_pristine=float(np.mean(ssim_w)),
        ssim_restored_vs_pristine=float(np.mean(ssim_r)),
        lpips_weathered_vs_pristine=float(np.mean(lpips_w)) if lpips_w else None,
        lpips_restored_vs_pristine=float(np.mean(lpips_r)) if lpips_r else None,
        top1_weathered=top1_w / n,
        top1_restored=top1_r / n,
        top5_weathered=top5_w / n,
        top5_restored=top5_r / n,
        top1_lift=(top1_r - top1_w) / max(1, top1_w) if top1_w else float(top1_r > 0),
    )

    (output / "rows.json").write_text(json.dumps(rows, indent=2))
    (output / "summary.json").write_text(json.dumps(asdict(res), indent=2))
    print(json.dumps(asdict(res), indent=2))
    return res


def _cli() -> None:
    p = argparse.ArgumentParser(description="Evaluate restoration quality and classification lift.")
    p.add_argument("--backend", default="http://localhost:7860")
    p.add_argument("--eval_set", type=Path, required=True,
                   help="Directory containing the filtered eval images.")
    p.add_argument("--metadata", type=Path, default=None,
                   help="Path to metadata.csv (default: <eval_set>/metadata.csv).")
    p.add_argument("--output", type=Path, default=Path("runs/eval"))
    p.add_argument("--n_per_type", type=int, default=5)
    p.add_argument("--severity", type=float, default=0.6)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    meta = args.metadata or (args.eval_set / "metadata.csv")
    run_eval(args.eval_set, meta, args.backend, args.output,
             n_per_type=args.n_per_type, severity=args.severity, seed=args.seed)


if __name__ == "__main__":
    _cli()
