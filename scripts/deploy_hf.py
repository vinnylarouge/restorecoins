#!/usr/bin/env python3
"""One-shot HuggingFace deployment for restorecoins.

Run this AFTER `hf auth login`. It:
  1. Creates the LoRA model repo (vinnylarouge/restorecoins-lora-v0.X) and
     uploads the latest trained checkpoint.
  2. Creates the HF Space (vinnylarouge/restorecoins) configured as a Docker
     SDK Space with ZeroGPU.
  3. Uploads the backend code to the Space.
  4. Sets the Space's secret/env-vars so the backend loads the right LoRA.
  5. Pushes the HF_TOKEN to the GitHub repo as a secret so future workflow
     runs deploy automatically on every push.

Usage:
    python scripts/deploy_hf.py                  # auto-detect best LoRA
    python scripts/deploy_hf.py --lora_version v0.2
    python scripts/deploy_hf.py --dry_run
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OWNER = "vinnylarouge"
DEFAULT_SPACE = "restorecoins"
DEFAULT_LORA_BASE = "restorecoins-lora"


def latest_lora_dir() -> tuple[Path, str]:
    """Find the most recent finalized LoRA in runs/. Returns (path, version_tag)."""
    candidates = []
    runs = REPO_ROOT / "runs"
    if not runs.exists():
        raise SystemExit(f"No runs/ directory under {REPO_ROOT}; train first.")
    for d in runs.iterdir():
        if not d.is_dir() or not d.name.startswith("lora"):
            continue
        final = d / "final"
        if (final / "pytorch_lora_weights.safetensors").exists():
            candidates.append((final, d.name.split("-")[-1]))  # e.g. v0.1
    if not candidates:
        raise SystemExit("No finalized LoRA found. Expected runs/lora-*/final/pytorch_lora_weights.safetensors")
    candidates.sort(key=lambda p_v: p_v[0].stat().st_mtime, reverse=True)
    return candidates[0]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--owner", default=DEFAULT_OWNER)
    p.add_argument("--space_name", default=DEFAULT_SPACE)
    p.add_argument("--lora_base", default=DEFAULT_LORA_BASE)
    p.add_argument("--lora_version", default=None,
                   help="e.g. v0.2; default = newest from runs/.")
    p.add_argument("--lora_dir", type=Path, default=None,
                   help="Override the path to the LoRA checkpoint dir.")
    p.add_argument("--skip_lora", action="store_true",
                   help="Skip LoRA upload (assume already on Hub).")
    p.add_argument("--skip_space", action="store_true",
                   help="Skip Space creation/push (just upload the LoRA).")
    p.add_argument("--skip_github_secret", action="store_true",
                   help="Don't push HF_TOKEN as GitHub secret.")
    p.add_argument("--dry_run", action="store_true")
    args = p.parse_args()

    from huggingface_hub import HfApi
    api = HfApi()
    try:
        user = api.whoami()["name"]
    except Exception as e:
        raise SystemExit(f"Not logged into HF. Run `hf auth login` first.\n{e}")
    print(f"[deploy_hf] logged in as: {user}")

    # --- LoRA model repo ---------------------------------------------------- #

    if args.skip_lora:
        if not args.lora_version:
            raise SystemExit("--skip_lora requires --lora_version")
        lora_repo = f"{args.owner}/{args.lora_base}-{args.lora_version}"
    else:
        if args.lora_dir:
            lora_dir = args.lora_dir
            version = args.lora_version or "manual"
        else:
            lora_dir, version = latest_lora_dir()
            if args.lora_version:
                version = args.lora_version
        print(f"[deploy_hf] LoRA source: {lora_dir} (version {version})")
        lora_repo = f"{args.owner}/{args.lora_base}-{version}"

        if args.dry_run:
            print(f"[dry_run] would: create_repo {lora_repo} (model)")
            print(f"[dry_run] would: upload_folder {lora_dir} -> {lora_repo}")
        else:
            api.create_repo(repo_id=lora_repo, repo_type="model", exist_ok=True, private=False)
            (lora_dir / "README.md").write_text(
                f"# restorecoins LoRA {version}\n\n"
                f"Coin-domain SDXL-inpaint LoRA, see "
                f"https://github.com/{args.owner}/restorecoins.\n\n"
                f"License: CreativeML Open RAIL-M (matches SDXL upstream).\n"
            )
            api.upload_folder(folder_path=str(lora_dir), repo_id=lora_repo,
                              repo_type="model", commit_message=f"upload {version}")
            print(f"[deploy_hf] uploaded LoRA to https://huggingface.co/{lora_repo}")

    # --- Space -------------------------------------------------------------- #

    space_repo = f"{args.owner}/{args.space_name}"
    if args.skip_space:
        print(f"[deploy_hf] skipping space step (use --skip_space=false to enable)")
    else:
        if args.dry_run:
            print(f"[dry_run] would: create_repo {space_repo} (space, docker)")
            print(f"[dry_run] would: upload_folder backend/ + training/ -> {space_repo}")
            print(f"[dry_run] would: set space secrets: RESTORECOINS_LORA_ID={lora_repo}")
        else:
            api.create_repo(repo_id=space_repo, repo_type="space",
                            space_sdk="docker", exist_ok=True, private=False)
            api.upload_folder(folder_path=str(REPO_ROOT / "backend"),
                              repo_id=space_repo, repo_type="space",
                              commit_message=f"backend@{_git_short_sha()}")
            api.upload_folder(folder_path=str(REPO_ROOT / "training"),
                              path_in_repo="training", repo_id=space_repo, repo_type="space",
                              allow_patterns=["synthetic_weathering.py", "datasets.py",
                                              "train_mask_proposer.py", "__init__.py"],
                              commit_message=f"training utils@{_git_short_sha()}")
            # Variables (visible) vs secrets (hidden). Model IDs are not secret.
            api.add_space_variable(repo_id=space_repo, key="RESTORECOINS_MODE", value="real")
            api.add_space_variable(repo_id=space_repo, key="RESTORECOINS_LORA_ID", value=lora_repo)
            print(f"[deploy_hf] Space pushed: https://huggingface.co/spaces/{space_repo}")
            print(f"[deploy_hf] First build will take ~5-10 min (downloading SDXL weights).")

    # --- GitHub secret ------------------------------------------------------ #

    if not args.skip_github_secret:
        token = _read_hf_token()
        if token and not args.dry_run:
            try:
                subprocess.run(
                    ["gh", "secret", "set", "HF_TOKEN", "-b", token,
                     "-R", f"{args.owner}/{args.space_name}"],
                    check=True, capture_output=True,
                )
                print(f"[deploy_hf] HF_TOKEN set as GitHub secret on {args.owner}/{args.space_name}")
            except subprocess.CalledProcessError as e:
                print(f"[deploy_hf] WARN: could not set GitHub secret: "
                      f"{e.stderr.decode()[:200]}")
        elif not token:
            print(f"[deploy_hf] WARN: no HF token in ~/.cache/huggingface/token; "
                  f"set HF_TOKEN GitHub secret manually for auto-deploy.")


def _git_short_sha() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                              cwd=REPO_ROOT, capture_output=True, text=True,
                              check=True, timeout=3).stdout.strip()
    except Exception:
        return "unknown"


def _read_hf_token() -> str | None:
    """Read the user's stored HF token to mirror it to GitHub secrets."""
    for path in [Path.home() / ".cache" / "huggingface" / "token",
                 Path.home() / ".huggingface" / "token"]:
        if path.exists():
            return path.read_text().strip()
    return os.environ.get("HF_TOKEN")


if __name__ == "__main__":
    main()
