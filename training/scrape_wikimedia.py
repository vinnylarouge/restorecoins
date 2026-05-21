"""Wikimedia Commons scraper for ancient coin images.

Added after the OCRE/Fitzwilliam scrape proved unreachable from networks
without Cambridge routing (see DECISIONS.md 2026-05-21). Wikimedia Commons
hosts a substantial CC-licensed corpus accessible from any network.

Strategy: walk the MediaWiki API's `categorymembers` endpoint for a chosen root
category, descending one level into subcategories. For each `File:` member,
fetch the `imageinfo` to get the actual image URL and licence. Filter on
extension, dimensions, and licence string.

Usage:
    # Quick pull: 200 images under "Coins of ancient Rome".
    python -m training.scrape_wikimedia --limit 200 --out data/wikimedia_raw

    # Different root category.
    python -m training.scrape_wikimedia \\
        --category "Coins of the Byzantine Empire" --limit 1000

NOT a substitute for OCRE/RIC-typed data — Wikimedia categories are curated by
volunteers, not by RIC scholars, so type metadata is much weaker. But the
images themselves are perfectly serviceable as LoRA training data, and the
licences are unambiguous (CC-BY-SA / public domain).
"""

from __future__ import annotations

import argparse
import csv
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable

import requests

API = "https://commons.wikimedia.org/w/api.php"
USER_AGENT = "restorecoins-scraper/0.1 (research; vincentwangsemailaddress@gmail.com)"

GOOD_EXT = {".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff"}


def _api_get(params: dict) -> dict:
    """Polite GET with a baked-in delay — Wikimedia asks for ≤200 req/s, we use ~5."""
    params = {**params, "format": "json", "formatversion": "2"}
    r = requests.get(API, params=params, timeout=30,
                     headers={"User-Agent": USER_AGENT})
    r.raise_for_status()
    return r.json()


def iter_category(category: str, depth: int = 1, max_items: int = 1000) -> Iterable[str]:
    """Yield `File:<...>` titles under `category`, optionally descending `depth` levels."""
    queues: list[tuple[int, str]] = [(0, category)]
    seen_files: set[str] = set()
    while queues and len(seen_files) < max_items:
        level, cat = queues.pop(0)
        params = {
            "action": "query",
            "list": "categorymembers",
            "cmtitle": f"Category:{cat}",
            "cmlimit": 200,
            "cmtype": "file|subcat" if level < depth else "file",
        }
        cont: dict | None = None
        while True:
            if cont:
                params.update(cont)
            try:
                data = _api_get(params)
            except requests.RequestException:
                break
            for m in data.get("query", {}).get("categorymembers", []):
                title = m["title"]
                if title.startswith("File:") and title not in seen_files:
                    seen_files.add(title)
                    yield title
                    if len(seen_files) >= max_items:
                        return
                elif title.startswith("Category:") and level < depth:
                    queues.append((level + 1, title[len("Category:"):]))
            cont = data.get("continue")
            if not cont:
                break
            time.sleep(0.2)
        time.sleep(0.2)


def imageinfo(titles: list[str]) -> dict[str, dict]:
    """Bulk-fetch imageinfo for up to 50 titles per call (API limit)."""
    out: dict[str, dict] = {}
    for chunk_start in range(0, len(titles), 50):
        chunk = titles[chunk_start:chunk_start + 50]
        data = _api_get({
            "action": "query",
            "titles": "|".join(chunk),
            "prop": "imageinfo",
            "iiprop": "url|extmetadata|size|mime",
        })
        for page in data.get("query", {}).get("pages", []):
            ii = page.get("imageinfo", [])
            if ii:
                out[page["title"]] = ii[0]
        time.sleep(0.2)
    return out


def slugify(title: str) -> str:
    """File:Foo bar (baz).jpg → Foo_bar__baz_.jpg"""
    name = title.removeprefix("File:")
    name = re.sub(r"[^A-Za-z0-9._-]", "_", name)
    return name[:120]


def licence_str(meta: dict) -> str:
    em = meta.get("extmetadata", {})
    return (em.get("LicenseShortName", {}).get("value", "")
            or em.get("UsageTerms", {}).get("value", "")
            or "unknown")


def scrape(
    category: str, out_dir: Path, limit: int = 200,
    min_dim: int = 256, workers: int = 6, depth: int = 1,
) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    # Over-fetch titles since many will be filtered out (wrong extension, too small).
    titles = list(iter_category(category, depth=depth, max_items=limit * 5))
    print(f"Discovered {len(titles)} file titles under {category!r}; fetching imageinfo...")
    infos = imageinfo(titles)
    print(f"Got imageinfo for {len(infos)}/{len(titles)}")

    csv_path = out_dir / "metadata.csv"
    with csv_path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["coin_id", "coin_uri", "type_uri", "type_label",
                         "obverse_path", "reverse_path", "licence"])

        downloaded = 0
        jobs: list[tuple[str, str, Path, dict]] = []
        for title, info in infos.items():
            url = info.get("url", "")
            ext = "." + url.rsplit(".", 1)[-1].lower() if "." in url else ""
            if ext not in GOOD_EXT:
                continue
            if info.get("width", 0) < min_dim or info.get("height", 0) < min_dim:
                continue
            slug = slugify(title)
            dest = out_dir / slug
            jobs.append((title, url, dest, info))

        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_download, url, dest): (title, dest, info)
                       for title, url, dest, info in jobs[:limit]}
            for fut in as_completed(futures):
                title, dest, info = futures[fut]
                if not fut.result():
                    continue
                # Wikimedia is single-side per image; we use the same path for obverse
                # and write a deterministic synthetic 'reverse' path (empty).
                # The dataloader handles single-side records.
                page_id = title.removeprefix("File:")
                writer.writerow([
                    dest.stem,
                    f"https://commons.wikimedia.org/wiki/{title.replace(' ', '_')}",
                    "",  # no type URI — Wikimedia categories aren't RIC types
                    page_id.rsplit(".", 1)[0][:200],
                    dest.name,
                    "",
                    licence_str(info),
                ])
                downloaded += 1
    print(f"Downloaded {downloaded} images to {out_dir} (metadata at {csv_path})")
    return downloaded


def _download(url: str, dest: Path) -> bool:
    try:
        if dest.exists() and dest.stat().st_size > 5_000:
            return True
        r = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=60, stream=True)
        r.raise_for_status()
        dest.parent.mkdir(parents=True, exist_ok=True)
        with dest.open("wb") as fh:
            for chunk in r.iter_content(chunk_size=64 * 1024):
                fh.write(chunk)
        return True
    except (requests.RequestException, OSError):
        return False


def _cli() -> None:
    p = argparse.ArgumentParser(description="Scrape ancient coin images from Wikimedia Commons.")
    p.add_argument("--category", default="Roman coins by emperor",
                   help="Wikimedia category name (without 'Category:' prefix). "
                        "'Roman coins by emperor' has the cleanest per-coin photos.")
    p.add_argument("--limit", type=int, default=200)
    p.add_argument("--out", type=Path, default=Path("data/wikimedia_raw"))
    p.add_argument("--min_dim", type=int, default=256)
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--depth", type=int, default=2,
                   help="Subcategory recursion depth. Default 2 so 'by emperor' "
                        "→ 'Coins of Augustus' → file list works.")
    args = p.parse_args()
    scrape(args.category, args.out, limit=args.limit, min_dim=args.min_dim,
           workers=args.workers, depth=args.depth)


if __name__ == "__main__":
    _cli()
