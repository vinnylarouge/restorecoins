"""Scrape OCRE coin specimens via the Nomisma SPARQL endpoint.

Spec §4.1: target ~10,000–20,000 high-grade specimens with type metadata.
Filter aggressively for image quality.

Why we wrote our own rather than vendor the Altaweel et al. scraper:
    The 2024 JCAA supplementary code is CC-BY for the article but the licence
    status of redistributed code is ambiguous. A fresh ~150-line SPARQL scraper
    is simpler than dependency-managing a fork.

The scraper has two phases:
    1. SPARQL → list of (coin URI, type URI, obverse/reverse image URLs).
    2. Download → `data/raw/<coin_id>_{obv,rev}.jpg` + a CSV of metadata.

Quality filtering happens in a separate pass (`filter_quality`) so re-filtering
doesn't re-hit the network.

Usage:
    # Smoke test: pull 50 coins, no filtering.
    python -m training.scrape_ocre --limit 50 --out data/raw

    # Full pull (slow — hours, ~5-10GB).
    python -m training.scrape_ocre --limit 20000 --out data/raw

    # Quality filter pass over an existing raw dir.
    python -m training.scrape_ocre --filter --in data/raw --out data/filtered
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import requests
from PIL import Image, UnidentifiedImageError

NOMISMA_ENDPOINT = "https://nomisma.org/query"
USER_AGENT = "restorecoins-scraper/0.1 (research; +https://github.com/vinnylarouge/restorecoins)"
TIMEOUT = 30
SPARQL_PAGE = 1000  # Nomisma's endpoint is fragile beyond a few thousand rows.

# Filtering thresholds for §4.1's "filter aggressively for image quality" pass.
MIN_DIM = 256          # both sides ≥ 256px
MAX_ASPECT = 1.6       # coins are round; tall/wide images are usually composites
MIN_BYTES = 5_000      # crops below this are usually 1×1 placeholder pixels
MAX_BYTES = 8_000_000  # over 8MB likely a multi-coin scan, not a single specimen


# --------------------------------------------------------------------------- #
# SPARQL                                                                      #
# --------------------------------------------------------------------------- #


SPARQL_QUERY = """\
PREFIX nm: <http://nomisma.org/id/>
PREFIX nmo: <http://nomisma.org/ontology#>
PREFIX dcterms: <http://purl.org/dc/terms/>
PREFIX foaf: <http://xmlns.com/foaf/0.1/>
PREFIX skos: <http://www.w3.org/2004/02/skos/core#>

SELECT DISTINCT ?coin ?type ?label ?obverse_thumb ?reverse_thumb
       ?obverse_full ?reverse_full
WHERE {
    ?coin a nmo:NumismaticObject ;
          nmo:hasTypeSeriesItem ?type ;
          nmo:hasObverse ?obv .
    ?obv foaf:depiction ?obverse_full .
    FILTER(STRSTARTS(STR(?type), "http://numismatics.org/ocre/id/"))
    # Skip image hosts we know are unreachable from this network — see
    # DECISIONS.md 2026-05-21. ANS-hosted images (numismatics.org/...) and
    # Met images are reachable.
    FILTER(!CONTAINS(STR(?obverse_full), "fitzmuseum"))
    FILTER(!CONTAINS(STR(?obverse_full), "localhost"))
    OPTIONAL { ?obv foaf:thumbnail ?obverse_thumb }
    OPTIONAL {
        ?coin nmo:hasReverse ?rev .
        OPTIONAL { ?rev foaf:depiction ?reverse_full }
        OPTIONAL { ?rev foaf:thumbnail ?reverse_thumb }
    }
    OPTIONAL { ?type skos:prefLabel ?label . FILTER(LANG(?label) = "en") }
}
LIMIT %d OFFSET %d
"""


@dataclass
class CoinRecord:
    coin_uri: str
    type_uri: str
    type_label: str
    obverse_url: str | None
    reverse_url: str | None

    @property
    def coin_id(self) -> str:
        """Filesystem-safe slug derived from the coin URI."""
        slug = self.coin_uri.rstrip("/").rsplit("/", 1)[-1]
        slug = re.sub(r"[^A-Za-z0-9._-]", "_", slug)
        if len(slug) > 80:
            slug = slug[:60] + "_" + hashlib.sha1(self.coin_uri.encode()).hexdigest()[:8]
        return slug or hashlib.sha1(self.coin_uri.encode()).hexdigest()[:16]


def sparql_select(query: str) -> list[dict]:
    """GET a SPARQL query, return the bindings list.

    Nomisma's endpoint rejects POST (HTTP 403) and only honours GET — verified
    against the live endpoint in May 2026. Older client libraries default to
    POST; don't.
    """
    resp = requests.get(
        NOMISMA_ENDPOINT,
        params={"query": query, "output": "json"},
        headers={"User-Agent": USER_AGENT, "Accept": "application/sparql-results+json"},
        timeout=TIMEOUT,
    )
    resp.raise_for_status()
    return resp.json().get("results", {}).get("bindings", [])


def iter_coins(limit: int, page_size: int = SPARQL_PAGE) -> Iterable[CoinRecord]:
    """Stream coin records, paginating through the SPARQL endpoint."""
    seen = 0
    offset = 0
    while seen < limit:
        page_limit = min(page_size, limit - seen)
        rows = sparql_select(SPARQL_QUERY % (page_limit, offset))
        if not rows:
            return
        for row in rows:
            yield CoinRecord(
                coin_uri=row["coin"]["value"],
                type_uri=row["type"]["value"],
                type_label=row.get("label", {}).get("value", ""),
                # Prefer full-res; thumbnail is the high-recall fallback.
                obverse_url=row.get("obverse_full", row.get("obverse_thumb", {})).get("value"),
                reverse_url=row.get("reverse_full", row.get("reverse_thumb", {})).get("value"),
            )
            seen += 1
            if seen >= limit:
                return
        offset += page_limit


# --------------------------------------------------------------------------- #
# Download                                                                    #
# --------------------------------------------------------------------------- #


def download_one(url: str, dest: Path) -> bool:
    """Save `url` to `dest`. Returns True on success, False on any failure."""
    if dest.exists() and dest.stat().st_size > MIN_BYTES:
        return True
    try:
        resp = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=TIMEOUT, stream=True)
        resp.raise_for_status()
        dest.parent.mkdir(parents=True, exist_ok=True)
        with dest.open("wb") as f:
            for chunk in resp.iter_content(chunk_size=64 * 1024):
                f.write(chunk)
        return True
    except (requests.RequestException, OSError):
        return False


def scrape(out_dir: Path, limit: int, workers: int = 8, csv_path: Path | None = None) -> int:
    """End-to-end scrape: SPARQL → image downloads → metadata CSV.

    Returns the number of *records* (not files) successfully captured. A record
    is captured if at least one side downloaded; downstream filtering decides
    whether single-sided records are usable.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = csv_path or (out_dir / "metadata.csv")

    records = list(iter_coins(limit))
    if not records:
        print("No records returned from SPARQL endpoint. Check the query/endpoint.", file=sys.stderr)
        return 0

    print(f"SPARQL returned {len(records)} coin records. Downloading images...")

    with csv_path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["coin_id", "coin_uri", "type_uri", "type_label",
                         "obverse_path", "reverse_path"])

        ok = 0
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures: dict = {}
            for rec in records:
                paths: dict[str, Path | None] = {"obverse": None, "reverse": None}
                if rec.obverse_url:
                    paths["obverse"] = out_dir / f"{rec.coin_id}_obv{_ext(rec.obverse_url)}"
                    futures[pool.submit(download_one, rec.obverse_url, paths["obverse"])] = (rec, "obverse", paths)
                if rec.reverse_url:
                    paths["reverse"] = out_dir / f"{rec.coin_id}_rev{_ext(rec.reverse_url)}"
                    futures[pool.submit(download_one, rec.reverse_url, paths["reverse"])] = (rec, "reverse", paths)

            results: dict[str, dict] = {}
            for fut in as_completed(futures):
                rec, side, paths = futures[fut]
                got = fut.result()
                bucket = results.setdefault(rec.coin_id, {"rec": rec, "obverse": None, "reverse": None})
                if got and paths[side] and paths[side].exists():
                    bucket[side] = paths[side]

            for coin_id, b in results.items():
                rec: CoinRecord = b["rec"]
                if b["obverse"] is None and b["reverse"] is None:
                    continue
                writer.writerow([
                    coin_id, rec.coin_uri, rec.type_uri, rec.type_label,
                    str(b["obverse"].relative_to(out_dir)) if b["obverse"] else "",
                    str(b["reverse"].relative_to(out_dir)) if b["reverse"] else "",
                ])
                ok += 1

    print(f"Saved {ok} records to {out_dir} (metadata at {csv_path})")
    return ok


def _ext(url: str) -> str:
    """Pick a sane file extension for an image URL."""
    suffix = Path(url.split("?", 1)[0]).suffix.lower()
    return suffix if suffix in {".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff"} else ".jpg"


# --------------------------------------------------------------------------- #
# Quality filter                                                              #
# --------------------------------------------------------------------------- #


def is_high_grade(path: Path) -> tuple[bool, str]:
    """Return (keep, reason). Reason is empty on keep, populated on reject."""
    try:
        size = path.stat().st_size
        if size < MIN_BYTES:
            return False, f"too_small_bytes({size})"
        if size > MAX_BYTES:
            return False, f"too_large_bytes({size})"
        with Image.open(path) as img:
            img.load()
            w, h = img.size
        if min(w, h) < MIN_DIM:
            return False, f"too_small_dim({w}x{h})"
        aspect = max(w, h) / max(1, min(w, h))
        if aspect > MAX_ASPECT:
            return False, f"non_coin_aspect({aspect:.2f})"
        return True, ""
    except (UnidentifiedImageError, OSError) as e:
        return False, f"unreadable({e.__class__.__name__})"


def filter_quality(in_dir: Path, out_dir: Path) -> int:
    """Copy high-grade images from `in_dir` to `out_dir`. Returns count kept."""
    out_dir.mkdir(parents=True, exist_ok=True)
    kept = 0
    rejected: dict[str, int] = {}
    for img_path in sorted(in_dir.glob("*")):
        if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff"}:
            continue
        keep, reason = is_high_grade(img_path)
        if keep:
            (out_dir / img_path.name).write_bytes(img_path.read_bytes())
            kept += 1
        else:
            rejected[reason] = rejected.get(reason, 0) + 1
    print(f"Kept {kept} high-grade images.")
    if rejected:
        print("Rejections by reason:")
        for r, c in sorted(rejected.items(), key=lambda x: -x[1]):
            print(f"  {c:>6}  {r}")
    return kept


# --------------------------------------------------------------------------- #
# CLI                                                                         #
# --------------------------------------------------------------------------- #


def _cli() -> None:
    p = argparse.ArgumentParser(description="Scrape and quality-filter OCRE coin images.")
    sub = p.add_subparsers(dest="mode", required=False)

    scr = p.add_argument_group("scrape")
    scr.add_argument("--limit", type=int, default=50, help="Max coin records to fetch.")
    scr.add_argument("--out", type=Path, default=Path("data/raw"), help="Output dir for scrape.")
    scr.add_argument("--workers", type=int, default=8)
    scr.add_argument("--csv", type=Path, default=None)

    flt = p.add_argument_group("filter")
    flt.add_argument("--filter", action="store_true",
                     help="Run the quality filter pass instead of scraping.")
    flt.add_argument("--in", dest="in_dir", type=Path, default=None,
                     help="Input dir for --filter.")

    probe = p.add_argument_group("probe")
    probe.add_argument("--probe", action="store_true",
                       help="Run a tiny SPARQL query and print the first record. "
                            "Useful when the endpoint or schema may have changed.")

    args = p.parse_args()

    if args.probe:
        rows = sparql_select(SPARQL_QUERY % (3, 0))
        print(f"Probe returned {len(rows)} rows. First row keys: "
              f"{sorted(rows[0].keys()) if rows else 'NONE'}")
        for r in rows[:3]:
            print({k: v.get("value", "") for k, v in r.items()})
        return

    if args.filter:
        if not args.in_dir:
            p.error("--filter requires --in")
        filter_quality(args.in_dir, args.out)
        return

    scrape(args.out, limit=args.limit, workers=args.workers, csv_path=args.csv)


if __name__ == "__main__":
    _cli()
