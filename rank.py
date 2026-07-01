#!/usr/bin/env python3
"""
rank.py — Main entry point for the Candidate Discovery Engine.

Usage:
    python rank.py --candidates ./data/raw/candidates.jsonl --out ./outputs/submission.csv

Runs in < 5 minutes on CPU, 16 GB RAM. No network calls during ranking.
All embeddings and models are pre-loaded from local disk.
"""

import argparse
import gzip
import json
import logging
import sys
import time
from pathlib import Path

import pandas as pd

from src.ranker.pipeline import RankingPipeline
from src.utils.validation import validate_output

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def load_candidates(candidates_path: str) -> list[dict]:
    """Load candidates from .jsonl or .jsonl.gz file."""
    path = Path(candidates_path)
    logger.info(f"Loading candidates from {path} ...")

    opener = gzip.open if path.suffix == ".gz" else open
    mode = "rt"

    candidates = []
    with opener(path, mode, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                candidates.append(json.loads(line))

    logger.info(f"Loaded {len(candidates):,} candidates.")
    return candidates


def main():
    parser = argparse.ArgumentParser(description="Rank candidates for the Redrob hackathon.")
    parser.add_argument(
        "--candidates",
        default="./data/raw/candidates.jsonl",
        help="Path to candidates.jsonl or candidates.jsonl.gz",
    )
    parser.add_argument(
        "--out",
        default="./outputs/submission.csv",
        help="Output CSV path",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=100,
        help="Number of top candidates to output (default: 100)",
    )
    parser.add_argument(
        "--jd",
        default="./docs/job_description.md",
        help="Path to job description markdown file",
    )
    args = parser.parse_args()

    start = time.time()

    # 1. Load candidates
    candidates = load_candidates(args.candidates)

    # 2. Load job description
    jd_path = Path(args.jd)
    if jd_path.exists():
        jd_text = jd_path.read_text(encoding="utf-8")
        logger.info(f"Loaded job description from {jd_path} ({len(jd_text)} chars).")
    else:
        logger.warning(f"JD file not found at {jd_path}. Using embedded JD.")
        jd_text = None  # Pipeline falls back to embedded JD

    # 3. Run ranking pipeline
    pipeline = RankingPipeline(jd_text=jd_text, top_n=args.top_n)
    results_df = pipeline.run(candidates)

    # 4. Write output
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(out_path, index=False)
    logger.info(f"Submission written to {out_path}")

    # 5. Validate
    errors = validate_output(results_df)
    if errors:
        logger.error(f"Validation issues found ({len(errors)}):")
        for e in errors:
            logger.error(f"  - {e}")
        sys.exit(1)
    else:
        logger.info("Output validated successfully ✓")

    elapsed = time.time() - start
    logger.info(f"Total runtime: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
