#!/usr/bin/env python3
"""Main entry point for preprocessing papers with multiprocessing."""

import argparse
import csv
import multiprocessing
from datetime import datetime
from pathlib import Path

from tqdm import tqdm

from .config import STATS_CSV_HEADER
from .text_processor import preprocess_text
from .utils import (
    PreprocessingStat,
    categorize_error,
    get_paper_id,
    should_skip_paper,
    write_error_file,
)


def process_paper(
    paper_path: Path, output_dir: Path, error_output_dir: Path, force: bool, retry_errors: bool
) -> PreprocessingStat:
    """
    Process a single paper.

    Args:
        paper_path: Path to input paper
        output_dir: Output directory for preprocessed papers
        error_output_dir: Error output directory
        force: Force reprocessing
        retry_errors: Retry papers with errors

    Returns:
        PreprocessingStat object with processing results
    """
    paper_id = get_paper_id(paper_path)
    started_time = datetime.now().isoformat()

    if should_skip_paper(paper_id, output_dir, error_output_dir, force, retry_errors):
        return None

    try:
        original_bytes = paper_path.read_bytes()
        try:
            text = original_bytes.decode("utf-8", errors="replace")
        except Exception:
            text = original_bytes.decode("latin-1", errors="replace")

        original_chars = len(text)

        preprocessed_text, stats = preprocess_text(text, original_bytes)
        preprocessed_chars = len(preprocessed_text)
        chars_removed = original_chars - preprocessed_chars

        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / f"{paper_id}.txt"
        output_file.write_text(preprocessed_text, encoding="utf-8")

        finished_time = datetime.now().isoformat()
        stat = PreprocessingStat(
            paper_id=paper_id,
            original_chars=original_chars,
            preprocessed_chars=preprocessed_chars,
            chars_removed=chars_removed,
            encoding_fixed=stats["encoding_fixed"],
            latex_patterns_found=stats["latex_patterns_found"],
            language_detected=stats["language_detected"],
            quality_score=stats["quality_score"],
            started_time=started_time,
            finished_time=finished_time,
            status="success",
            error="",
        )

        return stat

    except Exception as e:
        error_message = categorize_error(e)
        write_error_file(error_output_dir, paper_id, error_message)

        finished_time = datetime.now().isoformat()
        stat = PreprocessingStat(
            paper_id=paper_id,
            original_chars=len(text) if "text" in locals() else 0,
            preprocessed_chars=0,
            chars_removed=0,
            encoding_fixed=False,
            latex_patterns_found=0,
            language_detected="unknown",
            quality_score=0.0,
            started_time=started_time,
            finished_time=finished_time,
            status="error",
            error=error_message,
        )

        return stat


def worker_process_paper(args):
    """
    Worker function for multiprocessing.

    Args:
        args: Tuple of (paper_path, output_dir, error_output_dir, force, retry_errors, stats_file)

    Returns:
        PreprocessingStat or None
    """
    paper_path, output_dir, error_output_dir, force, retry_errors, stats_file = args

    result = process_paper(paper_path, output_dir, error_output_dir, force, retry_errors)

    if result:
        write_stat_to_csv(result, stats_file)

    return result


def write_stat_to_csv(stat: PreprocessingStat, stats_file: Path):
    """Write a single stat to CSV file."""
    stats_file.parent.mkdir(parents=True, exist_ok=True)

    write_header = not stats_file.exists()

    with open(stats_file, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(STATS_CSV_HEADER)
        writer.writerow(stat.to_csv_row())


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Preprocess scientific papers")
    parser.add_argument("--input", type=str, required=True, help="Input directory containing papers")
    parser.add_argument("--output", type=str, required=True, help="Output directory for preprocessed papers")
    parser.add_argument("--error-output", type=str, required=True, help="Error output directory")
    parser.add_argument("--stats-output", type=str, required=True, help="Statistics output directory")
    parser.add_argument("--workers", type=int, default=8, help="Number of worker processes (default: 8)")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of papers to process")
    parser.add_argument("--force", action="store_true", help="Force reprocessing even if already preprocessed")
    parser.add_argument("--retry-errors", action="store_true", help="Retry papers that failed in an earlier run")

    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)
    error_output_dir = Path(args.error_output)
    stats_output_dir = Path(args.stats_output)
    stats_file = stats_output_dir / "stats.csv"

    paper_files = sorted(input_dir.glob("*.txt"))
    if args.limit:
        paper_files = paper_files[: args.limit]

    print(f"Found {len(paper_files)} papers to process")
    print(f"Using {args.workers} workers")
    print(f"Output: {output_dir}")
    print(f"Error output: {error_output_dir}")
    print(f"Stats output: {stats_file}")

    worker_args = [
        (paper_path, output_dir, error_output_dir, args.force, args.retry_errors, stats_file)
        for paper_path in paper_files
    ]

    success_count = 0
    error_count = 0
    skipped_count = 0

    with multiprocessing.Pool(processes=args.workers) as pool:
        results = list(
            tqdm(
                pool.imap_unordered(worker_process_paper, worker_args),
                total=len(worker_args),
                desc="Preprocessing papers",
            )
        )

    for result in results:
        if result is None:
            skipped_count += 1
        elif result.status == "success":
            success_count += 1
        else:
            error_count += 1

    print("\nProcessing complete:")
    print(f"  Success: {success_count}")
    print(f"  Errors: {error_count}")
    print(f"  Skipped: {skipped_count}")
    print(f"  Total: {len(paper_files)}")


if __name__ == "__main__":
    main()
