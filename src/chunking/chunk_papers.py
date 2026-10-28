#!/usr/bin/env python3
"""Main script for semantic chunking of papers using multiprocessing."""

import argparse
import csv
import multiprocessing as mp
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

from tqdm import tqdm

from .semantic_chunker import process_paper
from .utils import (
    ProcessingResult,
    get_chunk_csv_header,
    get_paper_csv_header,
    is_paper_failed,
    is_paper_processed,
)


def worker_process_paper(args):
    """Wrapper for multiprocessing - unpacks arguments and calls process_paper."""
    (
        paper_path,
        output_dir,
        error_output_dir,
        chunk_size,
        force,
        retry_errors,
        chunk_stats_file,
        paper_stats_file,
        device,
    ) = args

    result = process_paper(paper_path, output_dir, error_output_dir, chunk_size, force, retry_errors, device)

    write_chunk_stats_to_csv(result, chunk_stats_file)

    write_paper_stats_to_csv(result, paper_stats_file)

    return result


def discover_papers(input_dir: Path) -> list[Path]:
    """Discover all paper files in input directory."""
    papers = sorted(input_dir.glob("*.txt"))
    return papers


def filter_papers(
    papers: list[Path], output_dir: Path, error_output_dir: Path, force: bool, retry_errors: bool
) -> tuple[list[Path], int, int]:
    """
    Filter papers based on processing status.

    Returns:
        (papers_to_process, num_successful, num_failed)
    """
    if force:
        return papers, 0, 0

    to_process = []
    num_successful = 0
    num_failed = 0

    for paper_path in papers:
        paper_id = paper_path.stem

        if is_paper_processed(paper_id, output_dir):
            num_successful += 1
            continue

        if is_paper_failed(paper_id, error_output_dir):
            num_failed += 1
            if not retry_errors:
                continue

        to_process.append(paper_path)

    return to_process, num_successful, num_failed


def write_chunk_stats_to_csv(result: ProcessingResult, stats_file: Path):
    """Write chunk statistics to CSV file immediately."""
    if result.status == "skipped" or not result.chunk_stats:
        return

    write_header = not stats_file.exists()

    with open(stats_file, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        if write_header:
            writer.writerow(get_chunk_csv_header())

        for chunk_stat in result.chunk_stats:
            writer.writerow(chunk_stat.to_csv_row())


def write_paper_stats_to_csv(result: ProcessingResult, stats_file: Path):
    """Write paper statistics to CSV file immediately."""
    if result.status == "skipped":
        return

    write_header = not stats_file.exists()

    with open(stats_file, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        if write_header:
            writer.writerow(get_paper_csv_header())

        writer.writerow(result.to_paper_csv_row())


def print_statistics(
    total_papers: int,
    already_successful: int,
    already_failed: int,
    results: list[ProcessingResult],
    start_time: datetime,
):
    """Print final statistics summary."""
    newly_processed = len([r for r in results if r.status != "skipped"])
    newly_successful = len([r for r in results if r.status == "success"])
    newly_failed = len([r for r in results if r.status == "error"])
    total_chunks = sum(r.num_chunks for r in results if r.status == "success")

    error_counts = Counter()
    for result in results:
        if result.status == "error" and result.error_type:
            error_counts[result.error_type.value] += 1

    successful_results = [r for r in results if r.status == "success"]
    if successful_results:
        avg_chunks = total_chunks / len(successful_results)

        all_chunk_stats = []
        for r in successful_results:
            all_chunk_stats.extend(r.chunk_stats)

        if all_chunk_stats:
            avg_chunk_tokens = sum(c.num_tokens for c in all_chunk_stats) / len(all_chunk_stats)
            avg_chunk_words = sum(c.num_words for c in all_chunk_stats) / len(all_chunk_stats)
            avg_chunk_chars = sum(c.num_characters for c in all_chunk_stats) / len(all_chunk_stats)
        else:
            avg_chunk_tokens = avg_chunk_words = avg_chunk_chars = 0
    else:
        avg_chunks = avg_chunk_tokens = avg_chunk_words = avg_chunk_chars = 0

    elapsed = (datetime.now() - start_time).total_seconds()
    papers_per_sec = newly_processed / elapsed if elapsed > 0 else 0

    print("\n" + "=" * 70)
    print("PROCESSING SUMMARY")
    print("=" * 70)
    print(f"Total papers in input:        {total_papers:,}")
    print(f"Already successful (skipped): {already_successful:,}")
    print(f"Already failed (skipped):     {already_failed:,}")
    print(f"Newly processed:              {newly_processed:,}")
    print(f"  - Newly successful:         {newly_successful:,}")
    print(f"  - Newly failed:             {newly_failed:,}")
    print(f"Total chunks created:         {total_chunks:,}")
    print()

    if error_counts:
        print("Failures by error type:")
        for error_type, count in error_counts.most_common():
            print(f"  - {error_type}: {count:,}")
        print()

    print(f"Processing rate:              {papers_per_sec:.2f} papers/sec")
    print(f"Total elapsed time:           {elapsed:.2f} seconds")

    if successful_results:
        print()
        print("Averages (successful papers only):")
        print(f"  - Chunks per paper:         {avg_chunks:.2f}")
        print()
        print("Averages per chunk:")
        print(f"  - Tokens per chunk:         {avg_chunk_tokens:.0f}")
        print(f"  - Words per chunk:          {avg_chunk_words:.0f}")
        print(f"  - Characters per chunk:     {avg_chunk_chars:.0f}")

    print("=" * 70)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Semantic chunking of papers using Chonkie", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--input", type=Path, required=True, help="Input directory containing paper files")
    parser.add_argument("--output", type=Path, required=True, help="Output directory for chunks")
    parser.add_argument("--error-output", type=Path, required=True, help="Output directory for error files")
    parser.add_argument("--stats-output", type=Path, required=True, help="Output directory for statistics CSV")
    parser.add_argument("--chunk-size", type=int, default=2048, help="Maximum tokens per chunk")
    parser.add_argument("--workers", type=int, default=mp.cpu_count() - 1, help="Number of worker processes")
    parser.add_argument("--force", action="store_true", help="Reprocess all papers, ignoring existing outputs")
    parser.add_argument("--retry-errors", action="store_true", help="Reprocess papers that failed in an earlier run")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of papers to process")

    args = parser.parse_args()

    if not args.input.exists():
        print(f"Error: Input directory does not exist: {args.input}", file=sys.stderr)
        sys.exit(1)

    args.output.mkdir(parents=True, exist_ok=True)
    args.error_output.mkdir(parents=True, exist_ok=True)
    args.stats_output.mkdir(parents=True, exist_ok=True)

    start_time = datetime.now()

    print("=" * 70)
    print("SEMANTIC CHUNKING - BIOGNOSIA RAG")
    print("=" * 70)
    print(f"Input directory:    {args.input}")
    print(f"Output directory:   {args.output}")
    print(f"Error directory:    {args.error_output}")
    print(f"Stats directory:    {args.stats_output}")
    print(f"Chunk size:         {args.chunk_size} tokens")
    print(f"Workers:            {args.workers}")
    print(f"Force reprocess:    {args.force}")
    print(f"Retry errors:       {args.retry_errors}")
    print("=" * 70)
    print()

    print("Discovering papers...")
    papers = discover_papers(args.input)
    print(f"Found {len(papers):,} paper files")

    print("Checking processing status...")
    papers_to_process, already_successful, already_failed = filter_papers(
        papers, args.output, args.error_output, args.force, args.retry_errors
    )

    if args.limit is not None and args.limit > 0:
        papers_to_process = papers_to_process[: args.limit]
        print(f"Limiting to {args.limit} papers (--limit parameter)")

    print(f"Already successful: {already_successful:,}")
    print(f"Already failed:     {already_failed:,}")
    print(f"To process:         {len(papers_to_process):,}")
    print()

    if not papers_to_process:
        print("No papers to process. Exiting.")
        return

    chunk_stats_file = args.stats_output / "chunk_stats.csv"
    paper_stats_file = args.stats_output / "paper_stats.csv"

    num_gpus = 0
    gpu_devices = []

    try:
        import os

        import torch

        if torch.cuda.is_available():
            num_gpus = torch.cuda.device_count()

            cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
            if cuda_visible:
                visible_devices = [int(d.strip()) for d in cuda_visible.split(",") if d.strip().isdigit()]
                if visible_devices:
                    num_gpus = len(visible_devices)
                    print(f"Using CUDA_VISIBLE_DEVICES: {cuda_visible}")

            if num_gpus > 0:
                print(f"Detected {num_gpus} available GPU(s)")
                print(f"CUDA version: {torch.version.cuda}")
                for i in range(num_gpus):
                    try:
                        device = torch.device(f"cuda:{i}")
                        torch.cuda.set_device(device)
                        print(
                            f"  GPU {i}: {torch.cuda.get_device_name(i)} - {torch.cuda.get_device_properties(i).total_memory / 1e9:.2f} GB"
                        )
                    except Exception as e:
                        print(f"  GPU {i}: Error accessing - {e}")

                gpu_devices = [f"cuda:{i % num_gpus}" for i in range(args.workers)]
            else:
                print("CUDA available but no GPUs detected, using CPU")
                gpu_devices = [None] * args.workers
        else:
            print("CUDA not available, using CPU")
            print(f"  PyTorch version: {torch.__version__}")
            gpu_devices = [None] * args.workers

    except ImportError as e:
        print(f"PyTorch not found: {e}")
        print("  Using CPU mode")
        gpu_devices = [None] * args.workers
    except Exception as e:
        print(f"GPU detection error: {e}")
        print("  Falling back to CPU mode")
        gpu_devices = [None] * args.workers

    worker_args = []
    for idx, paper_path in enumerate(papers_to_process):
        device = gpu_devices[idx % len(gpu_devices)]
        worker_args.append(
            (
                paper_path,
                args.output,
                args.error_output,
                args.chunk_size,
                args.force,
                args.retry_errors,
                chunk_stats_file,
                paper_stats_file,
                device,
            )
        )

    if num_gpus > 0:
        gpu_distribution = {}
        for device in gpu_devices:
            gpu_distribution[device] = gpu_distribution.get(device, 0) + 1
        print("\nWorker-GPU distribution:")
        for device, count in sorted(gpu_distribution.items()):
            print(f"  {device}: {count} workers")
    else:
        print(f"\nAll {args.workers} workers using CPU")

    print(f"Processing papers with {args.workers} workers...")
    print(f"Chunk stats will be written to: {chunk_stats_file}")
    print(f"Paper stats will be written to: {paper_stats_file}")
    print()
    results = []

    with mp.Pool(processes=args.workers) as pool:
        for result in tqdm(
            pool.imap_unordered(worker_process_paper, worker_args),
            total=len(worker_args),
            desc="Processing",
            unit="paper",
        ):
            results.append(result)

    print("\nProcessing completed!")

    print_statistics(len(papers), already_successful, already_failed, results, start_time)


if __name__ == "__main__":
    main()
