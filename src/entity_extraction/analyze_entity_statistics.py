"""
Compute per-entity occurrence statistics from the n-gram matching output.

Reads every TSV file under the input directory and writes the statistics tables
(one per entity prefix, plus GLOBAL and GLOBAL_summary tables) next to the output file.
"""

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
from tqdm import tqdm


def discover_entity_files(input_dir: Path) -> list[Path]:
    """
    Discover all *.tsv files in input directory.

    Args:
        input_dir: Directory containing entity matching results

    Returns:
        List of paths to TSV files
    """
    if not input_dir.exists():
        print(f"Error: Input directory not found: {input_dir}")
        return []

    tsv_files = list(input_dir.glob("**/*.tsv"))
    print(f"Found {len(tsv_files)} TSV files to process")

    return tsv_files


def aggregate_entities(
    file_paths: list[Path], case_insensitive: bool = False, min_similarity: float = 0.0
) -> tuple[dict[str, dict], int, int]:
    """
    Aggregate entity matches across all files.

    Args:
        file_paths: List of TSV file paths to process
        case_insensitive: If True, normalize extracted text to lowercase
        min_similarity: Minimum similarity threshold (0.0-1.0). Only matches with
                       similarity_score_label >= threshold are included

    Returns:
        Tuple of (entity_stats, total_chunks, total_papers) where:
        - entity_stats: Dictionary mapping entity_name to aggregated statistics
        - total_chunks: Total number of unique chunks processed
        - total_papers: Total number of unique papers processed

        entity_stats format:
        {
            'entity_name': {
                'entity_label': str,
                'count': int,
                'extracted_texts': Set[str],
                'chunks': Set[str],
                'papers': Set[str],
                'scores': List[float]
            }
        }
    """
    entity_stats = defaultdict(
        lambda: {
            "entity_label": "",
            "count": 0,
            "extracted_texts": set(),
            "chunks": set(),
            "papers": set(),
            "scores": [],
        }
    )

    all_chunks = set()
    all_papers = set()

    total_matches = 0
    filtered_matches = 0

    print("\nAggregating entity statistics...")
    for file_path in tqdm(file_paths, desc="Processing files", unit="file"):
        try:
            chunk_id = file_path.stem
            paper_id = file_path.parent.name

            chunk_key = f"{paper_id}/{chunk_id}"
            all_chunks.add(chunk_key)
            all_papers.add(paper_id)

            with open(file_path, encoding="utf-8") as f:
                reader = csv.DictReader(f, delimiter="\t")

                for row in reader:
                    similarity_label = float(row["similarity_score_label"])

                    if similarity_label < min_similarity:
                        filtered_matches += 1
                        continue

                    entity_name = row["entity_name"].strip()
                    entity_label = row["entity_label"].strip()
                    extracted_text = row["extracted_text"].strip()

                    if case_insensitive:
                        extracted_text = extracted_text.lower()

                    if not entity_stats[entity_name]["entity_label"]:
                        entity_stats[entity_name]["entity_label"] = entity_label

                    entity_stats[entity_name]["count"] += 1
                    entity_stats[entity_name]["extracted_texts"].add(extracted_text)
                    entity_stats[entity_name]["chunks"].add(f"{paper_id}/{chunk_id}")
                    entity_stats[entity_name]["papers"].add(paper_id)
                    entity_stats[entity_name]["scores"].append(similarity_label)

                    total_matches += 1

        except Exception as e:
            print(f"\nWarning: Failed to process {file_path}: {e}")
            continue

    print("\nAggregation complete:")
    print(f"  Total matches: {total_matches:,}")
    if filtered_matches > 0:
        print(f"  Filtered out (similarity < {min_similarity}): {filtered_matches:,}")
    print(f"  Unique entities: {len(entity_stats):,}")
    print(f"  Total chunks: {len(all_chunks):,}")
    print(f"  Total papers: {len(all_papers):,}")

    return dict(entity_stats), len(all_chunks), len(all_papers)


def write_statistics(entity_stats: dict, output_file: Path, total_chunks: int, total_papers: int):
    """
    Write aggregated statistics to TSV files (one per entity type prefix).

    Rows are sorted by pct_paper, pct_chunk and count in descending order.

    Args:
        entity_stats: Dictionary of aggregated entity statistics
        output_file: Path to output TSV file (will be modified with prefix)
        total_chunks: Total number of unique chunks processed
        total_papers: Total number of unique papers processed
    """
    output_file.parent.mkdir(parents=True, exist_ok=True)

    print("\nGrouping entities by prefix...")
    entities_by_prefix = defaultdict(list)

    for entity_name, stats in entity_stats.items():
        prefix = entity_name.split(":", 1)[0] if ":" in entity_name else "OTHER"
        entities_by_prefix[prefix].append((entity_name, stats))

    print(f"Found {len(entities_by_prefix)} entity prefixes: {', '.join(sorted(entities_by_prefix.keys()))}")

    print("\nSorting entities within each prefix by pct_paper, pct_chunk, count...")
    for prefix in entities_by_prefix:
        entities_by_prefix[prefix] = sorted(
            entities_by_prefix[prefix],
            key=lambda x: (
                len(x[1]["papers"]) / total_papers * 100.0 if total_papers > 0 else 0.0,
                len(x[1]["chunks"]) / total_chunks * 100.0 if total_chunks > 0 else 0.0,
                x[1]["count"],
            ),
            reverse=True,
        )

    print("\nWriting statistics to separate files by prefix...")

    output_base = output_file.stem
    output_dir = output_file.parent
    output_ext = output_file.suffix

    written_files = []

    print("\nWriting global statistics file...")
    global_output_file = output_dir / f"{output_base}_GLOBAL{output_ext}"

    all_entities_combined = []
    for entities in entities_by_prefix.values():
        all_entities_combined.extend(entities)

    all_entities_sorted = sorted(
        all_entities_combined,
        key=lambda x: (
            len(x[1]["papers"]) / total_papers * 100.0 if total_papers > 0 else 0.0,
            len(x[1]["chunks"]) / total_chunks * 100.0 if total_chunks > 0 else 0.0,
            x[1]["count"],
        ),
        reverse=True,
    )

    with open(global_output_file, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")

        writer.writerow(
            [
                "entity_name",
                "entity_label",
                "count",
                "n_extracted_text",
                "n_chunk",
                "n_paper",
                "pct_chunk",
                "pct_paper",
                "score_min",
                "score_max",
                "score_avg",
                "score_std",
                "all_extracted_text",
            ]
        )

        for entity_name, stats in tqdm(all_entities_sorted, desc="Writing GLOBAL", unit="entity"):
            entity_label = stats["entity_label"]
            count = stats["count"]
            extracted_texts = stats["extracted_texts"]
            chunks = stats["chunks"]
            papers = stats["papers"]
            scores = stats["scores"]

            n_extracted_text = len(extracted_texts)
            n_chunk = len(chunks)
            n_paper = len(papers)
            all_extracted_text = "|".join(sorted(extracted_texts))

            pct_chunk = (n_chunk / total_chunks * 100.0) if total_chunks > 0 else 0.0
            pct_paper = (n_paper / total_papers * 100.0) if total_papers > 0 else 0.0

            if scores:
                score_min = float(np.min(scores))
                score_max = float(np.max(scores))
                score_avg = float(np.mean(scores))
                score_std = float(np.std(scores))
            else:
                score_min = score_max = score_avg = score_std = 0.0

            writer.writerow(
                [
                    entity_name,
                    entity_label,
                    count,
                    n_extracted_text,
                    n_chunk,
                    n_paper,
                    f"{pct_chunk:.2f}",
                    f"{pct_paper:.2f}",
                    f"{score_min:.4f}",
                    f"{score_max:.4f}",
                    f"{score_avg:.4f}",
                    f"{score_std:.4f}",
                    all_extracted_text,
                ]
            )

    written_files.append(global_output_file)
    print(f"  GLOBAL: {len(all_entities_sorted)} entities -> {global_output_file.name}")

    print("\nWriting global summary statistics file (without variant texts)...")
    global_summary_file = output_dir / f"{output_base}_GLOBAL_summary{output_ext}"

    with open(global_summary_file, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")

        writer.writerow(
            [
                "entity_name",
                "entity_label",
                "count",
                "n_extracted_text",
                "n_chunk",
                "n_paper",
                "pct_chunk",
                "pct_paper",
                "score_min",
                "score_max",
                "score_avg",
                "score_std",
            ]
        )

        for entity_name, stats in tqdm(all_entities_sorted, desc="Writing GLOBAL_summary", unit="entity"):
            entity_label = stats["entity_label"]
            count = stats["count"]
            n_extracted_text = len(stats["extracted_texts"])
            n_chunk = len(stats["chunks"])
            n_paper = len(stats["papers"])
            scores = stats["scores"]

            pct_chunk = (n_chunk / total_chunks * 100.0) if total_chunks > 0 else 0.0
            pct_paper = (n_paper / total_papers * 100.0) if total_papers > 0 else 0.0

            if scores:
                score_min = float(np.min(scores))
                score_max = float(np.max(scores))
                score_avg = float(np.mean(scores))
                score_std = float(np.std(scores))
            else:
                score_min = score_max = score_avg = score_std = 0.0

            writer.writerow(
                [
                    entity_name,
                    entity_label,
                    count,
                    n_extracted_text,
                    n_chunk,
                    n_paper,
                    f"{pct_chunk:.2f}",
                    f"{pct_paper:.2f}",
                    f"{score_min:.4f}",
                    f"{score_max:.4f}",
                    f"{score_avg:.4f}",
                    f"{score_std:.4f}",
                ]
            )

    written_files.append(global_summary_file)
    print(f"  GLOBAL_summary: {len(all_entities_sorted)} entities -> {global_summary_file.name}")

    for prefix in sorted(entities_by_prefix.keys()):
        sorted_entities = entities_by_prefix[prefix]

        prefix_output_file = output_dir / f"{output_base}_{prefix}{output_ext}"

        with open(prefix_output_file, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f, delimiter="\t")

            writer.writerow(
                [
                    "entity_name",
                    "entity_label",
                    "count",
                    "n_extracted_text",
                    "n_chunk",
                    "n_paper",
                    "pct_chunk",
                    "pct_paper",
                    "score_min",
                    "score_max",
                    "score_avg",
                    "score_std",
                    "all_extracted_text",
                ]
            )

            for entity_name, stats in tqdm(sorted_entities, desc=f"Writing {prefix}", unit="entity"):
                entity_label = stats["entity_label"]
                count = stats["count"]
                extracted_texts = stats["extracted_texts"]
                chunks = stats["chunks"]
                papers = stats["papers"]
                scores = stats["scores"]

                n_extracted_text = len(extracted_texts)
                n_chunk = len(chunks)
                n_paper = len(papers)
                all_extracted_text = "|".join(sorted(extracted_texts))

                pct_chunk = (n_chunk / total_chunks * 100.0) if total_chunks > 0 else 0.0
                pct_paper = (n_paper / total_papers * 100.0) if total_papers > 0 else 0.0

                if scores:
                    score_min = float(np.min(scores))
                    score_max = float(np.max(scores))
                    score_avg = float(np.mean(scores))
                    score_std = float(np.std(scores))
                else:
                    score_min = score_max = score_avg = score_std = 0.0

                writer.writerow(
                    [
                        entity_name,
                        entity_label,
                        count,
                        n_extracted_text,
                        n_chunk,
                        n_paper,
                        f"{pct_chunk:.2f}",
                        f"{pct_paper:.2f}",
                        f"{score_min:.4f}",
                        f"{score_max:.4f}",
                        f"{score_avg:.4f}",
                        f"{score_std:.4f}",
                        all_extracted_text,
                    ]
                )

        written_files.append(prefix_output_file)
        print(f"  {prefix}: {len(sorted_entities)} entities -> {prefix_output_file.name}")

    print(f"\nStatistics written to {len(written_files)} files in {output_dir}")
    print(f"  - GLOBAL file (with all variant texts): {global_output_file.name}")
    print(f"  - GLOBAL summary file (variant count only): {global_summary_file.name}")
    print(f"  - {len(entities_by_prefix)} prefix-specific files")

    print("\nTop 10 entities by distribution (pct_paper, pct_chunk, count):")
    print(
        f"{'Rank':<6} {'Count':<10} {'Variants':<10} {'Chunks':<10} {'Papers':<10} "
        f"{'%Paper':<10} {'%Chunk':<10} {'Prefix':<10} {'Entity Name':<30} {'Label'}"
    )
    print("-" * 150)

    for rank, (entity_name, stats) in enumerate(all_entities_sorted[:10], 1):
        count = stats["count"]
        n_variants = len(stats["extracted_texts"])
        n_chunks = len(stats["chunks"])
        n_papers = len(stats["papers"])
        pct_paper = (n_papers / total_papers * 100.0) if total_papers > 0 else 0.0
        pct_chunk = (n_chunks / total_chunks * 100.0) if total_chunks > 0 else 0.0
        entity_label = stats["entity_label"]
        prefix = entity_name.split(":", 1)[0] if ":" in entity_name else "OTHER"

        display_name = entity_name if len(entity_name) <= 30 else entity_name[:27] + "..."

        print(
            f"{rank:<6} {count:<10,} {n_variants:<10} {n_chunks:<10} {n_papers:<10} "
            f"{pct_paper:<10.2f} {pct_chunk:<10.2f} {prefix:<10} {display_name:<30} {entity_label}"
        )


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Analyze entity extraction statistics from n-gram matching results")
    parser.add_argument(
        "--input", type=Path, required=True, help="Input directory containing entity matching TSV files"
    )
    parser.add_argument("--output", type=Path, required=True, help="Output TSV file for statistics")
    parser.add_argument(
        "--case-insensitive", action="store_true", help="Normalize extracted text to lowercase for aggregation"
    )
    parser.add_argument(
        "--min-similarity",
        type=float,
        default=0.0,
        help=(
            "Minimum similarity threshold (0.0-1.0). Only include matches where "
            "similarity_score_label >= threshold. Default: 0.0 (no filtering)"
        ),
    )

    args = parser.parse_args()

    print("=" * 80)
    print("Entity Extraction Statistics Analyzer")
    print("=" * 80)
    print(f"Input directory: {args.input}")
    print(f"Output file: {args.output}")
    print(f"Case-insensitive: {args.case_insensitive}")
    print(f"Min similarity: {args.min_similarity}")
    print()

    files = discover_entity_files(args.input)

    if not files:
        print("No TSV files found. Exiting.")
        return

    entity_stats, total_chunks, total_papers = aggregate_entities(
        files, case_insensitive=args.case_insensitive, min_similarity=args.min_similarity
    )

    if not entity_stats:
        print("No entities found. Exiting.")
        return

    write_statistics(entity_stats, args.output, total_chunks, total_papers)

    print("\n" + "=" * 80)
    print("Analysis complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
