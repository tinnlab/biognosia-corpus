"""Core semantic chunking logic using Chonkie."""

import shutil
from datetime import datetime
from pathlib import Path

from chonkie import SemanticChunker

from .utils import ChunkStat, ErrorType, ProcessingResult, is_paper_failed, is_paper_processed, write_error_file

_chunker = None


def get_chunker(chunk_size: int = 2048, device: str = None) -> SemanticChunker:
    """
    Get or create SemanticChunker instance for this process.

    Args:
        chunk_size: Maximum tokens per chunk
        device: Device to use for embeddings (e.g., 'cuda:0', 'cuda:1', etc.)
                If None, will auto-detect available GPU or use CPU

    Returns:
        SemanticChunker instance
    """
    global _chunker
    if _chunker is None:
        if device is None:
            import torch

            device = "cuda" if torch.cuda.is_available() else "cpu"

        import os

        pid = os.getpid()
        print(f"[PID {pid}] Initializing semantic chunker on device: {device}")

        _chunker = SemanticChunker(
            embedding_model="minishlab/potion-base-32M",
            chunk_size=chunk_size,
            similarity_window=4,
            skip_window=0,
            min_sentences_per_chunk=10,
            min_characters_per_sentence=24,
            model_kwargs={"device": device},
        )
    return _chunker


def count_tokens(text: str, chunker: SemanticChunker) -> int:
    """Count tokens in text using the chunker's tokenizer."""
    try:
        tokens = chunker.tokenizer.encode(text)
        return len(tokens)
    except Exception:
        return len(text) // 4


def merge_to_target_tokens(chunks, chunker, target_tokens: int = 2048, min_chars: int = 512):
    """
    Merge consecutive chunks to create segments with similar token counts near target.

    Strategy: Calculate total tokens, determine optimal number of segments,
    then merge chunks to create evenly-sized segments.

    Args:
        chunks: List of Chunk objects from Chonkie
        chunker: SemanticChunker instance for token counting
        target_tokens: Target number of tokens per chunk (default 2048)
        min_chars: Minimum characters per chunk (default 512)

    Returns:
        List of merged chunks with similar token counts
    """
    if not chunks:
        return chunks

    class MergedChunk:
        def __init__(self, text):
            self.text = text

    chunk_tokens = [count_tokens(chunk.text, chunker) for chunk in chunks]
    total_tokens = sum(chunk_tokens)

    num_segments = max(1, round(total_tokens / target_tokens))

    tokens_per_segment = total_tokens / num_segments

    merged = []
    current_text = ""
    current_tokens = 0

    for i, chunk in enumerate(chunks):
        chunk_text = chunk.text
        chunk_token_count = chunk_tokens[i]

        if current_text:
            current_text += " " + chunk_text
        else:
            current_text = chunk_text
        current_tokens += chunk_token_count

        is_last_chunk = i == len(chunks) - 1

        if is_last_chunk:
            merged_chunk = MergedChunk(current_text)
            merged.append(merged_chunk)
        elif current_tokens >= tokens_per_segment:
            if i + 1 < len(chunks):
                next_tokens = chunk_tokens[i + 1]
                current_distance = abs(current_tokens - tokens_per_segment)
                future_distance = abs(current_tokens + next_tokens - tokens_per_segment)

                if current_distance <= future_distance:
                    merged_chunk = MergedChunk(current_text)
                    merged.append(merged_chunk)
                    current_text = ""
                    current_tokens = 0

    final_merged = []
    i = 0
    while i < len(merged):
        current_chunk = merged[i]

        if len(current_chunk.text) < min_chars and i < len(merged) - 1:
            next_chunk = merged[i + 1]
            merged_text = current_chunk.text + " " + next_chunk.text
            final_merged.append(MergedChunk(merged_text))
            i += 2
        elif len(current_chunk.text) < min_chars and i == len(merged) - 1 and final_merged:
            last = final_merged.pop()
            merged_text = last.text + " " + current_chunk.text
            final_merged.append(MergedChunk(merged_text))
            i += 1
        else:
            final_merged.append(current_chunk)
            i += 1

    return final_merged


def process_paper(
    paper_path: Path,
    output_dir: Path,
    error_output_dir: Path,
    chunk_size: int,
    force: bool = False,
    retry_errors: bool = False,
    device: str = None,
) -> ProcessingResult:
    """
    Process a single paper: read, chunk, and save.

    Args:
        paper_path: Path to input paper file
        output_dir: Directory for successful chunk output
        error_output_dir: Directory for error files
        chunk_size: Maximum tokens per chunk
        force: Reprocess even if already done
        retry_errors: Reprocess if the paper failed in an earlier run

    Returns:
        ProcessingResult with statistics and status
    """
    started_time = datetime.now().isoformat()
    paper_id = paper_path.stem
    num_characters = 0
    num_chunks = 0
    error_type = None
    error_message = ""

    try:
        if not force:
            if is_paper_processed(paper_id, output_dir):
                return ProcessingResult(
                    paper_id=paper_id,
                    status="skipped",
                    started_time=started_time,
                    finished_time=datetime.now().isoformat(),
                )

            if not retry_errors and is_paper_failed(paper_id, error_output_dir):
                return ProcessingResult(
                    paper_id=paper_id,
                    status="skipped",
                    started_time=started_time,
                    finished_time=datetime.now().isoformat(),
                )

        try:
            text = paper_path.read_text(encoding="utf-8")
        except Exception as e:
            error_type = ErrorType.READ_ERROR
            error_message = f"Failed to read file: {str(e)}"
            raise

        num_characters = len(text)
        len(text.split())

        if num_characters < 10:
            error_type = ErrorType.EMPTY_FILE
            error_message = f"File too short: {num_characters} characters"
            raise ValueError(error_message)

        chunker = get_chunker(chunk_size, device)

        count_tokens(text, chunker)

        try:
            chunks = chunker.chunk(text)
            chunks = merge_to_target_tokens(chunks, chunker, target_tokens=chunk_size, min_chars=512)
        except Exception as e:
            error_type = ErrorType.CHUNKING_ERROR
            error_message = f"Chunking failed: {str(e)}"
            raise

        temp_dir = output_dir / f".tmp_{paper_id}"
        temp_dir.mkdir(parents=True, exist_ok=True)

        chunk_stats = []
        timestamp = datetime.now().isoformat()

        try:
            for i, chunk in enumerate(chunks, start=1):
                chunk_file = temp_dir / f"{i}.txt"
                try:
                    chunk_text = chunk.text.strip()

                    chunk_file.write_text(chunk_text, encoding="utf-8")

                    chunk_tokens = count_tokens(chunk_text, chunker)
                    chunk_words = len(chunk_text.split())
                    chunk_chars = len(chunk_text)

                    chunk_stat = ChunkStat(
                        paper_id=paper_id,
                        chunk_id=i,
                        num_tokens=chunk_tokens,
                        num_words=chunk_words,
                        num_characters=chunk_chars,
                        timestamp=timestamp,
                    )
                    chunk_stats.append(chunk_stat)

                except Exception as e:
                    error_type = ErrorType.WRITE_ERROR
                    error_message = f"Failed to write chunk {i}: {str(e)}"
                    raise

            num_chunks = len(chunks)

            final_dir = output_dir / paper_id
            if final_dir.exists():
                shutil.rmtree(final_dir)
            temp_dir.rename(final_dir)

        except Exception:
            if temp_dir.exists():
                shutil.rmtree(temp_dir)
            raise

        finished_time = datetime.now().isoformat()
        return ProcessingResult(
            paper_id=paper_id,
            status="success",
            chunk_stats=chunk_stats,
            num_chunks=num_chunks,
            started_time=started_time,
            finished_time=finished_time,
        )

    except Exception as e:
        finished_time = datetime.now().isoformat()

        if error_type is None:
            error_type = ErrorType.UNKNOWN_ERROR
            error_message = str(e)

        write_error_file(
            paper_id=paper_id,
            paper_path=paper_path,
            error_type=error_type,
            error_message=error_message,
            error_output_dir=error_output_dir,
        )

        return ProcessingResult(
            paper_id=paper_id,
            status="error",
            chunk_stats=[],
            num_chunks=0,
            started_time=started_time,
            finished_time=finished_time,
            error_type=error_type,
            error_message=error_message,
        )
