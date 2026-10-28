"""Utility functions and constants for semantic chunking."""

import traceback
from datetime import datetime
from enum import StrEnum
from pathlib import Path


class ErrorType(StrEnum):
    """Error categories for chunking failures."""

    READ_ERROR = "READ_ERROR"
    EMPTY_FILE = "EMPTY_FILE"
    CHUNKING_ERROR = "CHUNKING_ERROR"
    WRITE_ERROR = "WRITE_ERROR"
    UNKNOWN_ERROR = "UNKNOWN_ERROR"


class ChunkStat:
    """Statistics for a single chunk."""

    def __init__(
        self,
        paper_id: str,
        chunk_id: int,
        num_tokens: int,
        num_words: int,
        num_characters: int,
        timestamp: str | None = None,
    ):
        self.paper_id = paper_id
        self.chunk_id = chunk_id
        self.num_tokens = num_tokens
        self.num_words = num_words
        self.num_characters = num_characters
        self.timestamp = timestamp or datetime.now().isoformat()

    def to_csv_row(self) -> list:
        """Convert to CSV row format."""
        return [
            self.paper_id,
            str(self.chunk_id),
            str(self.num_tokens),
            str(self.num_words),
            str(self.num_characters),
            self.timestamp,
        ]


class ProcessingResult:
    """Result of processing a single paper."""

    def __init__(
        self,
        paper_id: str,
        status: str,
        chunk_stats: list = None,
        num_chunks: int = 0,
        started_time: str | None = None,
        finished_time: str | None = None,
        error_type: ErrorType | None = None,
        error_message: str = "",
    ):
        self.paper_id = paper_id
        self.status = status
        self.chunk_stats = chunk_stats or []
        self.num_chunks = num_chunks
        self.started_time = started_time or datetime.now().isoformat()
        self.finished_time = finished_time or datetime.now().isoformat()
        self.error_type = error_type
        self.error_message = error_message

    def to_dict(self) -> dict:
        """Convert to dictionary format."""
        return {
            "paper_id": self.paper_id,
            "status": self.status,
            "num_chunks": self.num_chunks,
            "started_time": self.started_time,
            "finished_time": self.finished_time,
            "error_type": self.error_type.value if self.error_type else None,
            "error_message": self.error_message,
        }

    def to_paper_csv_row(self) -> list:
        """Convert to paper-level CSV row format."""
        if self.chunk_stats:
            total_tokens = sum(c.num_tokens for c in self.chunk_stats)
            total_words = sum(c.num_words for c in self.chunk_stats)
            total_chars = sum(c.num_characters for c in self.chunk_stats)
            avg_tokens = total_tokens / len(self.chunk_stats) if self.chunk_stats else 0
            avg_words = total_words / len(self.chunk_stats) if self.chunk_stats else 0
            avg_chars = total_chars / len(self.chunk_stats) if self.chunk_stats else 0
        else:
            total_tokens = total_words = total_chars = 0
            avg_tokens = avg_words = avg_chars = 0

        return [
            self.paper_id,
            str(self.num_chunks),
            str(total_tokens),
            str(total_words),
            str(total_chars),
            str(int(avg_tokens)),
            str(int(avg_words)),
            str(int(avg_chars)),
            self.started_time,
            self.finished_time,
            self.status,
            self.error_message,
        ]


def write_error_file(
    paper_id: str, paper_path: Path, error_type: ErrorType, error_message: str, error_output_dir: Path
) -> None:
    """Write error details to error output file."""
    error_file = error_output_dir / f"{paper_id}.txt"

    tb = traceback.format_exc()

    content = f"""Error Type: {error_type.value}
Timestamp: {datetime.now().isoformat()}
Paper ID: {paper_id}
Paper Path: {paper_path}

Error Message:
{error_message}

Full Traceback:
{tb}
"""

    try:
        error_file.write_text(content)
    except Exception as e:
        print(f"Warning: Failed to write error file for {paper_id}: {e}")


def is_paper_processed(paper_id: str, output_dir: Path) -> bool:
    """Check if paper was successfully processed."""
    paper_dir = output_dir / paper_id
    if not paper_dir.exists():
        return False

    txt_files = list(paper_dir.glob("*.txt"))
    return len(txt_files) > 0


def is_paper_failed(paper_id: str, error_output_dir: Path) -> bool:
    """Check if the paper failed in an earlier run."""
    error_file = error_output_dir / f"{paper_id}.txt"
    return error_file.exists()


def get_chunk_csv_header() -> list:
    """Get CSV header row for chunk statistics."""
    return ["id", "chunk", "num_tokens", "num_words", "num_characters", "timestamp"]


def get_paper_csv_header() -> list:
    """Get CSV header row for paper statistics."""
    return [
        "id",
        "num_chunks",
        "total_tokens",
        "total_words",
        "total_characters",
        "avg_tokens_per_chunk",
        "avg_words_per_chunk",
        "avg_chars_per_chunk",
        "started_time",
        "finished_time",
        "status",
        "error",
    ]
