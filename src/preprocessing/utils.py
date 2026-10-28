"""Utility classes and functions for preprocessing."""

from dataclasses import dataclass
from pathlib import Path


@dataclass
class PreprocessingStat:
    """Statistics for a single preprocessed paper."""

    paper_id: str
    original_chars: int
    preprocessed_chars: int
    chars_removed: int
    encoding_fixed: bool
    latex_patterns_found: int
    language_detected: str
    quality_score: float
    started_time: str
    finished_time: str
    status: str
    error: str = ""

    def to_csv_row(self) -> list[str]:
        """Convert to CSV row."""
        return [
            self.paper_id,
            str(self.original_chars),
            str(self.preprocessed_chars),
            str(self.chars_removed),
            str(self.encoding_fixed),
            str(self.latex_patterns_found),
            self.language_detected,
            f"{self.quality_score:.4f}",
            self.started_time,
            self.finished_time,
            self.status,
            self.error,
        ]


class ProcessingError(Exception):
    """Base exception for preprocessing errors."""

    pass


class ValidationError(ProcessingError):
    """Exception for validation failures."""

    pass


class EncodingError(ProcessingError):
    """Exception for encoding issues."""

    pass


def get_paper_id(file_path: Path) -> str:
    """Extract paper ID from file path."""
    return file_path.stem


def should_skip_paper(paper_id: str, output_dir: Path, error_output_dir: Path, force: bool, retry_errors: bool) -> bool:
    """
    Determine if a paper should be skipped.

    Args:
        paper_id: Paper identifier
        output_dir: Output directory for preprocessed papers
        error_output_dir: Error output directory
        force: Force reprocessing even if already processed
        retry_errors: Retry papers that failed in an earlier run

    Returns:
        True if paper should be skipped, False otherwise
    """
    if force:
        return False

    output_file = output_dir / f"{paper_id}.txt"
    error_file = error_output_dir / f"{paper_id}.txt"

    if output_file.exists():
        return True

    if error_file.exists() and not retry_errors:
        return True

    return False


def write_error_file(error_output_dir: Path, paper_id: str, error_message: str):
    """Write error information to error output directory."""
    error_output_dir.mkdir(parents=True, exist_ok=True)
    error_file = error_output_dir / f"{paper_id}.txt"
    error_file.write_text(error_message, encoding="utf-8")


def categorize_error(error: Exception) -> str:
    """Categorize error for reporting."""
    if isinstance(error, ValidationError):
        return f"ValidationError: {str(error)}"
    elif isinstance(error, EncodingError):
        return f"EncodingError: {str(error)}"
    elif isinstance(error, UnicodeDecodeError):
        return f"UnicodeDecodeError: {str(error)}"
    elif isinstance(error, FileNotFoundError):
        return f"FileNotFoundError: {str(error)}"
    else:
        return f"{type(error).__name__}: {str(error)}"
