"""Core text preprocessing using established packages."""

import chardet
import ftfy
import regex as re
from langdetect import LangDetectException, detect
from pylatexenc.latex2text import LatexNodes2Text
from unidecode import unidecode

from .config import GREEK_FULL_NAMES, GREEK_PATTERNS, PREPROCESSING_CONFIG
from .utils import ValidationError

_latex_converter = None


def get_latex_converter() -> LatexNodes2Text:
    """Get or create LaTeX converter instance."""
    global _latex_converter
    if _latex_converter is None:
        _latex_converter = LatexNodes2Text()
    return _latex_converter


def fix_encoding(text: str, original_bytes: bytes = None) -> tuple[str, bool]:
    """
    Fix encoding issues using chardet and ftfy.

    Args:
        text: Input text (may have encoding issues)
        original_bytes: Optional original bytes for re-decoding

    Returns:
        (fixed_text, was_fixed) tuple
    """
    original_text = text
    was_fixed = False

    if original_bytes:
        try:
            detected = chardet.detect(original_bytes)
            if detected["encoding"] and detected["confidence"] > 0.7:
                text = original_bytes.decode(detected["encoding"], errors="replace")
                was_fixed = True
        except Exception:
            pass

    fixed_text = ftfy.fix_text(text)
    if fixed_text != original_text:
        was_fixed = True

    return fixed_text, was_fixed


def clean_latex(text: str) -> tuple[str, int]:
    """
    Clean LaTeX notation using manual patterns before pylatexenc.

    Args:
        text: Input text with LaTeX notation

    Returns:
        (cleaned_text, num_patterns_found) tuple
    """
    original_text = text

    text = re.sub(r"\\cdot\s*", "", text)

    text = re.sub(r"\\lor\s*", " or ", text)
    text = re.sub(r"\\land\s*", " and ", text)

    text = re.sub(r"\\times\s*", "x", text)
    text = re.sub(r"\\pm\s*", "+/-", text)

    text = re.sub(r"\\mathsf\{([^}]+)\}", r"\1", text)
    text = re.sub(r"\\mathbf\{([^}]+)\}", r"\1", text)
    text = re.sub(r"\\mathrm\{([^}]+)\}", r"\1", text)
    text = re.sub(r"\\mathit\{([^}]+)\}", r"\1", text)

    text = re.sub(
        r"\\up(alpha|beta|gamma|delta|epsilon|zeta|eta|theta|iota|kappa|lambda|mu|nu|xi|omicron|pi|rho|sigma|tau|upsilon|phi|chi|psi|omega)",
        r"\\\1",
        text,
    )

    converter = get_latex_converter()

    try:
        cleaned_text = converter.latex_to_text(text)
    except Exception:
        cleaned_text = text
        cleaned_text = re.sub(r"\$([^\$]+)\$", r"\1", cleaned_text)

    num_patterns = len(re.findall(r"[\$\\]", original_text))

    return cleaned_text, num_patterns


def convert_unicode_to_ascii(text: str) -> str:
    """
    Convert Unicode characters to ASCII using unidecode.

    Args:
        text: Input text with Unicode characters

    Returns:
        ASCII-only text
    """
    for greek_char, name in GREEK_FULL_NAMES.items():
        text = text.replace(greek_char, name)

    text = unidecode(text)

    return text


def refine_greek_letters(text: str) -> str:
    """
    Refine Greek letter conversions in scientific contexts.

    After unidecode converts α->a, β->b, etc., this function
    replaces single letters with full names in scientific contexts.

    Args:
        text: Input text (already ASCII from unidecode)

    Returns:
        Text with refined Greek letter names
    """
    for pattern, replacement in GREEK_PATTERNS.items():
        text = re.sub(pattern, replacement, text)

    return text


def remove_citations(text: str) -> str:
    """
    Remove citation references from text.

    Removes patterns like:
    - [1]
    - [1,2,3,4]
    - [1-3]
    - [1,2,5-8]
    - [1, 2, 3]
    - [1 - 3]

    Args:
        text: Input text

    Returns:
        Text with citations removed
    """
    text = re.sub(r"\[\s*\d+(?:\s*[-,]\s*\d+)*\s*\]", "", text)

    return text


def normalize_whitespace(text: str) -> str:
    """
    Normalize whitespace using regex.

    Args:
        text: Input text

    Returns:
        Text with normalized whitespace
    """
    text = re.sub(r"(TNF|TGF|IL|NF|IFN|PGC)- +(alpha|beta|gamma|delta)", r"\1-\2", text)

    text = re.sub(r"(IL|TNF|TGF|NF|IFN)- +(\d)", r"\1-\2", text)

    text = re.sub(r" +", " ", text)

    text = text.replace("\t", " ")

    text = text.replace("\r\n", "\n").replace("\r", "\n")

    text = re.sub(r"\n{3,}", "\n\n", text)

    text = re.sub(r" +\n", "\n", text)

    text = text.strip()

    return text


def fix_hyphenation(text: str) -> str:
    """
    Fix end-of-line hyphenation.

    Args:
        text: Input text

    Returns:
        Text with fixed hyphenation
    """
    text = re.sub(r"(\w+)-\s*\n\s*(\w+)", r"\1\2", text)

    return text


def remove_duplicate_paragraphs(text: str) -> str:
    """
    Remove duplicate paragraphs using fuzzy matching.

    Args:
        text: Input text

    Returns:
        Text with duplicates removed
    """
    try:
        from rapidfuzz import fuzz
    except ImportError:
        return text

    paragraphs = text.split("\n\n")
    if len(paragraphs) <= 1:
        return text

    unique_paragraphs = []
    seen_paragraphs = []

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        is_duplicate = False
        for seen in seen_paragraphs:
            similarity = fuzz.ratio(para, seen)
            if similarity > 95:
                is_duplicate = True
                break

        if not is_duplicate:
            unique_paragraphs.append(para)
            seen_paragraphs.append(para)

    return "\n\n".join(unique_paragraphs)


def validate_paper(text: str, min_length: int, max_length: int, check_language: bool) -> tuple[str, float]:
    """
    Validate paper quality and language.

    Args:
        text: Input text
        min_length: Minimum character length
        max_length: Maximum character length
        check_language: Whether to check language

    Returns:
        (language_code, quality_score) tuple

    Raises:
        ValidationError: If validation fails
    """
    text_len = len(text)
    if text_len < min_length:
        raise ValidationError(f"Text too short: {text_len} chars (min: {min_length})")
    if text_len > max_length:
        raise ValidationError(f"Text too long: {text_len} chars (max: {max_length})")

    language = "unknown"
    if check_language and text_len > 50:
        try:
            sample = text[:1000]
            language = detect(sample)
            if language != "en":
                raise ValidationError(f"Non-English language detected: {language}")
        except LangDetectException:
            language = "unknown"

    quality_score = calculate_quality_score(text)

    return language, quality_score


def calculate_quality_score(text: str) -> float:
    """
    Calculate quality score for text.

    Args:
        text: Input text

    Returns:
        Quality score (0-1)
    """
    if not text:
        return 0.0

    words = re.findall(r"\b\w+\b", text)
    if not words:
        return 0.0

    avg_word_len = sum(len(w) for w in words) / len(words)

    alphanum_chars = sum(1 for c in text if c.isalnum() or c.isspace())
    alphanum_ratio = alphanum_chars / len(text)

    word_score = min(1.0, avg_word_len / 6.0)
    char_score = alphanum_ratio

    quality_score = word_score * 0.3 + char_score * 0.7

    return min(1.0, max(0.0, quality_score))


def preprocess_text(text: str, original_bytes: bytes = None) -> tuple[str, dict]:
    """
    Main preprocessing pipeline.

    Args:
        text: Input text
        original_bytes: Optional original bytes for encoding detection

    Returns:
        (preprocessed_text, stats_dict) tuple

    Raises:
        ValidationError: If validation fails
    """
    config = PREPROCESSING_CONFIG
    stats = {
        "encoding_fixed": False,
        "latex_patterns_found": 0,
        "language_detected": "unknown",
        "quality_score": 0.0,
    }

    if config["fix_encoding"]:
        text, encoding_fixed = fix_encoding(text, original_bytes)
        stats["encoding_fixed"] = encoding_fixed

    if config["clean_latex"]:
        text, latex_count = clean_latex(text)
        stats["latex_patterns_found"] = latex_count

    text = remove_citations(text)

    if config["unicode_to_ascii"]:
        text = convert_unicode_to_ascii(text)
        text = refine_greek_letters(text)

    if config["normalize_whitespace"]:
        text = normalize_whitespace(text)

    if config["fix_hyphenation"]:
        text = fix_hyphenation(text)

    if config["remove_duplicates"]:
        text = remove_duplicate_paragraphs(text)

    language, quality_score = validate_paper(text, config["min_length"], config["max_length"], config["check_language"])
    stats["language_detected"] = language
    stats["quality_score"] = quality_score

    if quality_score < config["quality_threshold"]:
        raise ValidationError(f"Quality score too low: {quality_score:.4f} (threshold: {config['quality_threshold']})")

    return text, stats
