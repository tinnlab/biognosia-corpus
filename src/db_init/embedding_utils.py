#!/usr/bin/env python3
"""
Shared embedding utilities for database insertion and entity processing.

This module provides reusable functions for computing embeddings with
masked mean pooling and L2 normalization, ensuring consistency across
all database insertion scripts, chunk processing, and entity mapping.

Standard embedding computation pattern:
1. Masked mean pooling: Average token embeddings over non-padding tokens
2. L2 normalization: Normalize embeddings to unit length

This ensures:
- Identical embedding computation across all scripts
- Accurate similarity search in Milvus
- Consistent entity mapping and chunk retrieval
- Single source of truth for embedding logic
"""

import numpy as np
import torch
import torch.nn.functional as F


def masked_mean_pooling(token_embeddings: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """
    Apply masked mean pooling to token embeddings.

    Only averages over non-padding tokens (where attention_mask=1).

    Args:
        token_embeddings: Token-level embeddings [batch_size, seq_len, hidden_dim]
        attention_mask: Attention mask [batch_size, seq_len]

    Returns:
        Pooled embeddings [batch_size, hidden_dim]

    Example:
        >>> outputs = model(**inputs)
        >>> pooled = masked_mean_pooling(outputs.last_hidden_state, inputs['attention_mask'])
    """
    attention_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()

    sum_embeddings = torch.sum(token_embeddings * attention_mask_expanded, dim=1)

    sum_mask = torch.clamp(attention_mask_expanded.sum(dim=1), min=1e-9)

    return sum_embeddings / sum_mask


def normalize_embeddings(embeddings: torch.Tensor, p: int = 2, dim: int = 1) -> torch.Tensor:
    """
    L2 normalize embeddings to unit length.

    Args:
        embeddings: Input embeddings [batch_size, hidden_dim]
        p: Norm degree (default: 2 for L2 norm)
        dim: Dimension along which to normalize (default: 1)

    Returns:
        Normalized embeddings [batch_size, hidden_dim]

    Example:
        >>> normalized = normalize_embeddings(embeddings)
    """
    return F.normalize(embeddings, p=p, dim=dim)


def compute_embeddings(
    model: torch.nn.Module,
    tokenizer,
    texts: list[str],
    device: str,
    batch_size: int = 32,
    max_length: int = 8192,
    use_masked_pooling: bool = True,
    normalize: bool = True,
) -> np.ndarray:
    """
    Compute embeddings for a list of texts.

    Args:
        model: HuggingFace transformer model
        tokenizer: HuggingFace tokenizer
        texts: List of text strings to embed
        device: Device to run computation on ('cuda:0', 'cpu', etc.)
        batch_size: Batch size for processing
        max_length: Maximum sequence length for tokenization
        use_masked_pooling: If True, use masked mean pooling; if False, use CLS token
        normalize: If True, L2 normalize the embeddings

    Returns:
        Embeddings as numpy array [num_texts, hidden_dim]

    Example:
        >>> embeddings = compute_embeddings(
        ...     model=content_model,
        ...     tokenizer=content_tokenizer,
        ...     texts=["text 1", "text 2"],
        ...     device="cuda:0"
        ... )
    """
    all_embeddings = []

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]

        inputs = tokenizer(batch_texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt")

        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)

            if use_masked_pooling:
                embeddings = masked_mean_pooling(outputs.last_hidden_state, inputs["attention_mask"])
            else:
                embeddings = outputs.last_hidden_state[:, 0, :]

            if normalize:
                embeddings = normalize_embeddings(embeddings)

            embeddings = embeddings.cpu().numpy()
            all_embeddings.append(embeddings)

    return np.vstack(all_embeddings) if len(all_embeddings) > 1 else all_embeddings[0]


def compute_dual_embeddings(
    content_model: torch.nn.Module,
    content_tokenizer,
    label_model: torch.nn.Module,
    label_tokenizer,
    contents: list[str],
    labels: list[str],
    device: str,
    batch_size: int = 32,
    content_max_length: int = 8192,
    label_max_length: int = 512,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute both content and label embeddings for a list of items.

    Args:
        content_model: Model for content embeddings (e.g., BGE-M3)
        content_tokenizer: Tokenizer for content model
        label_model: Model for label embeddings (e.g., MedCPT)
        label_tokenizer: Tokenizer for label model
        contents: List of content strings
        labels: List of label strings
        device: Device to run computation on
        batch_size: Batch size for processing
        content_max_length: Max length for content tokenization
        label_max_length: Max length for label tokenization

    Returns:
        Tuple of (content_embeddings, label_embeddings) as numpy arrays

    Example:
        >>> content_embs, label_embs = compute_dual_embeddings(
        ...     content_model=content_model,
        ...     content_tokenizer=content_tokenizer,
        ...     label_model=label_model,
        ...     label_tokenizer=label_tokenizer,
        ...     contents=["content 1", "content 2"],
        ...     labels=["label 1", "label 2"],
        ...     device="cuda:0"
        ... )
    """
    content_embeddings = compute_embeddings(
        model=content_model,
        tokenizer=content_tokenizer,
        texts=contents,
        device=device,
        batch_size=batch_size,
        max_length=content_max_length,
        use_masked_pooling=True,
        normalize=True,
    )

    label_embeddings = compute_embeddings(
        model=label_model,
        tokenizer=label_tokenizer,
        texts=labels,
        device=device,
        batch_size=batch_size,
        max_length=label_max_length,
        use_masked_pooling=True,
        normalize=True,
    )

    return content_embeddings, label_embeddings
