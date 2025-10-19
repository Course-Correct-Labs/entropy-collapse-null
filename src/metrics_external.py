"""
External metrics: ΔI drift, n-gram novelty, character entropy.

These metrics are computed from generated text and measure
observable behavioral properties.
"""

from typing import List, Set
from collections import Counter
import numpy as np


def compute_delta_i_drift(text1: str, text2: str, n: int = 3) -> float:
    """
    Compute ΔI drift between two text segments using n-gram overlap.

    ΔI = 1 - (intersection / union) of n-gram sets

    Args:
        text1: First text segment
        text2: Second text segment
        n: N-gram size (default 3)

    Returns:
        ΔI drift value [0, 1]
    """
    def get_ngrams(text: str, n: int) -> Set[str]:
        tokens = text.split()
        return set(' '.join(tokens[i:i+n]) for i in range(len(tokens) - n + 1))

    ngrams1 = get_ngrams(text1, n)
    ngrams2 = get_ngrams(text2, n)

    if not ngrams1 or not ngrams2:
        return 1.0

    intersection = len(ngrams1 & ngrams2)
    union = len(ngrams1 | ngrams2)

    if union == 0:
        return 0.0

    return 1.0 - (intersection / union)


def compute_ngram_novelty(text: str, n: int = 3) -> float:
    """
    Compute n-gram novelty (fraction of unique n-grams).

    Novelty = unique_ngrams / total_ngrams

    Args:
        text: Input text
        n: N-gram size (default 3)

    Returns:
        Novelty value [0, 1]
    """
    tokens = text.split()
    ngrams = [' '.join(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]

    if not ngrams:
        return 0.0

    unique = len(set(ngrams))
    total = len(ngrams)

    return unique / total


def compute_char_entropy(text: str) -> float:
    """
    Compute character-level Shannon entropy.

    H = -sum(p(c) * log2(p(c))) for each character c

    Args:
        text: Input text

    Returns:
        Character entropy (bits)
    """
    if not text:
        return 0.0

    # Count character frequencies
    counts = Counter(text)
    total = len(text)

    # Compute probabilities
    probs = np.array([count / total for count in counts.values()])

    # Shannon entropy
    entropy = -np.sum(probs * np.log2(probs))

    return float(entropy)


def detect_repetition(text: str, window_size: int = 50, threshold: float = 0.8) -> bool:
    """
    Detect excessive repetition in text.

    Args:
        text: Input text
        window_size: Size of sliding window
        threshold: Overlap threshold for repetition

    Returns:
        True if repetition detected
    """
    tokens = text.split()

    if len(tokens) < window_size * 2:
        return False

    # Compare consecutive windows
    for i in range(len(tokens) - window_size * 2):
        window1 = set(tokens[i:i+window_size])
        window2 = set(tokens[i+window_size:i+window_size*2])

        if not window1 or not window2:
            continue

        overlap = len(window1 & window2) / len(window1 | window2)

        if overlap > threshold:
            return True

    return False
