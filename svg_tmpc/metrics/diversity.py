# svg_tmpc/metrics/diversity.py
"""Lexical-diversity metrics: distinct-n and pairwise self-BLEU."""

from __future__ import annotations

from typing import List, Sequence

from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu


def _tokenize(text: str) -> List[str]:
    return text.split()


def _ngrams(tokens: Sequence[str], n: int) -> List[tuple]:
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    if len(tokens) < n:
        return []
    return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


def distinct_n(texts: Sequence[str], n: int) -> float:
    """Ratio of unique n-grams to total n-grams across the corpus.

    Returns 0.0 when no n-grams exist (e.g. all texts shorter than n).
    """
    total: List[tuple] = []
    for text in texts:
        total.extend(_ngrams(_tokenize(text), n))
    if not total:
        return 0.0
    return len(set(total)) / len(total)


def self_bleu(texts: Sequence[str], max_n: int = 4) -> float:
    """Mean sentence-BLEU of each text against the rest of the corpus.

    Lower self-BLEU implies higher diversity. Returns 0.0 when fewer than two
    non-empty texts are present.
    """
    if max_n < 1:
        raise ValueError(f"max_n must be >= 1, got {max_n}")
    tokenized = [_tokenize(t) for t in texts]
    tokenized = [tok for tok in tokenized if tok]
    if len(tokenized) < 2:
        return 0.0

    weights = tuple([1.0 / max_n] * max_n)
    smoothing = SmoothingFunction().method1
    scores: List[float] = []
    for i, hyp in enumerate(tokenized):
        refs = [tokenized[j] for j in range(len(tokenized)) if j != i]
        score = sentence_bleu(refs, hyp, weights=weights, smoothing_function=smoothing)
        scores.append(float(score))
    return sum(scores) / len(scores)
