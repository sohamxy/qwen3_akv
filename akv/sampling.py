"""Sampling strategies for next-token generation from raw logits."""

import numpy as np


def sample_next_token(
    logits: np.ndarray,
    temperature: float = 0.7,
    top_p: float = 0.8,
    top_k: int = 20,
) -> int:
    """Samples next token ID using Top-K + Top-P + Temperature sampling.
    
    Args:
        logits: 1D array of logits for the vocabulary.
        temperature: Sampling temperature. <= 0.0 means greedy decoding.
        top_p: Nucleus sampling threshold in (0.0, 1.0].
        top_k: Top-K filtering count. <= 0 or >= vocab_size disables filtering.
        
    Returns:
        Sampled token ID (integer).
    """
    vocab_size = logits.shape[0]

    # Greedy decoding
    if temperature <= 0.0 or top_k == 1:
        return int(np.argmax(logits))

    # 1. Top-K filtering
    if 0 < top_k < vocab_size:
        top_k_indices = np.argpartition(logits, -top_k)[-top_k:]
        top_k_logits = logits[top_k_indices]
    else:
        top_k_indices = np.arange(vocab_size)
        top_k_logits = logits

    # 2. Temperature scaling & Softmax in float64 for numerical stability
    scaled = (top_k_logits / temperature).astype(np.float64)
    scaled -= np.max(scaled)
    probs = np.exp(scaled)
    sum_probs = np.sum(probs)

    if sum_probs <= 0.0 or np.isnan(sum_probs):
        # Fallback to argmax if logits produced zero or NaN probabilities
        return int(top_k_indices[np.argmax(top_k_logits)])

    probs /= sum_probs

    # 3. Top-P (nucleus) filtering
    if 0.0 < top_p < 1.0:
        sort_order = np.argsort(probs)[::-1]
        sorted_probs = probs[sort_order]
        sorted_indices = top_k_indices[sort_order]
        cumulative_probs = np.cumsum(sorted_probs)

        cutoff_idx = np.searchsorted(cumulative_probs, top_p) + 1
        kept_probs = sorted_probs[:cutoff_idx]
        kept_indices = sorted_indices[:cutoff_idx]

        kept_sum = np.sum(kept_probs)
        if kept_sum <= 0.0 or np.isnan(kept_sum):
            return int(sorted_indices[0])

        kept_probs = kept_probs / kept_sum
        return int(np.random.choice(kept_indices, p=kept_probs))

    return int(np.random.choice(top_k_indices, p=probs))
