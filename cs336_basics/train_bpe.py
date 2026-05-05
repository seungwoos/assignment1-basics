import os
from collections import Counter

from .pretokenization import compute_pretokens
from .bpe_utils import merge_pair_in_tuple


def build_vocab(special_tokens: list[str]) -> dict[int, bytes]:
    """Build the initial BPE vocabulary: 256 byte tokens followed by special tokens.

    Args:
        special_tokens: Tokens to reserve in the vocabulary, appended starting at id 256.

    Returns:
        A dictionary mapping token ids to their `bytes` representation.

    Example:
        >>> vocab = build_vocab(["<|endoftext|>"])
        >>> vocab[0], vocab[256]
        (b'\\x00', b'<|endoftext|>')
    """
    vocab: dict[int, bytes] = {idx: bytes([idx]) for idx in range(256)}
    for i, token in enumerate(special_tokens):
        vocab[256 + i] = token.encode("utf-8")

    return vocab


def _compute_initial_pair_counts(pretokens: Counter[tuple[bytes, ...]]) -> Counter[tuple[bytes, bytes]]:
    """Count the frequency of every adjacent byte pair across all pre-tokens.

    Args:
        pretokens: Pre-token frequencies.

    Returns:
        A counter mapping each `(left, right)` byte pair to its total frequency.
    """
    pairs: Counter[tuple[bytes, bytes]] = Counter()
    for pretoken, freq in pretokens.items():
        for i in range(len(pretoken) - 1):
            pair = pretoken[i : i + 2]
            pairs[pair] += freq

    return pairs


def _apply_merge_step(
    pretokens: Counter[tuple[bytes, ...]], pairs: Counter[tuple[bytes, bytes]], merge_pair: tuple[bytes, bytes]
) -> Counter[tuple[bytes, ...]]:
    """Apply a single BPE merge to all pre-tokens and update pair counts in place.

    Args:
        pretokens: Current pre-token frequencies.
        pairs: Current adjacent pair frequencies. Mutated in place.
        merge_pair: The `(left, right)` byte pair to merge.

    Returns:
        The updated pre-token frequencies after applying the merge.
    """
    new_pretokens: Counter[tuple[bytes, ...]] = Counter()
    for seq, freq in pretokens.items():
        new_seq = merge_pair_in_tuple(seq, merge_pair)
        if new_seq == seq:
            new_pretokens[seq] += freq
            continue

        for pair in zip(seq, seq[1:]):
            pairs[pair] -= freq
            if pairs[pair] == 0:
                del pairs[pair]

        for pair in zip(new_seq, new_seq[1:]):
            pairs[pair] += freq

        new_pretokens[new_seq] += freq

    return new_pretokens


def merge(
    pretokens: Counter[tuple[bytes, ...]], vocab_size: int, vocab: dict[int, bytes]
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Run the BPE merge loop until the vocabulary reaches `vocab_size`.

    Repeatedly merges the most frequent adjacent byte pair into a new token. Ties are broken
    by taking the lexicographically greatest pair.

    Args:
        pretokens: Pre-token frequencies from :func:`compute_pretokens`.
        vocab_size: Target final vocabulary size. Stops early if no pairs remain.
        vocab: Initial vocabulary from :func:`build_vocab`. Mutated in place.

    Returns:
        A tuple `(vocab, merges)` of the final vocabulary and the ordered list of merge rules.
    """
    pairs = _compute_initial_pair_counts(pretokens)
    merges: list[tuple[bytes, bytes]] = []
    while len(vocab) < vocab_size:
        if not pairs:
            break

        most_freq_pair = max(pairs, key=lambda k: (pairs[k], k))
        vocab[len(vocab)] = most_freq_pair[0] + most_freq_pair[1]
        merges.append(most_freq_pair)
        pretokens = _apply_merge_step(pretokens, pairs, most_freq_pair)

    return vocab, merges


def train_bpe(
    input_path: str | os.PathLike, vocab_size: int, special_tokens: list[str], **kwargs
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Train a byte-level BPE tokenizer on the given corpus.

    Builds the initial vocabulary, pre-tokenizes the corpus, and runs the BPE merge loop.

    Args:
        input_path: Path to a UTF-8 encoded text corpus.
        vocab_size: Target final vocabulary size, including byte and special tokens.
        special_tokens: Tokens preserved as atomic units during pre-tokenization.
        **kwargs: Reserved for future options; currently ignored.

    Returns:
        A tuple `(vocab, merges)` of the trained vocabulary and the ordered list of merge rules.
    """
    with open(input_path, encoding="utf-8") as f:
        corpus = f.read()

    vocab = build_vocab(special_tokens)
    pretokens = compute_pretokens(corpus, special_tokens)
    vocab, merges = merge(pretokens, vocab_size, vocab)

    return vocab, merges


if __name__ == "__main__":
    from pathlib import Path

    input_path = Path(__file__).parent.parent / "tests" / "fixtures" / "tinystories_sample.txt"
    special_tokens = ["<|endoftext|>"]

    vocab, merges = train_bpe(input_path=input_path, vocab_size=500, special_tokens=special_tokens)

    print(merges)
