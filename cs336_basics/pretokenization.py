from collections import Counter
from collections.abc import Iterator

import regex as re

# GPT-2 regex pattern
PAT_STR = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


def _to_byte_tuple(word: str) -> tuple[bytes, ...]:
    """Convert a string into a tuple of single-byte `bytes` objects via UTF-8 encoding.

    Args:
        word: The input string to convert.

    Returns:
        A tuple where each element is one UTF-8 byte of the input.

    Example:
        >>> _to_byte_tuple("hi")
        (b'h', b'i')
    """
    return tuple(bytes([b]) for b in word.encode("utf-8"))


def iter_pretokens(text: str, special_tokens: list[str]) -> Iterator[tuple[str, bytes | tuple[bytes, ...]]]:
    """Yield `(kind, value)` pairs for each special token and pre-token in `text`, in order.

    Args:
        text: The input string to pre-tokenize.
        special_tokens: Tokens preserved as atomic units. Longer tokens take precedence on overlap.

    Yields:
        `("special", token_bytes)` for special tokens, or `("text", byte_tuple)` for pre-tokens
            produced by the GPT-2 style regex.
    """
    special_tokens = special_tokens or []
    special_set = set(special_tokens)

    if special_tokens:
        sorted_special_tokens = sorted(special_tokens, key=len, reverse=True)
        safe_pattern = "(" + "|".join(re.escape(w) for w in sorted_special_tokens) + ")"
        chunks = re.split(safe_pattern, text)
    else:
        chunks = [text]

    for chunk in chunks:
        if not chunk:
            continue
        if chunk in special_set:
            yield ("special", chunk.encode("utf-8"))
        else:
            for m in re.finditer(PAT_STR, chunk):
                yield ("text", _to_byte_tuple(m.group()))


def compute_pretokens(corpus: str, special_tokens: list[str]) -> Counter[tuple[bytes, ...]]:
    """Pre-tokenize a corpus and count occurrences of each pre-token.

    Special tokens are excluded from the counts since they are never merged.

    Args:
        corpus: The training corpus to pre-tokenize.
        special_tokens: Tokens used as hard boundaries when splitting the corpus.

    Returns:
        A counter mapping each pre-token to its frequency in the corpus.
    """
    counter = Counter()
    for kind, value in iter_pretokens(corpus, special_tokens):
        if kind == "text":
            counter[value] += 1
    return counter
