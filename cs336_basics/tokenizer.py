from collections.abc import Iterable

from .pretokenization import iter_pretokens
from .bpe_utils import merge_pair_in_tuple


class Tokenizer:
    """Byte-level BPE tokenizer that encodes text to token ids and decodes back."""

    def __init__(
        self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str] | None = None
    ):
        """Initialize the tokenizer from a trained vocabulary and merge list.

        Args:
            vocab: Mapping from token id to its `bytes` representation.
            merges: Ordered list of `(left, right)` byte pairs to apply during encoding.
            special_tokens: Tokens preserved as atomic units during encoding.
        """
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens
        self.byte_to_id = {token_bytes: token_id for token_id, token_bytes in self.vocab.items()}

    @classmethod
    def from_files(cls, vocab_filepath: str, merges_filepath: str, special_tokens=None):
        """Construct a tokenizer by loading the vocabulary and merges from disk.

        Args:
            vocab_filepath: Path to the serialized vocabulary file.
            merges_filepath: Path to the serialized merges file.
            special_tokens: Tokens preserved as atomic units during encoding.

        Returns:
            A new :class:`Tokenizer` instance.
        """
        raise NotImplementedError("Will be implemented later.")

    def _apply_merge(self, pretoken: tuple[bytes, ...]) -> tuple[bytes, ...]:
        """Apply all merge rules in order to a single pre-token.

        Args:
            pretoken: A tuple of `bytes` representing one pre-token.

        Returns:
            The pre-token after all applicable merges have been applied.
        """
        for merge in self.merges:
            pretoken = merge_pair_in_tuple(pretoken, merge)
        return pretoken

    def encode(self, text: str) -> list[int]:
        """Encode a string into a list of token ids.

        Args:
            text: The input string to encode.

        Returns:
            The token ids corresponding to `text`.
        """
        token_ids = []
        for kind, value in iter_pretokens(text, self.special_tokens):
            if kind == "special":
                token_ids.append(self.byte_to_id[value])
            else:
                merged = self._apply_merge(value)
                for tok in merged:
                    token_ids.append(self.byte_to_id[tok])
        return token_ids

    def encode_iterable(self, iterable: Iterable[str]) -> Iterable[int]:
        """Lazily encode an iterable of strings, yielding token ids one by one.

        Args:
            iterable: An iterable of string chunks (e.g., lines from a file).

        Yields:
            Token ids from each encoded chunk in order.
        """
        for chunk in iterable:
            yield from self.encode(chunk)

    def decode(self, ids: list[int]) -> str:
        """Decode a list of token ids back into a string.

        Invalid UTF-8 byte sequences are replaced with the Unicode replacement character.

        Args:
            ids: The token ids to decode.

        Returns:
            The decoded string.
        """
        out = [self.vocab[idx] for idx in ids]
        output_bytes = b"".join(out)
        return output_bytes.decode("utf-8", errors="replace")
