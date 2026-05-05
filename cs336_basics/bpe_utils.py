def merge_pair_in_tuple(seq: tuple[bytes, ...], pair: tuple[bytes, bytes]) -> tuple[bytes, ...]:
    """Merge every non-overlapping occurrence of `pair` inside `seq`.

    Args:
        seq: The token sequence to scan.
        pair: The adjacent token pair to merge into a single concatenated token.

    Returns:
        A new tuple with each occurrence of `pair` replaced by `pair[0] + pair[1]`.

    Example:
        >>> merge_pair_in_tuple((b'l', b'o', b'l', b'o'), (b'l', b'o'))
        (b'lo', b'lo')
    """
    i = 0
    result = []

    while i < len(seq):
        if i < len(seq) - 1 and (seq[i], seq[i + 1]) == pair:
            result.append(seq[i] + seq[i + 1])
            i += 2
        else:
            result.append(seq[i])
            i += 1

    return tuple(result)
