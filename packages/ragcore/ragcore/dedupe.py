"""Content fingerprinting and near-duplicate detection (requirement #9).

Two layers, because real corpora produce two kinds of duplicate:

* **Exact** — the same paragraph ingested from two sources, or a document re-fetched
  unchanged. Caught by :func:`content_sha256` over normalised text.
* **Near** — a policy restated with a new date, a page reproduced with different
  boilerplate, an email quoted inside its own reply. Caught by :func:`simhash64`, a
  locality-sensitive hash whose Hamming distance approximates textual distance.

Normalisation matters more than the hash: without collapsing whitespace and case, a
re-export with different line wrapping looks like new content and the exact layer
never fires.

:func:`dedupe_chunks` always **returns** its drops with a reason rather than
discarding them, so ``RetrievalResult.dropped`` can explain why a chunk the user
expected to see is missing.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from collections.abc import Callable

__all__ = [
    "SIMHASH_BITS",
    "content_sha256",
    "dedupe_chunks",
    "hamming64",
    "is_near_duplicate",
    "normalise_text",
    "shingles",
    "simhash64",
    "simhash_hex",
]

#: Width of the simhash. 64 bits is the standard choice: wide enough that unrelated
#: texts sit far apart, narrow enough to compare with one XOR and a popcount.
SIMHASH_BITS = 64

_MASK64 = (1 << SIMHASH_BITS) - 1

#: Runs of anything that is not a letter or digit become a single space. Unicode-aware
#: so accented and non-Latin text tokenises rather than collapsing to nothing.
_NON_WORD = re.compile(r"[^\w]+", re.UNICODE)


def normalise_text(text: str) -> str:
    """Canonicalise text so cosmetic differences do not defeat hashing.

    Applies NFKC normalisation (so a full-width or ligature form matches its plain
    equivalent), lowercases, replaces every non-word run with a single space, and
    strips. The result is what both hash layers operate on.

    Args:
        text: Raw text.

    Returns:
        The normalised text.
    """
    folded = unicodedata.normalize("NFKC", text).casefold()
    return _NON_WORD.sub(" ", folded).strip()


def content_sha256(text: str) -> str:
    """Hash text for exact-duplicate detection.

    Hashes the **normalised** form, so two chunks differing only in whitespace, case
    or Unicode composition share a hash. Stored in
    ``ChunkPayload.content_sha256`` and indexed, so ingestion can ask Qdrant whether
    a chunk is already present.

    Args:
        text: Raw text.

    Returns:
        A 64-character lowercase hex digest.
    """
    return hashlib.sha256(normalise_text(text).encode()).hexdigest()


def shingles(text: str, *, width: int = 4) -> list[str]:
    """Split normalised text into overlapping word shingles.

    Shingles rather than bare words are what make simhash sensitive to word *order*:
    "the cat sat on the mat" and "the mat sat on the cat" share every unigram but
    almost no 4-gram.

    Args:
        text: Raw text; normalised internally.
        width: Words per shingle (``settings.dedupe_simhash_shingle``).

    Returns:
        The shingles, in order. A text shorter than ``width`` words yields its words
        as single-word shingles rather than nothing, so short chunks still hash to
        something meaningful.

    Raises:
        ValueError: If ``width`` is not positive.
    """
    if width <= 0:
        msg = f"shingle width must be positive, got {width}"
        raise ValueError(msg)
    tokens = normalise_text(text).split()
    if not tokens:
        return []
    if len(tokens) <= width:
        return tokens
    return [
        " ".join(tokens[index : index + width])
        for index in range(len(tokens) - width + 1)
    ]


def _hash_shingle(shingle: str) -> int:
    """Hash one shingle to 64 bits.

    Uses BLAKE2b rather than :func:`hash`, whose per-process randomisation would make
    a stored simhash meaningless across restarts.

    Args:
        shingle: The shingle text.

    Returns:
        An unsigned 64-bit integer.
    """
    digest = hashlib.blake2b(shingle.encode(), digest_size=8).digest()
    return int.from_bytes(digest, "big")


def simhash64(text: str, *, shingle: int = 4) -> int:
    """Compute the 64-bit simhash of a text.

    Each shingle is hashed, then every bit position votes ``+1`` or ``-1`` depending on
    whether it is set in that shingle's hash. The output bit is 1 where the vote is
    positive. Similar texts share most shingles, so their votes — and therefore most
    output bits — agree.

    Args:
        text: Raw text.
        shingle: Words per shingle (``settings.dedupe_simhash_shingle``).

    Returns:
        An unsigned 64-bit integer. Empty or word-free text hashes to 0, which callers
        should treat as "no fingerprint" rather than as a value to compare.
    """
    pieces = shingles(text, width=shingle)
    if not pieces:
        return 0
    votes = [0] * SIMHASH_BITS
    for piece in pieces:
        value = _hash_shingle(piece)
        for bit in range(SIMHASH_BITS):
            if value >> bit & 1:
                votes[bit] += 1
            else:
                votes[bit] -= 1
    result = 0
    for bit in range(SIMHASH_BITS):
        if votes[bit] > 0:
            result |= 1 << bit
    return result & _MASK64


def simhash_hex(text: str, *, shingle: int = 4) -> str:
    """Compute the simhash as 16 hex characters.

    This is the form stored in ``ChunkPayload.simhash``, whose validator requires
    exactly 16 lowercase hex characters.

    Args:
        text: Raw text.
        shingle: Words per shingle (``settings.dedupe_simhash_shingle``).

    Returns:
        A zero-padded 16-character lowercase hex string.
    """
    return f"{simhash64(text, shingle=shingle):016x}"


def hamming64(a: int, b: int) -> int:
    """Hamming distance between two 64-bit integers.

    Args:
        a: First value.
        b: Second value.

    Returns:
        The number of differing bit positions, 0..64.
    """
    return ((a ^ b) & _MASK64).bit_count()


def _parse_hex(value: str) -> int | None:
    """Parse a stored hex simhash.

    Args:
        value: A 16-character hex string, or an empty/blank string.

    Returns:
        The integer value, or None when the string is empty or not hexadecimal — a
        missing fingerprint must never be treated as the value zero, or every
        unfingerprinted chunk would look like a duplicate of every other.
    """
    text = value.strip()
    if not text:
        return None
    try:
        return int(text, 16) & _MASK64
    except ValueError:
        return None


def is_near_duplicate(a: str, b: str, *, max_distance: int = 3) -> bool:
    """Whether two hex simhashes are within the near-duplicate threshold.

    Args:
        a: First hex simhash (as stored in ``ChunkPayload.simhash``).
        b: Second hex simhash.
        max_distance: Inclusive Hamming distance threshold
            (``settings.dedupe_max_distance``).

    Returns:
        True when both fingerprints are present and their distance is at or below the
        threshold. False when either is missing or unparseable.
    """
    left = _parse_hex(a)
    right = _parse_hex(b)
    if left is None or right is None:
        return False
    return hamming64(left, right) <= max_distance


def dedupe_chunks[T](
    chunks: list[T],
    *,
    key: Callable[[T], str],
    simhash_key: Callable[[T], str],
    max_distance: int = 3,
) -> tuple[list[T], list[tuple[T, str]]]:
    """Drop exact and near duplicates, returning every drop with its reason.

    Order is significant: the first occurrence wins, so callers should pass candidates
    already sorted best-first (fusion score descending) and the survivor will be the
    higher-ranked copy.

    Args:
        chunks: Candidates to deduplicate.
        key: Extracts the exact content hash, e.g.
            ``lambda c: c.payload.content_sha256``.
        simhash_key: Extracts the hex simhash, e.g. ``lambda c: c.payload.simhash``.
            Return an empty string to opt a candidate out of near-duplicate checking
            (the ingestion pipeline does this for chunks below
            ``settings.dedupe_min_chunk_chars``, which are too short to compare
            meaningfully).
        max_distance: Inclusive Hamming threshold for the near-duplicate layer
            (``settings.dedupe_max_distance``).

    Returns:
        A ``(kept, dropped)`` pair, where ``dropped`` holds ``(chunk, reason)`` tuples
        with reasons ``"duplicate:sha256"`` or ``"duplicate:simhash:<distance>"``.
        Drops are always returned for audit, never silently discarded — they become
        ``RetrievedChunk.dropped_reason`` in ``RetrievalResult.dropped``.

    Example:
        >>> kept, dropped = dedupe_chunks(
        ...     ["a", "b"], key=lambda s: s, simhash_key=lambda _: ""
        ... )
        >>> len(kept), len(dropped)
        (2, 0)
    """
    kept: list[T] = []
    dropped: list[tuple[T, str]] = []
    seen_hashes: set[str] = set()
    kept_simhashes: list[int] = []

    for chunk in chunks:
        exact = key(chunk).strip()
        if exact and exact in seen_hashes:
            dropped.append((chunk, "duplicate:sha256"))
            continue

        fingerprint = _parse_hex(simhash_key(chunk))
        if fingerprint is not None:
            distance: int | None = None
            for existing in kept_simhashes:
                candidate_distance = hamming64(fingerprint, existing)
                if candidate_distance <= max_distance:
                    distance = candidate_distance
                    break
            if distance is not None:
                dropped.append((chunk, f"duplicate:simhash:{distance}"))
                continue
            kept_simhashes.append(fingerprint)

        if exact:
            seen_hashes.add(exact)
        kept.append(chunk)

    return kept, dropped
