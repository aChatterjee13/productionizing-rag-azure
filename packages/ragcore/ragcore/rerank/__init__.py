"""Cross-encoder reranking (``Xenova/bge-reranker-v2-m3``) with a no-op fallback.

Importing this package is cheap — the FastEmbed cross-encoder is only pulled in when
:func:`get_reranker` first builds a :class:`CrossEncoderReranker`.
"""

from ragcore.rerank.base import (
    Reranker,
    RerankResult,
    get_reranker,
    reset_rerankers,
)
from ragcore.rerank.cross_encoder import CrossEncoderReranker, NoopReranker

__all__ = [
    "CrossEncoderReranker",
    "NoopReranker",
    "RerankResult",
    "Reranker",
    "get_reranker",
    "reset_rerankers",
]
