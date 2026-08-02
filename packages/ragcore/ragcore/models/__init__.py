"""Data models shared by every service.

Import from this package rather than from the individual modules so a future
reorganisation stays internal:

    from ragcore.models import ChunkPayload, Principal, RetrievalResult
"""

from ragcore.models.acl import ADMIN_ROLE, AccessControl, Classification, Principal
from ragcore.models.chat import (
    ChatRequest,
    ContextStats,
    GuardrailAction,
    GuardrailEvent,
    GuardrailKind,
    GuardrailStage,
    Message,
    Role,
    SSEEvent,
    ToolCall,
)
from ragcore.models.chunk import ChunkPayload, SourceType
from ragcore.models.document import (
    BlockKind,
    IngestAction,
    IngestManifest,
    IngestManifestEntry,
    IngestRunSummary,
    IngestStatus,
    IngestTrigger,
    ParsedBlock,
    ParsedDocument,
    SourceConfig,
    SourceDocument,
)
from ragcore.models.eval import (
    EvalCategory,
    EvalResult,
    EvalRun,
    GoldenItem,
    MetricScores,
)
from ragcore.models.memory import (
    LongTermMemory,
    MemoryKind,
    SemanticCacheEntry,
    UserProfile,
    normalize_query,
)
from ragcore.models.retrieval import (
    Citation,
    MetadataFilter,
    RetrievalResult,
    RetrievalStage,
    RetrievedChunk,
)
from ragcore.models.tool import (
    McpServerSpec,
    RestToolSpec,
    ToolAuth,
    ToolKind,
    ToolResult,
    ToolSpec,
)

__all__ = [
    "ADMIN_ROLE",
    "AccessControl",
    "BlockKind",
    "ChatRequest",
    "ChunkPayload",
    "Citation",
    "Classification",
    "ContextStats",
    "EvalCategory",
    "EvalResult",
    "EvalRun",
    "GoldenItem",
    "GuardrailAction",
    "GuardrailEvent",
    "GuardrailKind",
    "GuardrailStage",
    "IngestAction",
    "IngestManifest",
    "IngestManifestEntry",
    "IngestRunSummary",
    "IngestStatus",
    "IngestTrigger",
    "LongTermMemory",
    "McpServerSpec",
    "MemoryKind",
    "Message",
    "MetadataFilter",
    "MetricScores",
    "ParsedBlock",
    "ParsedDocument",
    "Principal",
    "RestToolSpec",
    "RetrievalResult",
    "RetrievalStage",
    "RetrievedChunk",
    "Role",
    "SSEEvent",
    "SemanticCacheEntry",
    "SourceConfig",
    "SourceDocument",
    "SourceType",
    "ToolAuth",
    "ToolCall",
    "ToolKind",
    "ToolResult",
    "ToolSpec",
    "UserProfile",
    "normalize_query",
]
