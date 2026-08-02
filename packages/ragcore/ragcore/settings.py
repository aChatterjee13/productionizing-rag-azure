"""Central, typed configuration for every service in the RAG platform.

`pydantic-settings` reads `RAG_*` environment variables and an optional `.env` file.
Nothing in this repository hard-codes a threshold, model name, limit or URL at a call
site: every tunable is a field here with a documented default.

Use :func:`get_settings` (LRU-cached) rather than constructing :class:`Settings`
directly, so a process shares one instance and one set of derived URLs.
"""

from __future__ import annotations

import math
from datetime import datetime
from functools import lru_cache
from typing import Literal
from zoneinfo import ZoneInfo

from pydantic import Field, computed_field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

Environment = Literal["local", "dev", "staging", "production"]
FusionStrategy = Literal["rrf", "dbsf"]
PIIRedactionMode = Literal["mask", "hash", "partial"]
Effort = Literal["low", "medium", "high", "xhigh", "max"]
LLMProviderName = Literal["anthropic", "azure_openai", "ollama"]
ClearanceLevel = Literal["public", "internal", "confidential", "restricted"]

#: Weekday indices treated as working days by default (Monday=0 ... Sunday=6).
DEFAULT_WORKING_DAYS: tuple[int, ...] = (0, 1, 2, 3, 4)


class Settings(BaseSettings):
    """Runtime configuration for ragcore and every service built on it.

    Field groups mirror the sections of `docs/CONTRACTS.md`: ``qdrant_*``,
    ``postgres_*``, ``redis_*``, ``anthropic_*``, ``entra_*``, ``langfuse_*``,
    ``azure_*``, ``embedding_*``, ``rerank_*``, ``retrieval_*``, ``qt_*``,
    ``citation_*``, ``context_*``, ``memory_*``, ``guardrail_*``, ``pii_*``,
    ``dedupe_*``, ``tool_*``, ``ingest_*``, ``eval_*`` and ``api_*``.
    """

    model_config = SettingsConfigDict(
        env_prefix="RAG_",
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        extra="ignore",
        case_sensitive=False,
        validate_default=True,
    )

    # ------------------------------------------------------------------ platform
    env: Environment = Field(
        default="local",
        description=(
            "Deployment environment. Guards unsafe local shortcuts: dev auth mode "
            "refuses to start when this is 'production'."
        ),
    )
    service_name: str = Field(
        default="rag-platform",
        description="Logical service name attached to logs, traces and metrics.",
    )
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO",
        description="Root log level for structlog and the stdlib logging bridge.",
    )
    log_json: bool = Field(
        default=True,
        description=(
            "Emit newline-delimited JSON logs (required for Azure Log Analytics). "
            "Set false for human-readable console output during local development."
        ),
    )
    debug: bool = Field(
        default=False,
        description="Enable verbose diagnostics. Never enable in production.",
    )

    # -------------------------------------------------------------------- qdrant
    qdrant_url: str = Field(
        default="http://localhost:6333",
        description="Qdrant REST endpoint, e.g. https://rag-qdrant.<region>.azurecontainerapps.io.",
    )
    qdrant_api_key: str | None = Field(
        default=None,
        description="Qdrant API key. Sourced from Key Vault in Azure; unset locally.",
    )
    qdrant_prefer_grpc: bool = Field(
        default=False,
        description="Use the gRPC port (6334) for search traffic instead of REST.",
    )
    qdrant_grpc_port: int = Field(
        default=6334,
        description="Qdrant gRPC port, used when qdrant_prefer_grpc is true.",
    )
    qdrant_timeout_seconds: float = Field(
        default=30.0,
        gt=0,
        description="Per-request timeout for Qdrant calls.",
    )
    qdrant_https: bool | None = Field(
        default=None,
        description="Force TLS on/off. None lets the client infer it from qdrant_url.",
    )
    qdrant_chunks_collection: str = Field(
        default="rag_chunks",
        description=(
            "Collection holding document chunks (named vectors dense + sparse)."
        ),
    )
    qdrant_memories_collection: str = Field(
        default="rag_memories",
        description="Collection holding long-term user memories (dense only).",
    )
    qdrant_cache_collection: str = Field(
        default="rag_semantic_cache",
        description="Collection holding semantic-cache probes (dense only).",
    )
    qdrant_hnsw_m: int = Field(
        default=0,
        ge=0,
        description=(
            "Global HNSW 'm'. Zero disables the global index so that tenant-scoped "
            "search uses the per-tenant payload index instead."
        ),
    )
    qdrant_hnsw_payload_m: int = Field(
        default=16,
        ge=4,
        description="Per-payload-partition HNSW 'm' used for tenant-scoped search.",
    )
    qdrant_hnsw_ef_construct: int = Field(
        default=128,
        ge=4,
        description="HNSW ef_construct for index build quality vs build time.",
    )
    qdrant_upsert_batch_size: int = Field(
        default=64,
        gt=0,
        description="Points per upsert request when writing chunks.",
    )
    qdrant_replication_factor: int = Field(
        default=1,
        ge=1,
        description="Qdrant replication factor for created collections.",
    )
    qdrant_shard_number: int = Field(
        default=1,
        ge=1,
        description="Qdrant shard count for created collections.",
    )

    # ------------------------------------------------------------------ postgres
    postgres_host: str = Field(
        default="localhost",
        description="PostgreSQL host, e.g. rag-pg-prod.postgres.database.azure.com.",
    )
    postgres_port: int = Field(default=5432, description="PostgreSQL port.")
    postgres_user: str = Field(default="rag", description="PostgreSQL user name.")
    postgres_password: str = Field(
        default="rag",
        description="PostgreSQL password. Sourced from Key Vault in Azure.",
    )
    postgres_db: str = Field(default="rag", description="PostgreSQL database name.")
    postgres_sslmode: Literal["disable", "prefer", "require", "verify-full"] = Field(
        default="prefer",
        description="TLS mode. Azure Database for PostgreSQL requires 'require'.",
    )
    postgres_pool_size: int = Field(
        default=10, gt=0, description="SQLAlchemy connection pool size."
    )
    postgres_max_overflow: int = Field(
        default=10, ge=0, description="Extra connections allowed above the pool size."
    )
    postgres_pool_timeout_seconds: float = Field(
        default=30.0, gt=0, description="Seconds to wait for a pooled connection."
    )
    postgres_pool_recycle_seconds: int = Field(
        default=1800,
        gt=0,
        description="Recycle connections older than this (Azure drops idle ones).",
    )
    postgres_echo: bool = Field(
        default=False, description="Echo SQL to logs. Debug only; very verbose."
    )
    postgres_statement_timeout_ms: int = Field(
        default=30_000,
        gt=0,
        description="Server-side statement_timeout applied to every session.",
    )
    database_url_override: str | None = Field(
        default=None,
        description=(
            "Full SQLAlchemy async URL. When set it wins over the postgres_* fields "
            "(used by tests to point at sqlite+aiosqlite)."
        ),
    )

    # --------------------------------------------------------------------- redis
    redis_url: str = Field(
        default="redis://localhost:6379/0",
        description="Redis URL for rate limiting, SSE fan-out and short-lived locks.",
    )
    redis_enabled: bool = Field(
        default=True,
        description=(
            "Disable to run without Redis; dependent features degrade to no-ops."
        ),
    )
    redis_ttl_seconds: int = Field(
        default=3600, gt=0, description="Default TTL for opaque Redis entries."
    )
    redis_lock_timeout_seconds: int = Field(
        default=300,
        gt=0,
        description="Lease duration for the ingestion single-runner lock.",
    )
    redis_session_prefix: str = Field(
        default="rag:session:",
        description=(
            "Redis key prefix for the short-term session window. Keys are "
            "'<prefix><tenant_id>:<session_id>' — tenant first."
        ),
    )

    # ------------------------------------------------------------- llm provider
    llm_provider: LLMProviderName = Field(
        default="anthropic",
        description=(
            "Which backend serves chat completions. 'anthropic' is the only one "
            "supporting extended thinking, context edits, remote MCP and prompt "
            "caching; enabling any of those against another provider is refused at "
            "startup rather than silently ignored."
        ),
    )
    llm_timeout_seconds: float = Field(
        default=60.0,
        gt=0,
        description="Request timeout for the OpenAI-compatible providers.",
    )
    llm_max_retries: int = Field(
        default=3,
        ge=0,
        description=(
            "Retry budget for the OpenAI-compatible providers. The SDK's own retries "
            "are disabled so this is the only one in effect."
        ),
    )
    llm_model_aliases: dict[str, str] = Field(
        default_factory=dict,
        description=(
            "Map a model id a call site asks for onto the one this provider serves, "
            "e.g. {'claude-opus-5': 'gpt-4o'}. Call sites name Claude models "
            "directly, so without an entry the request reaches the backend unchanged "
            "and Azure/Ollama would reject it."
        ),
    )

    # ------------------------------------------------------------- azure openai
    azure_openai_endpoint: str | None = Field(
        default=None,
        description=(
            "Azure OpenAI resource endpoint, e.g. "
            "'https://my-resource.openai.azure.com'. Required when "
            "llm_provider='azure_openai'."
        ),
    )
    azure_openai_api_key: str | None = Field(
        default=None,
        description=(
            "Azure OpenAI key. Omit to use managed identity via the SDK's own "
            "credential chain."
        ),
    )
    azure_openai_api_version: str = Field(
        default="2024-10-21",
        description="Azure OpenAI REST api-version sent on every request.",
    )

    # ------------------------------------------------------------------- ollama
    ollama_base_url: str = Field(
        default="http://localhost:11434",
        description=(
            "Ollama daemon root. '/v1' is appended to reach its OpenAI-compatible "
            "surface."
        ),
    )

    # ----------------------------------------------------------------- anthropic
    anthropic_api_key: str | None = Field(
        default=None,
        description="Anthropic API key. Omit to let the SDK resolve env/OAuth profile.",
    )
    anthropic_base_url: str | None = Field(
        default=None, description="Override the Anthropic API base URL (proxies/tests)."
    )
    anthropic_model_main: str = Field(
        default="claude-opus-5",
        description=(
            "Answer generation, agentic tool loop, contradiction resolution. "
            "Exact model id, no date suffix."
        ),
    )
    anthropic_model_fast: str = Field(
        default="claude-sonnet-5",
        description="Query transformation, summarisation, memory extraction.",
    )
    anthropic_model_cheap: str = Field(
        default="claude-haiku-4-5",
        description="Classification: routing, OOD, PII verification (200K context).",
    )
    anthropic_effort: Effort = Field(
        default="high",
        description=(
            "Default output_config.effort for generation. Nested inside output_config "
            "by the client, never sent top-level."
        ),
    )
    anthropic_effort_fast: Effort = Field(
        default="medium",
        description="Effort for MODEL_FAST structured calls (transform, summarise).",
    )
    anthropic_effort_cheap: Effort = Field(
        default="low", description="Effort for MODEL_CHEAP classification calls."
    )
    anthropic_thinking: bool = Field(
        default=True,
        description=(
            "Use adaptive thinking on MODEL_MAIN. Thinking is on by default on "
            "claude-opus-5; disabling is only valid at effort 'high' or lower."
        ),
    )
    anthropic_thinking_display: Literal["omitted", "summarized"] = Field(
        default="summarized",
        description=(
            "Whether to receive summarised reasoning. 'omitted' is the API default "
            "and yields empty thinking text, which looks like a stall in the UI."
        ),
    )
    anthropic_max_tokens: int = Field(
        default=16_000,
        gt=0,
        description="max_tokens for non-streaming generation calls.",
    )
    anthropic_max_tokens_streaming: int = Field(
        default=64_000,
        gt=0,
        description="max_tokens for streamed generation calls.",
    )
    anthropic_max_tokens_classify: int = Field(
        default=256,
        gt=0,
        description="max_tokens for single-label classification calls.",
    )
    anthropic_max_tokens_structured: int = Field(
        default=4_000,
        gt=0,
        description="max_tokens for structured-output calls (query transform, memory).",
    )
    anthropic_cache_system: bool = Field(
        default=True,
        description=(
            "Put cache_control on the final system block. Minimum cacheable prefix "
            "on claude-opus-5 is 512 tokens."
        ),
    )
    anthropic_fallbacks_enabled: bool = Field(
        default=True,
        description=(
            "Opt into server-side refusal fallbacks "
            "(betas=['server-side-fallback-2026-07-01'], fallbacks='default')."
        ),
    )
    anthropic_max_retries: int = Field(
        default=4,
        ge=0,
        description="Retries for RateLimitError/5xx with jittered exponential backoff.",
    )
    anthropic_retry_base_delay_seconds: float = Field(
        default=1.0, gt=0, description="Base delay for the retry backoff."
    )
    anthropic_retry_max_delay_seconds: float = Field(
        default=30.0, gt=0, description="Ceiling for the retry backoff."
    )
    anthropic_timeout_seconds: float = Field(
        default=600.0,
        gt=0,
        description="Per-request timeout. Long-horizon turns can run many minutes.",
    )
    anthropic_price_per_mtok: dict[str, tuple[float, float]] = Field(
        default={
            "claude-opus-5": (5.0, 25.0),
            "claude-sonnet-5": (3.0, 15.0),
            "claude-haiku-4-5": (1.0, 5.0),
        },
        description=(
            "USD per million (input, output) tokens per model, for cost accounting. "
            "Cache reads bill at 0.1x input, cache writes at 1.25x."
        ),
    )
    anthropic_cache_read_multiplier: float = Field(
        default=0.1, gt=0, description="Cost multiplier applied to cache-read tokens."
    )
    anthropic_cache_write_multiplier: float = Field(
        default=1.25, gt=0, description="Cost multiplier applied to cache-write tokens."
    )

    # --------------------------------------------------------------------- entra
    entra_tenant_id: str | None = Field(
        default=None,
        description=(
            "Microsoft Entra ID directory (tenant) GUID used to build the issuer."
        ),
    )
    entra_client_id: str | None = Field(
        default=None,
        description=(
            "API application (client) ID. Expected 'aud' claim on inbound JWTs."
        ),
    )
    entra_audience: str | None = Field(
        default=None,
        description=(
            "Explicit expected audience. Defaults to entra_client_id, or "
            "api://<client_id> when the token uses the URI form."
        ),
    )
    entra_authority_host: str = Field(
        default="https://login.microsoftonline.com",
        description="Entra authority host used for JWKS and issuer construction.",
    )
    entra_jwks_cache_seconds: int = Field(
        default=3600,
        gt=0,
        description="How long a fetched JWKS document stays valid before refetch.",
    )
    entra_allowed_algorithms: list[str] = Field(
        default=["RS256"],
        description="Accepted JWT signing algorithms. RS256 only; never allow 'none'.",
    )
    entra_leeway_seconds: int = Field(
        default=60,
        ge=0,
        description="Clock-skew tolerance when validating exp/nbf.",
    )
    entra_admin_role: str = Field(
        default="rag.admin",
        description=(
            "App role granting admin endpoints. Must match "
            "ragcore.models.acl.ADMIN_ROLE."
        ),
    )
    entra_roles_claim: str = Field(
        default="roles", description="Token claim carrying app roles."
    )
    entra_groups_claim: str = Field(
        default="groups", description="Token claim carrying Entra group object IDs."
    )
    entra_dev_mode: bool = Field(
        default=False,
        description=(
            "Accept an unsigned dev header instead of a validated JWT. Logs a loud "
            "warning and refuses to start when env == 'production'."
        ),
    )
    entra_dev_principal_header: str = Field(
        default="x-dev-principal",
        description="Header carrying the JSON dev principal when entra_dev_mode is on.",
    )
    entra_clearance_roles: dict[str, str] = Field(
        default={
            "rag.admin": "restricted",
            "rag.restricted": "restricted",
            "rag.confidential": "confidential",
            "rag.internal": "internal",
            "rag.public": "public",
        },
        description=(
            "App role -> clearance ceiling. The strongest matching rule across a "
            "caller's roles and groups wins; a role that is not a key contributes "
            "nothing."
        ),
    )
    entra_clearance_groups: dict[str, str] = Field(
        default={},
        description=(
            "Entra group object id -> clearance ceiling. Empty by default: group "
            "ids are deployment-specific, so an operator maps them per tenant."
        ),
    )
    entra_default_classification: ClearanceLevel = Field(
        default="internal",
        description=(
            "Clearance for a caller matching no role or group rule. Mirrors "
            "AccessControl.classification's own default."
        ),
    )
    entra_accept_v1_issuer: bool = Field(
        default=True,
        description=(
            "Accept the v1.0 issuer 'https://sts.windows.net/{tid}/' alongside the "
            "v2.0 one. Some app registrations still mint v1.0 tokens."
        ),
    )
    entra_required_scope: str | None = Field(
        default=None,
        description=(
            "Optional 'scp' every caller must present; a miss is 403. None "
            "disables the check — app-role-only tokens carry no 'scp' at all."
        ),
    )
    entra_pin_tenant: bool = Field(
        default=True,
        description=(
            "Reject a token whose 'tid' is not the pinned directory. Only turn off "
            "for a genuinely multi-directory registration."
        ),
    )
    entra_jwks_refresh_min_seconds: float = Field(
        default=60.0,
        gt=0,
        description=(
            "Floor between two JWKS refetches triggered by an unknown 'kid', and "
            "how long a failed fetch is remembered. Without it a stream of "
            "unknown-key tokens becomes a DoS on the Microsoft endpoint."
        ),
    )
    entra_jwks_timeout_seconds: float = Field(
        default=10.0, gt=0, description="Per-request timeout for the JWKS fetch."
    )
    entra_group_overage_lookup: bool = Field(
        default=True,
        description=(
            "Resolve the group-overage claim through Microsoft Graph. With this "
            "off a caller whose token overflowed has no groups, which fails closed."
        ),
    )
    entra_group_cache_seconds: float = Field(
        default=900.0, gt=0, description="TTL for a resolved group-overage lookup."
    )
    entra_group_page_size: int = Field(
        default=999, gt=0, description="Page size for the Graph transitive group query."
    )
    entra_group_lookup_timeout_seconds: float = Field(
        default=10.0, gt=0, description="Per-request timeout for Microsoft Graph calls."
    )
    entra_group_lookup_path: str = Field(
        default="/me/transitiveMemberOf/microsoft.graph.group",
        description=(
            "Graph path for the transitive group lookup, relative to "
            "azure_graph_base_url. '/me/' becomes '/users/{oid}/' for the "
            "application-identity call."
        ),
    )
    entra_client_secret: str | None = Field(
        default=None,
        description=(
            "Client-credentials secret for the Graph call, used only when managed "
            "identity is unavailable. Source from Key Vault; never commit one."
        ),
    )
    entra_managed_identity_endpoint: str = Field(
        default="http://169.254.169.254/metadata/identity/oauth2/token",
        description="Azure IMDS endpoint, the managed-identity token source.",
    )
    entra_managed_identity_api_version: str = Field(
        default="2018-02-01", description="API version sent to the IMDS token endpoint."
    )
    entra_graph_token_skew_seconds: float = Field(
        default=60.0,
        ge=0,
        description="Refresh a Graph token this many seconds before it expires.",
    )

    # ------------------------------------------------------------------ langfuse
    langfuse_enabled: bool = Field(
        default=False,
        description=(
            "Enable Langfuse tracing. When false (or keys absent) the tracer degrades "
            "to a no-op; the app never fails because observability is down."
        ),
    )
    langfuse_host: str = Field(
        default="http://localhost:3000", description="Langfuse base URL."
    )
    langfuse_public_key: str | None = Field(
        default=None, description="Langfuse public key (pk-lf-...)."
    )
    langfuse_secret_key: str | None = Field(
        default=None, description="Langfuse secret key (sk-lf-...)."
    )
    langfuse_sample_rate: float = Field(
        default=1.0, ge=0.0, le=1.0, description="Fraction of traces to record."
    )
    langfuse_flush_interval_seconds: float = Field(
        default=1.0, gt=0, description="Background flush interval for the Langfuse SDK."
    )
    langfuse_debug: bool = Field(
        default=False, description="Verbose Langfuse SDK logging."
    )
    langfuse_release: str | None = Field(
        default=None, description="Release identifier (git SHA) attached to traces."
    )

    # --------------------------------------------------------------------- azure
    azure_blob_account_url: str | None = Field(
        default=None,
        description="Blob endpoint, e.g. https://ragstorprod.blob.core.windows.net.",
    )
    azure_blob_container: str = Field(
        default="rag-documents",
        description="Container holding source documents for the blob connector.",
    )
    azure_blob_raw_container: str = Field(
        default="rag-raw",
        description="Container where ingestion archives the raw fetched bytes.",
    )
    azure_blob_connection_string: str | None = Field(
        default=None,
        description="Connection string. Prefer managed identity; used locally only.",
    )
    azure_key_vault_url: str | None = Field(
        default=None,
        description="Key Vault URI, e.g. https://rag-kv-prod.vault.azure.net/.",
    )
    azure_use_managed_identity: bool = Field(
        default=True,
        description=(
            "Authenticate to Azure with DefaultAzureCredential/managed identity."
        ),
    )
    azure_client_id: str | None = Field(
        default=None,
        description=(
            "User-assigned managed identity client ID, when not system-assigned."
        ),
    )
    azure_graph_scope: str = Field(
        default="https://graph.microsoft.com/.default",
        description="Scope requested for Microsoft Graph (SharePoint/OneDrive delta).",
    )
    azure_graph_base_url: str = Field(
        default="https://graph.microsoft.com/v1.0",
        description="Microsoft Graph base URL used by the SharePoint connector.",
    )
    azure_storage_queue_name: str = Field(
        default="rag-ingest",
        description="Storage queue driving the queue-triggered ingestion function.",
    )
    azure_log_analytics_workspace_id: str | None = Field(
        default=None, description="Log Analytics workspace ID for diagnostics."
    )

    # ----------------------------------------------------------------- embedding
    embedding_model: str = Field(
        default="BAAI/bge-m3",
        description="FastEmbed dense model. 1024-dim, cosine distance.",
    )
    embedding_dim: int = Field(
        default=1024,
        gt=0,
        description="Dense vector dimensionality. Must match embedding_model.",
    )
    embedding_sparse_model: str = Field(
        default="Qdrant/bm25",
        description=(
            "FastEmbed sparse model. True BM25; requires the IDF modifier on the "
            "Qdrant sparse vector params."
        ),
    )
    embedding_batch_size: int = Field(
        default=32,
        gt=0,
        description="Texts per FastEmbed batch. FastEmbed is sync and CPU-bound.",
    )
    embedding_cache_dir: str | None = Field(
        default=None,
        description="Directory for downloaded FastEmbed model weights.",
    )
    embedding_threads: int | None = Field(
        default=None,
        description="ONNX runtime thread count. None lets FastEmbed decide.",
    )
    embedding_max_chars: int = Field(
        default=8_000,
        gt=0,
        description="Hard cap on characters sent to the embedder per text.",
    )

    # -------------------------------------------------------------------- rerank
    rerank_enabled: bool = Field(
        default=True,
        description="Disable to select NoopReranker for cheap CI runs.",
    )
    rerank_model: str = Field(
        default="Xenova/bge-reranker-v2-m3",
        description="FastEmbed TextCrossEncoder model used for reranking.",
    )
    rerank_top_n: int = Field(
        default=8,
        gt=0,
        description="Documents kept after cross-encoder reranking.",
    )
    rerank_candidate_limit: int = Field(
        default=50,
        gt=0,
        description="Maximum candidates fed into the cross-encoder per query.",
    )
    rerank_batch_size: int = Field(
        default=16, gt=0, description="Pairs per cross-encoder batch."
    )
    rerank_min_score: float | None = Field(
        default=None,
        description=(
            "Drop chunks scoring below this after rerank. None keeps everything and "
            "lets the OOD gate decide."
        ),
    )

    # ----------------------------------------------------------------- retrieval
    retrieval_top_n: int = Field(
        default=8,
        gt=0,
        description="Chunks handed to context assembly after rerank and MMR.",
    )
    retrieval_limit: int = Field(
        default=30,
        gt=0,
        description="Fused candidates requested from the Qdrant Query API per query.",
    )
    retrieval_prefetch_limit: int = Field(
        default=60,
        gt=0,
        description="Candidates per prefetch branch (dense and sparse) before fusion.",
    )
    retrieval_fusion: FusionStrategy = Field(
        default="rrf",
        description="Server-side fusion: Reciprocal Rank Fusion or Distribution-Based.",
    )
    retrieval_mmr_enabled: bool = Field(
        default=True,
        description="Apply Maximal Marginal Relevance diversification after rerank.",
    )
    retrieval_mmr_lambda: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="MMR trade-off: 1.0 pure relevance, 0.0 pure diversity.",
    )
    retrieval_max_per_document: int = Field(
        default=3,
        gt=0,
        description="Cap on chunks from any single document in the final set.",
    )
    retrieval_include_deleted: bool = Field(
        default=False,
        description="Never enable outside forensic tooling; soft-deleted chunks leak.",
    )
    retrieval_contextual_header_enabled: bool = Field(
        default=True,
        description=(
            "Prepend the contextual header to chunk text at embed time (stored "
            "separately in the payload)."
        ),
    )
    retrieval_snippet_chars: int = Field(
        default=1_200,
        gt=0,
        description=(
            "Characters of chunk text rendered into the numbered-source prompt."
        ),
    )
    retrieval_fusion_score_scale: float = Field(
        default=0.05,
        gt=0,
        description=(
            "Divisor mapping a server-side fusion score onto the [0, 1] relevance "
            "scale guardrail_ood_min_score is expressed in, used only when no "
            "cross-encoder score is available. 0.05 is the RRF reference; set it "
            "near 2.0 when retrieval_fusion is 'dbsf'."
        ),
    )
    retrieval_query_concurrency: int = Field(
        default=4,
        gt=0,
        description=(
            "Probes (rewritten query, sub-questions, HyDE passage) embedded and "
            "searched in parallel."
        ),
    )
    retrieval_mmr_embed_fallback: bool = Field(
        default=True,
        description=(
            "Re-embed a candidate locally when Qdrant did not return its dense "
            "vector. Qdrant is tried first either way."
        ),
    )

    # ----------------------------------------------------- query transform (#10)
    qt_enabled: bool = Field(
        default=True,
        description="Run the MODEL_FAST query-transformation stage.",
    )
    qt_max_subqueries: int = Field(
        default=3,
        ge=0,
        description="Maximum decomposed sub-questions produced by the transformer.",
    )
    qt_hyde_enabled: bool = Field(
        default=True,
        description="Allow a HyDE passage when the query is sparse or abstract.",
    )
    qt_hyde_max_chars: int = Field(
        default=800, gt=0, description="Cap on the generated HyDE passage length."
    )
    qt_hyde_max_query_words: int = Field(
        default=12,
        gt=0,
        description=(
            "At or below this many words a query is 'sparse' enough that lexical "
            "matching is unreliable, which is the case HyDE exists for."
        ),
    )
    qt_rewrite_enabled: bool = Field(
        default=True,
        description="Resolve pronouns/ellipsis against the short-term window.",
    )
    qt_history_turns: int = Field(
        default=6,
        ge=0,
        description="Recent turns given to the transformer for context resolution.",
    )
    qt_max_query_chars: int = Field(
        default=4_000, gt=0, description="Input size cap for the transform call."
    )

    # ------------------------------------------------------- context (#3 and #5)
    context_budget_tokens: int = Field(
        default=120_000,
        gt=0,
        description="Total prompt token budget the assembler packs into.",
    )
    context_reserve_output_tokens: int = Field(
        default=16_000,
        gt=0,
        description="Tokens held back from the budget for the model's own output.",
    )
    context_compact_at_ratio: float = Field(
        default=0.75,
        gt=0.0,
        le=1.0,
        description=(
            "Fraction of budget at which compaction runs. Proactive and periodic, "
            "not only on hard overflow."
        ),
    )
    context_max_history_turns: int = Field(
        default=20,
        ge=0,
        description="Recent turns eligible for the live window before suppression.",
    )
    context_min_live_turns: int = Field(
        default=4,
        ge=1,
        description="Turns never suppressed, however tight the budget gets.",
    )
    context_tool_result_ttl_turns: int = Field(
        default=3,
        ge=0,
        description=(
            "Tool results older than this many turns are cleared via the "
            "clear_tool_uses_20250919 context edit."
        ),
    )
    context_summary_max_tokens: int = Field(
        default=1_200,
        gt=0,
        description="Cap on the rolling MODEL_FAST conversation summary.",
    )
    context_retrieved_budget_ratio: float = Field(
        default=0.55,
        gt=0.0,
        le=1.0,
        description="Share of the budget reserved for retrieved chunks.",
    )
    context_memory_budget_ratio: float = Field(
        default=0.08,
        ge=0.0,
        le=1.0,
        description="Share of the budget reserved for long-term memory.",
    )
    context_compaction_enabled: bool = Field(
        default=True,
        description=(
            "Fold suppressed turns into the rolling summary instead of dropping."
        ),
    )
    context_compact_every_n_turns: int = Field(
        default=6,
        gt=0,
        description=(
            "Periodic suppression (requirement #5): compaction fires every N "
            "appended turns even when the budget is nowhere near full."
        ),
    )
    context_token_count_concurrency: int = Field(
        default=8,
        gt=0,
        description="Parallel count_tokens measurements during exact fitting.",
    )
    context_token_cache_entries: int = Field(
        default=4_096,
        gt=0,
        description=(
            "Bound on the token-count memo, in entries. Keyed by content digest, "
            "never by content."
        ),
    )
    context_fit_max_passes: int = Field(
        default=3,
        gt=0,
        description=(
            "Exact-measurement shed passes allowed before the assembler strips the "
            "prompt down to the mandatory system + question payload."
        ),
    )
    context_min_retrieved_chunks: int = Field(
        default=1,
        ge=0,
        description=(
            "Retrieved chunks kept even under extreme pressure, when any were "
            "retrieved at all."
        ),
    )
    context_duplicate_penalty: float = Field(
        default=0.35,
        ge=0.0,
        le=1.0,
        description=(
            "Marginal-value multiplier applied to a simhash near-duplicate chunk, "
            "which makes it the first thing shed."
        ),
    )
    context_cache_history_breakpoint: bool = Field(
        default=True,
        description=(
            "Place a second cache_control breakpoint at the end of the stable "
            "history prefix."
        ),
    )
    context_cache_history_min_tokens: int = Field(
        default=1_024,
        gt=0,
        description=(
            "Minimum measured history tokens before that breakpoint is worth "
            "writing. The API minimum cacheable prefix on claude-opus-5 is 512."
        ),
    )

    # ------------------------------------------------------------- memory (#2/#5)
    memory_enabled: bool = Field(
        default=True, description="Master switch for long-term memory read and write."
    )
    memory_top_k: int = Field(
        default=6,
        ge=0,
        description="Long-term memories injected per turn (salience-weighted).",
    )
    memory_min_salience: float = Field(
        default=0.25,
        ge=0.0,
        le=1.0,
        description="Memories below this salience are not retrieved.",
    )
    memory_salience_decay_per_day: float = Field(
        default=0.01,
        ge=0.0,
        description="Salience subtracted per day since last use.",
    )
    memory_salience_boost_on_hit: float = Field(
        default=0.05,
        ge=0.0,
        description="Salience added when a memory is reused in an answer.",
    )
    memory_max_per_user: int = Field(
        default=500,
        gt=0,
        description="Cap on stored memories per user; lowest salience is pruned first.",
    )
    memory_ttl_days: int | None = Field(
        default=365,
        description="Default expiry for new memories. None means no expiry.",
    )
    memory_extract_enabled: bool = Field(
        default=True,
        description="Run the MODEL_FAST memory write-back stage after answering.",
    )
    memory_dedupe_threshold: float = Field(
        default=0.9,
        ge=0.0,
        le=1.0,
        description="Cosine similarity above which a new memory supersedes an old one.",
    )
    memory_cache_enabled: bool = Field(
        default=True,
        description="Enable the semantic cache for similar-query retrieval reuse.",
    )
    memory_cache_threshold: float = Field(
        default=0.94,
        ge=0.0,
        le=1.0,
        description=(
            "Minimum cosine similarity for a semantic-cache hit. A hit also requires "
            "an exact filter_fingerprint match."
        ),
    )
    memory_cache_ttl_seconds: int = Field(
        default=86_400,
        gt=0,
        description="TTL for semantic-cache entries.",
    )
    memory_cache_max_chunk_ids: int = Field(
        default=32,
        gt=0,
        description="Chunk ids stored per cache entry (never the rendered answer).",
    )
    memory_profile_refresh_turns: int = Field(
        default=10,
        gt=0,
        description="Rebuild the rolling persona summary every N turns.",
    )
    memory_cache_max_entries: int = Field(
        default=500,
        gt=0,
        description=(
            "Semantic-cache entries kept per (tenant_id, filter_fingerprint) "
            "bucket before LRU-ish eviction by hit_count then last_used_at."
        ),
    )
    memory_consolidate_batch_size: int = Field(
        default=200,
        gt=0,
        description="Memories scanned per consolidation pass, per user.",
    )
    memory_recall_oversample: int = Field(
        default=4,
        gt=0,
        description="Candidate oversampling factor for salience-weighted recall.",
    )
    memory_recall_similarity_weight: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description=(
            "Blend of dense similarity versus decayed salience when ranking recall "
            "candidates. 1.0 is pure similarity, 0.0 is pure salience."
        ),
    )
    memory_profile_min_memories: int = Field(
        default=3,
        gt=0,
        description=(
            "Memories required before the rolling profile is regenerated by MODEL_FAST."
        ),
    )

    # --------------------------------------------- citations (#9, stage 11)
    citation_fuzzy_threshold: float = Field(
        default=0.55,
        ge=0.0,
        le=1.0,
        description=(
            "Similarity a paraphrased sentence must reach against its best-matching "
            "window of the cited chunk to count as supported."
        ),
    )
    citation_quote_min_ratio: float = Field(
        default=0.9,
        ge=0.0,
        le=1.0,
        description=(
            "Similarity an explicitly quoted literal must reach. Much stricter: the "
            "answer prompt promises verbatim quoting, so a near-miss is a fabrication."
        ),
    )
    citation_token_recall_threshold: float = Field(
        default=0.6,
        ge=0.0,
        le=1.0,
        description=(
            "Fraction of a span's content tokens that must appear in the matched "
            "window for token recall to stand in for character similarity."
        ),
    )
    citation_number_check: bool = Field(
        default=True,
        description=(
            "Require every multi-digit number a cited sentence asserts to occur in "
            "the cited chunk. Skipped when a sentence cites several sources."
        ),
    )
    citation_min_span_chars: int = Field(
        default=12,
        gt=0,
        description=(
            "Spans shorter than this carry too little signal to verify alone and "
            "borrow the preceding sentence first."
        ),
    )
    citation_max_span_chars: int = Field(
        default=600,
        gt=0,
        description=(
            "Hard cap on the span compared against a chunk, so one runaway sentence "
            "cannot dominate verification cost."
        ),
    )
    citation_min_claim_words: int = Field(
        default=5,
        gt=0,
        description=(
            "Sentences with at least this many words count as factual claims for "
            "coverage."
        ),
    )
    citation_min_coverage: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description=(
            "Fraction of claim sentences that must carry a verified citation before "
            "the verdict is 'grounded' rather than 'partial'."
        ),
    )
    citation_anchor_tokens: int = Field(
        default=3,
        gt=0,
        description="Rare tokens used to anchor the fuzzy window search in a chunk.",
    )
    citation_max_windows: int = Field(
        default=24,
        gt=0,
        description=(
            "Upper bound on candidate windows scored per span. Bounds the O(n*m) work."
        ),
    )
    citation_strip_unresolved_markers: bool = Field(
        default=True,
        description=(
            "Remove '[n]' markers from the answer when no citation survived for them."
        ),
    )

    # ---------------------------------------------------------- guardrails (#9)
    guardrail_input_max_chars: int = Field(
        default=16_000,
        gt=0,
        description="Hard size cap on a single user turn (stage 1 input guard).",
    )
    guardrail_injection_enabled: bool = Field(
        default=True,
        description="Run prompt-injection and jailbreak heuristics on the user turn.",
    )
    guardrail_injection_block_threshold: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="Injection score at or above which the turn is blocked outright.",
    )
    guardrail_injection_warn_threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Injection score at or above which a warning event is emitted.",
    )
    guardrail_ood_enabled: bool = Field(
        default=True, description="Enable the out-of-domain gate (stage 6)."
    )
    guardrail_ood_min_score: float = Field(
        default=0.35,
        ge=0.0,
        le=1.0,
        description=(
            "Retrieval max score below which the query is treated as out of domain "
            "and answered with an explicit refusal."
        ),
    )
    guardrail_min_groundedness: float = Field(
        default=0.6,
        ge=0.0,
        le=1.0,
        description=(
            "citation_validity below which an uncertainty notice is appended to the "
            "answer (stage 11 groundedness gate)."
        ),
    )
    guardrail_contradiction_enabled: bool = Field(
        default=True,
        description="Cluster retrieved chunks by claim and surface disagreements.",
    )
    guardrail_contradiction_min_similarity: float = Field(
        default=0.82,
        ge=0.0,
        le=1.0,
        description="Similarity above which two chunks are treated as the same claim.",
    )
    guardrail_contradiction_recency_days: int = Field(
        default=180,
        gt=0,
        description=(
            "Age gap beyond which the newer effective_from wins a contradiction "
            "outright rather than surfacing both."
        ),
    )
    guardrail_doc_type_authority: dict[str, int] = Field(
        default={
            "policy": 100,
            "standard": 90,
            "contract": 85,
            "handbook": 70,
            "runbook": 60,
            "faq": 40,
            "email": 20,
            "note": 10,
        },
        description="Authority weight per doc_type used to break contradiction ties.",
    )
    guardrail_output_pii_scan: bool = Field(
        default=True,
        description="Scan the generated answer for PII egress before persisting.",
    )
    guardrail_enforce_classification_on_output: bool = Field(
        default=True,
        description=(
            "Defence in depth: never emit content above the principal's clearance "
            "even if a filter bug let it through."
        ),
    )
    guardrail_input_min_chars: int = Field(
        default=1,
        ge=0,
        description="Below this the turn is refused with a 'clarify' guardrail event.",
    )
    guardrail_input_truncate: bool = Field(
        default=False,
        description=(
            "Truncate rather than refuse a turn over guardrail_input_max_chars."
        ),
    )
    guardrail_input_normalise_unicode: bool = Field(
        default=True,
        description="NFKC-normalise and strip invisible characters before scanning.",
    )
    guardrail_input_credential_entities: list[str] = Field(
        default=["API_KEY", "JWT"],
        description=(
            "PII entity types masked out of the prompt text as well as the "
            "persisted text, because a leaked credential must never reach a model."
        ),
    )
    guardrail_language_detect_enabled: bool = Field(
        default=True, description="Run the dependency-free language guess on the turn."
    )
    guardrail_language_min_confidence: float = Field(
        default=0.35,
        ge=0.0,
        le=1.0,
        description="Below this confidence the language policy is not enforced.",
    )
    guardrail_allowed_languages: list[str] = Field(
        default=[],
        description=(
            "Permitted turn languages. Empty means all. A detected language outside "
            "the list warns; it never blocks."
        ),
    )
    guardrail_injection_scan_retrieved: bool = Field(
        default=True,
        description=(
            "Scan retrieved chunk text for indirect injection. A poisoned document "
            "is the threat that actually matters."
        ),
    )
    guardrail_injection_retrieved_block_threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description=(
            "Quarantine threshold for retrieved text. Deliberately stricter (lower) "
            "than guardrail_injection_block_threshold for the user turn."
        ),
    )
    guardrail_injection_quarantine_retrieved: bool = Field(
        default=True,
        description="Drop a flagged chunk. False keeps it and only warns.",
    )
    guardrail_injection_max_scan_chars: int = Field(
        default=20_000,
        gt=0,
        description="Cap on characters inspected per passage by the injection scan.",
    )
    guardrail_injection_classifier_enabled: bool = Field(
        default=False,
        description=(
            "Optional MODEL_CHEAP adjudication of a borderline injection score."
        ),
    )
    guardrail_injection_classifier_benign_factor: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Score multiplier applied when the classifier says 'benign'.",
    )
    guardrail_ood_mean_score_min: float = Field(
        default=0.2,
        ge=0.0,
        le=1.0,
        description="Mean top-k relevance floor for the out-of-domain gate.",
    )
    guardrail_ood_min_candidates: int = Field(
        default=1,
        ge=0,
        description=(
            "Fewer surviving candidates than this is automatically out of domain."
        ),
    )
    guardrail_ood_collapse_enabled: bool = Field(
        default=True, description="Enable score-distribution collapse detection."
    )
    guardrail_ood_collapse_top_k: int = Field(
        default=5, gt=0, description="Window of scores the collapse test looks at."
    )
    guardrail_ood_collapse_spread: float = Field(
        default=0.05,
        ge=0.0,
        le=1.0,
        description=(
            "Max-minus-min within the window at or below which the score "
            "distribution counts as 'collapsed'."
        ),
    )
    guardrail_ood_collapse_max_score: float = Field(
        default=0.55,
        ge=0.0,
        le=1.0,
        description="Collapse only counts when the best score is at or below this.",
    )
    guardrail_ood_classifier_enabled: bool = Field(
        default=False,
        description="Optional MODEL_CHEAP adjudication when the OOD signals disagree.",
    )
    guardrail_ood_coverage_sample: int = Field(
        default=512,
        gt=0,
        description="Chunks scrolled per tenant to build the domain-coverage summary.",
    )
    guardrail_ood_coverage_ttl_seconds: int = Field(
        default=900, gt=0, description="TTL for the per-tenant coverage cache."
    )
    guardrail_ood_coverage_max_items: int = Field(
        default=8, gt=0, description="Facet values listed per refusal message."
    )
    guardrail_ood_llm_refusal: bool = Field(
        default=False,
        description=(
            "Phrase the refusal with MODEL_FAST instead of the deterministic template."
        ),
    )
    guardrail_contradiction_min_chunks: int = Field(
        default=2,
        gt=0,
        description="Below this many retrieved chunks stage 7 is skipped entirely.",
    )
    guardrail_contradiction_max_clusters: int = Field(
        default=6, gt=0, description="Claim clusters kept for adjudication."
    )
    guardrail_contradiction_max_pairs: int = Field(
        default=4,
        gt=0,
        description=(
            "Cross-document pairs adjudicated per turn — the model-call budget for "
            "stage 7."
        ),
    )
    guardrail_contradiction_llm_enabled: bool = Field(
        default=True,
        description="Use the structured MODEL_MAIN contradiction detector.",
    )
    guardrail_contradiction_snippet_chars: int = Field(
        default=600, gt=0, description="Passage length sent to the adjudicator."
    )
    guardrail_output_block_below_groundedness: float = Field(
        default=0.2,
        ge=0.0,
        le=1.0,
        description=(
            "Hard floor under guardrail_min_groundedness: below it the answer is "
            "withheld rather than annotated with an uncertainty notice."
        ),
    )
    guardrail_output_leak_span_chars: int = Field(
        default=60,
        gt=0,
        description=(
            "Verbatim overlap with an over-clearance chunk treated as a leak by the "
            "defence-in-depth check."
        ),
    )
    guardrail_output_pii_ignore_entities: list[str] = Field(
        default=["DATE_TIME", "LOCATION"],
        description=(
            "Entity types not redacted on egress: an effective date and an office "
            "name are policy content, and masking them would gut every answer."
        ),
    )
    guardrail_refusal_check_enabled: bool = Field(
        default=True, description="Assess the quality of a refusal before emitting it."
    )
    guardrail_refusal_min_chars: int = Field(
        default=40,
        gt=0,
        description="Shorter than this, a refusal is reported as bare.",
    )

    # ----------------------------------------------------------------------- pii
    pii_enabled: bool = Field(
        default=True, description="Run PII detection on ingest and on chat turns."
    )
    pii_use_presidio: bool = Field(
        default=True,
        description=(
            "Use Presidio when importable. Falls back to the regex-only recogniser "
            "set with a warning when unavailable."
        ),
    )
    pii_language: str = Field(
        default="en", description="Default language passed to the analyzer."
    )
    pii_score_threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Minimum recogniser confidence for a finding to count.",
    )
    pii_redaction_mode: PIIRedactionMode = Field(
        default="mask",
        description=(
            "mask -> <EMAIL_ADDRESS>; hash -> stable HMAC so redacted values still "
            "join; partial -> keep the last 4 characters."
        ),
    )
    pii_hash_secret: str = Field(
        default="change-me-pii-hmac-secret",
        description="HMAC key for pii_redaction_mode='hash'. Source from Key Vault.",
    )
    pii_partial_keep_chars: int = Field(
        default=4, ge=0, description="Trailing characters kept in 'partial' mode."
    )
    pii_entities: list[str] = Field(
        default=[
            "EMAIL_ADDRESS",
            "PHONE_NUMBER",
            "CREDIT_CARD",
            "IBAN_CODE",
            "US_SSN",
            "PERSON",
            "LOCATION",
            "IP_ADDRESS",
            "DATE_TIME",
            "AADHAAR",
            "PAN",
            "SWIFT_CODE",
            "API_KEY",
            "JWT",
        ],
        description="Entity types the detector reports.",
    )
    pii_llm_verify_enabled: bool = Field(
        default=False,
        description="Second-pass MODEL_CHEAP verification of borderline findings.",
    )
    pii_llm_verify_min_score: float = Field(
        default=0.4,
        ge=0.0,
        le=1.0,
        description="Only findings at or above this score go to LLM verification.",
    )
    pii_block_on_egress: bool = Field(
        default=False,
        description=(
            "Block rather than redact when PII is found in generated output. "
            "Redaction is the default so answers stay useful."
        ),
    )

    # -------------------------------------------------------------------- dedupe
    dedupe_enabled: bool = Field(
        default=True, description="Run exact-hash and simhash dedupe over candidates."
    )
    dedupe_simhash_shingle: int = Field(
        default=4, gt=0, description="Token shingle width for simhash64."
    )
    dedupe_max_distance: int = Field(
        default=3,
        ge=0,
        description="Hamming distance at or below which two simhashes are duplicates.",
    )
    dedupe_min_chunk_chars: int = Field(
        default=64,
        gt=0,
        description="Chunks shorter than this skip simhash (too noisy to compare).",
    )

    # -------------------------------------------------------------- chunking (#6)
    chunk_target_tokens: int = Field(
        default=512, gt=0, description="Target chunk size in tokens."
    )
    chunk_overlap_tokens: int = Field(
        default=64, ge=0, description="Token overlap between adjacent chunks."
    )
    chunk_min_tokens: int = Field(
        default=64,
        gt=0,
        description="Chunks below this are merged into a neighbour.",
    )
    chunk_max_tokens: int = Field(
        default=1_024, gt=0, description="Hard cap before a chunk is force-split."
    )
    chunk_respect_headings: bool = Field(
        default=True,
        description="Prefer heading boundaries so section_path stays meaningful.",
    )
    chunk_contextual_header_enabled: bool = Field(
        default=True,
        description="Generate a contextual header (title + section path) per chunk.",
    )
    chunk_summary_enabled: bool = Field(
        default=False,
        description="Generate a per-chunk MODEL_FAST summary. Costly at ingest scale.",
    )

    # ----------------------------------------------------------- tool loop (#4)
    tool_enabled: bool = Field(
        default=True, description="Allow the Claude tool-use loop for un-indexed data."
    )
    tool_max_iterations: int = Field(
        default=6, gt=0, description="Hard bound on tool-loop round trips per turn."
    )
    tool_timeout_seconds: float = Field(
        default=20.0, gt=0, description="Per-invocation timeout for a REST tool."
    )
    tool_max_result_chars: int = Field(
        default=8_000,
        gt=0,
        description="Tool results longer than this are truncated with a marker.",
    )
    tool_registry_path: str | None = Field(
        default=None,
        description="Optional YAML/JSON file of declarative REST tool definitions.",
    )
    tool_mcp_enabled: bool = Field(
        default=False,
        description=(
            "Enable remote MCP servers via the mcp_servers + mcp_toolset connector "
            "on the beta endpoint."
        ),
    )
    tool_mcp_beta_flag: str = Field(
        default="mcp-client-2025-11-20",
        description="Beta flag required by the MCP connector.",
    )
    tool_allow_insecure_http: bool = Field(
        default=False,
        description="Permit plain http:// tool endpoints. Never enable in production.",
    )

    # ----------------------------------------------------------- ingestion (#1)
    ingest_enabled: bool = Field(
        default=True,
        description="Master switch for scheduled ingestion. False skips timer runs.",
    )
    ingest_cron: str = Field(
        default="0 30 2 * * *",
        description=(
            "Six-field NCRONTAB schedule as Azure Functions requires "
            "(sec min hour day month dow). Default 02:30 daily."
        ),
    )
    ingest_timezone: str = Field(
        default="UTC",
        description="IANA timezone the cron and working-hours guard are evaluated in.",
    )
    ingest_working_hours_start: int = Field(
        default=8,
        ge=0,
        le=23,
        description="First hour of the working day (local to ingest_timezone).",
    )
    ingest_working_hours_end: int = Field(
        default=18,
        ge=0,
        le=24,
        description=(
            "End hour of the working day, exclusive. A value below the start hour "
            "means the window wraps past midnight."
        ),
    )
    ingest_working_days: list[int] = Field(
        default=list(DEFAULT_WORKING_DAYS),
        description="Weekday indices counted as working days (Monday=0, Sunday=6).",
    )
    ingest_force: bool = Field(
        default=False,
        description=(
            "Override the working-hours guard. A scheduled run inside working hours "
            "refuses to start unless this is set."
        ),
    )
    ingest_max_parallel_docs: int = Field(
        default=8,
        gt=0,
        description="Documents processed concurrently within one ingestion run.",
    )
    ingest_batch_size: int = Field(
        default=32,
        gt=0,
        description="Documents per Durable Functions activity batch.",
    )
    ingest_max_document_bytes: int = Field(
        default=25 * 1024 * 1024,
        gt=0,
        description="Skip source documents larger than this.",
    )
    ingest_delete_missing: bool = Field(
        default=True,
        description=(
            "Soft-delete chunks whose source document disappeared from the source "
            "(is_deleted=true rather than a hard purge)."
        ),
    )
    ingest_reindex_on_acl_change: bool = Field(
        default=True,
        description="Rewrite chunk payloads when only the ACL changed, skipping embed.",
    )
    ingest_http_user_agent: str = Field(
        default="productionizing-rag-azure-crawler/0.1",
        description="User-Agent sent by the HTTP/sitemap crawler.",
    )
    ingest_http_max_pages: int = Field(
        default=500, gt=0, description="Page cap per crawl run."
    )
    ingest_http_timeout_seconds: float = Field(
        default=20.0, gt=0, description="Per-request timeout for the HTTP connector."
    )
    ingest_http_concurrency: int = Field(
        default=4, gt=0, description="Concurrent HTTP fetches during a crawl."
    )
    ingest_sql_batch_size: int = Field(
        default=500, gt=0, description="Rows per fetch for the SQL watermark connector."
    )
    ingest_manifest_container: str = Field(
        default="rag-manifests",
        description="Blob container storing per-source ingest manifests.",
    )
    ingest_retry_attempts: int = Field(
        default=3, ge=0, description="Retries per document before it is marked failed."
    )
    ingest_local_root: str = Field(
        default="./data/documents",
        description="Root directory for the local-filesystem connector.",
    )
    ingest_function_url: str | None = Field(
        default=None,
        description=(
            "HTTP trigger of the ingestion Function App, e.g. "
            "https://<app>.azurewebsites.net/api/ingest/trigger. When unset, "
            "POST /admin/ingest/trigger runs the crawl as a local background task."
        ),
    )
    ingest_function_key: str | None = Field(
        default=None,
        description=(
            "Azure Functions host key sent as 'x-functions-key' with the dispatch. "
            "Omit for an anonymous trigger or when network policy authorises it."
        ),
    )

    # ------------------------------------------------------------ evaluation (#8)
    eval_enabled: bool = Field(
        default=True, description="Allow evaluation runs to be triggered."
    )
    eval_golden_path: str = Field(
        default="services/eval/golden/golden_set.yaml",
        description="Golden question set with personas, expectations and categories.",
    )
    eval_max_concurrency: int = Field(
        default=4, gt=0, description="Golden items evaluated in parallel."
    )
    eval_judge_model: str = Field(
        default="claude-sonnet-5",
        description="Model used as the RAGAS judge for reference-free metrics.",
    )
    eval_similarity_model: str = Field(
        default="BAAI/bge-m3",
        description="Sentence-embedding model for semantic-similarity scoring.",
    )
    eval_gate_enabled: bool = Field(
        default=True, description="Fail CI when aggregate scores miss the thresholds."
    )
    eval_min_faithfulness: float = Field(
        default=0.8, ge=0.0, le=1.0, description="Gate threshold for faithfulness."
    )
    eval_min_answer_relevancy: float = Field(
        default=0.75, ge=0.0, le=1.0, description="Gate threshold for answer relevancy."
    )
    eval_min_context_precision: float = Field(
        default=0.7, ge=0.0, le=1.0, description="Gate threshold for context precision."
    )
    eval_min_context_recall: float = Field(
        default=0.7, ge=0.0, le=1.0, description="Gate threshold for context recall."
    )
    eval_min_answer_correctness: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Gate threshold for answer correctness.",
    )
    eval_min_semantic_similarity: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="Gate threshold for embedding cosine similarity to ground truth.",
    )
    eval_min_citation_validity: float = Field(
        default=0.9,
        ge=0.0,
        le=1.0,
        description="Gate threshold for verified citation spans.",
    )
    eval_max_acl_leak: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description=(
            "Maximum tolerated ACL leak rate. Zero: any cross-tenant or "
            "over-clearance chunk fails the gate."
        ),
    )
    eval_min_refusal_correct: float = Field(
        default=0.9,
        ge=0.0,
        le=1.0,
        description="Gate threshold for correctly refusing OOD questions.",
    )
    eval_max_latency_ms: float = Field(
        default=20_000.0,
        gt=0,
        description="Per-item latency budget recorded against the gate.",
    )
    eval_sample_size: int | None = Field(
        default=None,
        description="Evaluate only the first N golden items. None runs the whole set.",
    )
    eval_pipeline_target: str = Field(
        default="eval.harness:run_turn",
        description=(
            "'module:attribute' of the coroutine one golden item is run through. "
            "eval.harness.run_turn drives the real orchestrator through all "
            "thirteen stages; RAG_EVAL_PIPELINE_TARGET overrides it."
        ),
    )
    eval_personas_path: str = Field(
        default="services/eval/golden/personas.yaml",
        description="Persona file, resolved relative to the repository root.",
    )
    eval_report_dir: str = Field(
        default="services/eval/reports",
        description=(
            "Where run JSON, markdown and HTML land. CI uploads this directory."
        ),
    )
    eval_item_timeout_seconds: float = Field(
        default=300.0,
        gt=0,
        description=(
            "Wall-clock ceiling for one golden item, so one wedged turn cannot hang "
            "a whole run."
        ),
    )
    eval_persist_results: bool = Field(
        default=True,
        description=(
            "Write eval_runs / eval_results rows. A database failure degrades to a "
            "warning: the gate's verdict must not depend on the reporting store."
        ),
    )
    eval_run_tenant_id: str | None = Field(
        default=None,
        description=(
            "Tenant recorded on the run row. None means the dominant tenant of the "
            "selection, which is what a single-tenant CI run wants."
        ),
    )
    eval_skip_unregistered_tools: bool = Field(
        default=True,
        description=(
            "Skip rather than fail a tool_required item whose expected tool is not "
            "registered for that persona."
        ),
    )
    eval_tool_required_live: bool = Field(
        default=False,
        description=(
            "Run tool_required items that name a REST or MCP tool. Off by default: "
            "the example registry points at unreachable hosts. The built-in tools "
            "always run."
        ),
    )
    eval_ragas_enabled: bool = Field(
        default=True,
        description="Use the RAGAS package for the LLM-only metrics when it imports.",
    )
    eval_judge_effort: Effort = Field(
        default="medium",
        description=(
            "output_config.effort for every judge call. Judging is classification, "
            "not a reasoning showcase."
        ),
    )
    eval_relevancy_probe_questions: int = Field(
        default=3,
        gt=0,
        description=(
            "Questions generated from the answer for the native answer-relevancy "
            "metric, each embedded and compared with the real question."
        ),
    )
    eval_correctness_f1_weight: float = Field(
        default=0.75,
        ge=0.0,
        le=1.0,
        description=(
            "Answer-correctness weight on statement F1 against ground truth. Must "
            "sum to 1.0 with eval_correctness_similarity_weight."
        ),
    )
    eval_correctness_similarity_weight: float = Field(
        default=0.25,
        ge=0.0,
        le=1.0,
        description=(
            "Answer-correctness weight on embedding similarity. Must sum to 1.0 "
            "with eval_correctness_f1_weight."
        ),
    )
    eval_judge_max_contexts: int = Field(
        default=12, gt=0, description="Contexts handed to a single judge call."
    )
    eval_judge_max_context_chars: int = Field(
        default=4_000,
        gt=0,
        description=(
            "Per-context character cap in a judge prompt. Judge prompts are the "
            "harness's main token cost."
        ),
    )
    eval_hard_min_acl_leak: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description=(
            "Hard floor on the leak-free rate. eval_max_acl_leak may tighten the "
            "gate but never loosen this: an ACL leak is a correctness failure."
        ),
    )
    eval_hard_min_refusal_correct: float = Field(
        default=0.95,
        ge=0.0,
        le=1.0,
        description=(
            "Hard floor on refusal correctness. eval_min_refusal_correct may "
            "tighten the gate but never loosen this."
        ),
    )
    eval_min_tool_correct: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description=(
            "Soft floor for tool_correct, lenient on purpose: whether a model "
            "reaches for a tool when the prompt already carries the answer is a "
            "behavioural preference, not a correctness property."
        ),
    )
    eval_min_retrieval_recall: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Aggregate floor for expected-document retrieval recall.",
    )
    eval_regression_tolerance: float = Field(
        default=0.02,
        ge=0.0,
        le=1.0,
        description=(
            "A baseline comparison flags a metric as a regression once it drops by "
            "more than this. Below it, the move is non-deterministic judge noise."
        ),
    )
    eval_report_worst_items: int = Field(
        default=10,
        ge=0,
        description="Worst-performing items shown in full, with their trace ids.",
    )
    eval_report_answer_chars: int = Field(
        default=600,
        gt=0,
        description=(
            "Characters of the emitted answer stored on a result. Clipping keeps a "
            "run artefact from becoming a copy of the corpus."
        ),
    )
    eval_score_prefix: str = Field(
        default="eval.",
        description="Prefix for every Langfuse score name pushed by the harness.",
    )

    # ----------------------------------------------------------------------- api
    api_host: str = Field(default="0.0.0.0", description="Bind address for the API.")  # noqa: S104
    api_port: int = Field(default=8000, description="Bind port for the API.")
    api_root_path: str = Field(
        default="", description="ASGI root_path when served behind a path prefix."
    )
    api_prefix: str = Field(
        default="/api/v1", description="Versioned prefix for every routed endpoint."
    )
    api_cors_origins: list[str] = Field(
        default=["http://localhost:5173"],
        description="Allowed browser origins for the React app.",
    )
    api_request_timeout_seconds: float = Field(
        default=300.0, gt=0, description="Server-side ceiling for one chat request."
    )
    api_max_upload_bytes: int = Field(
        default=25 * 1024 * 1024,
        gt=0,
        description="Maximum multipart upload size on POST /documents.",
    )
    api_rate_limit_per_minute: int = Field(
        default=60,
        gt=0,
        description="Per-principal request budget per minute (Redis-backed).",
    )
    api_sse_keepalive_seconds: float = Field(
        default=15.0,
        gt=0,
        description="Interval between SSE comment heartbeats.",
    )
    api_metrics_enabled: bool = Field(
        default=True, description="Expose Prometheus text on /metrics."
    )
    api_request_id_header: str = Field(
        default="x-request-id",
        description=(
            "Request-correlation header. Echoed on every response and bound into "
            "structlog, so a log line, a trace and a browser entry can be joined."
        ),
    )
    api_trace_id_header: str = Field(
        default="x-trace-id",
        description="Response header carrying the Langfuse trace id, when one is open.",
    )
    api_cors_allow_credentials: bool = Field(
        default=True,
        description=(
            "Send Access-Control-Allow-Credentials. MSAL tokens travel in an "
            "Authorization header rather than a cookie, but the opt-in is explicit."
        ),
    )
    api_cors_allow_headers: list[str] = Field(
        default=["*"], description="Request headers the browser may send cross-origin."
    )
    api_cors_expose_headers: list[str] = Field(
        default=["x-request-id", "x-trace-id", "retry-after"],
        description=(
            "Response headers readable by the web app. The correlation headers must "
            "be exposed or the client cannot report a request id."
        ),
    )
    api_cors_max_age_seconds: int = Field(
        default=600, ge=0, description="Pre-flight cache lifetime."
    )
    api_gzip_min_bytes: int = Field(
        default=1_024,
        gt=0,
        description=(
            "Below this, compression costs more than it saves. SSE responses are "
            "never compressed: buffering would defeat token streaming."
        ),
    )
    api_rate_limit_enabled: bool = Field(
        default=True, description="Enforce the per-principal Redis rate limit."
    )
    api_rate_limit_burst: int = Field(
        default=20,
        gt=0,
        description=(
            "Bucket depth above api_rate_limit_per_minute. A chat turn is one "
            "request but a page load is several."
        ),
    )
    api_rate_limit_prefix: str = Field(
        default="rag:ratelimit:",
        description=(
            "Redis key prefix; keys are '<prefix><tenant_id>:<user_id>' — tenant "
            "first, so one tenant cannot evict another's counters."
        ),
    )
    api_sse_retry_ms: int = Field(
        default=2_000,
        gt=0,
        description="The one 'retry:' hint sent at the start of an SSE stream.",
    )
    api_sse_heartbeat_comment: str = Field(
        default="heartbeat",
        description=(
            "Comment text sent every api_sse_keepalive_seconds. A bare comment "
            "keeps proxies from timing the connection out without emitting an event."
        ),
    )
    api_problem_type_base: str = Field(
        default="https://productionizing-rag-azure.dev/problems",
        description=(
            "Base URI for RFC 7807 'type' members. A stable, dereferenceable-looking "
            "URI is what makes a problem type machine-identifiable."
        ),
    )
    api_docs_enabled: bool = Field(
        default=True,
        description=(
            "Serve /docs, /redoc and /openapi.json. Additionally forced off when "
            "env == 'production' unless explicitly re-enabled."
        ),
    )
    api_default_page_size: int = Field(
        default=50, gt=0, description="Default 'limit' on every list endpoint."
    )
    api_max_page_size: int = Field(
        default=200,
        gt=0,
        description=(
            "Hard ceiling on 'limit'; an unbounded page is a denial of service."
        ),
    )
    api_warm_models: bool = Field(
        default=True,
        description=(
            "Load the FastEmbed dense, sparse and rerank models during startup "
            "rather than on the first turn. Disable for a fast-booting CI container."
        ),
    )
    api_ensure_collections: bool = Field(
        default=True,
        description=(
            "Create the Qdrant collections during startup. Idempotent; disable when "
            "a separate bootstrap job owns the schema."
        ),
    )
    api_readiness_timeout_seconds: float = Field(
        default=5.0,
        gt=0,
        description="Seconds the readiness probe waits on each dependency.",
    )

    # ------------------------------------------------------------------ validation
    @field_validator("ingest_working_days")
    @classmethod
    def _validate_working_days(cls, value: list[int]) -> list[int]:
        """Reject weekday indices outside 0..6.

        Args:
            value: Candidate weekday indices.

        Returns:
            The de-duplicated, sorted weekday list.

        Raises:
            ValueError: If any index is outside the Monday..Sunday range.
        """
        for day in value:
            if not 0 <= day <= 6:
                msg = f"ingest_working_days entries must be 0..6, got {day}"
                raise ValueError(msg)
        return sorted(set(value))

    @field_validator("ingest_timezone")
    @classmethod
    def _validate_timezone(cls, value: str) -> str:
        """Ensure the configured timezone exists in the IANA database.

        Args:
            value: IANA timezone name.

        Returns:
            The validated timezone name.

        Raises:
            ValueError: If the timezone is unknown to the platform tz database.
        """
        try:
            ZoneInfo(value)
        except Exception as exc:
            msg = f"unknown ingest_timezone {value!r}"
            raise ValueError(msg) from exc
        return value

    @field_validator("ingest_cron")
    @classmethod
    def _validate_cron(cls, value: str) -> str:
        """Require the six-field NCRONTAB form Azure Functions expects.

        Args:
            value: Candidate NCRONTAB expression.

        Returns:
            The validated expression.

        Raises:
            ValueError: If the expression does not have exactly six fields.
        """
        fields = value.split()
        if len(fields) != 6:
            msg = (
                "ingest_cron must be a six-field NCRONTAB expression "
                f"(sec min hour day month dow), got {value!r}"
            )
            raise ValueError(msg)
        return value

    def _validate_llm_provider(self) -> None:
        """Refuse a provider asked for a capability it does not have.

        Extended thinking, prompt caching, remote MCP and context compaction are
        Anthropic features. A deployment that switches provider without switching
        these off would quietly do less than its configuration claims, so this fails
        at construction instead — the same posture as ``entra_dev_mode`` in
        production.

        Raises:
            ValueError: If an Anthropic-only feature is enabled against another
                provider, if Azure OpenAI has no endpoint, or if a non-Anthropic
                provider has no alias for a model slot a call site will ask for.
        """
        if self.llm_provider == "anthropic":
            return

        unsupported = [
            name
            for name, enabled in (
                ("anthropic_thinking", self.anthropic_thinking),
                ("anthropic_cache_system", self.anthropic_cache_system),
                ("tool_mcp_enabled", self.tool_mcp_enabled),
                ("context_compaction_enabled", self.context_compaction_enabled),
            )
            if enabled
        ]
        if unsupported:
            msg = (
                f"llm_provider={self.llm_provider!r} cannot honour "
                f"{', '.join(sorted(unsupported))}: extended thinking, prompt "
                "caching, remote MCP and context compaction are Anthropic-only. "
                "Set them false, or keep llm_provider='anthropic'."
            )
            raise ValueError(msg)

        if self.llm_provider == "azure_openai" and not self.azure_openai_endpoint:
            msg = (
                "azure_openai_endpoint is required when llm_provider='azure_openai': "
                "there is no default resource to fall back to"
            )
            raise ValueError(msg)

        # Call sites name Claude models directly, so without an alias the request
        # would reach the backend as e.g. 'claude-opus-5' and be rejected there —
        # at request time, one failed turn at a time, rather than here.
        slots = {
            "anthropic_model_main": self.anthropic_model_main,
            "anthropic_model_fast": self.anthropic_model_fast,
            "anthropic_model_cheap": self.anthropic_model_cheap,
        }
        unmapped = sorted(
            f"{field}={model!r}"
            for field, model in slots.items()
            if model not in self.llm_model_aliases
        )
        if unmapped:
            msg = (
                f"llm_provider={self.llm_provider!r} needs llm_model_aliases entries "
                f"for every model slot; missing: {', '.join(unmapped)}. Map each onto "
                "a deployment or model this provider serves."
            )
            raise ValueError(msg)

    @property
    def llm_model_main(self) -> str:
        """Model slot for answer generation and the agentic tool loop.

        Returns:
            The configured main model id, before alias mapping. The active provider
            maps it through :attr:`llm_model_aliases`.
        """
        return self.anthropic_model_main

    @property
    def llm_model_fast(self) -> str:
        """Model slot for query transformation, summarisation and extraction.

        Returns:
            The configured fast model id, before alias mapping.
        """
        return self.anthropic_model_fast

    @property
    def llm_model_cheap(self) -> str:
        """Model slot for classification.

        Returns:
            The configured cheap model id, before alias mapping.
        """
        return self.anthropic_model_cheap

    @model_validator(mode="after")
    def _validate_consistency(self) -> Settings:
        """Cross-field guards that catch unsafe or self-contradicting config.

        Returns:
            The validated settings instance.

        Raises:
            ValueError: If dev auth is enabled in production, if a provider is asked
                for a capability it does not have, or if a derived budget/limit
                relationship is impossible to satisfy.
        """
        self._validate_llm_provider()
        if self.entra_dev_mode and self.env == "production":
            msg = (
                "entra_dev_mode=true is refused when env='production': unsigned "
                "principals must never be accepted in production"
            )
            raise ValueError(msg)
        if self.context_reserve_output_tokens >= self.context_budget_tokens:
            msg = (
                "context_reserve_output_tokens must be smaller than "
                "context_budget_tokens"
            )
            raise ValueError(msg)
        if self.retrieval_top_n > self.retrieval_limit:
            msg = "retrieval_top_n cannot exceed retrieval_limit"
            raise ValueError(msg)
        if self.chunk_min_tokens > self.chunk_target_tokens:
            msg = "chunk_min_tokens cannot exceed chunk_target_tokens"
            raise ValueError(msg)
        if self.chunk_target_tokens > self.chunk_max_tokens:
            msg = "chunk_target_tokens cannot exceed chunk_max_tokens"
            raise ValueError(msg)
        if self.chunk_overlap_tokens >= self.chunk_target_tokens:
            msg = "chunk_overlap_tokens must be smaller than chunk_target_tokens"
            raise ValueError(msg)
        if (
            self.guardrail_injection_warn_threshold
            > self.guardrail_injection_block_threshold
        ):
            msg = (
                "guardrail_injection_warn_threshold must not exceed "
                "guardrail_injection_block_threshold"
            )
            raise ValueError(msg)
        if (
            self.guardrail_injection_retrieved_block_threshold
            > self.guardrail_injection_block_threshold
        ):
            msg = (
                "guardrail_injection_retrieved_block_threshold must not exceed "
                "guardrail_injection_block_threshold: retrieved text is untrusted "
                "and cannot be judged more leniently than the user's own turn"
            )
            raise ValueError(msg)
        if (
            self.guardrail_output_block_below_groundedness
            > self.guardrail_min_groundedness
        ):
            msg = (
                "guardrail_output_block_below_groundedness must not exceed "
                "guardrail_min_groundedness: the block floor sits underneath the "
                "threshold that merely appends an uncertainty notice"
            )
            raise ValueError(msg)
        if self.citation_min_span_chars > self.citation_max_span_chars:
            msg = "citation_min_span_chars cannot exceed citation_max_span_chars"
            raise ValueError(msg)
        if self.api_default_page_size > self.api_max_page_size:
            msg = "api_default_page_size cannot exceed api_max_page_size"
            raise ValueError(msg)
        weight_total = (
            self.eval_correctness_f1_weight + self.eval_correctness_similarity_weight
        )
        if not math.isclose(weight_total, 1.0, rel_tol=0.0, abs_tol=1e-9):
            msg = (
                "eval_correctness_f1_weight and eval_correctness_similarity_weight "
                f"must sum to 1.0, got {weight_total}"
            )
            raise ValueError(msg)
        if self.tool_allow_insecure_http and self.env == "production":
            msg = "tool_allow_insecure_http=true is refused when env='production'"
            raise ValueError(msg)
        return self

    # ------------------------------------------------------------ derived values
    @property
    def is_production(self) -> bool:
        """Whether this process runs in the production environment.

        Returns:
            True when :attr:`env` is ``"production"``.
        """
        return self.env == "production"

    @property
    def is_local(self) -> bool:
        """Whether this process runs against local docker-compose infrastructure.

        Returns:
            True when :attr:`env` is ``"local"``.
        """
        return self.env == "local"

    @computed_field  # type: ignore[prop-decorator]
    @property
    def database_url(self) -> str:
        """SQLAlchemy async URL for PostgreSQL.

        Returns:
            ``database_url_override`` when set, otherwise a
            ``postgresql+asyncpg://`` URL assembled from the ``postgres_*`` fields.
        """
        if self.database_url_override:
            return self.database_url_override
        return (
            f"postgresql+asyncpg://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    @property
    def alembic_database_url(self) -> str:
        """URL Alembic should use for migrations.

        Returns:
            The same async URL as :attr:`database_url`; ``env.py`` runs migrations
            through an async engine so no driver swap is needed.
        """
        return self.database_url

    @property
    def entra_issuer(self) -> str | None:
        """Expected ``iss`` claim for inbound Entra ID tokens.

        Returns:
            The v2.0 issuer URL, or None when ``entra_tenant_id`` is unset.
        """
        if not self.entra_tenant_id:
            return None
        return f"{self.entra_authority_host}/{self.entra_tenant_id}/v2.0"

    @property
    def entra_jwks_url(self) -> str | None:
        """JWKS endpoint used to validate token signatures.

        Returns:
            The discovery keys URL, or None when ``entra_tenant_id`` is unset.
        """
        if not self.entra_tenant_id:
            return None
        return f"{self.entra_authority_host}/{self.entra_tenant_id}/discovery/v2.0/keys"

    @property
    def expected_audience(self) -> str | None:
        """Audience the API accepts on inbound tokens.

        Returns:
            ``entra_audience`` when set, else ``entra_client_id``, else None.
        """
        return self.entra_audience or self.entra_client_id

    @property
    def context_prompt_budget_tokens(self) -> int:
        """Token budget available to the prompt after reserving output space.

        Returns:
            ``context_budget_tokens`` minus ``context_reserve_output_tokens``.
        """
        return self.context_budget_tokens - self.context_reserve_output_tokens

    @property
    def context_compact_threshold_tokens(self) -> int:
        """Prompt size at which proactive compaction fires.

        Returns:
            ``context_compact_at_ratio`` of the usable prompt budget, in tokens.
        """
        return int(self.context_prompt_budget_tokens * self.context_compact_at_ratio)

    @property
    def langfuse_ready(self) -> bool:
        """Whether tracing can actually reach Langfuse.

        Returns:
            True only when tracing is enabled and both API keys are present.
        """
        return bool(
            self.langfuse_enabled
            and self.langfuse_public_key
            and self.langfuse_secret_key
        )

    def price_for_model(self, model: str) -> tuple[float, float]:
        """Look up USD-per-million-token pricing for a model.

        Args:
            model: Exact Anthropic model id, e.g. ``"claude-opus-5"``.

        Returns:
            A ``(input_price, output_price)`` pair in USD per million tokens.
            Ollama is local inference and costs nothing per token, so it reports
            zero rather than the Anthropic fallback rate, which would otherwise make
            a self-hosted deployment look like the most expensive one. Unknown models
            on a paid provider fall back to the MODEL_MAIN price so cost accounting
            over-reports rather than silently reporting zero.
        """
        if self.llm_provider == "ollama":
            return (0.0, 0.0)
        known = self.anthropic_price_per_mtok
        if model in known:
            return known[model]
        return known.get(self.anthropic_model_main, (0.0, 0.0))

    def is_within_working_hours(self, now: datetime | None = None) -> bool:
        """Whether a moment falls inside configured working hours.

        The scheduled-ingestion guard uses this to refuse a delta refresh during
        business hours unless :attr:`ingest_force` is set. Non-working weekdays are
        always outside the window, and a window whose end hour is below its start
        hour is treated as wrapping past midnight.

        Args:
            now: Moment to test. Naive datetimes are interpreted in
                :attr:`ingest_timezone`; aware datetimes are converted to it.
                Defaults to the current time in that timezone.

        Returns:
            True when the moment is inside working hours.
        """
        tz = ZoneInfo(self.ingest_timezone)
        moment = datetime.now(tz) if now is None else now
        moment = (
            moment.replace(tzinfo=tz)
            if moment.tzinfo is None
            else moment.astimezone(tz)
        )

        if moment.weekday() not in self.ingest_working_days:
            return False

        hour = moment.hour + moment.minute / 60.0
        start = float(self.ingest_working_hours_start)
        end = float(self.ingest_working_hours_end)
        if start == end:
            return False
        if start < end:
            return start <= hour < end
        # Window wraps past midnight, e.g. start=22, end=6.
        return hour >= start or hour < end

    def may_start_scheduled_ingest(
        self, now: datetime | None = None
    ) -> tuple[bool, str]:
        """Decide whether a scheduled ingestion run is allowed to start.

        Args:
            now: Moment the run is starting. See :meth:`is_within_working_hours`.

        Returns:
            A ``(allowed, reason)`` pair. ``reason`` is a short machine-friendly
            token suitable for logging and for ``ingest_runs.status`` bookkeeping:
            ``"ok"``, ``"disabled"``, ``"forced"`` or ``"working_hours"``.
        """
        if not self.ingest_enabled:
            return False, "disabled"
        if not self.is_within_working_hours(now):
            return True, "ok"
        if self.ingest_force:
            return True, "forced"
        return False, "working_hours"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings singleton.

    Returns:
        A cached :class:`Settings` instance built from the environment and `.env`.
        Call ``get_settings.cache_clear()`` in tests after mutating the environment.
    """
    return Settings()
