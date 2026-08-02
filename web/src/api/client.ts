/**
 * HTTP client for the FastAPI surface described in `docs/CONTRACTS.md`.
 *
 * Responsibilities:
 *
 * * attach `Authorization: Bearer <Entra JWT>` to every request (or the unsigned
 *   dev-principal header when `VITE_DEV_MODE` is on, mirroring `entra_dev_mode`);
 * * retry exactly once on `401` after forcing a silent token refresh, because an
 *   access token can expire between two keystrokes;
 * * surface failures as a typed {@link ApiError} carrying the FastAPI `detail`;
 * * keep every tunable (base URL, prefix, timeout, page size) in {@link apiEnv}
 *   rather than at a call site.
 *
 * `fetch` is used rather than axios so the same authorization path serves both JSON
 * requests and the SSE stream in `./sse.ts` — `EventSource` cannot send headers.
 */

import type {
  ChatRequest,
  ChatResponse,
  CompactionResult,
  DocumentProvenance,
  DocumentSummary,
  EvalRun,
  EvalRunRequest,
  FeedbackRequest,
  HealthResponse,
  IngestRunSummary,
  IngestTriggerRequest,
  LongTermMemory,
  Message,
  MetadataFilter,
  Principal,
  ProfileUpdate,
  RetrievalResult,
  ScheduleInfo,
  SessionSummary,
  SourceConfigSummary,
  UploadOptions,
  UserProfile,
} from './types';

// ------------------------------------------------------------------ config

/** Resolved client configuration. Every value is env-driven with a default. */
export interface ApiEnv {
  /** Origin of the API, or `''` for same-origin (dev proxy / co-hosted nginx). */
  baseUrl: string;
  /** Versioned route prefix; must match ragcore `api_prefix`. */
  prefix: string;
  /** Timeout for non-streaming requests, in milliseconds. */
  timeoutMs: number;
  /** Milliseconds of stream silence tolerated before the stream is failed. */
  streamIdleTimeoutMs: number;
  /** Transport retries for a stream that died before its first token. */
  streamMaxRetries: number;
  /** Base backoff between stream retries, in milliseconds. */
  streamRetryBaseMs: number;
  /** Default page size for list endpoints. */
  pageSize: number;
  /** Whether the unsigned dev-principal header replaces bearer auth. */
  devMode: boolean;
  /** Header name carrying the dev principal (`entra_dev_principal_header`). */
  devPrincipalHeader: string;
  /** JSON `Principal` sent in dev mode. */
  devPrincipal: string;
}

function readNumber(raw: string | undefined, fallback: number): number {
  const value = Number(raw);
  return Number.isFinite(value) && value > 0 ? value : fallback;
}

function readBoolean(raw: string | undefined, fallback: boolean): boolean {
  if (raw === undefined || raw === '') return fallback;
  return raw === 'true' || raw === '1' || raw === 'yes';
}

function trimTrailingSlash(value: string): string {
  return value.endsWith('/') ? value.slice(0, -1) : value;
}

/** Client configuration resolved from `import.meta.env`. */
export const apiEnv: ApiEnv = {
  baseUrl: trimTrailingSlash(import.meta.env.VITE_API_BASE_URL ?? ''),
  prefix: trimTrailingSlash(import.meta.env.VITE_API_PREFIX ?? '/api/v1'),
  timeoutMs: readNumber(import.meta.env.VITE_API_TIMEOUT_MS, 30_000),
  streamIdleTimeoutMs: readNumber(import.meta.env.VITE_STREAM_IDLE_TIMEOUT_MS, 120_000),
  streamMaxRetries: readNumber(import.meta.env.VITE_STREAM_MAX_RETRIES, 2),
  streamRetryBaseMs: readNumber(import.meta.env.VITE_STREAM_RETRY_BASE_MS, 500),
  pageSize: readNumber(import.meta.env.VITE_PAGE_SIZE, 50),
  devMode: readBoolean(import.meta.env.VITE_DEV_MODE, false),
  devPrincipalHeader: import.meta.env.VITE_DEV_PRINCIPAL_HEADER ?? 'x-dev-principal',
  devPrincipal: import.meta.env.VITE_DEV_PRINCIPAL ?? '',
};

// ------------------------------------------------------------------- errors

/** A non-2xx API response, or a transport failure. */
export class ApiError extends Error {
  /** HTTP status, or 0 for a transport-level failure. */
  readonly status: number;

  /** FastAPI `detail`, when the body carried one. */
  readonly detail: unknown;

  /**
   * Build an API error.
   *
   * @param message Human-readable, already-safe message.
   * @param status HTTP status code, or 0 when the request never completed.
   * @param detail Parsed `detail` field from the error body, if any.
   */
  constructor(message: string, status = 0, detail: unknown = null) {
    super(message);
    this.name = 'ApiError';
    this.status = status;
    this.detail = detail;
  }

  /** Whether the caller is unauthenticated or the token expired. */
  get isUnauthorized(): boolean {
    return this.status === 401;
  }

  /** Whether the principal lacks the role or clearance for this route. */
  get isForbidden(): boolean {
    return this.status === 403;
  }

  /** Whether the route or entity does not exist. */
  get isNotFound(): boolean {
    return this.status === 404;
  }

  /** Whether the endpoint is not implemented by this API build. */
  get isUnavailableRoute(): boolean {
    return this.status === 404 || this.status === 405 || this.status === 501;
  }
}

// ------------------------------------------------------- token plumbing

/** Supplies a bearer token; `null` means "no token available". */
export type TokenProvider = (options?: { forceRefresh?: boolean }) => Promise<
  string | null
>;

let tokenProvider: TokenProvider = async () => null;

/**
 * Register the function that mints access tokens.
 *
 * Called once by `AuthProvider` so non-React modules (this client, the SSE reader)
 * never need to reach into React context.
 *
 * @param provider Token provider backed by MSAL, or one returning null in dev mode.
 */
export function setTokenProvider(provider: TokenProvider): void {
  tokenProvider = provider;
}

/** Signals raised when the caller must re-authenticate interactively. */
export type AuthFailureHandler = () => void;

let authFailureHandler: AuthFailureHandler = () => {};

/**
 * Register a callback invoked when a request stays unauthorized after a refresh.
 *
 * @param handler Callback that triggers interactive sign-in.
 */
export function setAuthFailureHandler(handler: AuthFailureHandler): void {
  authFailureHandler = handler;
}

/**
 * Turn an API path plus query parameters into a fetchable URL.
 *
 * A path starting with `http` is used verbatim; anything else is prefixed with the
 * configured base URL and versioned prefix. Empty, null and undefined query values
 * are dropped so an unset filter never becomes `?tag=`.
 *
 * @param path Path relative to the versioned prefix, or an absolute URL.
 * @param query Optional query parameters; arrays repeat the key.
 * @returns The resolved URL.
 */
export function resolveUrl(path: string, query?: QueryParams): string {
  const absolute = /^https?:\/\//i.test(path);
  const base = absolute ? path : `${apiEnv.baseUrl}${apiEnv.prefix}${path}`;
  if (!query) return base;
  const search = new URLSearchParams();
  for (const [key, value] of Object.entries(query)) {
    if (value === undefined || value === null || value === '') continue;
    if (Array.isArray(value)) {
      for (const item of value) search.append(key, String(item));
    } else {
      search.append(key, String(value));
    }
  }
  const encoded = search.toString();
  return encoded ? `${base}?${encoded}` : base;
}

async function buildHeaders(
  base: HeadersInit | undefined,
  forceRefresh: boolean,
): Promise<Headers> {
  const headers = new Headers(base);
  const token = await tokenProvider({ forceRefresh });
  if (token) {
    headers.set('Authorization', `Bearer ${token}`);
  } else if (apiEnv.devMode && apiEnv.devPrincipal) {
    // Mirrors `entra_dev_mode`: unsigned, local-only, refused in production.
    headers.set(apiEnv.devPrincipalHeader, apiEnv.devPrincipal);
  }
  return headers;
}

/**
 * Link a caller signal to an internal timeout without needing `AbortSignal.any`.
 *
 * @param signal Caller-supplied signal, if any.
 * @param timeoutMs Timeout in milliseconds, or 0 to disable.
 * @returns The combined signal plus a cleanup function.
 */
function linkSignals(
  signal: AbortSignal | undefined,
  timeoutMs: number,
): { signal: AbortSignal; cleanup: () => void } {
  const controller = new AbortController();
  const abort = (reason?: unknown): void => controller.abort(reason);
  let timer: ReturnType<typeof setTimeout> | undefined;

  if (signal) {
    if (signal.aborted) abort(signal.reason);
    else signal.addEventListener('abort', () => abort(signal.reason), { once: true });
  }
  if (timeoutMs > 0) {
    timer = setTimeout(
      () => abort(new ApiError('Request timed out', 0)),
      timeoutMs,
    );
  }
  return {
    signal: controller.signal,
    cleanup: () => {
      if (timer !== undefined) clearTimeout(timer);
    },
  };
}

/**
 * Perform an authorized `fetch`, retrying once on 401 after a forced refresh.
 *
 * Exported so the SSE reader shares one authorization path with JSON requests.
 *
 * @param url Fully resolved URL, as returned by {@link resolveUrl}.
 * @param init Standard fetch init. `body` must be re-sendable (string or FormData).
 * @returns The raw {@link Response}; status is not inspected here.
 * @throws ApiError If the request never reaches the server.
 */
export async function authorizedFetch(
  url: string,
  init: RequestInit = {},
): Promise<Response> {
  const send = async (forceRefresh: boolean): Promise<Response> =>
    fetch(url, {
      ...init,
      headers: await buildHeaders(init.headers, forceRefresh),
      credentials: 'omit',
      mode: 'cors',
    });

  let response: Response;
  try {
    response = await send(false);
  } catch (error) {
    if (init.signal?.aborted) throw error;
    throw new ApiError(
      error instanceof Error ? error.message : 'Network request failed',
      0,
    );
  }

  if (response.status !== 401 || init.signal?.aborted) return response;

  // A single retry after a forced silent refresh. If it still fails the token is
  // genuinely dead and interactive sign-in is required.
  try {
    response = await send(true);
  } catch (error) {
    if (init.signal?.aborted) throw error;
    throw new ApiError(
      error instanceof Error ? error.message : 'Network request failed',
      0,
    );
  }
  if (response.status === 401) authFailureHandler();
  return response;
}

/** Query parameters accepted by {@link apiRequest}. */
export type QueryParams = Record<
  string,
  string | number | boolean | undefined | null | Array<string | number>
>;

/** Options for {@link apiRequest}. */
export interface RequestOptions {
  method?: 'GET' | 'POST' | 'PUT' | 'PATCH' | 'DELETE';
  query?: QueryParams;
  body?: unknown;
  formData?: FormData;
  signal?: AbortSignal;
  timeoutMs?: number;
}

async function readErrorBody(response: Response): Promise<{
  message: string;
  detail: unknown;
}> {
  const fallback = `${response.status} ${response.statusText || 'Request failed'}`;
  let text = '';
  try {
    text = await response.text();
  } catch {
    return { message: fallback, detail: null };
  }
  if (!text) return { message: fallback, detail: null };
  try {
    const parsed: unknown = JSON.parse(text);
    if (parsed && typeof parsed === 'object') {
      const record = parsed as Record<string, unknown>;
      const detail = record.detail ?? record.message ?? record.error;
      if (typeof detail === 'string' && detail) {
        return { message: detail, detail };
      }
      return { message: fallback, detail: detail ?? parsed };
    }
  } catch {
    return { message: text.slice(0, 500) || fallback, detail: text };
  }
  return { message: fallback, detail: text };
}

/**
 * Issue a JSON request against the API.
 *
 * @param path Path relative to the versioned prefix, e.g. `/sessions`.
 * @param options Method, query, body and cancellation options.
 * @returns The parsed response body, or `undefined` for `204 No Content`.
 * @throws ApiError On any non-2xx response or transport failure.
 */
export async function apiRequest<T>(
  path: string,
  options: RequestOptions = {},
): Promise<T> {
  const { method = 'GET', query, body, formData, signal, timeoutMs } = options;
  const linked = linkSignals(signal, timeoutMs ?? apiEnv.timeoutMs);
  const headers = new Headers();
  headers.set('Accept', 'application/json');

  let payload: BodyInit | undefined;
  if (formData) {
    payload = formData; // Let the browser set the multipart boundary.
  } else if (body !== undefined) {
    headers.set('Content-Type', 'application/json');
    payload = JSON.stringify(body);
  }

  try {
    const response = await authorizedFetch(resolveUrl(path, query), {
      method,
      headers,
      body: payload,
      signal: linked.signal,
    });
    if (!response.ok) {
      const { message, detail } = await readErrorBody(response);
      throw new ApiError(message, response.status, detail);
    }
    if (response.status === 204) return undefined as T;
    const text = await response.text();
    if (!text) return undefined as T;
    return JSON.parse(text) as T;
  } finally {
    linked.cleanup();
  }
}

const LIST_ENVELOPE_KEYS = [
  'items',
  'results',
  'data',
  'sessions',
  'documents',
  'memories',
  'runs',
  'sources',
  'messages',
];

/**
 * Accept either a bare JSON array or a `{ items: [...] }`-style envelope.
 *
 * List endpoints are the one place where the contract does not pin the envelope, so
 * the client tolerates both rather than guessing.
 *
 * @param raw Parsed response body.
 * @returns The extracted array, or `[]` when nothing list-shaped was found.
 */
export function unwrapList<T>(raw: unknown): T[] {
  if (Array.isArray(raw)) return raw as T[];
  if (raw && typeof raw === 'object') {
    const record = raw as Record<string, unknown>;
    for (const key of LIST_ENVELOPE_KEYS) {
      const value = record[key];
      if (Array.isArray(value)) return value as T[];
    }
  }
  return [];
}

// -------------------------------------------------------------- endpoints

/**
 * Resolve the calling principal.
 *
 * @param signal Optional cancellation signal.
 * @returns The `Principal` the API derived from the token claims.
 */
export function getMe(signal?: AbortSignal): Promise<Principal> {
  return apiRequest<Principal>('/me', { signal });
}

/**
 * Probe API readiness.
 *
 * `/readyz` sits outside the versioned prefix and needs no bearer token.
 *
 * @param signal Optional cancellation signal.
 * @returns The readiness payload.
 */
export async function getReadiness(signal?: AbortSignal): Promise<HealthResponse> {
  const response = await fetch(`${apiEnv.baseUrl}/readyz`, { signal });
  if (!response.ok) {
    throw new ApiError(`readyz returned ${response.status}`, response.status);
  }
  return (await response.json()) as HealthResponse;
}

/**
 * Retrieval without generation.
 *
 * @param query The search text.
 * @param filters Facet filter composed on top of the mandatory ACL filter.
 * @param topN Result ceiling; the API default applies when omitted.
 * @param signal Optional cancellation signal.
 * @returns The full `RetrievalResult`, including audited drops.
 */
export function search(
  query: string,
  filters?: MetadataFilter | null,
  topN?: number,
  signal?: AbortSignal,
): Promise<RetrievalResult> {
  return apiRequest<RetrievalResult>('/search', {
    method: 'POST',
    body: { query, filters: filters ?? null, top_n: topN ?? null },
    signal,
  });
}

/**
 * Non-streaming chat, used only as a fallback when streaming is unavailable.
 *
 * @param request The chat request; `stream` is forced to false.
 * @param signal Optional cancellation signal.
 * @returns The complete answer with citations and stats.
 */
export function chatOnce(
  request: ChatRequest,
  signal?: AbortSignal,
): Promise<ChatResponse> {
  return apiRequest<ChatResponse>('/chat', {
    method: 'POST',
    body: { ...request, stream: false },
    signal,
    timeoutMs: 0,
  });
}

/**
 * List the caller's chat sessions, newest first.
 *
 * @param signal Optional cancellation signal.
 * @returns Session summaries.
 */
export async function listSessions(signal?: AbortSignal): Promise<SessionSummary[]> {
  const raw = await apiRequest<unknown>('/sessions', {
    query: { limit: apiEnv.pageSize },
    signal,
  });
  return unwrapList<SessionSummary>(raw);
}

/**
 * Fetch one session.
 *
 * @param sessionId Session id.
 * @param signal Optional cancellation signal.
 * @returns The session summary.
 */
export function getSession(
  sessionId: string,
  signal?: AbortSignal,
): Promise<SessionSummary> {
  return apiRequest<SessionSummary>(`/sessions/${encodeURIComponent(sessionId)}`, {
    signal,
  });
}

/**
 * Delete a session and its messages.
 *
 * @param sessionId Session id.
 */
export function deleteSession(sessionId: string): Promise<void> {
  return apiRequest<void>(`/sessions/${encodeURIComponent(sessionId)}`, {
    method: 'DELETE',
  });
}

/**
 * List a session's messages, including suppressed turns.
 *
 * @param sessionId Session id.
 * @param signal Optional cancellation signal.
 * @returns Messages in ascending chronological order.
 */
export async function listMessages(
  sessionId: string,
  signal?: AbortSignal,
): Promise<Message[]> {
  const raw = await apiRequest<unknown>(
    `/sessions/${encodeURIComponent(sessionId)}/messages`,
    { query: { include_suppressed: true }, signal },
  );
  return unwrapList<Message>(raw);
}

/**
 * Force context compaction for a session.
 *
 * @param sessionId Session id.
 * @returns What the compaction changed.
 */
export function compactSession(sessionId: string): Promise<CompactionResult> {
  return apiRequest<CompactionResult>(
    `/sessions/${encodeURIComponent(sessionId)}/compact`,
    { method: 'POST', timeoutMs: 0 },
  );
}

/**
 * List documents visible to the caller.
 *
 * @param query Optional filters understood by the API (`q`, `source_type`, ...).
 * @param signal Optional cancellation signal.
 * @returns Document summaries.
 */
export async function listDocuments(
  query?: QueryParams,
  signal?: AbortSignal,
): Promise<DocumentSummary[]> {
  const raw = await apiRequest<unknown>('/documents', {
    query: { limit: apiEnv.pageSize, ...query },
    signal,
  });
  return unwrapList<DocumentSummary>(raw);
}

/**
 * Upload a document for ingestion.
 *
 * @param file The file to upload.
 * @param options Classification and metadata applied to the new document.
 * @returns The created document summary.
 */
export function uploadDocument(
  file: File,
  options: UploadOptions = {},
): Promise<DocumentSummary> {
  const form = new FormData();
  form.append('file', file);
  if (options.title) form.append('title', options.title);
  if (options.doc_type) form.append('doc_type', options.doc_type);
  if (options.language) form.append('language', options.language);
  if (options.classification) form.append('classification', options.classification);
  for (const tag of options.tags ?? []) form.append('tags', tag);
  for (const role of options.allowed_roles ?? []) form.append('allowed_roles', role);
  for (const group of options.allowed_groups ?? []) form.append('allowed_groups', group);
  return apiRequest<DocumentSummary>('/documents', {
    method: 'POST',
    formData: form,
    timeoutMs: 0,
  });
}

/**
 * Soft-delete a document and its chunks.
 *
 * @param documentId Document id.
 */
export function deleteDocument(documentId: string): Promise<void> {
  return apiRequest<void>(`/documents/${encodeURIComponent(documentId)}`, {
    method: 'DELETE',
  });
}

/**
 * Re-run parsing, chunking and embedding for one document.
 *
 * @param documentId Document id.
 * @returns The ingest run summary the reindex produced.
 */
export function reindexDocument(documentId: string): Promise<IngestRunSummary> {
  return apiRequest<IngestRunSummary>(
    `/documents/${encodeURIComponent(documentId)}/reindex`,
    { method: 'POST', timeoutMs: 0 },
  );
}

/**
 * Fetch a document's full provenance chain.
 *
 * @param documentId Document id.
 * @param signal Optional cancellation signal.
 * @returns The provenance mapping, including lineage records when present.
 */
export function getDocumentLineage(
  documentId: string,
  signal?: AbortSignal,
): Promise<DocumentProvenance> {
  return apiRequest<DocumentProvenance>(
    `/documents/${encodeURIComponent(documentId)}/lineage`,
    { signal },
  );
}

/**
 * Fetch the caller's rolling profile.
 *
 * @param signal Optional cancellation signal.
 * @returns The user profile.
 */
export function getMemoryProfile(signal?: AbortSignal): Promise<UserProfile> {
  return apiRequest<UserProfile>('/memory/profile', { signal });
}

/**
 * Update editable profile fields (Addendum B).
 *
 * @param update Fields to change.
 * @returns The stored profile.
 */
export function updateMemoryProfile(update: ProfileUpdate): Promise<UserProfile> {
  return apiRequest<UserProfile>('/memory/profile', { method: 'PUT', body: update });
}

/**
 * List the caller's long-term memories.
 *
 * @param signal Optional cancellation signal.
 * @returns Memories, newest or most salient first depending on the API.
 */
export async function listMemories(signal?: AbortSignal): Promise<LongTermMemory[]> {
  const raw = await apiRequest<unknown>('/memory/items', {
    query: { limit: apiEnv.pageSize },
    signal,
  });
  return unwrapList<LongTermMemory>(raw);
}

/**
 * Delete one long-term memory.
 *
 * @param memoryId Memory id.
 */
export function deleteMemory(memoryId: string): Promise<void> {
  return apiRequest<void>(`/memory/items/${encodeURIComponent(memoryId)}`, {
    method: 'DELETE',
  });
}

/**
 * Toggle long-term memory consent.
 *
 * Turning consent off soft-deletes existing memories server-side and skips stage 13
 * write-back entirely.
 *
 * @param consent Whether long-term memory may be stored.
 * @returns The updated profile.
 */
export function setMemoryConsent(consent: boolean): Promise<UserProfile> {
  return apiRequest<UserProfile>('/memory/consent', {
    method: 'PUT',
    body: { memory_consent: consent },
  });
}

/**
 * Submit thumbs-up/down feedback for an answer.
 *
 * @param feedback Rating plus optional comment.
 */
export function sendFeedback(feedback: FeedbackRequest): Promise<void> {
  return apiRequest<void>('/feedback', { method: 'POST', body: feedback });
}

/**
 * List evaluation runs.
 *
 * @param signal Optional cancellation signal.
 * @returns Runs, newest first.
 */
export async function listEvalRuns(signal?: AbortSignal): Promise<EvalRun[]> {
  const raw = await apiRequest<unknown>('/eval/runs', {
    query: { limit: apiEnv.pageSize },
    signal,
  });
  return unwrapList<EvalRun>(raw);
}

/**
 * Fetch one evaluation run with its per-item results.
 *
 * @param runId Run id.
 * @param signal Optional cancellation signal.
 * @returns The run detail.
 */
export function getEvalRun(runId: string, signal?: AbortSignal): Promise<EvalRun> {
  return apiRequest<EvalRun>(`/eval/runs/${encodeURIComponent(runId)}`, { signal });
}

/**
 * Start an evaluation run against the golden set.
 *
 * @param request Optional golden-set path and sample size.
 * @returns The created (possibly still running) run.
 */
export function startEvalRun(request: EvalRunRequest = {}): Promise<EvalRun> {
  return apiRequest<EvalRun>('/eval/runs', {
    method: 'POST',
    body: request,
    timeoutMs: 0,
  });
}

/**
 * List ingest runs (admin).
 *
 * @param query Optional `source_id` / `limit` filters.
 * @param signal Optional cancellation signal.
 * @returns Ingest run summaries, newest first.
 */
export async function listIngestRuns(
  query?: QueryParams,
  signal?: AbortSignal,
): Promise<IngestRunSummary[]> {
  const raw = await apiRequest<unknown>('/admin/ingest/runs', {
    query: { limit: apiEnv.pageSize, ...query },
    signal,
  });
  return unwrapList<IngestRunSummary>(raw);
}

/**
 * Trigger an ingestion run (admin).
 *
 * @param request Source selection and force/full-scan flags.
 * @returns The run summary the trigger produced or queued.
 */
export function triggerIngest(
  request: IngestTriggerRequest = {},
): Promise<IngestRunSummary> {
  return apiRequest<IngestRunSummary>('/admin/ingest/trigger', {
    method: 'POST',
    body: request,
    timeoutMs: 0,
  });
}

/**
 * List configured ingestion sources (admin).
 *
 * @param signal Optional cancellation signal.
 * @returns Source configurations for the caller's tenant.
 */
export async function listSources(
  signal?: AbortSignal,
): Promise<SourceConfigSummary[]> {
  const raw = await apiRequest<unknown>('/admin/sources', { signal });
  return unwrapList<SourceConfigSummary>(raw);
}

/**
 * Read the ingestion schedule and working-hours guard (admin).
 *
 * @param signal Optional cancellation signal.
 * @returns Cron, timezone and whether a scheduled run may start now.
 */
export function getSchedule(signal?: AbortSignal): Promise<ScheduleInfo> {
  return apiRequest<ScheduleInfo>('/admin/schedule', { signal });
}
