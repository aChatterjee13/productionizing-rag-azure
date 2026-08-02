/**
 * Server-sent-event reader for `POST /api/v1/chat`.
 *
 * `EventSource` is deliberately not used: it can only issue GET requests and cannot
 * attach an `Authorization` header, so the stream is read from `fetch` +
 * `ReadableStream` instead. That also means the same bearer-token and 401-retry logic
 * in `./client.ts` covers streaming and non-streaming calls alike.
 *
 * The parser implements the SSE wire format from the HTML spec: `field: value` lines,
 * `data` accumulation across lines, comment lines starting with `:`, and dispatch on
 * a blank line. Unknown event names are surfaced as `{ type: 'unknown' }` so the
 * caller can ignore them without the stream breaking — required by the contract.
 */

import { ApiError, authorizedFetch, resolveUrl } from './client';
import type {
  ChatRequest,
  ChatStreamEvent,
  Citation,
  ContextStats,
  DoneEventPayload,
  ErrorEventPayload,
  GuardrailEvent,
  RetrievalResult,
  SessionEventPayload,
  ThinkingEventPayload,
  TokenEventPayload,
  ToolCallEventPayload,
  ToolResultEventPayload,
  UsageEventPayload,
} from './types';

/** One decoded SSE frame. */
export interface SSEFrame {
  /** The `event:` field, defaulting to `message` when absent. */
  event: string;
  /** Concatenated `data:` lines, newline-joined. */
  data: string;
  /** The `id:` field, when present. */
  id?: string;
  /** The `retry:` field in milliseconds, when present. */
  retry?: number;
}

/** Incremental SSE parser. Feed it decoded text; it emits complete frames. */
export interface SSEParser {
  /**
   * Consume a chunk of decoded text.
   *
   * @param chunk Text decoded from the response body.
   */
  push(chunk: string): void;
  /** Flush a trailing frame that was not terminated by a blank line. */
  flush(): void;
}

/**
 * Create an incremental SSE parser.
 *
 * @param onFrame Called once per complete frame, in arrival order.
 * @returns A parser with `push` and `flush`.
 */
export function createSSEParser(onFrame: (frame: SSEFrame) => void): SSEParser {
  let buffer = '';
  let eventName = '';
  let dataLines: string[] = [];
  let lastId: string | undefined;
  let retry: number | undefined;
  let started = false;

  const reset = (): void => {
    eventName = '';
    dataLines = [];
    retry = undefined;
    started = false;
  };

  const dispatch = (): void => {
    if (!started) {
      reset();
      return;
    }
    const frame: SSEFrame = {
      event: eventName || 'message',
      data: dataLines.join('\n'),
    };
    if (lastId !== undefined) frame.id = lastId;
    if (retry !== undefined) frame.retry = retry;
    reset();
    onFrame(frame);
  };

  const handleLine = (line: string): void => {
    if (line === '') {
      dispatch();
      return;
    }
    if (line.startsWith(':')) return; // Comment / keep-alive.
    const colon = line.indexOf(':');
    const field = colon === -1 ? line : line.slice(0, colon);
    let value = colon === -1 ? '' : line.slice(colon + 1);
    if (value.startsWith(' ')) value = value.slice(1);
    started = true;
    switch (field) {
      case 'event':
        eventName = value;
        break;
      case 'data':
        dataLines.push(value);
        break;
      case 'id':
        if (!value.includes('\0')) lastId = value;
        break;
      case 'retry': {
        const parsed = Number(value);
        if (Number.isInteger(parsed) && parsed >= 0) retry = parsed;
        break;
      }
      default:
        break; // Ignore unknown fields, per spec.
    }
  };

  return {
    push(chunk: string): void {
      buffer += chunk;
      // Normalise CRLF / CR line endings, then consume whole lines only.
      buffer = buffer.replace(/\r\n/g, '\n').replace(/\r/g, '\n');
      let index = buffer.indexOf('\n');
      while (index !== -1) {
        handleLine(buffer.slice(0, index));
        buffer = buffer.slice(index + 1);
        index = buffer.indexOf('\n');
      }
    },
    flush(): void {
      if (buffer) {
        handleLine(buffer);
        buffer = '';
      }
      dispatch();
    },
  };
}

function parseJson(frame: SSEFrame): unknown {
  if (!frame.data) return {};
  try {
    return JSON.parse(frame.data) as unknown;
  } catch {
    // A non-JSON payload is still useful for `token` and `thinking`.
    return frame.data;
  }
}

function asRecord(value: unknown): Record<string, unknown> {
  return value && typeof value === 'object' && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : {};
}

function asText(value: unknown): string {
  if (typeof value === 'string') return value;
  const record = asRecord(value);
  const text = record.text ?? record.delta ?? record.content;
  return typeof text === 'string' ? text : '';
}

/**
 * Map a decoded frame onto the typed {@link ChatStreamEvent} union.
 *
 * Event names not in the contract become `{ type: 'unknown' }`; the caller ignores
 * them, which is what lets the server add events without breaking older builds.
 *
 * @param frame The decoded SSE frame.
 * @returns The typed event.
 */
export function toChatStreamEvent(frame: SSEFrame): ChatStreamEvent {
  const payload = parseJson(frame);
  switch (frame.event) {
    case 'session':
      return { type: 'session', data: asRecord(payload) as unknown as SessionEventPayload };
    case 'retrieval':
      return { type: 'retrieval', data: asRecord(payload) as unknown as RetrievalResult };
    case 'thinking':
      return {
        type: 'thinking',
        data: { text: asText(payload) } satisfies ThinkingEventPayload,
      };
    case 'tool_call':
      return {
        type: 'tool_call',
        data: asRecord(payload) as unknown as ToolCallEventPayload,
      };
    case 'tool_result':
      return {
        type: 'tool_result',
        data: asRecord(payload) as unknown as ToolResultEventPayload,
      };
    case 'token':
      return { type: 'token', data: { text: asText(payload) } satisfies TokenEventPayload };
    case 'citations': {
      const citations = Array.isArray(payload)
        ? (payload as Citation[])
        : ((asRecord(payload).citations as Citation[] | undefined) ?? []);
      return { type: 'citations', data: citations };
    }
    case 'context_stats':
      return {
        type: 'context_stats',
        data: asRecord(payload) as unknown as ContextStats,
      };
    case 'guardrail':
      return { type: 'guardrail', data: asRecord(payload) as unknown as GuardrailEvent };
    case 'usage':
      return { type: 'usage', data: asRecord(payload) as unknown as UsageEventPayload };
    case 'done':
      return { type: 'done', data: asRecord(payload) as unknown as DoneEventPayload };
    case 'error':
      return { type: 'error', data: asRecord(payload) as unknown as ErrorEventPayload };
    default:
      return { type: 'unknown', name: frame.event, data: payload };
  }
}

/** Options for {@link streamChat}. */
export interface StreamChatOptions {
  /** Chat request body; `stream` is forced to true. */
  request: ChatRequest;
  /** Called for every decoded event, including unknown ones. */
  onEvent: (event: ChatStreamEvent) => void;
  /** Abort signal wired to the composer's stop button. */
  signal?: AbortSignal;
  /** Milliseconds of silence tolerated before the stream is failed. */
  idleTimeoutMs?: number;
}

/**
 * Open the chat SSE stream and pump events until the server closes it.
 *
 * @param options Request body, event sink, cancellation and idle timeout.
 * @throws ApiError On a non-2xx response, a missing body, or an idle timeout.
 * @throws DOMException When the caller aborts (`AbortError`), which is not an error.
 */
export async function streamChat(options: StreamChatOptions): Promise<void> {
  const { request, onEvent, signal, idleTimeoutMs = 0 } = options;

  const controller = new AbortController();
  const abortWithCaller = (): void => controller.abort(signal?.reason);
  if (signal) {
    if (signal.aborted) abortWithCaller();
    else signal.addEventListener('abort', abortWithCaller, { once: true });
  }

  let idleTimer: ReturnType<typeof setTimeout> | undefined;
  let idleTimedOut = false;
  const armIdleTimer = (): void => {
    if (idleTimeoutMs <= 0) return;
    if (idleTimer !== undefined) clearTimeout(idleTimer);
    idleTimer = setTimeout(() => {
      idleTimedOut = true;
      controller.abort();
    }, idleTimeoutMs);
  };

  try {
    const response = await authorizedFetch(resolveUrl('/chat'), {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        Accept: 'text/event-stream',
        'Cache-Control': 'no-store',
      },
      body: JSON.stringify({ ...request, stream: true }),
      signal: controller.signal,
    });

    if (!response.ok) {
      let detail = '';
      try {
        detail = await response.text();
      } catch {
        detail = '';
      }
      throw new ApiError(
        detail.slice(0, 500) || `Chat stream failed with ${response.status}`,
        response.status,
        detail,
      );
    }
    if (!response.body) {
      throw new ApiError('Chat stream returned no body', response.status);
    }

    const parser = createSSEParser((frame) => onEvent(toChatStreamEvent(frame)));
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    armIdleTimer();

    for (;;) {
      const result = await reader.read();
      if (result.done) break;
      armIdleTimer();
      parser.push(decoder.decode(result.value, { stream: true }));
    }
    parser.push(decoder.decode());
    parser.flush();
  } catch (error) {
    if (idleTimedOut) {
      throw new ApiError(
        `Chat stream idle for more than ${Math.round(idleTimeoutMs / 1000)}s`,
        0,
      );
    }
    throw error;
  } finally {
    if (idleTimer !== undefined) clearTimeout(idleTimer);
    if (signal) signal.removeEventListener('abort', abortWithCaller);
  }
}

/**
 * Whether an error is a caller-initiated abort rather than a real failure.
 *
 * @param error The thrown value.
 * @returns True for `AbortError`.
 */
export function isAbortError(error: unknown): boolean {
  return (
    (error instanceof DOMException && error.name === 'AbortError') ||
    (error instanceof Error && error.name === 'AbortError')
  );
}
