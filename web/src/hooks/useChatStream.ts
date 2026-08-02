/**
 * Drive `POST /api/v1/chat` and fold its SSE events into the chat store.
 *
 * Every event name in the contract is handled; unknown names are ignored so a newer
 * API can add events without breaking this build.
 *
 * Reconnect policy: a transport failure is retried (with jittered backoff) **only**
 * while no token has been received. Once the answer has started streaming a silent
 * retry would duplicate text, so the turn is failed and `retry()` is offered instead —
 * an explicit, user-initiated resend rather than a hidden one.
 */

import { useCallback, useEffect, useRef } from 'react';

import { apiEnv, ApiError } from '../api/client';
import { isAbortError, streamChat } from '../api/sse';
import type { ChatRequest, ChatStreamEvent } from '../api/types';
import { useChatStore } from '../store/chat';
import { filterForRequest, useSettingsStore } from '../store/settings';

/** What {@link useChatStream} returns. */
export interface ChatStreamController {
  /** True while a turn is in flight. */
  streaming: boolean;
  /** Last stream error, or null. */
  error: string | null;
  /** Send a prompt and stream the answer. */
  send: (prompt: string) => Promise<void>;
  /** Abort the in-flight turn. */
  stop: () => void;
  /** Resend the last prompt after a failure. */
  retry: () => Promise<void>;
}

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

/**
 * Stream a chat turn and keep the store in step with the pipeline.
 *
 * @returns The stream controller.
 */
export function useChatStream(): ChatStreamController {
  const abortRef = useRef<AbortController | null>(null);
  const streaming = useChatStore((state) => state.streaming);
  const error = useChatStore((state) => state.streamError);

  useEffect(
    () => () => {
      abortRef.current?.abort();
    },
    [],
  );

  const run = useCallback(async (prompt: string): Promise<void> => {
    const chat = useChatStore.getState();
    const settings = useSettingsStore.getState();
    const request: ChatRequest = {
      message: prompt,
      session_id: chat.activeSessionId,
      filters: filterForRequest(settings.filters),
      allow_tools: settings.allowTools,
      stream: true,
    };

    let attempt = 0;
    for (;;) {
      const controller = new AbortController();
      abortRef.current = controller;
      // Held on an object so the event handler can mutate it without the compiler
      // narrowing the outer `let` bindings to their initial values.
      const flags = { tokens: false, done: false, error: '' };

      const handle = (event: ChatStreamEvent): void => {
        const store = useChatStore.getState();
        switch (event.type) {
          case 'session':
            if (event.data.session_id) {
              store.setSessionId(event.data.session_id, event.data.title);
              request.session_id = event.data.session_id;
            }
            break;
          case 'retrieval':
            store.setRetrieval(event.data);
            break;
          case 'thinking':
            store.appendThinking(event.data.text ?? event.data.delta ?? '');
            break;
          case 'tool_call':
            store.startTool({
              tool_call_id: event.data.tool_call_id,
              tool_name: event.data.tool_name,
              kind: event.data.kind ?? 'rest',
              arguments: event.data.arguments ?? {},
              latency_ms: null,
              result_summary: null,
              error_message: null,
              http_status: null,
            });
            break;
          case 'tool_result':
            store.finishTool(event.data.tool_call_id, {
              latency_ms: event.data.latency_ms ?? null,
              result_summary: event.data.result_summary ?? null,
              error_message: event.data.error_message ?? null,
              http_status: event.data.http_status ?? null,
              isError: event.data.is_error === true,
              ...(event.data.kind ? { kind: event.data.kind } : {}),
              ...(event.data.tool_name ? { tool_name: event.data.tool_name } : {}),
            });
            break;
          case 'token':
            if (event.data.text) {
              flags.tokens = true;
              store.appendToken(event.data.text);
            }
            break;
          case 'citations':
            store.setCitations(event.data);
            break;
          case 'context_stats':
            store.setContextStats(event.data);
            break;
          case 'guardrail':
            store.addGuardrail(event.data);
            break;
          case 'usage':
            store.setUsage(event.data);
            break;
          case 'done':
            flags.done = true;
            store.finishTurn({
              messageId: event.data.message_id,
              refused: event.data.refused,
              traceId: event.data.trace_id ?? null,
            });
            break;
          case 'error':
            flags.error =
              event.data.detail ??
              event.data.message ??
              event.data.error ??
              'The server reported an error.';
            break;
          case 'unknown':
            // Forward compatibility: the contract requires unknown events be ignored.
            break;
          default:
            break;
        }
      };

      try {
        await streamChat({
          request,
          onEvent: handle,
          signal: controller.signal,
          idleTimeoutMs: apiEnv.streamIdleTimeoutMs,
        });
        if (flags.error) {
          useChatStore.getState().failTurn(flags.error);
        } else if (!flags.done) {
          // Stream closed without `done`: keep whatever arrived, stop the spinner.
          useChatStore.getState().finishTurn({});
        }
        return;
      } catch (cause) {
        if (isAbortError(cause) || controller.signal.aborted) {
          useChatStore.getState().finishTurn({});
          return;
        }
        const status = cause instanceof ApiError ? cause.status : 0;
        const retryable = !flags.tokens && (status === 0 || status >= 500);
        if (retryable && attempt < apiEnv.streamMaxRetries) {
          attempt += 1;
          const backoff =
            apiEnv.streamRetryBaseMs * 2 ** (attempt - 1) + Math.random() * 200;
          useChatStore.getState().clearTurnSignals();
          await sleep(backoff);
          continue;
        }
        const message =
          cause instanceof ApiError
            ? cause.message
            : cause instanceof Error
              ? cause.message
              : 'The chat stream failed.';
        useChatStore.getState().failTurn(message);
        return;
      } finally {
        if (abortRef.current === controller) abortRef.current = null;
      }
    }
  }, []);

  const send = useCallback(
    async (prompt: string): Promise<void> => {
      const trimmed = prompt.trim();
      if (!trimmed) return;
      if (useChatStore.getState().streaming) return;
      useChatStore.getState().beginTurn(trimmed);
      await run(trimmed);
    },
    [run],
  );

  const retry = useCallback(async (): Promise<void> => {
    const state = useChatStore.getState();
    if (state.streaming || !state.lastPrompt) return;
    // Drop the failed assistant placeholder and its user turn, then resend.
    const trimmed = state.lastPrompt;
    const kept = state.messages.slice(0, Math.max(0, state.messages.length - 2));
    state.setMessages(kept);
    state.clearTurnSignals();
    state.beginTurn(trimmed);
    await run(trimmed);
  }, [run]);

  const stop = useCallback((): void => {
    abortRef.current?.abort();
  }, []);

  return { streaming, error, send, stop, retry };
}
