/**
 * Chat state: sessions, the visible transcript, and everything the RAG pipeline
 * reported about the current turn.
 *
 * The pipeline signals are kept as first-class state rather than being folded into
 * message text, because requirement #7 asks for the pipeline to be *legible*: the
 * retrieval funnel, the context budget, the tool trace and the guardrail decisions
 * each render from their own slice.
 */

import { create } from 'zustand';

import type {
  Citation,
  ContextStats,
  GuardrailEvent,
  Message,
  RetrievalResult,
  Role,
  SessionSummary,
  ToolCall,
  UsageEventPayload,
} from '../api/types';

/** One tool invocation as the UI sees it, spanning `tool_call` and `tool_result`. */
export interface ToolTraceEntry {
  tool_call_id: string;
  tool_name: string;
  kind: string;
  arguments: Record<string, unknown>;
  status: 'running' | 'ok' | 'error';
  latency_ms: number | null;
  result_summary: string | null;
  error_message: string | null;
  http_status: number | null;
  started_at: string;
  finished_at: string | null;
}

/** A transcript entry: a persisted `Message` plus streaming-only fields. */
export interface ChatUiMessage {
  message_id: string;
  session_id: string;
  role: Role;
  content: string;
  citations: Citation[];
  tool_calls: ToolCall[];
  token_count: number;
  created_at: string;
  suppressed: boolean;
  pinned: boolean;
  /** True while tokens are still arriving for this message. */
  streaming: boolean;
  /** Set when the turn failed; rendered inline instead of as a toast. */
  error: string | null;
  /** Accumulated `thinking` events for this turn. */
  thinking: string;
  /** True when the answer was an explicit out-of-corpus refusal. */
  refused: boolean;
}

/** Placeholder id used for the assistant turn until `done` supplies the real one. */
export const PENDING_MESSAGE_ID = 'pending-assistant';

function nowIso(): string {
  return new Date().toISOString();
}

function localId(prefix: string): string {
  const random = Math.random().toString(36).slice(2, 10);
  return `${prefix}-${Date.now().toString(36)}-${random}`;
}

/**
 * Convert an API `Message` into a transcript entry.
 *
 * @param message The persisted message.
 * @returns The UI message.
 */
export function toUiMessage(message: Message): ChatUiMessage {
  return {
    message_id: message.message_id,
    session_id: message.session_id,
    role: message.role,
    content: message.content,
    citations: message.citations ?? [],
    tool_calls: message.tool_calls ?? [],
    token_count: message.token_count ?? 0,
    created_at: message.created_at,
    suppressed: message.suppressed ?? false,
    pinned: message.pinned ?? false,
    streaming: false,
    error: null,
    thinking: '',
    refused: false,
  };
}

/** Shape of the chat store. */
export interface ChatState {
  sessions: SessionSummary[];
  sessionsLoading: boolean;
  activeSessionId: string | null;
  messages: ChatUiMessage[];
  messagesLoading: boolean;

  streaming: boolean;
  thinking: string;
  retrieval: RetrievalResult | null;
  contextStats: ContextStats | null;
  guardrails: GuardrailEvent[];
  tools: ToolTraceEntry[];
  usage: UsageEventPayload | null;
  streamError: string | null;
  lastPrompt: string | null;
  traceId: string | null;

  /** Marker (`"[3]"`) whose source is open in the drawer. */
  selectedMarker: string | null;
  drawerOpen: boolean;

  setSessions: (sessions: SessionSummary[]) => void;
  setSessionsLoading: (loading: boolean) => void;
  removeSession: (sessionId: string) => void;
  patchSession: (sessionId: string, patch: Partial<SessionSummary>) => void;
  selectSession: (sessionId: string | null) => void;
  setMessages: (messages: ChatUiMessage[]) => void;
  setMessagesLoading: (loading: boolean) => void;

  beginTurn: (prompt: string) => void;
  appendToken: (text: string) => void;
  appendThinking: (text: string) => void;
  setRetrieval: (retrieval: RetrievalResult) => void;
  setCitations: (citations: Citation[]) => void;
  setContextStats: (stats: ContextStats) => void;
  addGuardrail: (event: GuardrailEvent) => void;
  startTool: (entry: Omit<ToolTraceEntry, 'status' | 'started_at' | 'finished_at'>) => void;
  finishTool: (
    toolCallId: string,
    patch: Partial<Pick<
      ToolTraceEntry,
      'latency_ms' | 'result_summary' | 'error_message' | 'http_status' | 'kind' | 'tool_name'
    >> & { isError?: boolean },
  ) => void;
  setUsage: (usage: UsageEventPayload) => void;
  setSessionId: (sessionId: string, title?: string) => void;
  finishTurn: (options: { messageId?: string; refused?: boolean; traceId?: string | null }) => void;
  failTurn: (message: string) => void;
  clearTurnSignals: () => void;
  selectMarker: (marker: string | null) => void;
  closeDrawer: () => void;
}

/**
 * Chat store: sessions, transcript and per-turn pipeline signals.
 *
 * Deliberately not persisted — history is authoritative on the server, and caching a
 * transcript locally would keep answer text in `localStorage` after a user's clearance
 * changed.
 */
export const useChatStore = create<ChatState>()((set, get) => ({
  sessions: [],
  sessionsLoading: false,
  activeSessionId: null,
  messages: [],
  messagesLoading: false,

  streaming: false,
  thinking: '',
  retrieval: null,
  contextStats: null,
  guardrails: [],
  tools: [],
  usage: null,
  streamError: null,
  lastPrompt: null,
  traceId: null,

  selectedMarker: null,
  drawerOpen: false,

  setSessions: (sessions) => set({ sessions }),
  setSessionsLoading: (sessionsLoading) => set({ sessionsLoading }),
  removeSession: (sessionId) =>
    set((state) => ({
      sessions: state.sessions.filter((item) => item.session_id !== sessionId),
      activeSessionId:
        state.activeSessionId === sessionId ? null : state.activeSessionId,
      messages: state.activeSessionId === sessionId ? [] : state.messages,
    })),
  patchSession: (sessionId, patch) =>
    set((state) => ({
      sessions: state.sessions.map((item) =>
        item.session_id === sessionId ? { ...item, ...patch } : item,
      ),
    })),
  selectSession: (activeSessionId) =>
    set({
      activeSessionId,
      messages: [],
      retrieval: null,
      contextStats: null,
      guardrails: [],
      tools: [],
      usage: null,
      streamError: null,
      thinking: '',
      selectedMarker: null,
      drawerOpen: false,
    }),
  setMessages: (messages) => set({ messages }),
  setMessagesLoading: (messagesLoading) => set({ messagesLoading }),

  beginTurn: (prompt) => {
    const sessionId = get().activeSessionId ?? '';
    const user: ChatUiMessage = {
      message_id: localId('local-user'),
      session_id: sessionId,
      role: 'user',
      content: prompt,
      citations: [],
      tool_calls: [],
      token_count: 0,
      created_at: nowIso(),
      suppressed: false,
      pinned: false,
      streaming: false,
      error: null,
      thinking: '',
      refused: false,
    };
    const assistant: ChatUiMessage = {
      ...user,
      message_id: PENDING_MESSAGE_ID,
      role: 'assistant',
      content: '',
      streaming: true,
    };
    set((state) => ({
      messages: [...state.messages, user, assistant],
      streaming: true,
      thinking: '',
      retrieval: null,
      contextStats: null,
      guardrails: [],
      tools: [],
      usage: null,
      streamError: null,
      lastPrompt: prompt,
      traceId: null,
      selectedMarker: null,
      drawerOpen: false,
    }));
  },

  appendToken: (text) => {
    if (!text) return;
    set((state) => ({
      messages: state.messages.map((message) =>
        message.message_id === PENDING_MESSAGE_ID
          ? { ...message, content: message.content + text }
          : message,
      ),
    }));
  },

  appendThinking: (text) => {
    if (!text) return;
    set((state) => ({ thinking: state.thinking + text }));
  },

  setRetrieval: (retrieval) => set({ retrieval }),

  setCitations: (citations) =>
    set((state) => ({
      messages: state.messages.map((message) =>
        message.message_id === PENDING_MESSAGE_ID ? { ...message, citations } : message,
      ),
    })),

  setContextStats: (contextStats) => set({ contextStats }),

  addGuardrail: (event) =>
    set((state) => ({ guardrails: [...state.guardrails, event] })),

  startTool: (entry) =>
    set((state) => ({
      tools: [
        ...state.tools.filter((item) => item.tool_call_id !== entry.tool_call_id),
        {
          ...entry,
          status: 'running',
          started_at: nowIso(),
          finished_at: null,
        },
      ],
    })),

  finishTool: (toolCallId, patch) =>
    set((state) => ({
      tools: state.tools.map((item) => {
        if (item.tool_call_id !== toolCallId) return item;
        const { isError, ...rest } = patch;
        return {
          ...item,
          ...rest,
          status: isError ? 'error' : 'ok',
          finished_at: nowIso(),
        };
      }),
    })),

  setUsage: (usage) => set({ usage }),

  setSessionId: (sessionId, title) =>
    set((state) => {
      const known = state.sessions.some((item) => item.session_id === sessionId);
      return {
        activeSessionId: sessionId,
        messages: state.messages.map((message) =>
          message.session_id ? message : { ...message, session_id: sessionId },
        ),
        sessions: known
          ? state.sessions
          : [
              {
                session_id: sessionId,
                title: title ?? 'New conversation',
                last_message_at: nowIso(),
                message_count: 0,
              },
              ...state.sessions,
            ],
      };
    }),

  finishTurn: ({ messageId, refused, traceId }) =>
    set((state) => ({
      streaming: false,
      traceId: traceId ?? state.traceId,
      messages: state.messages.map((message) =>
        message.message_id === PENDING_MESSAGE_ID
          ? {
              ...message,
              message_id: messageId ?? localId('local-assistant'),
              streaming: false,
              refused: refused === true,
            }
          : message,
      ),
    })),

  failTurn: (message) =>
    set((state) => ({
      streaming: false,
      streamError: message,
      messages: state.messages.map((entry) =>
        entry.message_id === PENDING_MESSAGE_ID
          ? { ...entry, streaming: false, error: message }
          : entry,
      ),
    })),

  clearTurnSignals: () =>
    set({
      streamError: null,
      thinking: '',
      guardrails: [],
      tools: [],
    }),

  selectMarker: (marker) => set({ selectedMarker: marker, drawerOpen: marker !== null }),
  closeDrawer: () => set({ drawerOpen: false }),
}));

/**
 * Find the citation a marker points at, searching newest assistant turn first.
 *
 * @param messages The transcript.
 * @param marker The marker text, e.g. `"[2]"`.
 * @returns The citation, or null when the marker is unresolved.
 */
export function findCitation(
  messages: ChatUiMessage[],
  marker: string | null,
): Citation | null {
  if (!marker) return null;
  for (let index = messages.length - 1; index >= 0; index -= 1) {
    const message = messages[index];
    const hit = message.citations.find((citation) => citation.marker === marker);
    if (hit) return hit;
  }
  return null;
}

/**
 * The citations of the most recent assistant turn.
 *
 * @param messages The transcript.
 * @returns Citations, or an empty list.
 */
export function latestCitations(messages: ChatUiMessage[]): Citation[] {
  for (let index = messages.length - 1; index >= 0; index -= 1) {
    const message = messages[index];
    if (message.role === 'assistant' && message.citations.length > 0) {
      return message.citations;
    }
  }
  return [];
}
