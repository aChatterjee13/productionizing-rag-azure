/**
 * Session list: switch, delete, and force context compaction.
 *
 * Compaction is exposed to the user because requirement #3/#5 makes it observable:
 * pressing it folds the oldest non-pinned turns into the rolling summary and the
 * context meter immediately shows the reclaimed budget.
 */

import { Loader2, MessageSquarePlus, Scissors, Trash2 } from 'lucide-react';
import { useCallback, useEffect, useState, type ReactNode } from 'react';

import {
  ApiError,
  compactSession,
  deleteSession,
  listMessages,
  listSessions,
} from '../api/client';
import { toUiMessage, useChatStore } from '../store/chat';
import { useSettingsStore } from '../store/settings';

function formatWhen(value: string | null | undefined): string {
  if (!value) return '';
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return '';
  const today = new Date();
  const sameDay = date.toDateString() === today.toDateString();
  return sameDay
    ? date.toLocaleTimeString(undefined, { hour: '2-digit', minute: '2-digit' })
    : date.toLocaleDateString(undefined, { month: 'short', day: 'numeric' });
}

/**
 * Render the session sidebar.
 *
 * @returns The sidebar.
 */
export default function SessionSidebar(): ReactNode {
  const sessions = useChatStore((state) => state.sessions);
  const loading = useChatStore((state) => state.sessionsLoading);
  const activeSessionId = useChatStore((state) => state.activeSessionId);
  const streaming = useChatStore((state) => state.streaming);
  const setView = useSettingsStore((state) => state.setView);

  const [busyId, setBusyId] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    const store = useChatStore.getState();
    store.setSessionsLoading(true);
    try {
      store.setSessions(await listSessions());
      setError(null);
    } catch (cause) {
      setError(cause instanceof ApiError ? cause.message : 'Could not load sessions.');
    } finally {
      useChatStore.getState().setSessionsLoading(false);
    }
  }, []);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  const openSession = useCallback(
    async (sessionId: string) => {
      const store = useChatStore.getState();
      if (store.streaming) return;
      store.selectSession(sessionId);
      setView('chat');
      store.setMessagesLoading(true);
      try {
        const messages = await listMessages(sessionId);
        useChatStore.getState().setMessages(messages.map(toUiMessage));
        setError(null);
      } catch (cause) {
        setError(cause instanceof ApiError ? cause.message : 'Could not load messages.');
      } finally {
        useChatStore.getState().setMessagesLoading(false);
      }
    },
    [setView],
  );

  const onDelete = useCallback(async (sessionId: string) => {
    const confirmed = window.confirm(
      'Delete this conversation and its messages? This cannot be undone.',
    );
    if (!confirmed) return;
    setBusyId(sessionId);
    try {
      await deleteSession(sessionId);
      useChatStore.getState().removeSession(sessionId);
      setError(null);
    } catch (cause) {
      setError(cause instanceof ApiError ? cause.message : 'Could not delete session.');
    } finally {
      setBusyId(null);
    }
  }, []);

  const onCompact = useCallback(async (sessionId: string) => {
    setBusyId(sessionId);
    try {
      const result = await compactSession(sessionId);
      const store = useChatStore.getState();
      if (result.context_stats) store.setContextStats(result.context_stats);
      store.patchSession(sessionId, {
        compaction_events: result.compaction_events,
        summary_tokens: result.summary_tokens,
      });
      const messages = await listMessages(sessionId);
      store.setMessages(messages.map(toUiMessage));
      setError(null);
    } catch (cause) {
      setError(
        cause instanceof ApiError ? cause.message : 'Could not compact this session.',
      );
    } finally {
      setBusyId(null);
    }
  }, []);

  return (
    <div className="flex h-full flex-col">
      <div className="flex items-center gap-2 border-b border-slate-200 p-3 dark:border-slate-800">
        <button
          type="button"
          className="btn btn-primary flex-1"
          disabled={streaming}
          onClick={() => {
            useChatStore.getState().selectSession(null);
            setView('chat');
          }}
        >
          <MessageSquarePlus aria-hidden="true" className="h-4 w-4" />
          New conversation
        </button>
      </div>

      {error ? (
        <p role="alert" className="px-3 py-2 text-xs text-rose-600 dark:text-rose-400">
          {error}
        </p>
      ) : null}

      <nav aria-label="Conversations" className="scroll-area min-h-0 flex-1 p-2">
        {loading && sessions.length === 0 ? (
          <p className="flex items-center gap-2 px-2 py-3 text-sm text-slate-500">
            <Loader2 aria-hidden="true" className="h-4 w-4 animate-spin" />
            Loading…
          </p>
        ) : null}

        {!loading && sessions.length === 0 ? (
          <p className="px-2 py-3 text-sm text-slate-500 dark:text-slate-400">
            No conversations yet. Ask something to start one.
          </p>
        ) : null}

        <ul className="space-y-1">
          {sessions.map((session) => {
            const selected = session.session_id === activeSessionId;
            const busy = busyId === session.session_id;
            return (
              <li key={session.session_id}>
                <div
                  className={`group rounded-lg border px-2 py-1.5 ${
                    selected
                      ? 'border-brand-300 bg-brand-50 dark:border-brand-800 dark:bg-brand-950/60'
                      : 'border-transparent hover:bg-slate-100 dark:hover:bg-slate-800'
                  }`}
                >
                  <button
                    type="button"
                    className="w-full text-left"
                    aria-current={selected ? 'true' : undefined}
                    disabled={streaming}
                    onClick={() => void openSession(session.session_id)}
                  >
                    <span className="block truncate text-sm font-medium">
                      {session.title || 'Untitled conversation'}
                    </span>
                    <span className="mt-0.5 flex items-center gap-2 text-[11px] text-slate-500 dark:text-slate-400">
                      <span>{formatWhen(session.last_message_at ?? session.updated_at)}</span>
                      {typeof session.message_count === 'number' ? (
                        <span>{session.message_count} turns</span>
                      ) : null}
                      {session.compaction_events ? (
                        <span title="Context compactions in this session">
                          {session.compaction_events}× compacted
                        </span>
                      ) : null}
                    </span>
                  </button>
                  <div className="mt-1 flex items-center gap-1 opacity-0 transition-opacity focus-within:opacity-100 group-hover:opacity-100">
                    <button
                      type="button"
                      className="btn btn-ghost btn-xs"
                      disabled={busy}
                      onClick={() => void onCompact(session.session_id)}
                      title="Force context compaction"
                      aria-label={`Force context compaction for ${session.title || 'this conversation'}`}
                    >
                      {busy ? (
                        <Loader2 aria-hidden="true" className="h-3 w-3 animate-spin" />
                      ) : (
                        <Scissors aria-hidden="true" className="h-3 w-3" />
                      )}
                      Compact
                    </button>
                    <button
                      type="button"
                      className="btn btn-ghost btn-xs text-rose-600 dark:text-rose-400"
                      disabled={busy}
                      onClick={() => void onDelete(session.session_id)}
                      aria-label={`Delete ${session.title || 'this conversation'}`}
                    >
                      <Trash2 aria-hidden="true" className="h-3 w-3" />
                      Delete
                    </button>
                  </div>
                </div>
              </li>
            );
          })}
        </ul>
      </nav>
    </div>
  );
}
