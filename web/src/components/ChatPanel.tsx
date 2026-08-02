/**
 * The chat surface.
 *
 * The transcript is on the left; the right rail exposes the pipeline — retrieval
 * funnel, context budget, tool trace — so an answer can always be traced back to the
 * chunks, the token budget and the tool calls that produced it.
 */

import { Eye, EyeOff, Gauge, RefreshCw, Wrench } from 'lucide-react';
import { useEffect, useRef, type ReactNode } from 'react';

import { useChatStream } from '../hooks/useChatStream';
import { findCitation, latestCitations, useChatStore } from '../store/chat';
import { useSettingsStore } from '../store/settings';
import CitationList from './CitationList';
import Composer from './Composer';
import ContextMeter from './ContextMeter';
import FilterBar from './FilterBar';
import GuardrailBanner from './GuardrailBanner';
import MessageBubble from './MessageBubble';
import RetrievalInspector from './RetrievalInspector';
import SourceDrawer from './SourceDrawer';
import ToolTrace from './ToolTrace';

function EmptyState(): ReactNode {
  return (
    <div className="mx-auto max-w-xl py-10 text-center">
      <h2 className="text-lg font-semibold">Ask the corpus</h2>
      <p className="mt-2 text-sm text-slate-600 dark:text-slate-400">
        Answers are grounded in documents you are cleared to read, and every claim
        carries a <span className="marker-chip">[n]</span> marker you can open to see
        the exact quoted span. If the corpus does not cover your question you will get
        an explicit refusal rather than a guess.
      </p>
    </div>
  );
}

function PanelToggle({
  label,
  active,
  onClick,
  icon: Icon,
}: {
  label: string;
  active: boolean;
  onClick: () => void;
  icon: typeof Gauge;
}): ReactNode {
  return (
    <button
      type="button"
      className={`btn btn-xs ${active ? '' : 'btn-ghost'}`}
      aria-pressed={active}
      onClick={onClick}
    >
      <Icon aria-hidden="true" className="h-3.5 w-3.5" />
      {label}
      {active ? (
        <Eye aria-hidden="true" className="h-3 w-3 opacity-60" />
      ) : (
        <EyeOff aria-hidden="true" className="h-3 w-3 opacity-60" />
      )}
    </button>
  );
}

/**
 * Render the chat view.
 *
 * @returns The chat surface.
 */
export default function ChatPanel(): ReactNode {
  const messages = useChatStore((state) => state.messages);
  const messagesLoading = useChatStore((state) => state.messagesLoading);
  const streaming = useChatStore((state) => state.streaming);
  const thinking = useChatStore((state) => state.thinking);
  const streamError = useChatStore((state) => state.streamError);
  const selectedMarker = useChatStore((state) => state.selectedMarker);
  const drawerOpen = useChatStore((state) => state.drawerOpen);
  const selectMarker = useChatStore((state) => state.selectMarker);
  const closeDrawer = useChatStore((state) => state.closeDrawer);

  const showRetrieval = useSettingsStore((state) => state.showRetrieval);
  const showContext = useSettingsStore((state) => state.showContext);
  const showTools = useSettingsStore((state) => state.showTools);
  const showFilters = useSettingsStore((state) => state.showFilters);
  const togglePanel = useSettingsStore((state) => state.togglePanel);

  const stream = useChatStream();
  const bottomRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ block: 'end', behavior: 'smooth' });
  }, [messages.length, streaming]);

  const citations = latestCitations(messages);
  const activeCitation = findCitation(messages, selectedMarker);

  return (
    <div className="flex h-full min-h-0 flex-col lg:flex-row">
      <section
        aria-label="Conversation"
        className="flex min-h-0 flex-1 flex-col border-b border-slate-200 dark:border-slate-800 lg:border-b-0 lg:border-r"
      >
        <div className="scroll-area min-h-0 flex-1 px-4 py-4">
          <GuardrailBanner />

          {messages.length === 0 && !messagesLoading ? <EmptyState /> : null}
          {messagesLoading ? (
            <p className="py-6 text-center text-sm text-slate-500">Loading history…</p>
          ) : null}

          <div
            aria-live="polite"
            aria-atomic="false"
            aria-busy={streaming}
            aria-relevant="additions text"
            className="space-y-4"
          >
            {messages.map((message) => (
              <MessageBubble
                key={message.message_id}
                message={message}
                selectedMarker={selectedMarker}
                onSelectMarker={selectMarker}
              />
            ))}
          </div>

          {streaming && thinking ? (
            <details className="mt-3 rounded-lg border border-slate-200 bg-slate-50 p-3 text-sm dark:border-slate-800 dark:bg-slate-900/60">
              <summary className="cursor-pointer text-xs font-semibold uppercase tracking-wide text-slate-500">
                Model reasoning
              </summary>
              <p className="mt-2 whitespace-pre-wrap text-slate-600 dark:text-slate-400">
                {thinking}
              </p>
            </details>
          ) : null}

          {streamError ? (
            <div
              role="alert"
              className="mt-3 flex flex-wrap items-center gap-3 rounded-lg border border-rose-200 bg-rose-50 p-3 text-sm text-rose-700 dark:border-rose-900 dark:bg-rose-950/50 dark:text-rose-300"
            >
              <span className="flex-1">{streamError}</span>
              <button
                type="button"
                className="btn btn-xs"
                onClick={() => void stream.retry()}
              >
                <RefreshCw aria-hidden="true" className="h-3 w-3" />
                Resend
              </button>
            </div>
          ) : null}

          {citations.length > 0 ? (
            <CitationList
              citations={citations}
              selectedMarker={selectedMarker}
              onSelect={selectMarker}
            />
          ) : null}

          <div ref={bottomRef} />
        </div>

        {showFilters ? <FilterBar /> : null}
        <Composer stream={stream} />
      </section>

      <aside
        aria-label="Pipeline inspector"
        className="flex w-full shrink-0 flex-col gap-3 overflow-y-auto p-3 lg:w-[26rem]"
      >
        <div className="flex flex-wrap items-center gap-1">
          <PanelToggle
            label="Retrieval"
            icon={Eye}
            active={showRetrieval}
            onClick={() => togglePanel('showRetrieval')}
          />
          <PanelToggle
            label="Context"
            icon={Gauge}
            active={showContext}
            onClick={() => togglePanel('showContext')}
          />
          <PanelToggle
            label="Tools"
            icon={Wrench}
            active={showTools}
            onClick={() => togglePanel('showTools')}
          />
        </div>
        {showContext ? <ContextMeter /> : null}
        {showRetrieval ? <RetrievalInspector onSelectMarker={selectMarker} /> : null}
        {showTools ? <ToolTrace /> : null}
      </aside>

      <SourceDrawer
        open={drawerOpen}
        citation={activeCitation}
        onClose={closeDrawer}
      />
    </div>
  );
}
