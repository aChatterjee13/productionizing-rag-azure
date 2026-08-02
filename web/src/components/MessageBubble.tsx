/**
 * One transcript turn.
 *
 * Assistant text streams in token by token and inline `[n]` markers are rendered as
 * clickable chips: pressing one opens that source in the drawer, which is what makes
 * "where did this sentence come from?" a one-click question. Fenced code blocks are
 * preserved verbatim. Nothing is ever injected as HTML.
 */

import {
  AlertTriangle,
  Check,
  Copy,
  EyeOff,
  Pin,
  ThumbsDown,
  ThumbsUp,
  User,
} from 'lucide-react';
import { useCallback, useState, type ReactNode } from 'react';

import { sendFeedback } from '../api/client';
import { PENDING_MESSAGE_ID, type ChatUiMessage } from '../store/chat';

const MARKER_PATTERN = /\[(\d{1,3})\]/g;

interface TextSegment {
  kind: 'text';
  value: string;
}

interface MarkerSegment {
  kind: 'marker';
  value: string;
}

type Segment = TextSegment | MarkerSegment;

/**
 * Split prose into plain text and `[n]` citation markers.
 *
 * @param text The prose to split.
 * @returns Ordered segments.
 */
export function splitMarkers(text: string): Segment[] {
  const segments: Segment[] = [];
  let cursor = 0;
  MARKER_PATTERN.lastIndex = 0;
  let match = MARKER_PATTERN.exec(text);
  while (match !== null) {
    if (match.index > cursor) {
      segments.push({ kind: 'text', value: text.slice(cursor, match.index) });
    }
    segments.push({ kind: 'marker', value: match[0] });
    cursor = match.index + match[0].length;
    match = MARKER_PATTERN.exec(text);
  }
  if (cursor < text.length) {
    segments.push({ kind: 'text', value: text.slice(cursor) });
  }
  return segments;
}

interface Block {
  kind: 'prose' | 'code';
  value: string;
}

/**
 * Split an answer into prose and fenced code blocks.
 *
 * @param content The raw answer text.
 * @returns Ordered blocks.
 */
export function splitBlocks(content: string): Block[] {
  const parts = content.split('```');
  return parts
    .map((value, index) => ({
      kind: index % 2 === 1 ? ('code' as const) : ('prose' as const),
      value,
    }))
    .filter((block) => block.value.length > 0);
}

function stripLanguageHint(code: string): string {
  const newline = code.indexOf('\n');
  if (newline === -1) return code;
  const first = code.slice(0, newline).trim();
  return /^[a-z0-9+#.-]{1,20}$/i.test(first) ? code.slice(newline + 1) : code;
}

function FeedbackButtons({ message }: { message: ChatUiMessage }): ReactNode {
  const [sent, setSent] = useState<1 | -1 | null>(null);
  const [failed, setFailed] = useState(false);

  const submit = useCallback(
    async (rating: 1 | -1) => {
      setSent(rating);
      setFailed(false);
      try {
        await sendFeedback({
          session_id: message.session_id || null,
          message_id: message.message_id,
          rating,
        });
      } catch {
        setFailed(true);
        setSent(null);
      }
    },
    [message.message_id, message.session_id],
  );

  return (
    <>
      <button
        type="button"
        className={`btn btn-ghost btn-xs ${sent === 1 ? 'text-emerald-600' : ''}`}
        aria-label="This answer was helpful"
        aria-pressed={sent === 1}
        onClick={() => void submit(1)}
      >
        <ThumbsUp aria-hidden="true" className="h-3 w-3" />
      </button>
      <button
        type="button"
        className={`btn btn-ghost btn-xs ${sent === -1 ? 'text-rose-600' : ''}`}
        aria-label="This answer was not helpful"
        aria-pressed={sent === -1}
        onClick={() => void submit(-1)}
      >
        <ThumbsDown aria-hidden="true" className="h-3 w-3" />
      </button>
      {failed ? (
        <span className="text-[11px] text-rose-600 dark:text-rose-400">
          Feedback failed
        </span>
      ) : null}
    </>
  );
}

/** Props for {@link MessageBubble}. */
export interface MessageBubbleProps {
  message: ChatUiMessage;
  selectedMarker: string | null;
  onSelectMarker: (marker: string | null) => void;
}

/**
 * Render one turn of the conversation.
 *
 * @param props.message The turn to render.
 * @param props.selectedMarker Marker currently open in the source drawer.
 * @param props.onSelectMarker Called when a marker chip is pressed.
 * @returns The bubble.
 */
export default function MessageBubble({
  message,
  selectedMarker,
  onSelectMarker,
}: MessageBubbleProps): ReactNode {
  const [copied, setCopied] = useState(false);
  const isUser = message.role === 'user';
  const markerSet = new Set(message.citations.map((citation) => citation.marker));

  const copy = useCallback(async () => {
    try {
      await navigator.clipboard.writeText(message.content);
      setCopied(true);
      window.setTimeout(() => setCopied(false), 1500);
    } catch {
      setCopied(false);
    }
  }, [message.content]);

  const blocks = splitBlocks(message.content);

  return (
    <article
      className={`flex gap-3 ${isUser ? 'justify-end' : 'justify-start'}`}
      aria-label={isUser ? 'Your message' : 'Assistant answer'}
    >
      {!isUser ? (
        <div
          aria-hidden="true"
          className="mt-1 hidden h-7 w-7 shrink-0 items-center justify-center rounded-full bg-brand-100 text-xs font-bold text-brand-700 sm:flex dark:bg-brand-950 dark:text-brand-300"
        >
          AI
        </div>
      ) : null}

      <div
        className={`min-w-0 max-w-[46rem] rounded-2xl px-4 py-3 text-[15px] leading-relaxed ${
          isUser
            ? 'bg-brand-600 text-white'
            : 'border border-slate-200 bg-white text-slate-800 dark:border-slate-800 dark:bg-slate-900 dark:text-slate-100'
        } ${message.suppressed ? 'opacity-60' : ''}`}
      >
        {message.suppressed || message.pinned ? (
          <p className="mb-1.5 flex items-center gap-2 text-[11px] uppercase tracking-wide opacity-80">
            {message.suppressed ? (
              <span
                className="inline-flex items-center gap-1"
                title="Folded into the rolling summary; still stored, just not in the live window"
              >
                <EyeOff aria-hidden="true" className="h-3 w-3" />
                suppressed
              </span>
            ) : null}
            {message.pinned ? (
              <span className="inline-flex items-center gap-1" title="Never suppressed">
                <Pin aria-hidden="true" className="h-3 w-3" />
                pinned
              </span>
            ) : null}
          </p>
        ) : null}

        {blocks.length === 0 && message.streaming ? (
          <p className="streaming-caret text-slate-400" aria-label="Answer streaming" />
        ) : null}

        {blocks.map((block, blockIndex) =>
          block.kind === 'code' ? (
            <pre
              key={`code-${blockIndex}`}
              className="scroll-x my-2 rounded-lg bg-slate-900 p-3 text-xs text-slate-100 dark:bg-slate-950"
            >
              <code>{stripLanguageHint(block.value)}</code>
            </pre>
          ) : (
            <p
              key={`prose-${blockIndex}`}
              className={`whitespace-pre-wrap break-words ${
                message.streaming && blockIndex === blocks.length - 1
                  ? 'streaming-caret'
                  : ''
              }`}
            >
              {splitMarkers(block.value).map((segment, segmentIndex) =>
                segment.kind === 'text' ? (
                  <span key={`t-${blockIndex}-${segmentIndex}`}>{segment.value}</span>
                ) : (
                  <button
                    key={`m-${blockIndex}-${segmentIndex}`}
                    type="button"
                    className={`marker-chip ${
                      selectedMarker === segment.value ? 'marker-chip-active' : ''
                    } ${markerSet.has(segment.value) ? '' : 'opacity-60'}`}
                    title={
                      markerSet.has(segment.value)
                        ? `Open source ${segment.value}`
                        : `${segment.value} could not be verified against a chunk`
                    }
                    aria-label={`Open source ${segment.value}`}
                    onClick={() => onSelectMarker(segment.value)}
                  >
                    {segment.value.replace(/[[\]]/g, '')}
                  </button>
                ),
              )}
            </p>
          ),
        )}

        {message.error ? (
          <p className="mt-2 flex items-start gap-1.5 text-sm text-rose-600 dark:text-rose-400">
            <AlertTriangle aria-hidden="true" className="mt-0.5 h-3.5 w-3.5 shrink-0" />
            {message.error}
          </p>
        ) : null}

        {!isUser && !message.streaming && message.content ? (
          <div className="mt-2 flex items-center gap-1 border-t border-slate-100 pt-2 dark:border-slate-800">
            <button
              type="button"
              className="btn btn-ghost btn-xs"
              onClick={() => void copy()}
              aria-label="Copy answer"
            >
              {copied ? (
                <Check aria-hidden="true" className="h-3 w-3" />
              ) : (
                <Copy aria-hidden="true" className="h-3 w-3" />
              )}
              {copied ? 'Copied' : 'Copy'}
            </button>
            {message.message_id !== PENDING_MESSAGE_ID ? (
              <FeedbackButtons message={message} />
            ) : null}
            {message.refused ? (
              <span className="badge ml-auto" title="Answered with an explicit refusal">
                out of corpus
              </span>
            ) : null}
          </div>
        ) : null}
      </div>

      {isUser ? (
        <div
          aria-hidden="true"
          className="mt-1 hidden h-7 w-7 shrink-0 items-center justify-center rounded-full bg-slate-200 text-slate-600 sm:flex dark:bg-slate-800 dark:text-slate-300"
        >
          <User className="h-4 w-4" />
        </div>
      ) : null}
    </article>
  );
}
