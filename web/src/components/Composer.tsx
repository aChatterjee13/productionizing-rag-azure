/**
 * Message composer.
 *
 * Keyboard contract: Enter sends, Shift+Enter inserts a newline, Escape stops an
 * in-flight stream. The tool switch maps to `ChatRequest.allow_tools` and the filter
 * button reveals the metadata facets posted as `ChatRequest.filters`.
 */

import { Filter, Send, Square, Wrench } from 'lucide-react';
import {
  useCallback,
  useEffect,
  useRef,
  useState,
  type KeyboardEvent,
  type ReactNode,
} from 'react';

import type { ChatStreamController } from '../hooks/useChatStream';
import { countActiveFacets, useSettingsStore } from '../store/settings';

const MAX_TEXTAREA_ROWS_PX = 220;

/** Props for {@link Composer}. */
export interface ComposerProps {
  stream: ChatStreamController;
}

/**
 * Render the composer.
 *
 * @param props.stream Controller returned by `useChatStream`.
 * @returns The composer.
 */
export default function Composer({ stream }: ComposerProps): ReactNode {
  const [value, setValue] = useState('');
  const textareaRef = useRef<HTMLTextAreaElement | null>(null);

  const allowTools = useSettingsStore((state) => state.allowTools);
  const setAllowTools = useSettingsStore((state) => state.setAllowTools);
  const showFilters = useSettingsStore((state) => state.showFilters);
  const toggleFilters = useSettingsStore((state) => state.toggleFilters);
  const filters = useSettingsStore((state) => state.filters);
  const activeFacets = countActiveFacets(filters);

  // Grow the textarea with its content, up to a ceiling.
  useEffect(() => {
    const element = textareaRef.current;
    if (!element) return;
    element.style.height = 'auto';
    element.style.height = `${Math.min(element.scrollHeight, MAX_TEXTAREA_ROWS_PX)}px`;
  }, [value]);

  const submit = useCallback(async () => {
    const prompt = value.trim();
    if (!prompt || stream.streaming) return;
    setValue('');
    await stream.send(prompt);
    textareaRef.current?.focus();
  }, [stream, value]);

  const onKeyDown = useCallback(
    (event: KeyboardEvent<HTMLTextAreaElement>) => {
      if (event.key === 'Enter' && !event.shiftKey && !event.nativeEvent.isComposing) {
        event.preventDefault();
        void submit();
        return;
      }
      if (event.key === 'Escape' && stream.streaming) {
        event.preventDefault();
        stream.stop();
      }
    },
    [stream, submit],
  );

  return (
    <form
      className="shrink-0 border-t border-slate-200 bg-white px-4 py-3 dark:border-slate-800 dark:bg-slate-900"
      onSubmit={(event) => {
        event.preventDefault();
        void submit();
      }}
    >
      <label htmlFor="composer-input" className="sr-only">
        Your question
      </label>
      <div className="flex items-end gap-2">
        <textarea
          id="composer-input"
          ref={textareaRef}
          className="textarea max-h-[220px] min-h-[2.75rem] resize-none"
          rows={1}
          value={value}
          placeholder="Ask about the indexed corpus…"
          aria-describedby="composer-hint"
          disabled={stream.streaming}
          onChange={(event) => setValue(event.target.value)}
          onKeyDown={onKeyDown}
        />
        {stream.streaming ? (
          <button
            type="button"
            className="btn btn-danger h-[2.75rem]"
            onClick={stream.stop}
            aria-label="Stop generating"
          >
            <Square aria-hidden="true" className="h-4 w-4" />
            Stop
          </button>
        ) : (
          <button
            type="submit"
            className="btn btn-primary h-[2.75rem]"
            disabled={value.trim().length === 0}
            aria-label="Send message"
          >
            <Send aria-hidden="true" className="h-4 w-4" />
            Send
          </button>
        )}
      </div>

      <div className="mt-2 flex flex-wrap items-center gap-2">
        <button
          type="button"
          className={`btn btn-xs ${allowTools ? '' : 'btn-ghost'}`}
          aria-pressed={allowTools}
          onClick={() => setAllowTools(!allowTools)}
          title="Allow REST and MCP tools for data that is not indexed"
        >
          <Wrench aria-hidden="true" className="h-3 w-3" />
          Tools {allowTools ? 'on' : 'off'}
        </button>

        <button
          type="button"
          className={`btn btn-xs ${showFilters ? '' : 'btn-ghost'}`}
          aria-pressed={showFilters}
          aria-controls="filter-bar"
          onClick={toggleFilters}
        >
          <Filter aria-hidden="true" className="h-3 w-3" />
          Filters
          {activeFacets > 0 ? (
            <span className="ml-1 rounded-full bg-brand-600 px-1.5 text-[10px] font-semibold text-white">
              {activeFacets}
            </span>
          ) : null}
        </button>

        <p
          id="composer-hint"
          className="ml-auto text-[11px] text-slate-500 dark:text-slate-400"
        >
          <span className="kbd">Enter</span> to send ·{' '}
          <span className="kbd">Shift</span>+<span className="kbd">Enter</span> for a new
          line
        </p>
      </div>
    </form>
  );
}
