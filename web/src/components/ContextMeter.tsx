/**
 * Live token-budget meter.
 *
 * Requirements #3 and #5 demand that context management be observable rather than
 * implicit, so this renders `ContextStats` verbatim: how the window was spent, how
 * many turns are live versus suppressed, how many compactions have happened, how many
 * tool results the `clear_tool_uses_20250919` edit removed, and how much of the prompt
 * was served from the Anthropic cache.
 */

import { Gauge, Layers, Recycle, Scissors, Snowflake } from 'lucide-react';
import type { ReactNode } from 'react';

import { useChatStore } from '../store/chat';
import { CONTEXT_WARN_RATIO } from '../store/settings';

interface Slice {
  key: string;
  label: string;
  tokens: number;
  className: string;
}

function formatTokens(value: number): string {
  if (value >= 1000) return `${(value / 1000).toFixed(1)}k`;
  return String(value);
}

function Stat({
  icon: Icon,
  label,
  value,
  hint,
}: {
  icon: typeof Gauge;
  label: string;
  value: string;
  hint: string;
}): ReactNode {
  return (
    <div
      className="flex items-center gap-2 rounded-lg border border-slate-200 px-2 py-1.5 dark:border-slate-800"
      title={hint}
    >
      <Icon aria-hidden="true" className="h-4 w-4 shrink-0 text-slate-400" />
      <div className="min-w-0">
        <p className="truncate text-[10px] uppercase tracking-wide text-slate-500 dark:text-slate-400">
          {label}
        </p>
        <p className="text-sm font-semibold tabular-nums">{value}</p>
      </div>
    </div>
  );
}

/**
 * Render the context meter for the current turn.
 *
 * @returns The meter panel.
 */
export default function ContextMeter(): ReactNode {
  const stats = useChatStore((state) => state.contextStats);
  const usage = useChatStore((state) => state.usage);

  if (!stats) {
    return (
      <section aria-label="Context budget" className="panel p-3">
        <h3 className="flex items-center gap-1.5 text-xs font-semibold uppercase tracking-wide text-slate-500">
          <Gauge aria-hidden="true" className="h-3.5 w-3.5" />
          Context budget
        </h3>
        <p className="mt-2 text-sm text-slate-500 dark:text-slate-400">
          The token budget, suppressed turns and compaction events appear here once a
          turn has run.
        </p>
      </section>
    );
  }

  const slices: Slice[] = [
    {
      key: 'system',
      label: 'system',
      tokens: stats.system_tokens,
      className: 'bg-slate-400 dark:bg-slate-500',
    },
    {
      key: 'summary',
      label: 'summary',
      tokens: stats.summary_tokens,
      className: 'bg-violet-400 dark:bg-violet-500',
    },
    {
      key: 'memory',
      label: 'memory',
      tokens: stats.memory_tokens,
      className: 'bg-amber-400 dark:bg-amber-500',
    },
    {
      key: 'retrieved',
      label: 'retrieved',
      tokens: stats.retrieved_tokens,
      className: 'bg-brand-500',
    },
    {
      key: 'history',
      label: 'history',
      tokens: stats.history_tokens,
      className: 'bg-emerald-400 dark:bg-emerald-500',
    },
  ];

  const budget = stats.budget_tokens > 0 ? stats.budget_tokens : stats.window_tokens;
  const utilisation = budget > 0 ? stats.window_tokens / budget : 0;
  const nearCompaction = utilisation >= CONTEXT_WARN_RATIO;
  const accounted = slices.reduce((sum, slice) => sum + slice.tokens, 0);
  const other = Math.max(0, stats.window_tokens - accounted);
  const cacheRatio =
    stats.window_tokens > 0
      ? Math.min(1, stats.cache_read_tokens / stats.window_tokens)
      : 0;

  return (
    <section aria-label="Context budget" className="panel">
      <div className="panel-header">
        <h3 className="flex items-center gap-1.5 text-xs font-semibold uppercase tracking-wide text-slate-500">
          <Gauge aria-hidden="true" className="h-3.5 w-3.5" />
          Context budget
        </h3>
        <span
          className={`badge ${
            nearCompaction
              ? 'border-amber-300 bg-amber-50 text-amber-800 dark:border-amber-800 dark:bg-amber-950 dark:text-amber-300'
              : ''
          }`}
          title={`Window ${stats.window_tokens} of ${budget} prompt tokens`}
        >
          {formatTokens(stats.window_tokens)} / {formatTokens(budget)} ·{' '}
          {(utilisation * 100).toFixed(0)}%
        </span>
      </div>

      <div className="p-3">
        <div
          className="flex h-3 w-full overflow-hidden rounded-full bg-slate-200 dark:bg-slate-800"
          role="meter"
          aria-valuemin={0}
          aria-valuemax={budget}
          aria-valuenow={stats.window_tokens}
          aria-label="Prompt tokens used against the budget"
        >
          {slices
            .filter((slice) => slice.tokens > 0)
            .map((slice) => (
              <div
                key={slice.key}
                className={slice.className}
                style={{ width: `${budget > 0 ? (slice.tokens / budget) * 100 : 0}%` }}
                title={`${slice.label}: ${slice.tokens} tokens`}
              />
            ))}
          {other > 0 ? (
            <div
              className="bg-slate-300 dark:bg-slate-600"
              style={{ width: `${budget > 0 ? (other / budget) * 100 : 0}%` }}
              title={`other: ${other} tokens`}
            />
          ) : null}
        </div>

        <ul className="mt-2 flex flex-wrap gap-x-3 gap-y-1 text-[11px] text-slate-600 dark:text-slate-400">
          {slices.map((slice) => (
            <li key={slice.key} className="flex items-center gap-1">
              <span
                aria-hidden="true"
                className={`h-2 w-2 rounded-full ${slice.className}`}
              />
              {slice.label} {formatTokens(slice.tokens)}
            </li>
          ))}
        </ul>

        {nearCompaction ? (
          <p className="mt-2 rounded-lg border border-amber-200 bg-amber-50 px-2 py-1.5 text-xs text-amber-800 dark:border-amber-900 dark:bg-amber-950/50 dark:text-amber-300">
            Past {Math.round(CONTEXT_WARN_RATIO * 100)}% of the budget — the next turn
            will compact the oldest non-pinned turns into the rolling summary.
          </p>
        ) : null}

        <div className="mt-3 grid grid-cols-2 gap-2">
          <Stat
            icon={Layers}
            label="live turns"
            value={String(stats.messages_live)}
            hint="Turns present in the live window"
          />
          <Stat
            icon={Snowflake}
            label="suppressed"
            value={String(stats.messages_suppressed)}
            hint="Turns folded into the rolling summary instead of sent"
          />
          <Stat
            icon={Scissors}
            label="compactions"
            value={String(stats.compaction_events)}
            hint="Compactions performed in this session"
          />
          <Stat
            icon={Recycle}
            label="cache read"
            value={`${formatTokens(stats.cache_read_tokens)} · ${(
              cacheRatio * 100
            ).toFixed(0)}%`}
            hint="Prompt tokens served from the Anthropic prompt cache (billed at 0.1x)"
          />
        </div>

        {typeof stats.tool_results_cleared === 'number' &&
        stats.tool_results_cleared > 0 ? (
          <p className="mt-2 text-xs text-slate-600 dark:text-slate-400">
            {stats.tool_results_cleared} old tool result
            {stats.tool_results_cleared === 1 ? '' : 's'} cleared by the context edit.
          </p>
        ) : null}

        {usage ? (
          <p className="mt-2 flex flex-wrap gap-1 text-[11px]">
            {usage.model ? <span className="chip">{usage.model}</span> : null}
            {typeof usage.input_tokens === 'number' ? (
              <span className="chip">in {formatTokens(usage.input_tokens)}</span>
            ) : null}
            {typeof usage.output_tokens === 'number' ? (
              <span className="chip">out {formatTokens(usage.output_tokens)}</span>
            ) : null}
            {typeof usage.cache_write_tokens === 'number' &&
            usage.cache_write_tokens > 0 ? (
              <span className="chip">
                cache write {formatTokens(usage.cache_write_tokens)}
              </span>
            ) : null}
            {typeof usage.cost_usd === 'number' ? (
              <span className="chip">${usage.cost_usd.toFixed(4)}</span>
            ) : null}
          </p>
        ) : null}
      </div>
    </section>
  );
}
