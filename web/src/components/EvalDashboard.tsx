/**
 * Evaluation dashboard (requirement #8).
 *
 * Aggregates come straight from `EvalRun.aggregate`, and the per-category tally is
 * derived from each result's `category` — a run where every `acl_negative` item fails
 * is a security regression, not a quality one, and the split has to make that obvious.
 */

import { ClipboardCheck, Loader2, PlayCircle, RefreshCw, ShieldX } from 'lucide-react';
import { useCallback, useEffect, useState, type ReactNode } from 'react';

import { ApiError, getEvalRun, listEvalRuns, startEvalRun } from '../api/client';
import type {
  CategoryTally,
  EvalResultRow,
  EvalRun,
  MetricScores,
} from '../api/types';

/** Metrics shown as headline cards, in display order. */
const HEADLINE_METRICS: Array<{ key: string; label: string; hint: string }> = [
  { key: 'faithfulness', label: 'Faithfulness', hint: 'Is the answer grounded?' },
  { key: 'answer_relevancy', label: 'Relevancy', hint: 'Does it address the question?' },
  { key: 'context_precision', label: 'Ctx precision', hint: 'Are chunks relevant?' },
  { key: 'context_recall', label: 'Ctx recall', hint: 'Was needed context retrieved?' },
  { key: 'answer_correctness', label: 'Correctness', hint: 'Matches ground truth?' },
  {
    key: 'semantic_similarity',
    label: 'Semantic sim.',
    hint: 'Sentence-embedding cosine vs ground truth',
  },
  {
    key: 'citation_validity',
    label: 'Citation validity',
    hint: 'Cited spans occur in cited chunks',
  },
  { key: 'acl_leak', label: 'ACL (1.0 = clean)', hint: '<1.0 means a chunk leaked' },
  { key: 'refusal_correct', label: 'Refusals', hint: 'Refusal behaviour matched' },
];

function scoresOf(result: EvalResultRow): Partial<MetricScores> {
  return result.scores ?? {};
}

function formatMetric(value: number | undefined): string {
  if (value === undefined || Number.isNaN(value)) return '—';
  return value <= 1 ? value.toFixed(3) : value.toFixed(1);
}

function tallyByCategory(results: EvalResultRow[]): CategoryTally[] {
  const map = new Map<string, CategoryTally>();
  for (const result of results) {
    const category = result.category ?? 'uncategorised';
    const entry = map.get(category) ?? { category, total: 0, passed: 0, failed: 0 };
    entry.total += 1;
    if (result.passed) entry.passed += 1;
    else entry.failed += 1;
    map.set(category, entry);
  }
  return [...map.values()].sort((a, b) => a.category.localeCompare(b.category));
}

function MetricCard({
  label,
  value,
  hint,
}: {
  label: string;
  value: string;
  hint: string;
}): ReactNode {
  return (
    <div
      className="rounded-lg border border-slate-200 p-2 dark:border-slate-800"
      title={hint}
    >
      <p className="truncate text-[10px] uppercase tracking-wide text-slate-500 dark:text-slate-400">
        {label}
      </p>
      <p className="text-lg font-semibold tabular-nums">{value}</p>
    </div>
  );
}

/**
 * Render the evaluation dashboard.
 *
 * @returns The dashboard.
 */
export default function EvalDashboard(): ReactNode {
  const [runs, setRuns] = useState<EvalRun[]>([]);
  const [selected, setSelected] = useState<EvalRun | null>(null);
  const [loading, setLoading] = useState(true);
  const [starting, setStarting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const load = useCallback(async () => {
    setLoading(true);
    try {
      const loaded = await listEvalRuns();
      setRuns(loaded);
      const first = loaded[0];
      if (first) setSelected(await getEvalRun(first.run_id));
      setError(null);
    } catch (cause) {
      setError(
        cause instanceof ApiError ? cause.message : 'Could not load evaluation runs.',
      );
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void load();
  }, [load]);

  const openRun = useCallback(async (runId: string) => {
    try {
      setSelected(await getEvalRun(runId));
      setError(null);
    } catch (cause) {
      setError(cause instanceof ApiError ? cause.message : 'Could not load that run.');
    }
  }, []);

  const onStart = useCallback(async () => {
    setStarting(true);
    try {
      const created = await startEvalRun({});
      setRuns((current) => [created, ...current]);
      setSelected(created);
      setError(null);
    } catch (cause) {
      setError(
        cause instanceof ApiError ? cause.message : 'Could not start an evaluation run.',
      );
    } finally {
      setStarting(false);
    }
  }, []);

  const results = selected?.results ?? [];
  const tallies = tallyByCategory(results);
  const aggregate: Record<string, number> = selected?.aggregate ?? {};

  return (
    <div className="scroll-area h-full p-4">
      <div className="mx-auto max-w-6xl space-y-4">
        <header className="flex flex-wrap items-center justify-between gap-2">
          <h1 className="flex items-center gap-2 text-lg font-semibold">
            <ClipboardCheck aria-hidden="true" className="h-5 w-5 text-brand-600" />
            Evaluation
          </h1>
          <div className="flex items-center gap-2">
            <button
              type="button"
              className="btn btn-xs"
              onClick={() => void load()}
              disabled={loading}
            >
              <RefreshCw aria-hidden="true" className="h-3 w-3" />
              Refresh
            </button>
            <button
              type="button"
              className="btn btn-primary btn-xs"
              onClick={() => void onStart()}
              disabled={starting}
            >
              {starting ? (
                <Loader2 aria-hidden="true" className="h-3 w-3 animate-spin" />
              ) : (
                <PlayCircle aria-hidden="true" className="h-3 w-3" />
              )}
              Run golden set
            </button>
          </div>
        </header>

        {error ? (
          <p
            role="alert"
            className="rounded-lg border border-rose-200 bg-rose-50 p-3 text-sm text-rose-700 dark:border-rose-900 dark:bg-rose-950/50 dark:text-rose-300"
          >
            {error}
          </p>
        ) : null}

        <section aria-label="Runs" className="panel p-4">
          <h2 className="text-sm font-semibold">Runs</h2>
          <div className="scroll-x mt-2">
            <table className="data-table">
              <caption className="sr-only">Evaluation runs, newest first</caption>
              <thead>
                <tr>
                  <th scope="col">Started</th>
                  <th scope="col">Gate</th>
                  <th scope="col" className="text-right">
                    Items
                  </th>
                  <th scope="col" className="text-right">
                    Passed
                  </th>
                  <th scope="col" className="text-right">
                    Failed
                  </th>
                  <th scope="col" className="text-right">
                    Cost
                  </th>
                  <th scope="col">Commit</th>
                  <th scope="col" />
                </tr>
              </thead>
              <tbody>
                {runs.map((run) => (
                  <tr
                    key={run.run_id}
                    className={run.run_id === selected?.run_id ? 'bg-brand-50/60 dark:bg-brand-950/40' : ''}
                  >
                    <td className="whitespace-nowrap text-xs">
                      {new Date(run.started_at).toLocaleString()}
                    </td>
                    <td>
                      <span
                        className={`badge ${
                          run.gate_passed
                            ? 'border-emerald-200 bg-emerald-50 text-emerald-700 dark:border-emerald-900 dark:bg-emerald-950 dark:text-emerald-300'
                            : 'border-rose-200 bg-rose-50 text-rose-700 dark:border-rose-900 dark:bg-rose-950 dark:text-rose-300'
                        }`}
                      >
                        {run.gate_passed ? 'passed' : 'failed'}
                      </span>
                    </td>
                    <td className="text-right tabular-nums">{run.item_count ?? '—'}</td>
                    <td className="text-right tabular-nums">{run.passed_count ?? '—'}</td>
                    <td className="text-right tabular-nums">{run.failed_count ?? '—'}</td>
                    <td className="text-right tabular-nums">
                      {typeof run.total_cost_usd === 'number'
                        ? `$${run.total_cost_usd.toFixed(3)}`
                        : '—'}
                    </td>
                    <td className="font-mono text-[11px]">
                      {run.git_sha ? run.git_sha.slice(0, 8) : '—'}
                    </td>
                    <td>
                      <button
                        type="button"
                        className="btn btn-ghost btn-xs"
                        onClick={() => void openRun(run.run_id)}
                      >
                        Open
                      </button>
                    </td>
                  </tr>
                ))}
                {!loading && runs.length === 0 ? (
                  <tr>
                    <td colSpan={8} className="py-4 text-center text-sm text-slate-500">
                      No evaluation runs yet.
                    </td>
                  </tr>
                ) : null}
              </tbody>
            </table>
          </div>
        </section>

        {selected ? (
          <>
            <section aria-label="Aggregate metrics" className="panel p-4">
              <div className="flex flex-wrap items-center justify-between gap-2">
                <h2 className="text-sm font-semibold">
                  Aggregate — run{' '}
                  <span className="font-mono text-xs">{selected.run_id}</span>
                </h2>
                <span className="badge">
                  pass rate{' '}
                  {typeof aggregate.pass_rate === 'number'
                    ? `${(aggregate.pass_rate * 100).toFixed(0)}%`
                    : '—'}
                </span>
              </div>
              <div className="mt-3 grid grid-cols-2 gap-2 sm:grid-cols-3 lg:grid-cols-5">
                {HEADLINE_METRICS.map((metric) => (
                  <MetricCard
                    key={metric.key}
                    label={metric.label}
                    hint={metric.hint}
                    value={formatMetric(aggregate[metric.key])}
                  />
                ))}
                <MetricCard
                  label="Latency"
                  hint="Mean end-to-end item latency"
                  value={
                    typeof aggregate.latency_ms === 'number'
                      ? `${aggregate.latency_ms.toFixed(0)}ms`
                      : '—'
                  }
                />
              </div>
              {typeof aggregate.acl_leak === 'number' && aggregate.acl_leak < 1 ? (
                <p
                  role="alert"
                  className="mt-3 flex items-center gap-2 rounded-lg border border-rose-300 bg-rose-50 p-3 text-sm font-semibold text-rose-700 dark:border-rose-800 dark:bg-rose-950/60 dark:text-rose-300"
                >
                  <ShieldX aria-hidden="true" className="h-4 w-4" />
                  An ACL leak was detected in this run. Treat it as a security incident,
                  not a quality regression.
                </p>
              ) : null}
            </section>

            <section aria-label="Per-category results" className="panel p-4">
              <h2 className="text-sm font-semibold">By category</h2>
              {tallies.length === 0 ? (
                <p className="mt-2 text-sm text-slate-500">
                  This run has no per-item results to break down.
                </p>
              ) : (
                <ul className="mt-3 space-y-2">
                  {tallies.map((tally) => {
                    const ratio = tally.total > 0 ? tally.passed / tally.total : 0;
                    return (
                      <li key={tally.category}>
                        <div className="flex items-center justify-between text-xs">
                          <span className="font-medium">{tally.category}</span>
                          <span className="tabular-nums text-slate-500">
                            {tally.passed}/{tally.total} passed
                          </span>
                        </div>
                        <div
                          className="mt-1 h-2 w-full overflow-hidden rounded-full bg-rose-200 dark:bg-rose-950"
                          role="meter"
                          aria-valuemin={0}
                          aria-valuemax={tally.total}
                          aria-valuenow={tally.passed}
                          aria-label={`${tally.category}: ${tally.passed} of ${tally.total} passed`}
                        >
                          <div
                            className="h-full bg-emerald-500"
                            style={{ width: `${ratio * 100}%` }}
                          />
                        </div>
                      </li>
                    );
                  })}
                </ul>
              )}
            </section>

            <section aria-label="Per-item results" className="panel p-4">
              <h2 className="text-sm font-semibold">Items ({results.length})</h2>
              <div className="scroll-x mt-2">
                <table className="data-table">
                  <caption className="sr-only">Per-item evaluation results</caption>
                  <thead>
                    <tr>
                      <th scope="col">Item</th>
                      <th scope="col">Category</th>
                      <th scope="col">Result</th>
                      <th scope="col" className="text-right">
                        Faith.
                      </th>
                      <th scope="col" className="text-right">
                        Sem. sim.
                      </th>
                      <th scope="col" className="text-right">
                        Cite
                      </th>
                      <th scope="col">Failures</th>
                    </tr>
                  </thead>
                  <tbody>
                    {results.map((result) => (
                      <tr key={result.item_id}>
                        <td className="max-w-[14rem]">
                          <span className="block truncate font-mono text-[11px]">
                            {result.item_id}
                          </span>
                          {result.question ? (
                            <span
                              className="block truncate text-xs text-slate-500"
                              title={result.question}
                            >
                              {result.question}
                            </span>
                          ) : null}
                        </td>
                        <td className="text-xs">{result.category ?? '—'}</td>
                        <td>
                          <span
                            className={`badge ${
                              result.passed
                                ? 'border-emerald-200 bg-emerald-50 text-emerald-700 dark:border-emerald-900 dark:bg-emerald-950 dark:text-emerald-300'
                                : 'border-rose-200 bg-rose-50 text-rose-700 dark:border-rose-900 dark:bg-rose-950 dark:text-rose-300'
                            }`}
                          >
                            {result.passed ? 'pass' : 'fail'}
                          </span>
                        </td>
                        <td className="text-right font-mono text-[11px] tabular-nums">
                          {formatMetric(scoresOf(result).faithfulness ?? undefined)}
                        </td>
                        <td className="text-right font-mono text-[11px] tabular-nums">
                          {formatMetric(scoresOf(result).semantic_similarity ?? undefined)}
                        </td>
                        <td className="text-right font-mono text-[11px] tabular-nums">
                          {formatMetric(scoresOf(result).citation_validity ?? undefined)}
                        </td>
                        <td className="max-w-[18rem] text-xs text-rose-600 dark:text-rose-400">
                          {(result.failures ?? []).join('; ')}
                        </td>
                      </tr>
                    ))}
                    {results.length === 0 ? (
                      <tr>
                        <td
                          colSpan={7}
                          className="py-4 text-center text-sm text-slate-500"
                        >
                          No per-item results were returned for this run.
                        </td>
                      </tr>
                    ) : null}
                  </tbody>
                </table>
              </div>
            </section>
          </>
        ) : null}
      </div>
    </div>
  );
}
