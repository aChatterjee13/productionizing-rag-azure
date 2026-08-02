/**
 * Ingestion administration: the schedule guard, a manual trigger, and run history.
 *
 * Requirement #1 puts the nightly delta refresh outside working hours, so the schedule
 * card surfaces the guard's own answer (`may_start` / `reason`) rather than restating
 * the cron: a run skipped as `working_hours` is a correct outcome, not a failure.
 */

import { CalendarClock, Loader2, PlayCircle, RefreshCw } from 'lucide-react';
import { useCallback, useEffect, useState, type ReactNode } from 'react';

import {
  ApiError,
  getSchedule,
  listIngestRuns,
  listSources,
  triggerIngest,
} from '../api/client';
import type { IngestRunSummary, ScheduleInfo, SourceConfigSummary } from '../api/types';

const STATUS_CLASS: Record<string, string> = {
  succeeded:
    'border-emerald-200 bg-emerald-50 text-emerald-700 dark:border-emerald-900 dark:bg-emerald-950 dark:text-emerald-300',
  partial:
    'border-amber-200 bg-amber-50 text-amber-800 dark:border-amber-900 dark:bg-amber-950 dark:text-amber-300',
  failed:
    'border-rose-200 bg-rose-50 text-rose-700 dark:border-rose-900 dark:bg-rose-950 dark:text-rose-300',
  running:
    'border-sky-200 bg-sky-50 text-sky-700 dark:border-sky-900 dark:bg-sky-950 dark:text-sky-300',
  skipped: '',
};

function formatWhen(value: string | null | undefined): string {
  if (!value) return '—';
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? '—' : date.toLocaleString();
}

function duration(run: IngestRunSummary): string {
  if (!run.finished_at) return '—';
  const start = new Date(run.started_at).getTime();
  const end = new Date(run.finished_at).getTime();
  if (Number.isNaN(start) || Number.isNaN(end) || end < start) return '—';
  const seconds = (end - start) / 1000;
  return seconds < 60 ? `${seconds.toFixed(1)}s` : `${(seconds / 60).toFixed(1)}m`;
}

/**
 * Render the ingestion administration view.
 *
 * @returns The ingestion panel.
 */
export default function AdminIngestion(): ReactNode {
  const [runs, setRuns] = useState<IngestRunSummary[]>([]);
  const [sources, setSources] = useState<SourceConfigSummary[]>([]);
  const [schedule, setSchedule] = useState<ScheduleInfo | null>(null);
  const [sourceId, setSourceId] = useState('');
  const [force, setForce] = useState(false);
  const [fullScan, setFullScan] = useState(false);
  const [loading, setLoading] = useState(true);
  const [triggering, setTriggering] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);

  const load = useCallback(async () => {
    setLoading(true);
    const results = await Promise.allSettled([
      listIngestRuns(sourceId ? { source_id: sourceId } : undefined),
      listSources(),
      getSchedule(),
    ]);
    const [runsResult, sourcesResult, scheduleResult] = results;
    if (runsResult.status === 'fulfilled') setRuns(runsResult.value);
    if (sourcesResult.status === 'fulfilled') setSources(sourcesResult.value);
    if (scheduleResult.status === 'fulfilled') setSchedule(scheduleResult.value);

    const failure = results.find((result) => result.status === 'rejected');
    if (failure && failure.status === 'rejected') {
      const cause: unknown = failure.reason;
      setError(
        cause instanceof ApiError
          ? cause.message
          : 'Some ingestion data could not be loaded.',
      );
    } else {
      setError(null);
    }
    setLoading(false);
  }, [sourceId]);

  useEffect(() => {
    void load();
  }, [load]);

  const onTrigger = useCallback(async () => {
    setTriggering(true);
    setNotice(null);
    try {
      const summary = await triggerIngest({
        source_id: sourceId || null,
        force,
        full_scan: fullScan,
      });
      setNotice(
        summary.skip_reason
          ? `Run ${summary.run_id} skipped: ${summary.skip_reason}.`
          : `Run ${summary.run_id} ${summary.status}: ${summary.documents_created} created, ` +
              `${summary.documents_updated} updated, ${summary.documents_deleted} deleted, ` +
              `${summary.chunks_upserted} chunks upserted.`,
      );
      await load();
      setError(null);
    } catch (cause) {
      setError(cause instanceof ApiError ? cause.message : 'Could not trigger a run.');
    } finally {
      setTriggering(false);
    }
  }, [sourceId, force, fullScan, load]);

  return (
    <div className="scroll-area h-full p-4">
      <div className="mx-auto max-w-6xl space-y-4">
        <header className="flex flex-wrap items-center justify-between gap-2">
          <h1 className="text-lg font-semibold">Ingestion</h1>
          <button
            type="button"
            className="btn btn-xs"
            onClick={() => void load()}
            disabled={loading}
          >
            <RefreshCw aria-hidden="true" className="h-3 w-3" />
            Refresh
          </button>
        </header>

        {error ? (
          <p
            role="alert"
            className="rounded-lg border border-rose-200 bg-rose-50 p-3 text-sm text-rose-700 dark:border-rose-900 dark:bg-rose-950/50 dark:text-rose-300"
          >
            {error}
          </p>
        ) : null}
        {notice ? (
          <p
            role="status"
            className="rounded-lg border border-emerald-200 bg-emerald-50 p-3 text-sm text-emerald-800 dark:border-emerald-900 dark:bg-emerald-950/50 dark:text-emerald-300"
          >
            {notice}
          </p>
        ) : null}

        <section aria-label="Schedule" className="panel p-4">
          <h2 className="flex items-center gap-2 text-sm font-semibold">
            <CalendarClock aria-hidden="true" className="h-4 w-4" />
            Schedule
          </h2>
          {schedule ? (
            <div className="mt-3 flex flex-wrap gap-2 text-sm">
              <span className="badge" title="Six-field NCRONTAB expression">
                cron {schedule.ingest_cron ?? '—'}
              </span>
              <span className="badge">tz {schedule.ingest_timezone ?? 'UTC'}</span>
              <span className="badge">
                working hours {schedule.ingest_working_hours_start ?? '—'}–
                {schedule.ingest_working_hours_end ?? '—'}
              </span>
              <span className={`badge ${schedule.ingest_enabled ? '' : 'opacity-60'}`}>
                {schedule.ingest_enabled ? 'enabled' : 'disabled'}
              </span>
              <span
                className={`badge ${
                  schedule.may_start
                    ? STATUS_CLASS.succeeded
                    : 'border-amber-200 bg-amber-50 text-amber-800 dark:border-amber-900 dark:bg-amber-950 dark:text-amber-300'
                }`}
                title="Guard result: a scheduled run inside working hours is refused unless forced"
              >
                {schedule.may_start ? 'may start now' : `blocked: ${schedule.reason ?? '—'}`}
              </span>
              {schedule.next_run_at ? (
                <span className="badge">next {formatWhen(schedule.next_run_at)}</span>
              ) : null}
            </div>
          ) : (
            <p className="mt-2 text-sm text-slate-500">Schedule unavailable.</p>
          )}
        </section>

        <section aria-label="Trigger a run" className="panel p-4">
          <h2 className="text-sm font-semibold">Trigger a delta run</h2>
          <div className="mt-3 grid gap-3 sm:grid-cols-4">
            <div className="sm:col-span-2">
              <label className="label" htmlFor="ingest-source">
                Source
              </label>
              <select
                id="ingest-source"
                className="select mt-1"
                value={sourceId}
                onChange={(event) => setSourceId(event.target.value)}
              >
                <option value="">All enabled sources</option>
                {sources.map((source) => (
                  <option key={source.source_id} value={source.source_id}>
                    {source.name} ({source.source_type})
                    {source.enabled ? '' : ' — disabled'}
                  </option>
                ))}
              </select>
            </div>
            <div className="flex items-end gap-3">
              <label className="flex items-center gap-2 text-sm" htmlFor="ingest-force">
                <input
                  id="ingest-force"
                  type="checkbox"
                  className="h-4 w-4 rounded border-slate-300"
                  checked={force}
                  onChange={(event) => setForce(event.target.checked)}
                />
                Force
              </label>
              <label className="flex items-center gap-2 text-sm" htmlFor="ingest-full">
                <input
                  id="ingest-full"
                  type="checkbox"
                  className="h-4 w-4 rounded border-slate-300"
                  checked={fullScan}
                  onChange={(event) => setFullScan(event.target.checked)}
                />
                Full scan
              </label>
            </div>
            <div className="flex items-end">
              <button
                type="button"
                className="btn btn-primary w-full"
                disabled={triggering}
                onClick={() => void onTrigger()}
              >
                {triggering ? (
                  <Loader2 aria-hidden="true" className="h-4 w-4 animate-spin" />
                ) : (
                  <PlayCircle aria-hidden="true" className="h-4 w-4" />
                )}
                Run now
              </button>
            </div>
          </div>
          <p className="mt-2 text-[11px] text-slate-500 dark:text-slate-400">
            Force overrides the working-hours guard. Without it a run started inside
            working hours is recorded as skipped with reason <code>working_hours</code>.
          </p>
        </section>

        <section aria-label="Run history" className="panel p-4">
          <h2 className="text-sm font-semibold">Run history</h2>
          {loading && runs.length === 0 ? (
            <p className="mt-2 flex items-center gap-2 text-sm text-slate-500">
              <Loader2 aria-hidden="true" className="h-4 w-4 animate-spin" />
              Loading…
            </p>
          ) : null}
          <div className="scroll-x mt-3">
            <table className="data-table">
              <caption className="sr-only">Ingestion runs, newest first</caption>
              <thead>
                <tr>
                  <th scope="col">Started</th>
                  <th scope="col">Status</th>
                  <th scope="col">Trigger</th>
                  <th scope="col">Source</th>
                  <th scope="col" className="text-right">
                    Seen
                  </th>
                  <th scope="col" className="text-right">
                    New
                  </th>
                  <th scope="col" className="text-right">
                    Upd
                  </th>
                  <th scope="col" className="text-right">
                    Del
                  </th>
                  <th scope="col" className="text-right">
                    Skip
                  </th>
                  <th scope="col" className="text-right">
                    Fail
                  </th>
                  <th scope="col" className="text-right">
                    Chunks
                  </th>
                  <th scope="col" className="text-right">
                    Dupes
                  </th>
                  <th scope="col" className="text-right">
                    PII
                  </th>
                  <th scope="col" className="text-right">
                    Took
                  </th>
                </tr>
              </thead>
              <tbody>
                {runs.map((run) => (
                  <tr key={run.run_id}>
                    <td className="whitespace-nowrap text-xs">
                      {formatWhen(run.started_at)}
                    </td>
                    <td>
                      <span className={`badge ${STATUS_CLASS[run.status] ?? ''}`}>
                        {run.status}
                      </span>
                      {run.skip_reason ? (
                        <span className="ml-1 chip">{run.skip_reason}</span>
                      ) : null}
                    </td>
                    <td className="text-xs">{run.trigger}</td>
                    <td className="max-w-[10rem] truncate font-mono text-[11px]">
                      {run.source_id ?? 'all'}
                    </td>
                    <td className="text-right tabular-nums">{run.documents_seen}</td>
                    <td className="text-right tabular-nums">{run.documents_created}</td>
                    <td className="text-right tabular-nums">{run.documents_updated}</td>
                    <td className="text-right tabular-nums">{run.documents_deleted}</td>
                    <td className="text-right tabular-nums">{run.documents_skipped}</td>
                    <td className="text-right tabular-nums">{run.documents_failed}</td>
                    <td className="text-right tabular-nums">{run.chunks_upserted}</td>
                    <td className="text-right tabular-nums">{run.duplicates_dropped}</td>
                    <td className="text-right tabular-nums">{run.pii_documents}</td>
                    <td className="text-right tabular-nums">{duration(run)}</td>
                  </tr>
                ))}
                {!loading && runs.length === 0 ? (
                  <tr>
                    <td colSpan={14} className="py-4 text-center text-sm text-slate-500">
                      No ingestion runs recorded yet.
                    </td>
                  </tr>
                ) : null}
              </tbody>
            </table>
          </div>

          {runs.some((run) => run.error_message) ? (
            <details className="mt-3">
              <summary className="cursor-pointer text-xs font-semibold uppercase tracking-wide text-slate-500">
                Errors
              </summary>
              <ul className="mt-2 space-y-1 text-xs text-rose-600 dark:text-rose-400">
                {runs
                  .filter((run) => run.error_message)
                  .map((run) => (
                    <li key={`${run.run_id}-error`}>
                      <span className="font-mono">{run.run_id}</span>: {run.error_message}
                    </li>
                  ))}
              </ul>
            </details>
          ) : null}
        </section>
      </div>
    </div>
  );
}
