/**
 * Retrieval inspector: the funnel, the scores and the drops.
 *
 * Requirement #6 asks for hybrid search with rerank and metadata filtering, and
 * requirement #9 asks for auditable drops. Both are only trustworthy if they are
 * visible, so this panel renders the raw counters from `RetrievalResult`:
 * candidates → after dedupe → after rerank → kept, each chunk's dense / sparse /
 * fusion / rerank / final score, and every `dropped_reason`.
 */

import { ChevronRight, Database, Filter, Timer, Zap } from 'lucide-react';
import type { ReactNode } from 'react';

import type { RetrievedChunk } from '../api/types';
import { latestCitations, useChatStore } from '../store/chat';

function score(value: number | null | undefined): string {
  return value === null || value === undefined ? '—' : value.toFixed(3);
}

function FunnelStep({
  label,
  value,
  hint,
}: {
  label: string;
  value: number;
  hint: string;
}): ReactNode {
  return (
    <div className="flex min-w-0 flex-col items-center px-1" title={hint}>
      <span className="text-lg font-semibold tabular-nums">{value}</span>
      <span className="text-[10px] uppercase tracking-wide text-slate-500 dark:text-slate-400">
        {label}
      </span>
    </div>
  );
}

function ChunkRow({
  chunk,
  marker,
  onSelectMarker,
}: {
  chunk: RetrievedChunk;
  marker: string | undefined;
  onSelectMarker: (marker: string) => void;
}): ReactNode {
  const payload = chunk.payload;
  return (
    <tr>
      <td>
        {marker ? (
          <button
            type="button"
            className="marker-chip"
            aria-label={`Open source ${marker}`}
            onClick={() => onSelectMarker(marker)}
          >
            {marker.replace(/[[\]]/g, '')}
          </button>
        ) : (
          <span className="text-[11px] text-slate-400">—</span>
        )}
      </td>
      <td className="max-w-[12rem]">
        <span className="block truncate text-xs font-medium" title={payload.title}>
          {payload.title || payload.document_id}
        </span>
        <span className="block truncate text-[11px] text-slate-500">
          {payload.section_path.join(' › ')}
          {payload.page !== null ? ` · p.${payload.page}` : ''}
        </span>
      </td>
      <td className="text-right font-mono text-[11px] tabular-nums">
        {score(chunk.dense_score)}
      </td>
      <td className="text-right font-mono text-[11px] tabular-nums">
        {score(chunk.sparse_score)}
      </td>
      <td className="text-right font-mono text-[11px] tabular-nums">
        {score(chunk.fusion_score)}
      </td>
      <td className="text-right font-mono text-[11px] tabular-nums">
        {score(chunk.rerank_score)}
      </td>
      <td className="text-right font-mono text-[11px] font-semibold tabular-nums">
        {score(chunk.final_score)}
      </td>
      <td>
        <span className="chip">{chunk.retrieval_stage}</span>
      </td>
    </tr>
  );
}

/** Props for {@link RetrievalInspector}. */
export interface RetrievalInspectorProps {
  onSelectMarker: (marker: string) => void;
}

/**
 * Render the retrieval inspector for the current turn.
 *
 * @param props.onSelectMarker Called when a chunk's marker chip is pressed.
 * @returns The inspector panel.
 */
export default function RetrievalInspector({
  onSelectMarker,
}: RetrievalInspectorProps): ReactNode {
  const retrieval = useChatStore((state) => state.retrieval);
  const messages = useChatStore((state) => state.messages);

  const markerByChunk = new Map<string, string>();
  for (const citation of latestCitations(messages)) {
    if (!markerByChunk.has(citation.chunk_id)) {
      markerByChunk.set(citation.chunk_id, citation.marker);
    }
  }

  if (!retrieval) {
    return (
      <section aria-label="Retrieval inspector" className="panel p-3">
        <h3 className="flex items-center gap-1.5 text-xs font-semibold uppercase tracking-wide text-slate-500">
          <Database aria-hidden="true" className="h-3.5 w-3.5" />
          Retrieval
        </h3>
        <p className="mt-2 text-sm text-slate-500 dark:text-slate-400">
          Ask something to see the hybrid search funnel, per-stage scores and every
          audited drop.
        </p>
      </section>
    );
  }

  const latencyMap: Record<string, number> = retrieval.latency_ms ?? {};
  const latency = Object.entries(latencyMap);
  const totalLatency = latency.reduce((sum, [, value]) => sum + value, 0);
  const chunks = retrieval.chunks ?? [];
  const dropped = retrieval.dropped ?? [];
  const queries = retrieval.queries_used ?? [];

  return (
    <section aria-label="Retrieval inspector" className="panel">
      <div className="panel-header">
        <h3 className="flex items-center gap-1.5 text-xs font-semibold uppercase tracking-wide text-slate-500">
          <Database aria-hidden="true" className="h-3.5 w-3.5" />
          Retrieval
        </h3>
        <div className="flex items-center gap-1">
          {retrieval.cache_hit ? (
            <span
              className="badge border-emerald-200 bg-emerald-50 text-emerald-700 dark:border-emerald-900 dark:bg-emerald-950 dark:text-emerald-300"
              title="Chunk ids came from the semantic cache and were re-filtered through the live ACL filter"
            >
              <Zap aria-hidden="true" className="h-3 w-3" />
              cache hit
            </span>
          ) : null}
          <span className="badge" title="Total retrieval latency">
            <Timer aria-hidden="true" className="h-3 w-3" />
            {totalLatency.toFixed(0)} ms
          </span>
        </div>
      </div>

      <div className="p-3">
        <div className="flex items-center justify-between rounded-lg border border-slate-200 p-2 dark:border-slate-800">
          <FunnelStep
            label="candidates"
            value={retrieval.total_candidates}
            hint="Fused dense + sparse candidates across every sub-question"
          />
          <ChevronRight aria-hidden="true" className="h-4 w-4 shrink-0 text-slate-400" />
          <FunnelStep
            label="deduped"
            value={retrieval.after_dedupe}
            hint="After exact-hash and simhash dedupe"
          />
          <ChevronRight aria-hidden="true" className="h-4 w-4 shrink-0 text-slate-400" />
          <FunnelStep
            label="reranked"
            value={retrieval.after_rerank}
            hint="After the cross-encoder rerank"
          />
          <ChevronRight aria-hidden="true" className="h-4 w-4 shrink-0 text-slate-400" />
          <FunnelStep
            label="kept"
            value={chunks.length}
            hint="Chunks packed into the prompt"
          />
        </div>

        {queries.length > 0 ? (
          <div className="mt-3">
            <h4 className="label">Queries issued</h4>
            <ul className="mt-1 space-y-1">
              {queries.map((query, index) => (
                <li
                  key={`${index}-${query}`}
                  className="truncate text-xs text-slate-600 dark:text-slate-400"
                  title={query}
                >
                  {index + 1}. {query}
                </li>
              ))}
            </ul>
          </div>
        ) : null}

        {latency.length > 0 ? (
          <div className="mt-3 flex flex-wrap gap-1">
            {latency.map(([stage, value]) => (
              <span key={stage} className="chip">
                {stage} {value.toFixed(0)}ms
              </span>
            ))}
          </div>
        ) : null}

        <div className="scroll-x mt-3">
          <table className="data-table">
            <caption className="sr-only">
              Retrieved chunks with their per-stage scores
            </caption>
            <thead>
              <tr>
                <th scope="col">#</th>
                <th scope="col">Chunk</th>
                <th scope="col" className="text-right">
                  Dense
                </th>
                <th scope="col" className="text-right">
                  Sparse
                </th>
                <th scope="col" className="text-right">
                  Fusion
                </th>
                <th scope="col" className="text-right">
                  Rerank
                </th>
                <th scope="col" className="text-right">
                  Final
                </th>
                <th scope="col">Stage</th>
              </tr>
            </thead>
            <tbody>
              {chunks.map((chunk) => (
                <ChunkRow
                  key={chunk.payload.chunk_id}
                  chunk={chunk}
                  marker={markerByChunk.get(chunk.payload.chunk_id)}
                  onSelectMarker={onSelectMarker}
                />
              ))}
            </tbody>
          </table>
        </div>

        {dropped.length > 0 ? (
          <details className="mt-3">
            <summary className="cursor-pointer text-xs font-semibold uppercase tracking-wide text-slate-500">
              <Filter aria-hidden="true" className="mr-1 inline h-3 w-3" />
              Dropped ({dropped.length})
            </summary>
            <ul className="mt-2 space-y-1">
              {dropped.map((chunk) => (
                <li
                  key={`${chunk.payload.chunk_id}-${chunk.dropped_reason ?? 'drop'}`}
                  className="flex items-center justify-between gap-2 text-xs"
                >
                  <span className="min-w-0 flex-1 truncate" title={chunk.payload.title}>
                    {chunk.payload.title || chunk.payload.chunk_id}
                  </span>
                  <span className="chip shrink-0">{chunk.dropped_reason ?? 'unknown'}</span>
                </li>
              ))}
            </ul>
          </details>
        ) : null}

        {Object.keys(retrieval.filter_applied ?? {}).length > 0 ? (
          <details className="mt-2">
            <summary className="cursor-pointer text-xs font-semibold uppercase tracking-wide text-slate-500">
              Filter applied
            </summary>
            <pre className="scroll-x mt-2 rounded bg-slate-100 p-2 text-[11px] dark:bg-slate-950">
              <code>{JSON.stringify(retrieval.filter_applied, null, 2)}</code>
            </pre>
          </details>
        ) : null}
      </div>
    </section>
  );
}
