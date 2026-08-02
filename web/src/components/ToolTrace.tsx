/**
 * Tool-call trace for the current turn.
 *
 * Requirement #4 lets the model reach un-indexed data through REST tools and remote
 * MCP servers. Because a tool answer has no citation to inspect, the trace itself is
 * the evidence: which tool, what kind, the arguments the model chose, the latency and
 * the error if it failed.
 */

import { AlertTriangle, CheckCircle2, Loader2, Wrench } from 'lucide-react';
import type { ReactNode } from 'react';

import { useChatStore, type ToolTraceEntry } from '../store/chat';

const KIND_LABEL: Record<string, string> = {
  retrieval: 'retrieval',
  rest: 'REST',
  mcp: 'MCP',
};

function StatusIcon({ status }: { status: ToolTraceEntry['status'] }): ReactNode {
  if (status === 'running') {
    return (
      <Loader2
        aria-hidden="true"
        className="h-3.5 w-3.5 shrink-0 animate-spin text-brand-500"
      />
    );
  }
  if (status === 'error') {
    return (
      <AlertTriangle aria-hidden="true" className="h-3.5 w-3.5 shrink-0 text-rose-500" />
    );
  }
  return (
    <CheckCircle2 aria-hidden="true" className="h-3.5 w-3.5 shrink-0 text-emerald-500" />
  );
}

/**
 * Render the tool trace.
 *
 * @returns The trace panel.
 */
export default function ToolTrace(): ReactNode {
  const tools = useChatStore((state) => state.tools);

  return (
    <section aria-label="Tool trace" className="panel">
      <div className="panel-header">
        <h3 className="flex items-center gap-1.5 text-xs font-semibold uppercase tracking-wide text-slate-500">
          <Wrench aria-hidden="true" className="h-3.5 w-3.5" />
          Tools
        </h3>
        {tools.length > 0 ? <span className="badge">{tools.length}</span> : null}
      </div>

      <div className="p-3">
        {tools.length === 0 ? (
          <p className="text-sm text-slate-500 dark:text-slate-400">
            No tools were called for this turn. Retrieval-only answers never leave the
            index.
          </p>
        ) : (
          <ol className="space-y-2">
            {tools.map((tool) => (
              <li
                key={tool.tool_call_id}
                className="rounded-lg border border-slate-200 p-2 dark:border-slate-800"
              >
                <div className="flex items-center gap-2">
                  <StatusIcon status={tool.status} />
                  <span className="min-w-0 flex-1 truncate font-mono text-xs font-semibold">
                    {tool.tool_name}
                  </span>
                  <span className="chip shrink-0">
                    {KIND_LABEL[tool.kind] ?? tool.kind}
                  </span>
                  {tool.latency_ms !== null ? (
                    <span className="shrink-0 text-[11px] tabular-nums text-slate-500">
                      {tool.latency_ms.toFixed(0)} ms
                    </span>
                  ) : null}
                </div>

                {Object.keys(tool.arguments).length > 0 ? (
                  <details className="mt-1.5">
                    <summary className="cursor-pointer text-[11px] uppercase tracking-wide text-slate-500">
                      Arguments
                    </summary>
                    <pre className="scroll-x mt-1 rounded bg-slate-100 p-2 text-[11px] dark:bg-slate-950">
                      <code>{JSON.stringify(tool.arguments, null, 2)}</code>
                    </pre>
                  </details>
                ) : null}

                {tool.result_summary ? (
                  <p className="mt-1.5 line-clamp-4 text-xs text-slate-600 dark:text-slate-400">
                    {tool.result_summary}
                  </p>
                ) : null}

                {tool.error_message ? (
                  <p className="mt-1.5 text-xs text-rose-600 dark:text-rose-400">
                    {tool.error_message}
                    {tool.http_status !== null ? ` (HTTP ${tool.http_status})` : ''}
                  </p>
                ) : null}
              </li>
            ))}
          </ol>
        )}
      </div>
    </section>
  );
}
