/**
 * Numbered sources for the latest answer.
 *
 * Each entry shows the title, the section breadcrumb, the page and the verified
 * quoted span. An unverified citation is flagged rather than hidden — stage 11 drops
 * spans it cannot locate in the cited chunk, and the UI must not paper over that.
 */

import { BadgeCheck, AlertTriangle, ExternalLink, FileText } from 'lucide-react';
import type { ReactNode } from 'react';

import type { Citation } from '../api/types';

/**
 * Whether a link can safely be rendered as an anchor.
 *
 * Only http(s) URIs become links; `blob://`, `sharepoint:` and file paths are shown
 * as text so a citation can never turn into a `javascript:` navigation.
 *
 * @param uri The citation's source URI.
 * @returns True when the URI is a web link.
 */
export function isWebLink(uri: string): boolean {
  return /^https?:\/\//i.test(uri);
}

/**
 * Render a heading breadcrumb.
 *
 * @param path The `section_path` list.
 * @returns The joined path, or an empty string.
 */
export function sectionLabel(path: string[]): string {
  return path.filter(Boolean).join(' › ');
}

/** Props for {@link CitationList}. */
export interface CitationListProps {
  citations: Citation[];
  selectedMarker: string | null;
  onSelect: (marker: string) => void;
}

/**
 * Render the source list under an answer.
 *
 * @param props.citations Citations from the latest assistant turn.
 * @param props.selectedMarker Marker currently open in the drawer.
 * @param props.onSelect Called when a source is chosen.
 * @returns The list.
 */
export default function CitationList({
  citations,
  selectedMarker,
  onSelect,
}: CitationListProps): ReactNode {
  return (
    <section aria-label="Sources" className="panel mt-4 p-3">
      <h3 className="mb-2 flex items-center gap-1.5 text-xs font-semibold uppercase tracking-wide text-slate-500 dark:text-slate-400">
        <FileText aria-hidden="true" className="h-3.5 w-3.5" />
        Sources ({citations.length})
      </h3>
      <ol className="space-y-2">
        {citations.map((citation) => {
          const selected = selectedMarker === citation.marker;
          const verified = Boolean(citation.quoted_span) && citation.char_start !== null;
          return (
            <li key={`${citation.marker}-${citation.chunk_id}`}>
              <div
                className={`rounded-lg border p-2 ${
                  selected
                    ? 'border-brand-400 bg-brand-50 dark:border-brand-700 dark:bg-brand-950/60'
                    : 'border-slate-200 dark:border-slate-800'
                }`}
              >
                <div className="flex items-start gap-2">
                  <button
                    type="button"
                    className="marker-chip mt-0.5"
                    aria-label={`Open source ${citation.marker}`}
                    onClick={() => onSelect(citation.marker)}
                  >
                    {citation.marker.replace(/[[\]]/g, '')}
                  </button>
                  <div className="min-w-0 flex-1">
                    <p className="truncate text-sm font-medium">
                      {citation.title || citation.document_id}
                    </p>
                    <p className="truncate text-xs text-slate-500 dark:text-slate-400">
                      {sectionLabel(citation.section_path)}
                      {citation.page !== null ? ` · p.${citation.page}` : ''}
                    </p>
                    {citation.quoted_span ? (
                      <blockquote className="mt-1.5 border-l-2 border-slate-300 pl-2 text-xs italic text-slate-600 dark:border-slate-700 dark:text-slate-400">
                        “{citation.quoted_span}”
                      </blockquote>
                    ) : null}
                    <div className="mt-1.5 flex flex-wrap items-center gap-2">
                      {verified ? (
                        <span
                          className="badge border-emerald-200 bg-emerald-50 text-emerald-700 dark:border-emerald-900 dark:bg-emerald-950 dark:text-emerald-300"
                          title="The quoted span was located verbatim in the cited chunk"
                        >
                          <BadgeCheck aria-hidden="true" className="h-3 w-3" />
                          verified
                        </span>
                      ) : (
                        <span
                          className="badge border-amber-200 bg-amber-50 text-amber-800 dark:border-amber-900 dark:bg-amber-950 dark:text-amber-300"
                          title="No span could be located in the cited chunk"
                        >
                          <AlertTriangle aria-hidden="true" className="h-3 w-3" />
                          unverified
                        </span>
                      )}
                      <span className="chip" title="Citation confidence">
                        {(citation.confidence * 100).toFixed(0)}%
                      </span>
                      {isWebLink(citation.source_uri) ? (
                        <a
                          className="inline-flex items-center gap-1 text-xs text-brand-700 underline dark:text-brand-300"
                          href={citation.source_uri}
                          target="_blank"
                          rel="noreferrer noopener"
                        >
                          <ExternalLink aria-hidden="true" className="h-3 w-3" />
                          Open source
                        </a>
                      ) : citation.source_uri ? (
                        <span className="truncate font-mono text-[11px] text-slate-500">
                          {citation.source_uri}
                        </span>
                      ) : null}
                    </div>
                  </div>
                </div>
              </div>
            </li>
          );
        })}
      </ol>
    </section>
  );
}
