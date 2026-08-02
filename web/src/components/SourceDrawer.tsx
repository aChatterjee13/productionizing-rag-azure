/**
 * Slide-over panel showing one cited source in full.
 *
 * Focus is moved into the drawer on open and returned to the trigger on close, and
 * Escape closes it — a citation chip is often reached by keyboard while reading, so
 * the panel has to be usable without a pointer.
 */

import { ExternalLink, Hash, X } from 'lucide-react';
import { useEffect, useRef, type ReactNode } from 'react';

import type { Citation, RetrievedChunk } from '../api/types';
import { useChatStore } from '../store/chat';
import { isWebLink, sectionLabel } from './CitationList';

function findChunk(
  chunks: RetrievedChunk[] | undefined,
  chunkId: string,
): RetrievedChunk | null {
  if (!chunks) return null;
  return chunks.find((chunk) => chunk.payload.chunk_id === chunkId) ?? null;
}

function Row({ label, value }: { label: string; value: ReactNode }): ReactNode {
  return (
    <div className="grid grid-cols-[8rem_1fr] gap-2 py-1 text-sm">
      <dt className="text-slate-500 dark:text-slate-400">{label}</dt>
      <dd className="min-w-0 break-words">{value}</dd>
    </div>
  );
}

/** Props for {@link SourceDrawer}. */
export interface SourceDrawerProps {
  open: boolean;
  citation: Citation | null;
  onClose: () => void;
}

/**
 * Render the source drawer.
 *
 * @param props.open Whether the drawer is visible.
 * @param props.citation The citation to show, or null.
 * @param props.onClose Called to dismiss the drawer.
 * @returns The drawer, or null when closed.
 */
export default function SourceDrawer({
  open,
  citation,
  onClose,
}: SourceDrawerProps): ReactNode {
  const retrieval = useChatStore((state) => state.retrieval);
  const closeRef = useRef<HTMLButtonElement | null>(null);
  const restoreRef = useRef<Element | null>(null);

  useEffect(() => {
    if (open) {
      restoreRef.current = document.activeElement;
      closeRef.current?.focus();
    } else if (restoreRef.current instanceof HTMLElement) {
      restoreRef.current.focus();
      restoreRef.current = null;
    }
  }, [open]);

  useEffect(() => {
    if (!open) return;
    const onKeyDown = (event: KeyboardEvent): void => {
      if (event.key === 'Escape') onClose();
    };
    window.addEventListener('keydown', onKeyDown);
    return () => window.removeEventListener('keydown', onKeyDown);
  }, [open, onClose]);

  if (!open || !citation) return null;

  const chunk = findChunk(retrieval?.chunks, citation.chunk_id);
  const payload = chunk?.payload;

  return (
    <div className="fixed inset-0 z-40 flex justify-end">
      <div
        className="absolute inset-0 bg-slate-900/40 backdrop-blur-sm"
        aria-hidden="true"
        onClick={onClose}
      />
      <aside
        role="dialog"
        aria-modal="true"
        aria-labelledby="source-drawer-title"
        className="relative flex h-full w-full max-w-lg animate-slide-in-right flex-col border-l border-slate-200 bg-white shadow-xl dark:border-slate-800 dark:bg-slate-900"
      >
        <header className="flex items-start justify-between gap-3 border-b border-slate-200 p-4 dark:border-slate-800">
          <div className="min-w-0">
            <p className="text-xs font-semibold uppercase tracking-wide text-slate-500">
              Source {citation.marker}
            </p>
            <h2 id="source-drawer-title" className="truncate text-lg font-semibold">
              {citation.title || citation.document_id}
            </h2>
            <p className="truncate text-xs text-slate-500 dark:text-slate-400">
              {sectionLabel(citation.section_path)}
              {citation.page !== null ? ` · page ${citation.page}` : ''}
            </p>
          </div>
          <button
            ref={closeRef}
            type="button"
            className="btn btn-ghost"
            onClick={onClose}
            aria-label="Close source panel"
          >
            <X aria-hidden="true" className="h-4 w-4" />
          </button>
        </header>

        <div className="scroll-area min-h-0 flex-1 p-4">
          {citation.quoted_span ? (
            <section aria-label="Verified quoted span">
              <h3 className="label">Verified span</h3>
              <blockquote className="mt-1 rounded-lg border-l-4 border-brand-500 bg-brand-50 p-3 text-sm dark:bg-brand-950/50">
                “{citation.quoted_span}”
              </blockquote>
              {citation.char_start !== null && citation.char_end !== null ? (
                <p className="mt-1 text-[11px] text-slate-500">
                  characters {citation.char_start}–{citation.char_end} of the chunk
                </p>
              ) : null}
            </section>
          ) : (
            <p className="rounded-lg border border-amber-200 bg-amber-50 p-3 text-sm text-amber-800 dark:border-amber-900 dark:bg-amber-950/50 dark:text-amber-300">
              No verbatim span could be located in this chunk, so the citation is
              flagged as unverified.
            </p>
          )}

          <section aria-label="Citation metadata" className="mt-4">
            <h3 className="label">Metadata</h3>
            <dl className="mt-1 divide-y divide-slate-100 dark:divide-slate-800">
              <Row
                label="Document"
                value={<span className="font-mono text-xs">{citation.document_id}</span>}
              />
              <Row
                label="Chunk"
                value={
                  <span className="inline-flex items-center gap-1 font-mono text-xs">
                    <Hash aria-hidden="true" className="h-3 w-3" />
                    {citation.chunk_id}
                  </span>
                }
              />
              <Row label="Confidence" value={`${(citation.confidence * 100).toFixed(0)}%`} />
              {payload ? (
                <>
                  <Row label="Source type" value={payload.source_type} />
                  <Row label="Doc type" value={payload.doc_type} />
                  <Row label="Classification" value={payload.classification} />
                  <Row label="Language" value={payload.language} />
                  <Row label="Author" value={payload.author ?? '—'} />
                  <Row
                    label="Modified"
                    value={
                      payload.source_modified_at
                        ? new Date(payload.source_modified_at).toLocaleString()
                        : '—'
                    }
                  />
                  <Row
                    label="Tags"
                    value={
                      payload.tags.length > 0 ? (
                        <span className="flex flex-wrap gap-1">
                          {payload.tags.map((tag) => (
                            <span key={tag} className="chip">
                              {tag}
                            </span>
                          ))}
                        </span>
                      ) : (
                        '—'
                      )
                    }
                  />
                  {payload.pii_redacted ? (
                    <Row
                      label="PII"
                      value={`redacted (${payload.pii_types.join(', ') || 'unlisted'})`}
                    />
                  ) : null}
                </>
              ) : null}
              <Row
                label="Scores"
                value={
                  chunk ? (
                    <span className="flex flex-wrap gap-1 text-xs">
                      {chunk.dense_score !== null ? (
                        <span className="chip">dense {chunk.dense_score.toFixed(3)}</span>
                      ) : null}
                      {chunk.sparse_score !== null ? (
                        <span className="chip">
                          sparse {chunk.sparse_score.toFixed(3)}
                        </span>
                      ) : null}
                      <span className="chip">fusion {chunk.fusion_score.toFixed(3)}</span>
                      {chunk.rerank_score !== null ? (
                        <span className="chip">
                          rerank {chunk.rerank_score.toFixed(3)}
                        </span>
                      ) : null}
                      <span className="chip">final {chunk.final_score.toFixed(3)}</span>
                    </span>
                  ) : (
                    'not in the current retrieval set'
                  )
                }
              />
            </dl>
          </section>

          {payload?.text ? (
            <section aria-label="Chunk text" className="mt-4">
              <h3 className="label">Chunk text</h3>
              <p className="mt-1 whitespace-pre-wrap rounded-lg border border-slate-200 p-3 text-sm dark:border-slate-800">
                {payload.text}
              </p>
            </section>
          ) : null}
        </div>

        <footer className="border-t border-slate-200 p-4 dark:border-slate-800">
          {isWebLink(citation.source_uri) ? (
            <a
              className="btn btn-primary w-full"
              href={citation.source_uri}
              target="_blank"
              rel="noreferrer noopener"
            >
              <ExternalLink aria-hidden="true" className="h-4 w-4" />
              Open the original document
            </a>
          ) : (
            <p className="break-all font-mono text-xs text-slate-500 dark:text-slate-400">
              {citation.source_uri || 'No source URI recorded'}
            </p>
          )}
        </footer>
      </aside>
    </div>
  );
}
