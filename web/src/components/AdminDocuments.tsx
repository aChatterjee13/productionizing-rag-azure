/**
 * Document administration: upload, reindex, delete and inspect lineage.
 *
 * Requirement #9 asks for complete traceability, so every row can expand into its
 * provenance chain (`GET /documents/{id}/lineage`) — chunk → document → source URI,
 * with the operations and metrics that produced each hop.
 */

import {
  FileUp,
  GitBranch,
  Loader2,
  RefreshCw,
  RotateCcw,
  Search,
  Trash2,
} from 'lucide-react';
import { useCallback, useEffect, useState, type ReactNode } from 'react';

import {
  ApiError,
  deleteDocument,
  getDocumentLineage,
  listDocuments,
  reindexDocument,
  uploadDocument,
} from '../api/client';
import { CLASSIFICATIONS, type Classification } from '../api/types';
import type { DocumentProvenance, DocumentSummary } from '../api/types';

function formatDate(value: string | null | undefined): string {
  if (!value) return '—';
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? '—' : date.toLocaleDateString();
}

function formatBytes(value: number | undefined): string {
  if (!value) return '—';
  if (value < 1024) return `${value} B`;
  if (value < 1024 * 1024) return `${(value / 1024).toFixed(0)} KB`;
  return `${(value / (1024 * 1024)).toFixed(1)} MB`;
}

function LineageView({ provenance }: { provenance: DocumentProvenance }): ReactNode {
  const records = provenance.records ?? [];
  const extras = Object.entries(provenance).filter(
    ([key]) => key !== 'records' && key !== 'document_id',
  );
  return (
    <div className="rounded-lg border border-slate-200 bg-slate-50 p-3 dark:border-slate-800 dark:bg-slate-900/60">
      {extras.length > 0 ? (
        <dl className="mb-3 grid gap-1 text-xs sm:grid-cols-2">
          {extras.map(([key, value]) => (
            <div key={key} className="flex gap-2">
              <dt className="shrink-0 text-slate-500">{key}</dt>
              <dd className="min-w-0 break-all font-mono">
                {typeof value === 'string' || typeof value === 'number'
                  ? String(value)
                  : JSON.stringify(value)}
              </dd>
            </div>
          ))}
        </dl>
      ) : null}

      {records.length === 0 ? (
        <p className="text-xs text-slate-500">No lineage records recorded yet.</p>
      ) : (
        <ol className="space-y-2">
          {records.map((record) => (
            <li key={record.lineage_id} className="text-xs">
              <p className="flex flex-wrap items-center gap-2">
                <span className="badge">{record.kind}</span>
                <span className="font-semibold">{record.operation}</span>
                <span className="text-slate-500">by {record.actor}</span>
                <span className="text-slate-500">{formatDate(record.created_at)}</span>
              </p>
              <p className="mt-0.5 break-all font-mono text-[11px] text-slate-500">
                {record.subject_id}
                {record.parents.length > 0 ? ` ← ${record.parents.join(', ')}` : ''}
              </p>
              {Object.keys(record.metrics ?? {}).length > 0 ? (
                <p className="mt-0.5 flex flex-wrap gap-1">
                  {Object.entries(record.metrics).map(([name, value]) => (
                    <span key={name} className="chip">
                      {name} {typeof value === 'number' ? value : String(value)}
                    </span>
                  ))}
                </p>
              ) : null}
            </li>
          ))}
        </ol>
      )}
    </div>
  );
}

/**
 * Render the document administration view.
 *
 * @returns The admin documents panel.
 */
export default function AdminDocuments(): ReactNode {
  const [documents, setDocuments] = useState<DocumentSummary[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const [query, setQuery] = useState('');
  const [busyId, setBusyId] = useState<string | null>(null);
  const [lineageId, setLineageId] = useState<string | null>(null);
  const [lineage, setLineage] = useState<DocumentProvenance | null>(null);

  const [file, setFile] = useState<File | null>(null);
  const [docType, setDocType] = useState('document');
  const [tags, setTags] = useState('');
  const [classification, setClassification] = useState<Classification>('internal');
  const [uploading, setUploading] = useState(false);

  const load = useCallback(async () => {
    setLoading(true);
    try {
      setDocuments(await listDocuments(query ? { q: query } : undefined));
      setError(null);
    } catch (cause) {
      setError(cause instanceof ApiError ? cause.message : 'Could not load documents.');
    } finally {
      setLoading(false);
    }
  }, [query]);

  useEffect(() => {
    void load();
  }, [load]);

  const onUpload = useCallback(async () => {
    if (!file) return;
    setUploading(true);
    setNotice(null);
    try {
      const created = await uploadDocument(file, {
        doc_type: docType.trim() || 'document',
        classification,
        tags: tags
          .split(',')
          .map((tag) => tag.trim())
          .filter(Boolean),
      });
      setNotice(
        `Uploaded ${created.title || created.document_id} — ${created.chunk_count} chunks indexed.`,
      );
      setFile(null);
      setTags('');
      await load();
      setError(null);
    } catch (cause) {
      setError(cause instanceof ApiError ? cause.message : 'Upload failed.');
    } finally {
      setUploading(false);
    }
  }, [file, docType, classification, tags, load]);

  const onReindex = useCallback(async (documentId: string) => {
    setBusyId(documentId);
    setNotice(null);
    try {
      const summary = await reindexDocument(documentId);
      setNotice(
        `Reindexed: ${summary.chunks_upserted} chunks upserted, ${summary.chunks_deleted} removed.`,
      );
      setError(null);
    } catch (cause) {
      setError(cause instanceof ApiError ? cause.message : 'Reindex failed.');
    } finally {
      setBusyId(null);
    }
  }, []);

  const onDelete = useCallback(async (documentId: string) => {
    if (!window.confirm('Delete this document and remove its chunks from the index?')) {
      return;
    }
    setBusyId(documentId);
    try {
      await deleteDocument(documentId);
      setDocuments((current) =>
        current.map((item) =>
          item.document_id === documentId ? { ...item, is_deleted: true } : item,
        ),
      );
      setError(null);
    } catch (cause) {
      setError(cause instanceof ApiError ? cause.message : 'Delete failed.');
    } finally {
      setBusyId(null);
    }
  }, []);

  const onLineage = useCallback(
    async (documentId: string) => {
      if (lineageId === documentId) {
        setLineageId(null);
        setLineage(null);
        return;
      }
      setBusyId(documentId);
      try {
        setLineage(await getDocumentLineage(documentId));
        setLineageId(documentId);
        setError(null);
      } catch (cause) {
        setError(cause instanceof ApiError ? cause.message : 'Could not load lineage.');
      } finally {
        setBusyId(null);
      }
    },
    [lineageId],
  );

  return (
    <div className="scroll-area h-full p-4">
      <div className="mx-auto max-w-6xl space-y-4">
        <header className="flex flex-wrap items-center justify-between gap-2">
          <h1 className="text-lg font-semibold">Documents</h1>
          <div className="flex items-center gap-2">
            <label className="sr-only" htmlFor="document-search">
              Search documents
            </label>
            <div className="relative">
              <Search
                aria-hidden="true"
                className="pointer-events-none absolute left-2 top-2 h-4 w-4 text-slate-400"
              />
              <input
                id="document-search"
                className="input pl-8"
                value={query}
                placeholder="Search title or URI"
                onChange={(event) => setQuery(event.target.value)}
                onKeyDown={(event) => {
                  if (event.key === 'Enter') void load();
                }}
              />
            </div>
            <button
              type="button"
              className="btn btn-xs"
              onClick={() => void load()}
              disabled={loading}
            >
              <RefreshCw aria-hidden="true" className="h-3 w-3" />
              Refresh
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
        {notice ? (
          <p
            role="status"
            className="rounded-lg border border-emerald-200 bg-emerald-50 p-3 text-sm text-emerald-800 dark:border-emerald-900 dark:bg-emerald-950/50 dark:text-emerald-300"
          >
            {notice}
          </p>
        ) : null}

        <section aria-label="Upload a document" className="panel p-4">
          <h2 className="flex items-center gap-2 text-sm font-semibold">
            <FileUp aria-hidden="true" className="h-4 w-4" />
            Upload and index
          </h2>
          <div className="mt-3 grid gap-3 sm:grid-cols-4">
            <div className="sm:col-span-2">
              <label className="label" htmlFor="upload-file">
                File
              </label>
              <input
                id="upload-file"
                type="file"
                className="input mt-1 file:mr-2 file:rounded file:border-0 file:bg-slate-200 file:px-2 file:py-1 file:text-xs dark:file:bg-slate-700"
                onChange={(event) => setFile(event.target.files?.[0] ?? null)}
              />
            </div>
            <div>
              <label className="label" htmlFor="upload-doc-type">
                Doc type
              </label>
              <input
                id="upload-doc-type"
                className="input mt-1"
                value={docType}
                onChange={(event) => setDocType(event.target.value)}
              />
            </div>
            <div>
              <label className="label" htmlFor="upload-classification">
                Classification
              </label>
              <select
                id="upload-classification"
                className="select mt-1"
                value={classification}
                onChange={(event) =>
                  setClassification(event.target.value as Classification)
                }
              >
                {CLASSIFICATIONS.map((level) => (
                  <option key={level} value={level}>
                    {level}
                  </option>
                ))}
              </select>
            </div>
            <div className="sm:col-span-3">
              <label className="label" htmlFor="upload-tags">
                Tags (comma separated)
              </label>
              <input
                id="upload-tags"
                className="input mt-1"
                value={tags}
                onChange={(event) => setTags(event.target.value)}
              />
            </div>
            <div className="flex items-end">
              <button
                type="button"
                className="btn btn-primary w-full"
                disabled={!file || uploading}
                onClick={() => void onUpload()}
              >
                {uploading ? (
                  <Loader2 aria-hidden="true" className="h-4 w-4 animate-spin" />
                ) : (
                  <FileUp aria-hidden="true" className="h-4 w-4" />
                )}
                Upload
              </button>
            </div>
          </div>
        </section>

        <section aria-label="Indexed documents" className="panel p-4">
          {loading && documents.length === 0 ? (
            <p className="flex items-center gap-2 text-sm text-slate-500">
              <Loader2 aria-hidden="true" className="h-4 w-4 animate-spin" />
              Loading…
            </p>
          ) : null}

          <div className="scroll-x">
            <table className="data-table">
              <caption className="sr-only">Indexed documents for your tenant</caption>
              <thead>
                <tr>
                  <th scope="col">Title</th>
                  <th scope="col">Source</th>
                  <th scope="col">Type</th>
                  <th scope="col">Class</th>
                  <th scope="col" className="text-right">
                    Chunks
                  </th>
                  <th scope="col" className="text-right">
                    Tokens
                  </th>
                  <th scope="col" className="text-right">
                    Size
                  </th>
                  <th scope="col" className="text-right">
                    Ver
                  </th>
                  <th scope="col">Modified</th>
                  <th scope="col">Actions</th>
                </tr>
              </thead>
              <tbody>
                {documents.map((document) => (
                  <tr
                    key={document.document_id}
                    className={document.is_deleted ? 'opacity-50' : ''}
                  >
                    <td className="max-w-[16rem]">
                      <span className="block truncate font-medium" title={document.title}>
                        {document.title || document.document_id}
                      </span>
                      <span
                        className="block truncate font-mono text-[11px] text-slate-500"
                        title={document.source_uri}
                      >
                        {document.source_uri}
                      </span>
                    </td>
                    <td>
                      <span className="chip">{document.source_type}</span>
                    </td>
                    <td className="text-xs">{document.doc_type}</td>
                    <td className="text-xs">{document.classification}</td>
                    <td className="text-right tabular-nums">{document.chunk_count}</td>
                    <td className="text-right tabular-nums">{document.token_count}</td>
                    <td className="text-right tabular-nums">
                      {formatBytes(document.size_bytes)}
                    </td>
                    <td className="text-right tabular-nums">{document.version}</td>
                    <td className="whitespace-nowrap text-xs">
                      {formatDate(document.source_modified_at ?? document.updated_at)}
                    </td>
                    <td>
                      <div className="flex flex-wrap gap-1">
                        <button
                          type="button"
                          className="btn btn-ghost btn-xs"
                          disabled={busyId === document.document_id}
                          onClick={() => void onReindex(document.document_id)}
                          aria-label={`Reindex ${document.title || document.document_id}`}
                        >
                          <RotateCcw aria-hidden="true" className="h-3 w-3" />
                          Reindex
                        </button>
                        <button
                          type="button"
                          className="btn btn-ghost btn-xs"
                          disabled={busyId === document.document_id}
                          aria-expanded={lineageId === document.document_id}
                          onClick={() => void onLineage(document.document_id)}
                        >
                          <GitBranch aria-hidden="true" className="h-3 w-3" />
                          Lineage
                        </button>
                        <button
                          type="button"
                          className="btn btn-ghost btn-xs text-rose-600 dark:text-rose-400"
                          disabled={busyId === document.document_id || document.is_deleted}
                          onClick={() => void onDelete(document.document_id)}
                          aria-label={`Delete ${document.title || document.document_id}`}
                        >
                          <Trash2 aria-hidden="true" className="h-3 w-3" />
                          Delete
                        </button>
                      </div>
                    </td>
                  </tr>
                ))}
                {lineageId && lineage ? (
                  <tr>
                    <td colSpan={10}>
                      <LineageView provenance={lineage} />
                    </td>
                  </tr>
                ) : null}
                {!loading && documents.length === 0 ? (
                  <tr>
                    <td colSpan={10} className="py-4 text-center text-sm text-slate-500">
                      No documents matched.
                    </td>
                  </tr>
                ) : null}
              </tbody>
            </table>
          </div>
        </section>
      </div>
    </div>
  );
}
