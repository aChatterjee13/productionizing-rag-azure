/**
 * Metadata facet filter.
 *
 * Everything here maps 1:1 onto `MetadataFilter`, which the API composes *inside*
 * `build_acl_filter` — these facets narrow what you can already see and can never
 * widen it, so an over-permissive filter here is still safe.
 */

import { FilterX } from 'lucide-react';
import { useCallback, useState, type ReactNode } from 'react';

import { CLASSIFICATIONS, type Classification } from '../api/types';
import { useSettingsStore } from '../store/settings';

/** Source types recognised by the ingestion layer (`SourceType`). */
const SOURCE_TYPES = ['blob', 'local', 'sharepoint', 'http', 'sql', 'upload'] as const;

function parseList(text: string): string[] | null {
  const items = text
    .split(',')
    .map((item) => item.trim())
    .filter((item) => item.length > 0);
  return items.length > 0 ? items : null;
}

function joinList(items: string[] | null | undefined): string {
  return (items ?? []).join(', ');
}

function toDateInput(value: string | null | undefined): string {
  if (!value) return '';
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return '';
  return date.toISOString().slice(0, 10);
}

function fromDateInput(value: string, endOfDay: boolean): string | null {
  if (!value) return null;
  const suffix = endOfDay ? 'T23:59:59.999Z' : 'T00:00:00.000Z';
  const parsed = new Date(`${value}${suffix}`);
  return Number.isNaN(parsed.getTime()) ? null : parsed.toISOString();
}

/**
 * Render the facet filter bar.
 *
 * @returns The filter bar.
 */
export default function FilterBar(): ReactNode {
  const filters = useSettingsStore((state) => state.filters);
  const patchFilters = useSettingsStore((state) => state.patchFilters);
  const clearFilters = useSettingsStore((state) => state.clearFilters);

  const [docTypes, setDocTypes] = useState(() => joinList(filters.doc_types));
  const [tags, setTags] = useState(() => joinList(filters.tags));
  const [authors, setAuthors] = useState(() => joinList(filters.authors));
  const [languages, setLanguages] = useState(() => joinList(filters.languages));
  const [section, setSection] = useState(() => filters.section_prefix ?? '');

  const commitText = useCallback(() => {
    patchFilters({
      doc_types: parseList(docTypes),
      tags: parseList(tags),
      authors: parseList(authors),
      languages: parseList(languages),
      section_prefix: section.trim() || null,
    });
  }, [patchFilters, docTypes, tags, authors, languages, section]);

  const toggleSourceType = useCallback(
    (value: string) => {
      const current = new Set(filters.source_types ?? []);
      if (current.has(value)) current.delete(value);
      else current.add(value);
      patchFilters({ source_types: current.size > 0 ? [...current] : null });
    },
    [filters.source_types, patchFilters],
  );

  const reset = useCallback(() => {
    setDocTypes('');
    setTags('');
    setAuthors('');
    setLanguages('');
    setSection('');
    clearFilters();
  }, [clearFilters]);

  return (
    <section
      id="filter-bar"
      aria-label="Metadata filters"
      className="shrink-0 border-t border-slate-200 bg-slate-50 px-4 py-3 dark:border-slate-800 dark:bg-slate-900/60"
    >
      <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
        <div>
          <label className="label" htmlFor="filter-doc-types">
            Doc types
          </label>
          <input
            id="filter-doc-types"
            className="input mt-1"
            value={docTypes}
            placeholder="policy, contract"
            onChange={(event) => setDocTypes(event.target.value)}
            onBlur={commitText}
          />
        </div>

        <div>
          <label className="label" htmlFor="filter-tags">
            Tags
          </label>
          <input
            id="filter-tags"
            className="input mt-1"
            value={tags}
            placeholder="hr, 2026"
            onChange={(event) => setTags(event.target.value)}
            onBlur={commitText}
          />
        </div>

        <div>
          <label className="label" htmlFor="filter-authors">
            Authors
          </label>
          <input
            id="filter-authors"
            className="input mt-1"
            value={authors}
            placeholder="A. Nolan"
            onChange={(event) => setAuthors(event.target.value)}
            onBlur={commitText}
          />
        </div>

        <div>
          <label className="label" htmlFor="filter-languages">
            Languages
          </label>
          <input
            id="filter-languages"
            className="input mt-1"
            value={languages}
            placeholder="en, de"
            onChange={(event) => setLanguages(event.target.value)}
            onBlur={commitText}
          />
        </div>

        <div>
          <label className="label" htmlFor="filter-section">
            Section starts with
          </label>
          <input
            id="filter-section"
            className="input mt-1"
            value={section}
            placeholder="Chapter 4"
            onChange={(event) => setSection(event.target.value)}
            onBlur={commitText}
          />
        </div>

        <div>
          <label className="label" htmlFor="filter-classification">
            Max classification
          </label>
          <select
            id="filter-classification"
            className="select mt-1"
            value={filters.max_classification ?? ''}
            onChange={(event) =>
              patchFilters({
                max_classification: event.target.value
                  ? (event.target.value as Classification)
                  : null,
              })
            }
          >
            <option value="">Your clearance ceiling</option>
            {CLASSIFICATIONS.map((level) => (
              <option key={level} value={level}>
                {level}
              </option>
            ))}
          </select>
        </div>

        <div className="sm:col-span-2 lg:col-span-1">
          <span className="label" id="filter-source-types-label">
            Source types
          </span>
          <div
            role="group"
            aria-labelledby="filter-source-types-label"
            className="mt-1 flex flex-wrap gap-1"
          >
            {SOURCE_TYPES.map((type) => {
              const active = (filters.source_types ?? []).includes(type);
              return (
                <button
                  key={type}
                  type="button"
                  className={`btn btn-xs ${active ? '' : 'btn-ghost'}`}
                  aria-pressed={active}
                  onClick={() => toggleSourceType(type)}
                >
                  {type}
                </button>
              );
            })}
          </div>
        </div>

        <div className="grid grid-cols-2 gap-2">
          <div>
            <label className="label" htmlFor="filter-date-from">
              Modified from
            </label>
            <input
              id="filter-date-from"
              type="date"
              className="input mt-1"
              value={toDateInput(filters.date_from)}
              onChange={(event) =>
                patchFilters({ date_from: fromDateInput(event.target.value, false) })
              }
            />
          </div>
          <div>
            <label className="label" htmlFor="filter-date-to">
              Modified to
            </label>
            <input
              id="filter-date-to"
              type="date"
              className="input mt-1"
              value={toDateInput(filters.date_to)}
              onChange={(event) =>
                patchFilters({ date_to: fromDateInput(event.target.value, true) })
              }
            />
          </div>
        </div>

        <div className="flex items-end justify-between gap-2">
          <label className="flex items-center gap-2 text-sm" htmlFor="filter-exclude-pii">
            <input
              id="filter-exclude-pii"
              type="checkbox"
              className="h-4 w-4 rounded border-slate-300"
              checked={filters.exclude_pii === true}
              onChange={(event) => patchFilters({ exclude_pii: event.target.checked })}
            />
            Exclude chunks with PII
          </label>
          <button type="button" className="btn btn-xs" onClick={reset}>
            <FilterX aria-hidden="true" className="h-3 w-3" />
            Clear
          </button>
        </div>
      </div>

      <p className="mt-2 text-[11px] text-slate-500 dark:text-slate-400">
        Facets narrow the ACL filter; they never widen it. A blank field means “no
        constraint”, and lists are comma separated.
      </p>
    </section>
  );
}
