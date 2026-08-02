/**
 * Persisted UI preferences: theme, active view, metadata filters and panel toggles.
 *
 * The metadata filter lives here rather than in the chat store because it survives a
 * session switch and is posted verbatim as `ChatRequest.filters` / `SearchRequest`.
 */

import { create } from 'zustand';
import { persist } from 'zustand/middleware';

import type { Classification, MetadataFilter } from '../api/types';

/** Theme selection; `system` follows `prefers-color-scheme`. */
export type ThemeMode = 'light' | 'dark' | 'system';

/** Top-level views. Admin views are additionally gated on the `rag.admin` role. */
export type ViewKey = 'chat' | 'memory' | 'documents' | 'ingestion' | 'eval';

/** localStorage key; the pre-paint script in `index.html` reads the same key. */
const STORAGE_KEY = 'rag.settings.v1';

function readRatio(raw: string | undefined, fallback: number): number {
  const value = Number(raw);
  return Number.isFinite(value) && value > 0 && value <= 1 ? value : fallback;
}

/**
 * Budget utilisation at which the context meter warns about imminent compaction.
 *
 * Mirrors ragcore's `context_compact_at_ratio` (default 0.75). Set
 * `VITE_CONTEXT_WARN_RATIO` to keep the two in step rather than hard-coding a
 * threshold in the meter.
 */
export const CONTEXT_WARN_RATIO = readRatio(
  import.meta.env.VITE_CONTEXT_WARN_RATIO,
  0.75,
);

/** An empty filter: every facet unset. */
export const EMPTY_FILTER: MetadataFilter = {
  doc_types: null,
  source_types: null,
  tags: null,
  authors: null,
  languages: null,
  document_ids: null,
  section_prefix: null,
  date_from: null,
  date_to: null,
  max_classification: null,
  exclude_pii: false,
};

/**
 * Whether a filter constrains anything — mirrors `MetadataFilter.is_empty`.
 *
 * @param filter The filter to test.
 * @returns True when no facet is set.
 */
export function isFilterEmpty(filter: MetadataFilter): boolean {
  return countActiveFacets(filter) === 0;
}

/**
 * Count the active facets in a filter, for the badge on the filter bar.
 *
 * @param filter The filter to inspect.
 * @returns Number of constrained facets.
 */
export function countActiveFacets(filter: MetadataFilter): number {
  let count = 0;
  const lists: Array<string[] | null | undefined> = [
    filter.doc_types,
    filter.source_types,
    filter.tags,
    filter.authors,
    filter.languages,
    filter.document_ids,
  ];
  for (const list of lists) if (list && list.length > 0) count += 1;
  if (filter.section_prefix) count += 1;
  if (filter.date_from) count += 1;
  if (filter.date_to) count += 1;
  if (filter.max_classification) count += 1;
  if (filter.exclude_pii) count += 1;
  return count;
}

/**
 * Drop null/empty facets so the request body stays close to the pydantic default.
 *
 * @param filter The UI filter.
 * @returns A filter safe to post, or null when nothing is constrained.
 */
export function filterForRequest(filter: MetadataFilter): MetadataFilter | null {
  if (isFilterEmpty(filter)) return null;
  const out: MetadataFilter = {};
  if (filter.doc_types?.length) out.doc_types = filter.doc_types;
  if (filter.source_types?.length) out.source_types = filter.source_types;
  if (filter.tags?.length) out.tags = filter.tags;
  if (filter.authors?.length) out.authors = filter.authors;
  if (filter.languages?.length) out.languages = filter.languages;
  if (filter.document_ids?.length) out.document_ids = filter.document_ids;
  if (filter.section_prefix) out.section_prefix = filter.section_prefix;
  if (filter.date_from) out.date_from = filter.date_from;
  if (filter.date_to) out.date_to = filter.date_to;
  if (filter.max_classification) out.max_classification = filter.max_classification;
  if (filter.exclude_pii) out.exclude_pii = true;
  return out;
}

/** Shape of the settings store. */
export interface SettingsState {
  theme: ThemeMode;
  view: ViewKey;
  filters: MetadataFilter;
  allowTools: boolean;
  sidebarOpen: boolean;
  showRetrieval: boolean;
  showContext: boolean;
  showTools: boolean;
  showFilters: boolean;
  setTheme: (theme: ThemeMode) => void;
  setView: (view: ViewKey) => void;
  setFilters: (filters: MetadataFilter) => void;
  patchFilters: (patch: Partial<MetadataFilter>) => void;
  clearFilters: () => void;
  setMaxClassification: (value: Classification | null) => void;
  setAllowTools: (allow: boolean) => void;
  toggleSidebar: () => void;
  togglePanel: (panel: 'showRetrieval' | 'showContext' | 'showTools') => void;
  toggleFilters: () => void;
}

/**
 * Apply a theme to the document root.
 *
 * @param mode The selected theme mode.
 */
export function applyTheme(mode: ThemeMode): void {
  if (typeof document === 'undefined') return;
  const dark =
    mode === 'dark' ||
    (mode === 'system' &&
      typeof window !== 'undefined' &&
      window.matchMedia('(prefers-color-scheme: dark)').matches);
  document.documentElement.classList.toggle('dark', dark);
}

/**
 * Persisted UI settings store.
 *
 * Rehydrated from `localStorage` under `rag.settings.v1`, the same key the pre-paint
 * script in `index.html` reads so the theme never flashes.
 */
export const useSettingsStore = create<SettingsState>()(
  persist(
    (set, get) => ({
      theme: 'system',
      view: 'chat',
      filters: { ...EMPTY_FILTER },
      allowTools: true,
      sidebarOpen: true,
      showRetrieval: true,
      showContext: true,
      showTools: true,
      showFilters: false,

      setTheme: (theme) => {
        applyTheme(theme);
        set({ theme });
      },
      setView: (view) => set({ view }),
      setFilters: (filters) => set({ filters }),
      patchFilters: (patch) => set({ filters: { ...get().filters, ...patch } }),
      clearFilters: () => set({ filters: { ...EMPTY_FILTER } }),
      setMaxClassification: (value) =>
        set({ filters: { ...get().filters, max_classification: value } }),
      setAllowTools: (allowTools) => set({ allowTools }),
      toggleSidebar: () => set({ sidebarOpen: !get().sidebarOpen }),
      togglePanel: (panel) => {
        const current = get();
        if (panel === 'showRetrieval') set({ showRetrieval: !current.showRetrieval });
        else if (panel === 'showContext') set({ showContext: !current.showContext });
        else set({ showTools: !current.showTools });
      },
      toggleFilters: () => set({ showFilters: !get().showFilters }),
    }),
    {
      name: STORAGE_KEY,
      version: 1,
      partialize: (state) => ({
        theme: state.theme,
        view: state.view,
        filters: state.filters,
        allowTools: state.allowTools,
        sidebarOpen: state.sidebarOpen,
        showRetrieval: state.showRetrieval,
        showContext: state.showContext,
        showTools: state.showTools,
        showFilters: state.showFilters,
      }),
      onRehydrateStorage: () => (state) => {
        applyTheme(state?.theme ?? 'system');
      },
    },
  ),
);

// Follow the OS theme while `system` is selected.
if (typeof window !== 'undefined' && window.matchMedia) {
  const query = window.matchMedia('(prefers-color-scheme: dark)');
  query.addEventListener('change', () => {
    if (useSettingsStore.getState().theme === 'system') applyTheme('system');
  });
}
