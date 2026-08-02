/**
 * Application shell: skip link, header, view tabs, collapsible session sidebar.
 *
 * The header shows who the API thinks you are — tenant, roles and clearance — because
 * in a multi-tenant RAG system "why can't I see that document?" is answered by the
 * principal, not by the query.
 */

import {
  ClipboardCheck,
  Database,
  MessageSquare,
  Monitor,
  Moon,
  PanelLeftClose,
  PanelLeftOpen,
  ShieldCheck,
  Sparkles,
  Sun,
  User,
} from 'lucide-react';
import type { ReactNode } from 'react';

import { SignOutButton, useAuth } from '../auth/AuthProvider';
import { usePrincipal } from '../hooks/usePrincipal';
import { useSettingsStore, type ThemeMode, type ViewKey } from '../store/settings';
import SessionSidebar from './SessionSidebar';

interface NavItem {
  key: ViewKey;
  label: string;
  icon: typeof MessageSquare;
  adminOnly: boolean;
}

const NAV_ITEMS: NavItem[] = [
  { key: 'chat', label: 'Chat', icon: MessageSquare, adminOnly: false },
  { key: 'memory', label: 'Memory', icon: Sparkles, adminOnly: false },
  { key: 'documents', label: 'Documents', icon: Database, adminOnly: true },
  { key: 'ingestion', label: 'Ingestion', icon: Monitor, adminOnly: true },
  { key: 'eval', label: 'Evaluation', icon: ClipboardCheck, adminOnly: true },
];

const THEME_ORDER: ThemeMode[] = ['system', 'light', 'dark'];

function ThemeToggle(): ReactNode {
  const theme = useSettingsStore((state) => state.theme);
  const setTheme = useSettingsStore((state) => state.setTheme);
  const next = THEME_ORDER[(THEME_ORDER.indexOf(theme) + 1) % THEME_ORDER.length];
  const Icon = theme === 'dark' ? Moon : theme === 'light' ? Sun : Monitor;
  return (
    <button
      type="button"
      className="btn btn-ghost"
      onClick={() => setTheme(next)}
      aria-label={`Theme: ${theme}. Switch to ${next}.`}
      title={`Theme: ${theme} (click for ${next})`}
    >
      <Icon aria-hidden="true" className="h-4 w-4" />
    </button>
  );
}

function PrincipalChip(): ReactNode {
  const { principal, isAdmin } = usePrincipal();
  const { devMode } = useAuth();
  if (!principal) return null;
  return (
    <div className="hidden items-center gap-2 md:flex">
      {devMode ? (
        <span
          className="badge border-amber-300 bg-amber-100 text-amber-800 dark:border-amber-800 dark:bg-amber-950 dark:text-amber-300"
          title="Unsigned dev principal — never enable in production"
        >
          dev mode
        </span>
      ) : null}
      <span className="badge" title="Your clearance ceiling from the token">
        <ShieldCheck aria-hidden="true" className="h-3 w-3" />
        {principal.max_classification}
      </span>
      {isAdmin ? <span className="badge">rag.admin</span> : null}
      <span
        className="flex items-center gap-1.5 text-sm text-slate-600 dark:text-slate-300"
        title={`tenant ${principal.tenant_id} · user ${principal.user_id}`}
      >
        <User aria-hidden="true" className="h-4 w-4" />
        <span className="max-w-[16ch] truncate">
          {principal.display_name ?? principal.email ?? principal.user_id}
        </span>
      </span>
    </div>
  );
}

/**
 * Render the app shell around a view.
 *
 * @param props.children The active view.
 * @returns The shell.
 */
export default function Layout({ children }: { children: ReactNode }): ReactNode {
  const view = useSettingsStore((state) => state.view);
  const setView = useSettingsStore((state) => state.setView);
  const sidebarOpen = useSettingsStore((state) => state.sidebarOpen);
  const toggleSidebar = useSettingsStore((state) => state.toggleSidebar);
  const { isAdmin } = usePrincipal();

  return (
    <div className="flex h-full flex-col">
      <a
        href="#main-content"
        className="sr-only focus:not-sr-only focus:absolute focus:left-2 focus:top-2 focus:z-50 focus:rounded focus:bg-white focus:px-3 focus:py-2 focus:text-sm dark:focus:bg-slate-900"
      >
        Skip to main content
      </a>

      <header className="flex shrink-0 flex-wrap items-center gap-2 border-b border-slate-200 bg-white px-3 py-2 dark:border-slate-800 dark:bg-slate-900">
        <button
          type="button"
          className="btn btn-ghost"
          onClick={toggleSidebar}
          aria-expanded={sidebarOpen}
          aria-controls="session-sidebar"
          aria-label={sidebarOpen ? 'Hide sessions' : 'Show sessions'}
        >
          {sidebarOpen ? (
            <PanelLeftClose aria-hidden="true" className="h-4 w-4" />
          ) : (
            <PanelLeftOpen aria-hidden="true" className="h-4 w-4" />
          )}
        </button>

        <span className="mr-2 hidden text-sm font-semibold tracking-tight sm:inline">
          Productionizing RAG
        </span>

        <nav aria-label="Views">
          <ul role="tablist" className="flex flex-wrap items-center gap-1">
            {NAV_ITEMS.map((item) => {
              const Icon = item.icon;
              const selected = view === item.key;
              const locked = item.adminOnly && !isAdmin;
              return (
                <li key={item.key}>
                  <button
                    type="button"
                    role="tab"
                    aria-selected={selected}
                    aria-controls="main-content"
                    disabled={locked}
                    title={
                      locked ? 'Requires the rag.admin role' : `Open ${item.label}`
                    }
                    onClick={() => setView(item.key)}
                    className={`btn btn-xs ${
                      selected
                        ? 'border-brand-600 bg-brand-50 text-brand-700 dark:border-brand-500 dark:bg-brand-950 dark:text-brand-300'
                        : 'btn-ghost'
                    }`}
                  >
                    <Icon aria-hidden="true" className="h-3.5 w-3.5" />
                    {item.label}
                  </button>
                </li>
              );
            })}
          </ul>
        </nav>

        <div className="ml-auto flex items-center gap-2">
          <PrincipalChip />
          <ThemeToggle />
          <SignOutButton />
        </div>
      </header>

      <div className="flex min-h-0 flex-1">
        {sidebarOpen ? (
          <div
            id="session-sidebar"
            className="hidden w-72 shrink-0 border-r border-slate-200 bg-white md:block dark:border-slate-800 dark:bg-slate-900"
          >
            <SessionSidebar />
          </div>
        ) : null}
        <main id="main-content" className="min-w-0 flex-1 overflow-hidden">
          {children}
        </main>
      </div>
    </div>
  );
}
