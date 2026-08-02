/**
 * Root component: resolves the principal, then routes between the five views.
 *
 * Routing is intentionally a single piece of persisted state rather than a router
 * dependency — the app has no deep-linkable sub-resources, and every view is driven
 * by the same session/principal context.
 */

import { AlertTriangle, Loader2 } from 'lucide-react';
import type { ReactNode } from 'react';

import AdminDocuments from './components/AdminDocuments';
import AdminIngestion from './components/AdminIngestion';
import ChatPanel from './components/ChatPanel';
import EvalDashboard from './components/EvalDashboard';
import Layout from './components/Layout';
import MemoryPanel from './components/MemoryPanel';
import { usePrincipal } from './hooks/usePrincipal';
import { useSettingsStore } from './store/settings';

function Centered({ children }: { children: ReactNode }): ReactNode {
  return (
    <div className="flex h-full items-center justify-center p-8 text-center">
      <div className="max-w-md">{children}</div>
    </div>
  );
}

/** Shown when a non-admin selects an admin view (defence in depth; the API also gates). */
function AdminOnly(): ReactNode {
  return (
    <Centered>
      <AlertTriangle
        aria-hidden="true"
        className="mx-auto h-8 w-8 text-amber-500"
      />
      <h2 className="mt-3 text-lg font-semibold">Administrator access required</h2>
      <p className="mt-1 text-sm text-slate-600 dark:text-slate-400">
        This view needs the <code className="font-mono">rag.admin</code> application
        role. Ask a directory administrator to assign it in Entra ID.
      </p>
    </Centered>
  );
}

/**
 * The application shell and view switch.
 *
 * @returns The rendered app.
 */
export default function App(): ReactNode {
  const view = useSettingsStore((state) => state.view);
  const { principal, loading, error, isAdmin } = usePrincipal();

  let content: ReactNode;
  if (loading && !principal) {
    content = (
      <Centered>
        <Loader2
          aria-hidden="true"
          className="mx-auto h-6 w-6 animate-spin text-brand-600"
        />
        <p className="mt-3 text-sm text-slate-600 dark:text-slate-400">
          Resolving your identity…
        </p>
      </Centered>
    );
  } else if (error) {
    content = (
      <Centered>
        <AlertTriangle aria-hidden="true" className="mx-auto h-8 w-8 text-rose-500" />
        <h2 className="mt-3 text-lg font-semibold">Cannot reach the API</h2>
        <p className="mt-1 text-sm text-slate-600 dark:text-slate-400">{error}</p>
      </Centered>
    );
  } else {
    switch (view) {
      case 'memory':
        content = <MemoryPanel />;
        break;
      case 'documents':
        content = isAdmin ? <AdminDocuments /> : <AdminOnly />;
        break;
      case 'ingestion':
        content = isAdmin ? <AdminIngestion /> : <AdminOnly />;
        break;
      case 'eval':
        content = isAdmin ? <EvalDashboard /> : <AdminOnly />;
        break;
      case 'chat':
      default:
        content = <ChatPanel />;
        break;
    }
  }

  return <Layout>{content}</Layout>;
}
