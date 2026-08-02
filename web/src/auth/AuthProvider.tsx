/**
 * Authentication boundary for the whole app.
 *
 * Initialises MSAL, wires {@link setTokenProvider} so the plain-`fetch` API client can
 * attach a bearer token without reaching into React context, and renders a sign-in
 * screen until an account is present. In dev mode it short-circuits: no MSAL, no
 * sign-in gate, and the client sends the unsigned dev-principal header instead.
 */

import { MsalProvider } from '@azure/msal-react';
import type { AccountInfo, PublicClientApplication } from '@azure/msal-browser';
import { EventType } from '@azure/msal-browser';
import { LogIn, LogOut, ShieldAlert, ShieldCheck } from 'lucide-react';
import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useState,
  type ReactNode,
} from 'react';

import { setAuthFailureHandler, setTokenProvider } from '../api/client';
import {
  acquireApiToken,
  authEnv,
  claimsFromAccount,
  entraConfigured,
  getActiveAccount,
  initializeMsal,
  login as msalLogin,
  logout as msalLogout,
  type TokenClaims,
} from './msal';

/** What {@link useAuth} exposes. */
export interface AuthState {
  /** True once MSAL has initialised (always true in dev mode). */
  ready: boolean;
  /** True when a token can be obtained, or dev mode is on. */
  authenticated: boolean;
  /** True when Entra is bypassed and the dev-principal header is used. */
  devMode: boolean;
  /** The MSAL account, or null. */
  account: AccountInfo | null;
  /** Display claims from the id token, or null. */
  claims: TokenClaims | null;
  /** Initialisation or sign-in error message, already safe to render. */
  error: string | null;
  /** Start interactive sign-in. */
  login: () => Promise<void>;
  /** Sign out and clear the token cache. */
  logout: () => Promise<void>;
}

const AuthContext = createContext<AuthState | null>(null);

/**
 * Read the ambient authentication state.
 *
 * @returns The current {@link AuthState}.
 * @throws Error When called outside {@link AuthProvider}.
 */
export function useAuth(): AuthState {
  const value = useContext(AuthContext);
  if (!value) throw new Error('useAuth must be used inside <AuthProvider>');
  return value;
}

function SignInScreen({
  onLogin,
  error,
}: {
  onLogin: () => void;
  error: string | null;
}): ReactNode {
  return (
    <main className="flex min-h-full items-center justify-center bg-slate-100 p-6 dark:bg-slate-950">
      <div className="panel w-full max-w-md p-8 text-center">
        <ShieldCheck
          aria-hidden="true"
          className="mx-auto h-10 w-10 text-brand-600 dark:text-brand-400"
        />
        <h1 className="mt-4 text-xl font-semibold text-slate-900 dark:text-slate-100">
          Productionizing RAG
        </h1>
        <p className="mt-2 text-sm text-slate-600 dark:text-slate-400">
          Sign in with your Microsoft Entra ID account. Your tenant, roles and groups
          come from the token and decide which documents you can see.
        </p>
        {error ? (
          <p
            role="alert"
            className="mt-4 flex items-start gap-2 rounded-md bg-rose-50 p-3 text-left text-sm text-rose-700 dark:bg-rose-950/60 dark:text-rose-300"
          >
            <ShieldAlert aria-hidden="true" className="mt-0.5 h-4 w-4 shrink-0" />
            <span>{error}</span>
          </p>
        ) : null}
        <button type="button" className="btn btn-primary mt-6 w-full" onClick={onLogin}>
          <LogIn aria-hidden="true" className="h-4 w-4" />
          Sign in with Microsoft
        </button>
      </div>
    </main>
  );
}

function LoadingScreen(): ReactNode {
  return (
    <main
      className="flex min-h-full items-center justify-center bg-slate-100 dark:bg-slate-950"
      aria-busy="true"
    >
      <p className="text-sm text-slate-600 dark:text-slate-400">Signing you in…</p>
    </main>
  );
}

/**
 * Provide authentication state to the tree.
 *
 * @param props.children The application.
 * @returns The provider, gated on a signed-in account outside dev mode.
 */
export function AuthProvider({ children }: { children: ReactNode }): ReactNode {
  const [instance, setInstance] = useState<PublicClientApplication | null>(null);
  const [ready, setReady] = useState<boolean>(!entraConfigured);
  const [account, setAccount] = useState<AccountInfo | null>(null);
  const [error, setError] = useState<string | null>(null);

  // One token provider for the whole app, registered before any request is issued.
  useEffect(() => {
    setTokenProvider(async (options) => acquireApiToken(options ?? {}));
    setAuthFailureHandler(() => {
      setError('Your session expired. Please sign in again.');
      setAccount(null);
    });
  }, []);

  useEffect(() => {
    if (!entraConfigured) {
      setReady(true);
      return;
    }
    let cancelled = false;
    initializeMsal()
      .then((pca) => {
        if (cancelled) return;
        setInstance(pca);
        setAccount(getActiveAccount());
        setReady(true);
      })
      .catch((cause: unknown) => {
        if (cancelled) return;
        setError(cause instanceof Error ? cause.message : 'Sign-in failed');
        setReady(true);
      });
    return () => {
      cancelled = true;
    };
  }, []);

  // Keep the rendered account in step with MSAL's own lifecycle events.
  useEffect(() => {
    if (!instance) return;
    const callbackId = instance.addEventCallback((message) => {
      if (
        message.eventType === EventType.LOGIN_SUCCESS ||
        message.eventType === EventType.ACQUIRE_TOKEN_SUCCESS ||
        message.eventType === EventType.SSO_SILENT_SUCCESS
      ) {
        setAccount(getActiveAccount());
        setError(null);
      } else if (message.eventType === EventType.LOGOUT_SUCCESS) {
        setAccount(null);
      }
    });
    return () => {
      if (callbackId) instance.removeEventCallback(callbackId);
    };
  }, [instance]);

  const login = useCallback(async () => {
    try {
      await msalLogin();
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : 'Sign-in failed');
    }
  }, []);

  const logout = useCallback(async () => {
    try {
      await msalLogout();
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : 'Sign-out failed');
    }
  }, []);

  const value = useMemo<AuthState>(
    () => ({
      ready,
      authenticated: !entraConfigured || account !== null,
      devMode: !entraConfigured,
      account,
      claims: claimsFromAccount(account),
      error,
      login,
      logout,
    }),
    [ready, account, error, login, logout],
  );

  let content: ReactNode;
  if (!ready) content = <LoadingScreen />;
  else if (!value.authenticated) {
    content = <SignInScreen onLogin={() => void login()} error={error} />;
  } else content = children;

  const tree = <AuthContext.Provider value={value}>{content}</AuthContext.Provider>;
  if (!instance) return tree;
  return <MsalProvider instance={instance}>{tree}</MsalProvider>;
}

/**
 * Sign-out button, rendered in the header.
 *
 * Hidden in dev mode, where there is no session to end.
 *
 * @returns The button, or null in dev mode.
 */
export function SignOutButton(): ReactNode {
  const { devMode, logout } = useAuth();
  if (devMode) return null;
  return (
    <button
      type="button"
      className="btn btn-ghost"
      onClick={() => void logout()}
      title={`Sign out of ${authEnv.tenantId || 'your tenant'}`}
    >
      <LogOut aria-hidden="true" className="h-4 w-4" />
      <span className="hidden sm:inline">Sign out</span>
    </button>
  );
}
