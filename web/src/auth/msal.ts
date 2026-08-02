/**
 * Microsoft Entra ID (MSAL) wiring.
 *
 * The app is a public client using the **authorization-code flow with PKCE** —
 * `@azure/msal-browser` v4 does this by default for a SPA app registration and never
 * needs a client secret. Access tokens for the API are acquired silently from the
 * MSAL cache and fall back to an interactive prompt only when the refresh token is
 * no longer usable.
 *
 * Configuration comes exclusively from `VITE_*` variables (see `web/.env.example`).
 * When `VITE_DEV_MODE` is on, or no client id is configured, MSAL is skipped entirely
 * and the API is called with the unsigned dev-principal header — the browser-side
 * mirror of ragcore's `entra_dev_mode`, which the API refuses in production.
 */

import {
  BrowserCacheLocation,
  InteractionRequiredAuthError,
  LogLevel,
  PublicClientApplication,
} from '@azure/msal-browser';
import type {
  AccountInfo,
  AuthenticationResult,
  Configuration,
  PopupRequest,
  RedirectRequest,
  SilentRequest,
} from '@azure/msal-browser';

/** Resolved Entra configuration. */
export interface AuthEnv {
  clientId: string;
  tenantId: string;
  authorityHost: string;
  apiScope: string;
  redirectUri: string;
  postLogoutRedirectUri: string;
  cacheLocation: BrowserCacheLocation;
  storeAuthStateInCookie: boolean;
  devMode: boolean;
}

function readBoolean(raw: string | undefined, fallback: boolean): boolean {
  if (raw === undefined || raw === '') return fallback;
  return raw === 'true' || raw === '1' || raw === 'yes';
}

const CACHE_LOCATIONS: Record<string, BrowserCacheLocation> = {
  localStorage: BrowserCacheLocation.LocalStorage,
  sessionStorage: BrowserCacheLocation.SessionStorage,
};

const origin = typeof window === 'undefined' ? '' : window.location.origin;

/** Entra configuration read from the environment. */
export const authEnv: AuthEnv = {
  clientId: import.meta.env.VITE_ENTRA_CLIENT_ID ?? '',
  tenantId: import.meta.env.VITE_ENTRA_TENANT_ID ?? '',
  authorityHost:
    import.meta.env.VITE_ENTRA_AUTHORITY_HOST ?? 'https://login.microsoftonline.com',
  apiScope: import.meta.env.VITE_ENTRA_API_SCOPE ?? '',
  redirectUri: import.meta.env.VITE_ENTRA_REDIRECT_URI ?? origin,
  postLogoutRedirectUri: import.meta.env.VITE_ENTRA_POST_LOGOUT_REDIRECT_URI ?? origin,
  cacheLocation:
    CACHE_LOCATIONS[import.meta.env.VITE_ENTRA_CACHE_LOCATION ?? 'sessionStorage'] ??
    BrowserCacheLocation.SessionStorage,
  storeAuthStateInCookie: readBoolean(
    import.meta.env.VITE_ENTRA_STORE_AUTH_STATE_IN_COOKIE,
    false,
  ),
  devMode: readBoolean(import.meta.env.VITE_DEV_MODE, false),
};

/**
 * Whether real Entra sign-in is active.
 *
 * Dev mode wins: with `VITE_DEV_MODE=true` the app never contacts Entra, so a
 * developer can run the stack without an app registration.
 */
export const entraConfigured: boolean =
  !authEnv.devMode && Boolean(authEnv.clientId && authEnv.tenantId);

/** Scopes needed for an API access token, plus the OIDC basics. */
export const apiScopes: string[] = authEnv.apiScope ? [authEnv.apiScope] : [];

/** Interactive sign-in request. */
export const loginRequest: RedirectRequest = {
  scopes: ['openid', 'profile', 'offline_access', ...apiScopes],
};

const msalConfig: Configuration = {
  auth: {
    clientId: authEnv.clientId,
    authority: `${authEnv.authorityHost}/${authEnv.tenantId}`,
    redirectUri: authEnv.redirectUri,
    postLogoutRedirectUri: authEnv.postLogoutRedirectUri,
    navigateToLoginRequestUrl: false,
  },
  cache: {
    cacheLocation: authEnv.cacheLocation,
    storeAuthStateInCookie: authEnv.storeAuthStateInCookie,
  },
  system: {
    loggerOptions: {
      // Never log PII: MSAL messages can contain tokens and user identifiers.
      piiLoggingEnabled: false,
      logLevel: LogLevel.Warning,
      loggerCallback: (level, message, containsPii) => {
        if (containsPii) return;
        if (level === LogLevel.Error) console.error(`[msal] ${message}`);
        else if (level === LogLevel.Warning) console.warn(`[msal] ${message}`);
      },
    },
  },
};

let instance: PublicClientApplication | null = null;
let initialization: Promise<PublicClientApplication | null> | null = null;

/**
 * Return the MSAL instance, or null when Entra is not configured.
 *
 * @returns The singleton instance, or null in dev mode.
 */
export function getMsalInstance(): PublicClientApplication | null {
  if (!entraConfigured) return null;
  if (!instance) instance = new PublicClientApplication(msalConfig);
  return instance;
}

/**
 * Initialise MSAL and complete any pending redirect.
 *
 * Must be awaited before the first `acquireToken*` call — MSAL v4 requires an
 * explicit `initialize()`.
 *
 * @returns The initialised instance, or null when Entra is not configured.
 */
export function initializeMsal(): Promise<PublicClientApplication | null> {
  if (!entraConfigured) return Promise.resolve(null);
  if (initialization) return initialization;
  initialization = (async () => {
    const pca = getMsalInstance();
    if (!pca) return null;
    await pca.initialize();
    const result = await pca.handleRedirectPromise();
    if (result?.account) {
      pca.setActiveAccount(result.account);
    } else if (!pca.getActiveAccount()) {
      const accounts = pca.getAllAccounts();
      if (accounts.length > 0 && accounts[0]) pca.setActiveAccount(accounts[0]);
    }
    return pca;
  })();
  return initialization;
}

/**
 * The signed-in account, if any.
 *
 * @returns The active account, or null.
 */
export function getActiveAccount(): AccountInfo | null {
  const pca = getMsalInstance();
  if (!pca) return null;
  const active = pca.getActiveAccount();
  if (active) return active;
  const accounts = pca.getAllAccounts();
  return accounts.length > 0 ? (accounts[0] ?? null) : null;
}

/** Claims the app reads from the id token. Never rendered as authorization truth. */
export interface TokenClaims {
  oid?: string;
  tid?: string;
  roles?: string[];
  groups?: string[];
  name?: string;
  preferred_username?: string;
}

/**
 * Extract display claims from an account.
 *
 * These drive the UI only — authorization is decided server-side from the validated
 * access token, and `GET /me` is the single source of truth for the principal.
 *
 * @param account The MSAL account, or null.
 * @returns The claims, or null.
 */
export function claimsFromAccount(account: AccountInfo | null): TokenClaims | null {
  if (!account?.idTokenClaims) return null;
  const claims = account.idTokenClaims as Record<string, unknown>;
  const asStrings = (value: unknown): string[] | undefined => {
    if (!Array.isArray(value)) return undefined;
    return value.filter((item): item is string => typeof item === 'string');
  };
  return {
    oid: typeof claims.oid === 'string' ? claims.oid : undefined,
    tid: typeof claims.tid === 'string' ? claims.tid : undefined,
    roles: asStrings(claims.roles),
    groups: asStrings(claims.groups),
    name: typeof claims.name === 'string' ? claims.name : undefined,
    preferred_username:
      typeof claims.preferred_username === 'string'
        ? claims.preferred_username
        : undefined,
  };
}

/**
 * Acquire an API access token.
 *
 * Order of attempts: silent from cache, silent with `forceRefresh`, interactive
 * popup, interactive redirect. Returns null in dev mode so the client falls back to
 * the dev-principal header.
 *
 * @param options `forceRefresh` bypasses the cached access token.
 * @returns The bearer token, or null when Entra is not configured.
 * @throws Error When interactive sign-in is required but cannot be started.
 */
export async function acquireApiToken(
  options: { forceRefresh?: boolean } = {},
): Promise<string | null> {
  if (!entraConfigured) return null;
  const pca = await initializeMsal();
  if (!pca) return null;

  const account = getActiveAccount();
  if (!account) {
    await pca.loginRedirect(loginRequest);
    return null;
  }

  const silent: SilentRequest = {
    account,
    scopes: apiScopes.length > 0 ? apiScopes : loginRequest.scopes,
    forceRefresh: options.forceRefresh === true,
  };

  try {
    const result: AuthenticationResult = await pca.acquireTokenSilent(silent);
    return result.accessToken;
  } catch (error) {
    if (!(error instanceof InteractionRequiredAuthError)) {
      // Network or configuration problem: surface it rather than looping on a prompt.
      throw error;
    }
  }

  const popup: PopupRequest = { account, scopes: silent.scopes };
  try {
    const result = await pca.acquireTokenPopup(popup);
    if (result.account) pca.setActiveAccount(result.account);
    return result.accessToken;
  } catch {
    // Popups are commonly blocked; a full-page redirect always works.
    await pca.acquireTokenRedirect({
      account,
      scopes: silent.scopes,
      redirectStartPage: window.location.href,
    });
    return null;
  }
}

/**
 * Start interactive sign-in.
 *
 * @returns A promise that settles when the redirect has been initiated.
 */
export async function login(): Promise<void> {
  const pca = await initializeMsal();
  if (!pca) return;
  await pca.loginRedirect(loginRequest);
}

/**
 * Sign out and clear the local token cache.
 *
 * @returns A promise that settles when the redirect has been initiated.
 */
export async function logout(): Promise<void> {
  const pca = await initializeMsal();
  if (!pca) return;
  const account = getActiveAccount();
  await pca.logoutRedirect({
    account: account ?? undefined,
    postLogoutRedirectUri: authEnv.postLogoutRedirectUri,
  });
}
