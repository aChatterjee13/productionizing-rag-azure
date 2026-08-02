/// <reference types="vite/client" />

/** Typed view of the `VITE_*` variables documented in `web/.env.example`. */
interface ImportMetaEnv {
  readonly VITE_API_BASE_URL?: string;
  readonly VITE_API_PROXY_TARGET?: string;
  readonly VITE_API_PREFIX?: string;
  readonly VITE_API_TIMEOUT_MS?: string;
  readonly VITE_STREAM_IDLE_TIMEOUT_MS?: string;
  readonly VITE_STREAM_MAX_RETRIES?: string;
  readonly VITE_STREAM_RETRY_BASE_MS?: string;
  readonly VITE_PAGE_SIZE?: string;
  readonly VITE_CONTEXT_WARN_RATIO?: string;
  readonly VITE_DEV_HOST?: string;
  readonly VITE_DEV_PORT?: string;
  readonly VITE_ENTRA_CLIENT_ID?: string;
  readonly VITE_ENTRA_TENANT_ID?: string;
  readonly VITE_ENTRA_AUTHORITY_HOST?: string;
  readonly VITE_ENTRA_API_SCOPE?: string;
  readonly VITE_ENTRA_REDIRECT_URI?: string;
  readonly VITE_ENTRA_POST_LOGOUT_REDIRECT_URI?: string;
  readonly VITE_ENTRA_CACHE_LOCATION?: string;
  readonly VITE_ENTRA_STORE_AUTH_STATE_IN_COOKIE?: string;
  readonly VITE_DEV_MODE?: string;
  readonly VITE_DEV_PRINCIPAL_HEADER?: string;
  readonly VITE_DEV_PRINCIPAL?: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}
