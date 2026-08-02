import react from '@vitejs/plugin-react';
import { defineConfig, loadEnv } from 'vite';

/**
 * Vite configuration.
 *
 * Every tunable is read from the environment (see `.env.example`) rather than being
 * hard-coded: the dev port, the dev host and the API proxy target all come from
 * `VITE_*` variables so the same build works locally, in docker-compose and in Azure
 * Static Web Apps.
 *
 * When `VITE_API_BASE_URL` is empty the app issues same-origin requests and the dev
 * server proxies `/api` to `VITE_API_PROXY_TARGET`. Setting `VITE_API_BASE_URL`
 * (as docker-compose does) bypasses the proxy and talks to the API directly, which
 * requires the API's `RAG_API_CORS_ORIGINS` to include this origin.
 */
export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), '');
  const proxyTarget =
    env.VITE_API_PROXY_TARGET || env.VITE_API_BASE_URL || 'http://localhost:8000';
  const port = Number(env.VITE_DEV_PORT || '5173');

  return {
    plugins: [react()],
    server: {
      host: env.VITE_DEV_HOST || 'localhost',
      port,
      strictPort: true,
      proxy: {
        '/api': {
          target: proxyTarget,
          changeOrigin: true,
          ws: false,
          // Server-sent events must not be buffered by the dev proxy.
          configure: (proxy) => {
            proxy.on('proxyReq', (proxyReq) => {
              proxyReq.setHeader('Accept-Encoding', 'identity');
            });
          },
        },
      },
    },
    preview: { port, strictPort: true },
    build: {
      outDir: 'dist',
      target: 'es2022',
      sourcemap: mode !== 'production',
      chunkSizeWarningLimit: 900,
    },
  };
});
