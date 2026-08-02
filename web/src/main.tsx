/**
 * Application entry point.
 *
 * MSAL is initialised inside `AuthProvider`; nothing here touches the network, so a
 * misconfigured Entra app registration still renders a usable error state.
 */

import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';

import App from './App';
import { AuthProvider } from './auth/AuthProvider';
import './index.css';
import { applyTheme, useSettingsStore } from './store/settings';

applyTheme(useSettingsStore.getState().theme);

const container = document.getElementById('root');
if (!container) {
  throw new Error('Root container #root is missing from index.html');
}

createRoot(container).render(
  <StrictMode>
    <AuthProvider>
      <App />
    </AuthProvider>
  </StrictMode>,
);
