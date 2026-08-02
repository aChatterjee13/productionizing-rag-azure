/**
 * Shared access to `GET /api/v1/me`.
 *
 * The `Principal` the API resolved from the validated token is the only authorization
 * truth the UI uses: role gating (`rag.admin`) and the clearance badge both read it,
 * never the raw id-token claims, which the browser could tamper with.
 *
 * A single module-level store backs every consumer via `useSyncExternalStore`, so a
 * dozen components asking for the principal produce one request.
 */

import { useCallback, useSyncExternalStore } from 'react';

import { ApiError, getMe } from '../api/client';
import { ADMIN_ROLE, CLASSIFICATIONS } from '../api/types';
import type { Classification, Principal } from '../api/types';

/** Shared `/me` state. */
export interface PrincipalSnapshot {
  principal: Principal | null;
  loading: boolean;
  error: string | null;
}

let snapshot: PrincipalSnapshot = { principal: null, loading: false, error: null };
let inFlight: Promise<void> | null = null;
const listeners = new Set<() => void>();

function emit(next: PrincipalSnapshot): void {
  snapshot = next;
  for (const listener of listeners) listener();
}

function subscribe(listener: () => void): () => void {
  listeners.add(listener);
  if (!snapshot.principal && !snapshot.loading && !snapshot.error) void load();
  return () => {
    listeners.delete(listener);
  };
}

function getSnapshot(): PrincipalSnapshot {
  return snapshot;
}

async function load(): Promise<void> {
  if (inFlight) return inFlight;
  emit({ ...snapshot, loading: true, error: null });
  inFlight = (async () => {
    try {
      const principal = await getMe();
      emit({ principal, loading: false, error: null });
    } catch (cause) {
      const message =
        cause instanceof ApiError
          ? `Could not resolve your identity (${cause.status || 'network'}): ${cause.message}`
          : 'Could not resolve your identity.';
      emit({ principal: null, loading: false, error: message });
    } finally {
      inFlight = null;
    }
  })();
  return inFlight;
}

/** What {@link usePrincipal} returns. */
export interface PrincipalState extends PrincipalSnapshot {
  /** True when the principal carries the `rag.admin` app role. */
  isAdmin: boolean;
  /** Numeric clearance rank, 0 (public) to 3 (restricted). */
  clearanceRank: number;
  /** Re-fetch `/me`, e.g. after a tenant switch. */
  reload: () => void;
}

/**
 * Resolve the calling principal.
 *
 * @returns The principal, loading and error state, plus role/clearance helpers.
 */
export function usePrincipal(): PrincipalState {
  const state = useSyncExternalStore(subscribe, getSnapshot, getSnapshot);
  const reload = useCallback(() => {
    void load();
  }, []);
  const principal = state.principal;
  return {
    ...state,
    isAdmin: principal ? principal.roles.includes(ADMIN_ROLE) : false,
    clearanceRank: principal ? classificationRank(principal.max_classification) : 0,
    reload,
  };
}

/**
 * Rank of a classification label, mirroring `Classification.rank`.
 *
 * @param value The label.
 * @returns 0 for public through 3 for restricted; 1 (internal) for unknown values.
 */
export function classificationRank(value: Classification | string): number {
  const index = CLASSIFICATIONS.indexOf(value as Classification);
  return index === -1 ? 1 : index;
}
