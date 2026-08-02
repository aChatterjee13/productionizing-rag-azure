/**
 * Long-term memory and profile management (requirement #2).
 *
 * Consent is the important control: turning it off skips stage 13 write-back entirely
 * and soft-deletes what is already stored, so the switch is presented as the primary
 * action rather than buried in a settings page.
 */

import {
  BrainCircuit,
  Loader2,
  RefreshCw,
  Save,
  ShieldOff,
  Sparkles,
  Trash2,
} from 'lucide-react';
import { useCallback, useEffect, useState, type ReactNode } from 'react';

import {
  ApiError,
  deleteMemory,
  getMemoryProfile,
  listMemories,
  setMemoryConsent,
  updateMemoryProfile,
} from '../api/client';
import type { LongTermMemory, UserProfile } from '../api/types';

const KIND_HINT: Record<string, string> = {
  preference: 'A standing instruction, e.g. “answer in bullet points”.',
  fact: 'A stable fact about you, e.g. “I work in the Munich office”.',
  entity: 'An entity you repeatedly ask about.',
  episode: 'A summary of a past resolved conversation.',
};

function formatDate(value: string | null | undefined): string {
  if (!value) return '—';
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? '—' : date.toLocaleString();
}

function errorText(cause: unknown, fallback: string): string {
  return cause instanceof ApiError ? cause.message : fallback;
}

/**
 * Render the memory view.
 *
 * @returns The panel.
 */
export default function MemoryPanel(): ReactNode {
  const [profile, setProfile] = useState<UserProfile | null>(null);
  const [memories, setMemories] = useState<LongTermMemory[]>([]);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [busyId, setBusyId] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const [style, setStyle] = useState('');
  const [language, setLanguage] = useState('');

  const load = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const [loadedProfile, loadedMemories] = await Promise.all([
        getMemoryProfile(),
        listMemories(),
      ]);
      setProfile(loadedProfile);
      setStyle(loadedProfile.preferred_style ?? '');
      setLanguage(loadedProfile.preferred_language ?? '');
      setMemories(loadedMemories);
    } catch (cause) {
      setError(errorText(cause, 'Could not load your memory.'));
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void load();
  }, [load]);

  const onToggleConsent = useCallback(async () => {
    if (!profile) return;
    const next = !profile.memory_consent;
    if (
      !next &&
      !window.confirm(
        'Turning long-term memory off deletes what is already stored and stops new ' +
          'memories being written. Continue?',
      )
    ) {
      return;
    }
    setSaving(true);
    try {
      const updated = await setMemoryConsent(next);
      setProfile(updated);
      setNotice(
        next
          ? 'Long-term memory is on. Durable preferences and facts will be stored.'
          : 'Long-term memory is off and stored memories were deleted.',
      );
      if (!next) setMemories([]);
      else await load();
      setError(null);
    } catch (cause) {
      setError(errorText(cause, 'Could not change your consent setting.'));
    } finally {
      setSaving(false);
    }
  }, [profile, load]);

  const onSaveProfile = useCallback(async () => {
    setSaving(true);
    setNotice(null);
    try {
      const updated = await updateMemoryProfile({
        preferred_style: style.trim() || null,
        preferred_language: language.trim() || null,
      });
      setProfile(updated);
      setNotice('Profile saved.');
      setError(null);
    } catch (cause) {
      if (cause instanceof ApiError && cause.isUnavailableRoute) {
        setError(
          'This API build does not expose PUT /memory/profile, so preferences can ' +
            'only be learned from conversation.',
        );
      } else {
        setError(errorText(cause, 'Could not save your profile.'));
      }
    } finally {
      setSaving(false);
    }
  }, [style, language]);

  const onDelete = useCallback(async (memoryId: string) => {
    setBusyId(memoryId);
    try {
      await deleteMemory(memoryId);
      setMemories((current) => current.filter((item) => item.memory_id !== memoryId));
      setError(null);
    } catch (cause) {
      setError(errorText(cause, 'Could not delete that memory.'));
    } finally {
      setBusyId(null);
    }
  }, []);

  return (
    <div className="scroll-area h-full p-4">
      <div className="mx-auto max-w-4xl space-y-4">
        <header className="flex items-center justify-between gap-2">
          <h1 className="flex items-center gap-2 text-lg font-semibold">
            <BrainCircuit aria-hidden="true" className="h-5 w-5 text-brand-600" />
            Personalisation and memory
          </h1>
          <button
            type="button"
            className="btn btn-xs"
            onClick={() => void load()}
            disabled={loading}
          >
            <RefreshCw aria-hidden="true" className="h-3 w-3" />
            Refresh
          </button>
        </header>

        {error ? (
          <p
            role="alert"
            className="rounded-lg border border-rose-200 bg-rose-50 p-3 text-sm text-rose-700 dark:border-rose-900 dark:bg-rose-950/50 dark:text-rose-300"
          >
            {error}
          </p>
        ) : null}
        {notice ? (
          <p
            role="status"
            className="rounded-lg border border-emerald-200 bg-emerald-50 p-3 text-sm text-emerald-800 dark:border-emerald-900 dark:bg-emerald-950/50 dark:text-emerald-300"
          >
            {notice}
          </p>
        ) : null}

        {loading ? (
          <p className="flex items-center gap-2 text-sm text-slate-500">
            <Loader2 aria-hidden="true" className="h-4 w-4 animate-spin" />
            Loading…
          </p>
        ) : null}

        <section aria-label="Consent" className="panel p-4">
          <div className="flex flex-wrap items-center justify-between gap-3">
            <div className="min-w-0">
              <h2 className="text-sm font-semibold">Long-term memory</h2>
              <p className="mt-1 text-sm text-slate-600 dark:text-slate-400">
                When on, durable preferences and facts are extracted after a turn and
                injected into later prompts. When off, nothing is stored and write-back
                is skipped entirely.
              </p>
            </div>
            <button
              type="button"
              className={`btn ${profile?.memory_consent ? 'btn-danger' : 'btn-primary'}`}
              disabled={saving || !profile}
              onClick={() => void onToggleConsent()}
              aria-pressed={profile?.memory_consent === true}
            >
              {profile?.memory_consent ? (
                <>
                  <ShieldOff aria-hidden="true" className="h-4 w-4" />
                  Turn memory off
                </>
              ) : (
                <>
                  <Sparkles aria-hidden="true" className="h-4 w-4" />
                  Turn memory on
                </>
              )}
            </button>
          </div>
        </section>

        <section aria-label="Profile" className="panel p-4">
          <h2 className="text-sm font-semibold">Profile</h2>
          <p className="mt-1 text-xs text-slate-500 dark:text-slate-400">
            The summary is maintained by the model; style and language are yours to set.
          </p>
          <div className="mt-3 grid gap-3 sm:grid-cols-2">
            <div>
              <label className="label" htmlFor="profile-style">
                Preferred answer style
              </label>
              <input
                id="profile-style"
                className="input mt-1"
                value={style}
                placeholder="concise bullet points"
                onChange={(event) => setStyle(event.target.value)}
              />
            </div>
            <div>
              <label className="label" htmlFor="profile-language">
                Preferred language (ISO 639-1)
              </label>
              <input
                id="profile-language"
                className="input mt-1"
                value={language}
                placeholder="en"
                maxLength={8}
                onChange={(event) => setLanguage(event.target.value)}
              />
            </div>
          </div>

          <div className="mt-3">
            <span className="label">Rolling summary</span>
            <p className="mt-1 whitespace-pre-wrap rounded-lg border border-slate-200 p-3 text-sm text-slate-600 dark:border-slate-800 dark:text-slate-400">
              {profile?.summary || 'No persona has been learned yet.'}
            </p>
          </div>

          {profile?.top_topics && profile.top_topics.length > 0 ? (
            <p className="mt-3 flex flex-wrap gap-1">
              {profile.top_topics.map((topic) => (
                <span key={topic} className="chip">
                  {topic}
                </span>
              ))}
            </p>
          ) : null}

          <div className="mt-3 flex items-center gap-2">
            <button
              type="button"
              className="btn btn-primary"
              disabled={saving || !profile}
              onClick={() => void onSaveProfile()}
            >
              <Save aria-hidden="true" className="h-4 w-4" />
              Save profile
            </button>
            <span className="text-xs text-slate-500">
              Updated {formatDate(profile?.updated_at)}
            </span>
          </div>
        </section>

        <section aria-label="Stored memories" className="panel">
          <div className="panel-header">
            <h2 className="text-sm font-semibold">Stored memories ({memories.length})</h2>
          </div>
          <div className="p-4">
            {memories.length === 0 ? (
              <p className="text-sm text-slate-500 dark:text-slate-400">
                Nothing stored yet.
              </p>
            ) : (
              <ul className="space-y-2">
                {memories.map((memory) => (
                  <li
                    key={memory.memory_id}
                    className="rounded-lg border border-slate-200 p-3 dark:border-slate-800"
                  >
                    <div className="flex items-start justify-between gap-3">
                      <div className="min-w-0">
                        <p className="flex flex-wrap items-center gap-2">
                          <span className="badge" title={KIND_HINT[memory.kind] ?? ''}>
                            {memory.kind}
                          </span>
                          <span className="chip" title="Salience, 0 to 1">
                            salience {memory.salience.toFixed(2)}
                          </span>
                          <span className="chip" title="Times injected into a prompt">
                            {memory.hit_count} hits
                          </span>
                          {memory.pii_redacted ? (
                            <span className="chip">PII redacted</span>
                          ) : null}
                        </p>
                        <p className="mt-1.5 text-sm">{memory.text}</p>
                        <p className="mt-1 text-[11px] text-slate-500 dark:text-slate-400">
                          created {formatDate(memory.created_at)} · last used{' '}
                          {formatDate(memory.last_used_at)}
                          {memory.expires_at
                            ? ` · expires ${formatDate(memory.expires_at)}`
                            : ''}
                        </p>
                      </div>
                      <button
                        type="button"
                        className="btn btn-danger btn-xs shrink-0"
                        disabled={busyId === memory.memory_id}
                        onClick={() => void onDelete(memory.memory_id)}
                        aria-label={`Delete memory: ${memory.text.slice(0, 40)}`}
                      >
                        {busyId === memory.memory_id ? (
                          <Loader2 aria-hidden="true" className="h-3 w-3 animate-spin" />
                        ) : (
                          <Trash2 aria-hidden="true" className="h-3 w-3" />
                        )}
                        Delete
                      </button>
                    </div>
                  </li>
                ))}
              </ul>
            )}
          </div>
        </section>
      </div>
    </div>
  );
}
