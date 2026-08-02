/**
 * Guardrail banners: PII redaction, injection heuristics, out-of-domain refusals,
 * contradictions, classification and groundedness.
 *
 * Guardrail detail strings are already redacted server-side (`GuardrailEvent.detail`
 * never carries the offending content, and `entities` carries entity *type* names, not
 * values), so they are safe to render verbatim.
 */

import {
  AlertTriangle,
  Ban,
  HelpCircle,
  EyeOff,
  GitCompare,
  Lock,
  ShieldAlert,
  ShieldCheck,
  X,
} from 'lucide-react';
import { useState, type ReactNode } from 'react';

import type { GuardrailEvent } from '../api/types';
import { useChatStore } from '../store/chat';

type Tone = 'info' | 'warn' | 'danger' | 'ok';

const KIND_ICON: Record<string, typeof ShieldAlert> = {
  pii: EyeOff,
  injection: ShieldAlert,
  ood: HelpCircle,
  contradiction: GitCompare,
  classification: Lock,
  groundedness: AlertTriangle,
  size: AlertTriangle,
};

const KIND_TITLE: Record<string, string> = {
  pii: 'Personal data redacted',
  injection: 'Prompt-injection heuristic fired',
  ood: 'Outside the indexed corpus',
  contradiction: 'Sources disagree',
  classification: 'Classification guard',
  groundedness: 'Low groundedness',
  size: 'Input size capped',
};

function toneFor(event: GuardrailEvent): Tone {
  if (event.action === 'block') return 'danger';
  if (event.action === 'warn' || event.action === 'clarify') return 'warn';
  if (event.action === 'redact') return 'info';
  return 'ok';
}

const TONE_CLASS: Record<Tone, string> = {
  info: 'border-sky-200 bg-sky-50 text-sky-900 dark:border-sky-900 dark:bg-sky-950/50 dark:text-sky-200',
  warn: 'border-amber-200 bg-amber-50 text-amber-900 dark:border-amber-900 dark:bg-amber-950/50 dark:text-amber-200',
  danger:
    'border-rose-200 bg-rose-50 text-rose-900 dark:border-rose-900 dark:bg-rose-950/50 dark:text-rose-200',
  ok: 'border-emerald-200 bg-emerald-50 text-emerald-900 dark:border-emerald-900 dark:bg-emerald-950/50 dark:text-emerald-200',
};

/**
 * Render every guardrail decision for the current turn.
 *
 * `allow` events are collapsed into a single "checks passed" badge so the banner area
 * stays quiet when nothing fired, while still proving the checks ran.
 *
 * @returns The banner stack, or null when there is nothing to show.
 */
export default function GuardrailBanner(): ReactNode {
  const guardrails = useChatStore((state) => state.guardrails);
  const [dismissed, setDismissed] = useState<string[]>([]);

  if (guardrails.length === 0) return null;

  const notable = guardrails.filter((event) => event.action !== 'allow');
  const passed = guardrails.length - notable.length;

  const visible = notable.filter(
    (event, index) => !dismissed.includes(`${index}-${event.kind}-${event.stage}`),
  );

  if (visible.length === 0 && passed === 0) return null;

  return (
    <div className="mb-4 space-y-2">
      {visible.map((event) => {
        const key = `${notable.indexOf(event)}-${event.kind}-${event.stage}`;
        const Icon = KIND_ICON[event.kind] ?? ShieldAlert;
        const tone = toneFor(event);
        return (
          <div
            key={key}
            role={tone === 'danger' ? 'alert' : 'status'}
            className={`flex items-start gap-2 rounded-lg border p-3 text-sm ${TONE_CLASS[tone]}`}
          >
            {event.action === 'block' ? (
              <Ban aria-hidden="true" className="mt-0.5 h-4 w-4 shrink-0" />
            ) : (
              <Icon aria-hidden="true" className="mt-0.5 h-4 w-4 shrink-0" />
            )}
            <div className="min-w-0 flex-1">
              <p className="font-semibold">
                {KIND_TITLE[event.kind] ?? event.kind}
                <span className="ml-2 font-normal opacity-70">
                  {event.stage} · {event.action}
                </span>
              </p>
              {event.detail ? <p className="mt-0.5">{event.detail}</p> : null}
              {(event.entities ?? []).length > 0 ? (
                <p className="mt-1 flex flex-wrap gap-1">
                  {(event.entities ?? []).map((entity) => (
                    <span key={entity} className="chip">
                      {entity}
                    </span>
                  ))}
                </p>
              ) : null}
              {typeof event.score === 'number' ? (
                <p className="mt-1 text-xs opacity-80">score {event.score.toFixed(3)}</p>
              ) : null}
            </div>
            <button
              type="button"
              className="btn btn-ghost btn-xs"
              aria-label="Dismiss this notice"
              onClick={() => setDismissed((current) => [...current, key])}
            >
              <X aria-hidden="true" className="h-3 w-3" />
            </button>
          </div>
        );
      })}

      {passed > 0 ? (
        <p className="flex items-center gap-1.5 text-xs text-emerald-700 dark:text-emerald-400">
          <ShieldCheck aria-hidden="true" className="h-3.5 w-3.5" />
          {passed} guardrail check{passed === 1 ? '' : 's'} passed
        </p>
      ) : null}
    </div>
  );
}
