// Shared gate for who may use real-time voice mode and how long the free
// trial's voice taste lasts. Used by voice_token (the STT-token mint endpoint
// — the ONLY door to a voice session) and voice_tts.
//
// Pro Max → unlimited (subject to the Phase 2 minute cap). Active 30-day trial
// → allowed (the metered 15-minute taste; the cap is enforced separately in
// Phase 2's voice_usage check). Everyone else → denied.

/** Must match the client's `trialDuration` (subscription_provider.dart). */
export const TRIAL_DURATION_MS = 30 * 24 * 60 * 60 * 1000

export function isTrialActive(
  trialStartedAt: string | null | undefined,
  nowMs: number,
): boolean {
  if (!trialStartedAt) return false
  const started = Date.parse(trialStartedAt)
  if (!Number.isFinite(started)) return false
  return nowMs < started + TRIAL_DURATION_MS
}

export interface VoiceAccess {
  allowed: boolean
  isTrial: boolean
  reason: string
}

/**
 * Decide whether a user may start a voice session.
 *   pro_max           → allowed (not trial)
 *   active 30d trial  → allowed (trial taste)
 *   otherwise         → denied
 */
export function voiceAccess(
  tier: string | null | undefined,
  trialStartedAt: string | null | undefined,
  nowMs: number,
): VoiceAccess {
  if (tier === 'pro_max') {
    return { allowed: true, isTrial: false, reason: 'pro_max' }
  }
  if (isTrialActive(trialStartedAt, nowMs)) {
    return { allowed: true, isTrial: true, reason: 'trial' }
  }
  return { allowed: false, isTrial: false, reason: 'requires_pro_max' }
}

// ── Monthly minute caps (the "never run at a loss" guard) ────────────────────
// Pro Max: 120 min/month (~$6 at Aura's ~$0.05/min — well under the ~$16
// margin). Trial: 15-minute conversion taste. Tunable here without a redeploy
// of the DB.

export const PRO_MAX_CAP_SECONDS = 120 * 60
export const TRIAL_CAP_SECONDS = 15 * 60

export function voiceCapSeconds(isTrial: boolean): number {
  return isTrial ? TRIAL_CAP_SECONDS : PRO_MAX_CAP_SECONDS
}

/** True once the user has consumed their monthly allowance. */
export function isOverCap(secondsUsed: number, isTrial: boolean): boolean {
  return secondsUsed >= voiceCapSeconds(isTrial)
}

/** First-of-month (UTC) `YYYY-MM-DD` — the metering period key. */
export function periodMonthKey(nowMs: number): string {
  const d = new Date(nowMs)
  const y = d.getUTCFullYear()
  const m = String(d.getUTCMonth() + 1).padStart(2, '0')
  return `${y}-${m}-01`
}
