// deno test supabase/functions/_shared/voice_access.test.ts
import { assertEquals } from "https://deno.land/std@0.224.0/assert/mod.ts"
import {
  isOverCap,
  isTrialActive,
  periodMonthKey,
  PRO_MAX_CAP_SECONDS,
  TRIAL_CAP_SECONDS,
  TRIAL_DURATION_MS,
  voiceAccess,
  voiceCapSeconds,
} from "./voice_access.ts"

const NOW = Date.parse("2026-05-31T12:00:00Z")
const iso = (ms: number) => new Date(ms).toISOString()

Deno.test("isTrialActive: within 30 days true, after false", () => {
  assertEquals(isTrialActive(iso(NOW - 1 * 86400000), NOW), true) // 1 day ago
  assertEquals(isTrialActive(iso(NOW - 29 * 86400000), NOW), true)
  assertEquals(isTrialActive(iso(NOW - 31 * 86400000), NOW), false)
  assertEquals(isTrialActive(null, NOW), false)
  assertEquals(isTrialActive("not-a-date", NOW), false)
})

Deno.test("isTrialActive: exact boundary is exclusive", () => {
  const started = NOW - TRIAL_DURATION_MS
  assertEquals(isTrialActive(iso(started), NOW), false) // exactly expired
  assertEquals(isTrialActive(iso(started + 1000), NOW), true) // 1s of trial left
})

Deno.test("voiceAccess: pro_max always allowed, not trial", () => {
  const a = voiceAccess("pro_max", null, NOW)
  assertEquals(a.allowed, true)
  assertEquals(a.isTrial, false)
})

Deno.test("voiceAccess: pro tier (no trial) denied", () => {
  assertEquals(voiceAccess("pro", null, NOW).allowed, false)
})

Deno.test("voiceAccess: free tier denied", () => {
  assertEquals(voiceAccess("free", null, NOW).allowed, false)
  assertEquals(voiceAccess(null, null, NOW).allowed, false)
})

Deno.test("voiceAccess: active trial allowed + flagged trial", () => {
  const a = voiceAccess("pro", iso(NOW - 5 * 86400000), NOW)
  assertEquals(a.allowed, true)
  assertEquals(a.isTrial, true)
})

Deno.test("voiceAccess: expired trial denied", () => {
  assertEquals(voiceAccess("pro", iso(NOW - 40 * 86400000), NOW).allowed, false)
})

Deno.test("voiceAccess: pro_max beats trial flag even with active trial", () => {
  const a = voiceAccess("pro_max", iso(NOW - 5 * 86400000), NOW)
  assertEquals(a.allowed, true)
  assertEquals(a.isTrial, false) // counts as paid, not trial
})

Deno.test("voiceCapSeconds: pro_max 120min, trial 15min", () => {
  assertEquals(voiceCapSeconds(false), PRO_MAX_CAP_SECONDS)
  assertEquals(voiceCapSeconds(true), TRIAL_CAP_SECONDS)
  assertEquals(PRO_MAX_CAP_SECONDS, 7200)
  assertEquals(TRIAL_CAP_SECONDS, 900)
})

Deno.test("isOverCap: at/over the cap blocks, under allows", () => {
  assertEquals(isOverCap(0, false), false)
  assertEquals(isOverCap(7199, false), false)
  assertEquals(isOverCap(7200, false), true) // exactly at cap → blocked
  assertEquals(isOverCap(7201, false), true)
  // trial
  assertEquals(isOverCap(899, true), false)
  assertEquals(isOverCap(900, true), true)
})

Deno.test("periodMonthKey: first-of-month UTC", () => {
  assertEquals(periodMonthKey(Date.parse("2026-05-31T23:59:59Z")), "2026-05-01")
  assertEquals(periodMonthKey(Date.parse("2026-01-01T00:00:00Z")), "2026-01-01")
  assertEquals(periodMonthKey(Date.parse("2026-12-15T12:00:00Z")), "2026-12-01")
})
