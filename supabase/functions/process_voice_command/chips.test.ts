// Tests for the suggestion-chip sanitizer.
//
// Run with:
//   deno test supabase/functions/process_voice_command/chips.test.ts

import {
  assertEquals,
  assert,
} from "https://deno.land/std@0.224.0/assert/mod.ts"

import { sanitizeSuggestionChips, FORBIDDEN_CHIP_PREFIXES } from "./chips.ts"

Deno.test("sanitize: keeps chips that start with allowed verbs", () => {
  const out = sanitizeSuggestionChips([
    "Open Mike's invoice",
    "Show overdue invoices",
    "Create a quote",
    "Log an expense",
  ])
  assertEquals(out.length, 4)
  assertEquals(out[0], "Open Mike's invoice")
  assertEquals(out[3], "Log an expense")
})

Deno.test("sanitize: drops 'Send' chips and tops up with safe defaults", () => {
  const out = sanitizeSuggestionChips([
    "Send Mike a reminder",
    "Show overdue invoices",
  ])
  // "Send Mike a reminder" is dropped, "Show overdue invoices" kept,
  // and one safe default is appended to reach the minimum of 2.
  assertEquals(out.length, 2)
  assertEquals(out[0], "Show overdue invoices")
  assert(!out.some((c) => c.toLowerCase().startsWith('send')))
})

Deno.test("sanitize: drops 'Message', 'Text', 'Email', 'Call' chips", () => {
  const out = sanitizeSuggestionChips([
    "Message Mike about payment",
    "Text Steven now",
    "Email Hikaru the invoice",
    "Call Joel Paulraj",
  ])
  // All four forbidden, so the result is just the safe defaults (top-up).
  assert(out.length >= 2)
  for (const chip of out) {
    const lower = chip.toLowerCase()
    assert(!lower.startsWith('send'))
    assert(!lower.startsWith('message'))
    assert(!lower.startsWith('text '))
    assert(!lower.startsWith('email'))
    assert(!lower.startsWith('call '))
  }
})

Deno.test("sanitize: drops 'Schedule' / 'Export' / 'Remind' / 'Notify'", () => {
  const out = sanitizeSuggestionChips([
    "Schedule a follow-up",
    "Export to CSV",
    "Remind me later",
    "Notify Steven by SMS",
    "Open Mike's invoice", // legit, should survive
  ])
  assert(out.includes("Open Mike's invoice"))
  for (const chip of out) {
    const lower = chip.toLowerCase()
    assert(!lower.startsWith('schedule'))
    assert(!lower.startsWith('export'))
    assert(!lower.startsWith('remind'))
    assert(!lower.startsWith('notify'))
  }
})

Deno.test("sanitize: caps at 4 chips even if more provided", () => {
  const out = sanitizeSuggestionChips([
    "Open A", "Open B", "Open C", "Open D", "Open E", "Open F",
  ])
  assertEquals(out.length, 4)
  assertEquals(out[3], "Open D")
})

Deno.test("sanitize: top-ups dedupe against already-present chips", () => {
  // The fallback pool includes "Show unpaid invoices". If the model
  // already returned that and one forbidden chip, we should not add
  // a duplicate during top-up.
  const out = sanitizeSuggestionChips([
    "Show unpaid invoices",
    "Send Mike a reminder",
  ])
  const lowered = out.map((c) => c.toLowerCase())
  // No duplicate of "show unpaid invoices"
  const showUnpaidCount = lowered.filter(
    (c) => c === 'show unpaid invoices'
  ).length
  assertEquals(showUnpaidCount, 1)
  assert(out.length >= 2)
})

Deno.test("sanitize: handles empty / null / non-array input", () => {
  assertEquals(sanitizeSuggestionChips(null).length, 2) // top-up to minimum
  assertEquals(sanitizeSuggestionChips(undefined).length, 2)
  assertEquals(sanitizeSuggestionChips("not-an-array").length, 2)
  assertEquals(sanitizeSuggestionChips([]).length, 2)
  // Non-strings inside the array are silently dropped.
  const out = sanitizeSuggestionChips([null, 42, { foo: 'bar' }, "Open Mike"])
  assertEquals(out.length, 2)
  assertEquals(out[0], "Open Mike")
})

Deno.test("sanitize: trims whitespace and drops empty strings", () => {
  const out = sanitizeSuggestionChips([
    "   ",
    "  Open Mike  ",
    "",
    "Show drafts",
  ])
  assertEquals(out.length, 2)
  assertEquals(out[0], "Open Mike")
  assertEquals(out[1], "Show drafts")
})

Deno.test("FORBIDDEN_CHIP_PREFIXES: every prefix is matched as expected", () => {
  // Smoke test that adding a new forbidden prefix doesn't accidentally
  // match a legitimate chip via a substring (e.g. adding 'remi' would
  // match "Remit"). All current prefixes should be word-initial concepts.
  for (const prefix of FORBIDDEN_CHIP_PREFIXES) {
    const trimmed = prefix.trim()
    const example = `${trimmed} something`
    const out = sanitizeSuggestionChips([example])
    // Should be filtered → only safe defaults remain
    assert(
      !out.some((c) => c.toLowerCase().startsWith(trimmed)),
      `Prefix "${prefix}" failed to filter chip "${example}"`
    )
  }
})
