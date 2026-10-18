// Tests for the pre-model grounding guard.
//
// Run locally with:
//   deno test supabase/functions/process_voice_command/grounding.test.ts
//
// The guard's promise: if the user asks about a customer we don't have
// in context, we short-circuit with a safe reply BEFORE the LLM sees
// the question. That stops Claude from fabricating balances, invoice
// dates, or payment history for people who aren't real — the single
// worst failure mode a bookkeeping app can ship with.
//
// These tests pin the three primitives that make that promise:
//   - extractCustomerNameFromDataQuestion: which utterances trigger a lookup?
//   - isKnownCustomer: does a transcript name match a row we have?
//   - buildUnknownCustomerResponse: is the short-circuit reply shaped
//     like the action payload the client expects?
//
// Plus the two helpers they rely on:
//   - normalizeText: lowercase, strip punctuation, collapse whitespace.
//   - levenshteinDistance: edit distance for typo tolerance.

import {
  assertEquals,
  assertNotEquals,
} from "https://deno.land/std@0.224.0/assert/mod.ts"

import {
  buildUnknownCustomerResponse,
  DATA_QUESTION_PATTERNS,
  extractCustomerNameFromDataQuestion,
  GROUNDING_STOP_WORDS,
  isKnownCustomer,
  levenshteinDistance,
  normalizeText,
} from "./grounding.ts"

// ─── normalizeText ───────────────────────────────────────────────────────────

Deno.test("normalizeText lowercases", () => {
  assertEquals(normalizeText("Mike Jones"), "mike jones")
})

Deno.test("normalizeText strips punctuation", () => {
  assertEquals(normalizeText("O'Brien, Mrs."), "o brien mrs")
})

Deno.test("normalizeText collapses whitespace", () => {
  assertEquals(normalizeText("  mike    jones  "), "mike jones")
})

Deno.test("normalizeText expands &", () => {
  assertEquals(normalizeText("Smith & Sons"), "smith and sons")
})

Deno.test("normalizeText preserves digits", () => {
  assertEquals(normalizeText("Job 1234"), "job 1234")
})

Deno.test("normalizeText returns empty on whitespace-only input", () => {
  assertEquals(normalizeText("   "), "")
})

// ─── levenshteinDistance ────────────────────────────────────────────────────

Deno.test("levenshteinDistance returns 0 on identity", () => {
  assertEquals(levenshteinDistance("mike", "mike"), 0)
})

Deno.test("levenshteinDistance returns length when one side empty", () => {
  assertEquals(levenshteinDistance("", "mike"), 4)
  assertEquals(levenshteinDistance("mike", ""), 4)
})

Deno.test("levenshteinDistance counts single substitution", () => {
  assertEquals(levenshteinDistance("mike", "mikr"), 1)
})

Deno.test("levenshteinDistance counts insertion and deletion", () => {
  assertEquals(levenshteinDistance("mike", "mikes"), 1)
  assertEquals(levenshteinDistance("mikes", "mike"), 1)
})

Deno.test("levenshteinDistance handles transposition as 2 edits", () => {
  // Levenshtein (not Damerau) counts "ab" -> "ba" as 2 subs, not 1.
  assertEquals(levenshteinDistance("ab", "ba"), 2)
})

// ─── extractCustomerNameFromDataQuestion ─────────────────────────────────────

Deno.test("extract returns null on conversational filler", () => {
  assertEquals(extractCustomerNameFromDataQuestion("hey what's up"), null)
})

Deno.test("extract returns null on empty input", () => {
  assertEquals(extractCustomerNameFromDataQuestion(""), null)
})

Deno.test("extract captures 'how much does X owe?'", () => {
  assertEquals(
    extractCustomerNameFromDataQuestion("How much does Mike owe?"),
    "Mike",
  )
})

Deno.test("extract captures 'how much does X owe' without punctuation", () => {
  assertEquals(
    extractCustomerNameFromDataQuestion("how much does Alice Smith owe"),
    "Alice Smith",
  )
})

Deno.test("extract captures 'what does X owe'", () => {
  assertEquals(
    extractCustomerNameFromDataQuestion("what does Bob Jones owe"),
    "Bob Jones",
  )
})

Deno.test("extract captures \"what is X's balance\"", () => {
  assertEquals(
    extractCustomerNameFromDataQuestion("what is Carol's balance"),
    "Carol",
  )
})

Deno.test("extract captures 'has X paid'", () => {
  assertEquals(
    extractCustomerNameFromDataQuestion("has Dave paid"),
    "Dave",
  )
})

Deno.test("extract captures 'did X pay'", () => {
  assertEquals(
    extractCustomerNameFromDataQuestion("did Eve pay"),
    "Eve",
  )
})

Deno.test("extract captures 'when did i last invoice X'", () => {
  assertEquals(
    extractCustomerNameFromDataQuestion("when did i last invoice Frank"),
    "Frank",
  )
})

Deno.test("extract rejects stop-word captures ('the customer')", () => {
  // "how much does the customer owe" should capture "the customer",
  // which gets rejected on the GROUNDING_STOP_WORDS pass.
  assertEquals(
    extractCustomerNameFromDataQuestion("how much does the customer owe"),
    null,
  )
})

Deno.test("extract rejects stop-word 'him'", () => {
  assertEquals(
    extractCustomerNameFromDataQuestion("has him paid"),
    null,
  )
})

Deno.test("extract rejects single-character capture", () => {
  assertEquals(
    extractCustomerNameFromDataQuestion("has X paid"),
    null,
  )
})

Deno.test("extract strips trailing punctuation from captured name", () => {
  assertEquals(
    extractCustomerNameFromDataQuestion("how much does Mike owe?"),
    "Mike",
  )
  assertEquals(
    extractCustomerNameFromDataQuestion("what does Mike Jones owe."),
    "Mike Jones",
  )
})

Deno.test("extract is case-insensitive on pattern keywords", () => {
  assertEquals(
    extractCustomerNameFromDataQuestion("HOW MUCH DOES Mike OWE"),
    "Mike",
  )
})

// ─── isKnownCustomer ─────────────────────────────────────────────────────────

Deno.test("isKnownCustomer returns false on empty knownNames", () => {
  assertEquals(isKnownCustomer("Mike", []), false)
})

Deno.test("isKnownCustomer returns false on empty name", () => {
  assertEquals(isKnownCustomer("", ["Mike Jones"]), false)
})

Deno.test("isKnownCustomer matches exact name", () => {
  assertEquals(isKnownCustomer("Mike Jones", ["Mike Jones"]), true)
})

Deno.test("isKnownCustomer matches case-insensitively", () => {
  assertEquals(isKnownCustomer("mike jones", ["Mike Jones"]), true)
})

Deno.test("isKnownCustomer matches after punctuation normalization", () => {
  assertEquals(isKnownCustomer("Mike Jones.", ["Mike Jones"]), true)
})

Deno.test("isKnownCustomer matches single-token first name against full name", () => {
  // User says "Mike", context has "Mike Jones" — should match via
  // token-level membership.
  assertEquals(isKnownCustomer("Mike", ["Mike Jones"]), true)
})

Deno.test("isKnownCustomer matches single-token last name against full name", () => {
  assertEquals(isKnownCustomer("Jones", ["Mike Jones"]), true)
})

Deno.test("isKnownCustomer tolerates one-edit typo on short name", () => {
  // "Johnsen" vs "Johnson" — one substitution, length 7 → threshold
  // max(1, round(7/5)) = max(1, 1) = 1. Matches.
  assertEquals(isKnownCustomer("Johnsen", ["Johnson"]), true)
})

Deno.test("isKnownCustomer tolerates moderate typos on long names", () => {
  // "Alexandre Dumas" vs "Alexander Dumas" — 2 subs, length 15 →
  // threshold max(1, round(15/5)) = 3. Matches.
  assertEquals(isKnownCustomer("Alexandre Dumas", ["Alexander Dumas"]), true)
})

Deno.test("isKnownCustomer rejects completely different names", () => {
  assertEquals(isKnownCustomer("Zebediah", ["Mike Jones"]), false)
})

Deno.test("isKnownCustomer rejects names with too many edits", () => {
  // "Smith" vs "Jones" — 4 subs + 0 inserts, length 5 → threshold 1.
  assertEquals(isKnownCustomer("Smith", ["Jones"]), false)
})

Deno.test("isKnownCustomer walks the full list", () => {
  assertEquals(
    isKnownCustomer("Carol", ["Alice", "Bob", "Carol", "Dave"]),
    true,
  )
})

Deno.test("isKnownCustomer ignores empty entries in knownNames", () => {
  assertEquals(isKnownCustomer("Mike", ["", "Mike Jones", ""]), true)
})

// ─── buildUnknownCustomerResponse ────────────────────────────────────────────

Deno.test("buildUnknownCustomerResponse returns action 'answer'", () => {
  const out = buildUnknownCustomerResponse("Mystery Person")
  assertEquals(out.action, "answer")
})

Deno.test("buildUnknownCustomerResponse echoes the queried name", () => {
  const out = buildUnknownCustomerResponse("Mystery Person")
  // We want the user to see *which* name we couldn't find, so they can
  // catch transcription errors and retry.
  assertEquals(out.response.includes("Mystery Person"), true)
})

Deno.test("buildUnknownCustomerResponse offers escape hatches", () => {
  const out = buildUnknownCustomerResponse("Mystery Person")
  // The suggestion chips must give the user a next step that isn't
  // "ask again with better transcription" — either open the clients
  // list (to verify the name is missing) or create a new customer.
  assertEquals(out.suggestionChips.includes("Open clients"), true)
})

Deno.test("buildUnknownCustomerResponse returns empty params", () => {
  // Params must be empty — the whole point of the guard is that we
  // never passed the name through to an action that might act on it.
  const out = buildUnknownCustomerResponse("Mystery Person")
  assertEquals(Object.keys(out.params).length, 0)
})

// ─── End-to-end guard scenarios ──────────────────────────────────────────────

Deno.test("guard fires on unknown customer (integration)", () => {
  // The intended call-site sequence in index.ts.
  const transcript = "How much does NobodyInDatabase owe?"
  const known = ["Alice Smith", "Bob Jones"]

  const extracted = extractCustomerNameFromDataQuestion(transcript)
  assertNotEquals(extracted, null)
  assertEquals(isKnownCustomer(extracted!, known), false)

  const response = buildUnknownCustomerResponse(extracted!)
  assertEquals(response.action, "answer")
  assertEquals(response.response.includes("NobodyInDatabase"), true)
})

Deno.test("guard does NOT fire on known customer with typo", () => {
  // User says "Allise" (one edit from "Alice") — guard should let it
  // through to the model for full fuzzy resolution.
  const transcript = "How much does Allise owe?"
  const known = ["Alice Smith"]

  const extracted = extractCustomerNameFromDataQuestion(transcript)
  assertEquals(extracted, "Allise")
  assertEquals(isKnownCustomer(extracted!, known), true)
})

Deno.test("guard does NOT fire on non-data-access questions", () => {
  // "What should I charge for a new deck install?" — doesn't reference
  // a specific customer's records. extract returns null, guard is
  // bypassed, model handles it.
  const transcript = "What should I charge for a new deck install?"
  assertEquals(extractCustomerNameFromDataQuestion(transcript), null)
})

// ─── Structural invariants (regression guards) ───────────────────────────────

Deno.test("DATA_QUESTION_PATTERNS all have one capture group", () => {
  // The extractor assumes match[1] is the name. If any future pattern
  // has zero or multiple capture groups, the extractor breaks silently.
  for (const pattern of DATA_QUESTION_PATTERNS) {
    const clone = new RegExp(pattern.source, pattern.flags)
    // Add a synthetic group to detect expected count: match empty,
    // then count groups via RegExp.source parsing.
    const groupCount = (clone.source.match(/\((?!\?)/g) || []).length
    assertEquals(
      groupCount,
      1,
      `Pattern has ${groupCount} capture groups (expected 1): ${pattern.source}`,
    )
  }
})

Deno.test("GROUNDING_STOP_WORDS is all-lowercase", () => {
  // The extractor compares with `.toLowerCase()`; if a stop-word were
  // uppercase, it'd never match.
  for (const word of GROUNDING_STOP_WORDS) {
    assertEquals(word, word.toLowerCase(), `Stop word not lowercase: ${word}`)
  }
})
