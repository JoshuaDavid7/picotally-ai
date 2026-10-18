// Tests for transcript-driven client-name resolution in create_invoice.
//
// Run locally with:
//   deno test supabase/functions/process_voice_command/client_name.test.ts
//
// Why this file exists separately from verifier.test.ts:
//   verifier.ts handles arithmetic correction in the assistant's prose
//   ("$915 + $915 = $1,860" → caught and replaced with $1,830). The bug
//   these tests pin is unrelated — it's in the post-LLM normalizer in
//   index.ts, where a regex hint extractor was clobbering the model's
//   correct `clientName` with a noun pulled from the work description.
//
// The bug, end-to-end:
//   Transcript: "Quote Daniel $750 for deck repair"
//   Model:     { clientName: "Daniel", description: "Deck repair", ... }
//   Normalizer was overwriting clientName → "Deck Repair" because the
//   "for <X>" regex misfired on the trailing phrase, and the override
//   gate trusted that hint unconditionally (bestMatchingName returns
//   the raw input as a fallback, so `matchedHint` was always truthy).
//
// What we test:
//   1. extractClientNameHintFromTranscript — pins the regex behavior
//      (it still extracts "Deck Repair" / "Kitchen Plumbing" — that
//      isn't fixed here; the regex is intentionally permissive).
//   2. normalizeAssistantResult — pins the user-visible behavior: the
//      model's clientName must SURVIVE the normalizer when it's already
//      a known customer, regardless of what the hint regex pulls out.
//      This is the regression test for the actual bug.

import {
  assertEquals,
} from "https://deno.land/std@0.224.0/assert/mod.ts"

import {
  extractClientNameHintFromTranscript,
  normalizeAssistantResult,
} from "./index.ts"

// ─── Regex hint extraction (pin the inputs to the override gate) ─────────────

Deno.test("hint extractor pulls work description when money breaks the name capture", () => {
  // The capture set in pattern 3 is `[a-z0-9\s&'-]`, which does not include
  // "$" — so "Quote Daniel $750 for deck repair" cannot be parsed as
  // "Quote <Name>" by pattern 3. The fallback pattern 4 (`\bfor <X>`) then
  // catches the trailing "for deck repair" and returns it as the alleged
  // client name. This test pins that quirk so a future regex change is
  // intentional, not accidental.
  assertEquals(
    extractClientNameHintFromTranscript("Quote Daniel $750 for deck repair"),
    "Deck Repair",
  )
})

Deno.test("hint extractor misfires similarly on multi-word client + money + work", () => {
  assertEquals(
    extractClientNameHintFromTranscript("Quote Daniel Mowery $1,250 for kitchen plumbing"),
    "Kitchen Plumbing",
  )
})

Deno.test("hint extractor correctly catches comma-introduced client name", () => {
  // The legitimate use case for the hint: model misses the name, regex
  // picks it up before the comma.
  assertEquals(
    extractClientNameHintFromTranscript("Invoice for Daniel Mowery, bathroom tile"),
    "Daniel Mowery",
  )
})

// ─── Normalizer override gate (the actual fix) ───────────────────────────────

// Shared business-context fixture used across the normalizer tests.
// Daniel Mowery is in the known-customer list; "Deck Repair" / "Kitchen
// Plumbing" are not. This matches the user's real data shape — Daniel
// Mowery is a customer with an overdue balance on the home dashboard.
const contextWithDaniel = {
  customers: [
    { name: "Daniel Mowery", balance: 5106 },
    { name: "Mike Smith", balance: 0 },
    { name: "Steven Chen", balance: 2300 },
  ],
  recentCustomers: ["Daniel Mowery", "Mike Smith", "Steven Chen"],
  jobs: [],
}

Deno.test("create_invoice keeps single-name AI clientName when it matches a known customer", () => {
  // Reproducer #1: "Quote Daniel $750 for deck repair".
  // Pre-fix: clientName was clobbered to "Deck Repair".
  // Post-fix: AI's "Daniel" resolves to "Daniel Mowery" via the
  // existing fuzzy match step and is not overridden.
  const result = normalizeAssistantResult(
    {
      action: "create_invoice",
      params: {
        clientName: "Daniel",
        type: "quote",
        laborType: "flat",
        laborAmount: 750,
        description: "Deck repair",
        laborHours: 1,
      },
      response: "",
      suggestionChips: [],
    },
    contextWithDaniel,
    "Quote Daniel $750 for deck repair",
  )

  const params = result.params as Record<string, unknown>
  assertEquals(params.clientName, "Daniel Mowery")
})

Deno.test("create_invoice keeps full-name AI clientName when the hint is a work description", () => {
  // Reproducer #2: "Quote Daniel Mowery $1,250 for kitchen plumbing".
  // Pre-fix: clientName was clobbered to "Kitchen Plumbing".
  // Post-fix: AI's "Daniel Mowery" is preserved because it matches a
  // known customer and the hint ("Kitchen Plumbing") does not.
  const result = normalizeAssistantResult(
    {
      action: "create_invoice",
      params: {
        clientName: "Daniel Mowery",
        type: "quote",
        laborType: "flat",
        laborAmount: 1250,
        description: "Kitchen plumbing",
        laborHours: 1,
      },
      response: "",
      suggestionChips: [],
    },
    contextWithDaniel,
    "Quote Daniel Mowery $1,250 for kitchen plumbing",
  )

  const params = result.params as Record<string, unknown>
  assertEquals(params.clientName, "Daniel Mowery")
})

Deno.test("create_invoice still uses transcript hint when AI clientName is missing", () => {
  // Confirms the override gate still fires for its original purpose:
  // the AI returned an empty clientName and the hint correctly
  // identifies a known customer from a "for <Name>," pattern.
  const result = normalizeAssistantResult(
    {
      action: "create_invoice",
      params: {
        clientName: "",
        type: "invoice",
        laborType: "flat",
        laborAmount: 500,
        description: "Bathroom tile",
        laborHours: 1,
      },
      response: "",
      suggestionChips: [],
    },
    contextWithDaniel,
    "Invoice for Daniel Mowery, bathroom tile",
  )

  const params = result.params as Record<string, unknown>
  assertEquals(params.clientName, "Daniel Mowery")
})

Deno.test("create_invoice still uses transcript hint when AI clientName is junk", () => {
  // "Create An" was a known bad output from the legacy Gemini-era
  // pipeline; the override gate's `aiNameLooksWrong` arm is what
  // recovers from it. Confirm the new gate still catches this class
  // of failure even though "Create An" does not literally appear in
  // the stop-token list (`an` is, but `create an` as a phrase is not).
  const result = normalizeAssistantResult(
    {
      action: "create_invoice",
      params: {
        clientName: "Create An",
        type: "invoice",
        laborType: "flat",
        laborAmount: 500,
        description: "Bathroom tile",
        laborHours: 1,
      },
      response: "",
      suggestionChips: [],
    },
    contextWithDaniel,
    "Invoice for Daniel Mowery, bathroom tile",
  )

  const params = result.params as Record<string, unknown>
  // "Create An" doesn't match a known customer; the hint "Daniel
  // Mowery" does → override fires.
  assertEquals(params.clientName, "Daniel Mowery")
})

Deno.test("create_invoice preserves AI clientName for a brand-new customer not in context", () => {
  // New-customer flow: AI returns "Bob Newman", who is not yet in the
  // customer list. The hint regex also extracts "Bob Newman" (comma
  // pattern). Both AI and hint disagree with the known list. The gate
  // should preserve the AI's name rather than dropping it.
  const result = normalizeAssistantResult(
    {
      action: "create_invoice",
      params: {
        clientName: "Bob Newman",
        type: "invoice",
        laborType: "flat",
        laborAmount: 500,
        description: "Fix sink",
        laborHours: 1,
      },
      response: "",
      suggestionChips: [],
    },
    contextWithDaniel,
    "Invoice for Bob Newman, fix sink",
  )

  const params = result.params as Record<string, unknown>
  assertEquals(params.clientName, "Bob Newman")
})
