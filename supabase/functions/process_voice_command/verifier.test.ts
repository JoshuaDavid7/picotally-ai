// Tests for the post-response arithmetic verifier.
//
// Run with:
//   deno test supabase/functions/process_voice_command/verifier.test.ts
//
// The verifier's job is narrow but high-trust: catch any "$X + $Y = $Z"
// expression where the math is wrong, and replace the claimed total
// with the correct sum. These tests pin both the correct-pass cases
// (must NOT be modified) and the catch-and-correct cases.

import {
  assertEquals,
  assert,
} from "https://deno.land/std@0.224.0/assert/mod.ts"

import { verifyArithmetic } from "./verifier.ts"

// ─── Correct expressions pass through unchanged ──────────────────────────────

Deno.test("correct simple sum is unchanged", () => {
  const input = "You spent $915 + $915 = $1,830 on pipes."
  const { corrected, mistakes } = verifyArithmetic(input)
  assertEquals(mistakes, 0)
  assertEquals(corrected, input)
})

Deno.test("correct 4-term sum is unchanged", () => {
  const input = "$915 + $915 + $15 + $15 = $1,860"
  const { corrected, mistakes } = verifyArithmetic(input)
  assertEquals(mistakes, 0)
  assertEquals(corrected, input)
})

Deno.test("correct sum with cents is unchanged", () => {
  const input = "$1,234.56 + $789.10 = $2,023.66"
  const { corrected, mistakes } = verifyArithmetic(input)
  assertEquals(mistakes, 0)
  assertEquals(corrected, input)
})

Deno.test("correct sum without dollar signs is unchanged", () => {
  const input = "915 + 915 = 1830"
  const { corrected, mistakes } = verifyArithmetic(input)
  assertEquals(mistakes, 0)
  assertEquals(corrected, input)
})

Deno.test("rounding noise within 1¢ does not trigger correction", () => {
  // 3 × $33.34 per-unit rounding leaves the line total at $100.02 even
  // though raw 3 × 33.333 = $99.999. The rounded display is correct;
  // verifier should not "fix" it.
  const input = "$33.34 + $33.34 + $33.34 = $100.02"
  const { mistakes } = verifyArithmetic(input)
  assertEquals(mistakes, 0)
})

// ─── Wrong expressions get corrected ─────────────────────────────────────────

Deno.test("simple wrong sum is corrected", () => {
  const input = "$915 + $915 = $1,860"
  const { corrected, mistakes } = verifyArithmetic(input)
  assertEquals(mistakes, 1)
  assert(corrected.includes("$1,830"))
  assert(!corrected.includes("$1,860"))
})

Deno.test("4-term wrong sum is corrected", () => {
  // The verifier's narrow job: the arithmetic itself is wrong. (The
  // Sonnet "double-check" bug was a wrong-VALUES bug — addends were
  // correctly summed but the model picked the wrong field for one;
  // that class of bug is solved by the query tools, not by this
  // verifier. This test confirms the verifier catches sloppy adding,
  // which is what it's for.)
  const input = "$915 + $915 + $15 + $15 = $2,000"
  const { corrected, mistakes, details } = verifyArithmetic(input)
  assertEquals(mistakes, 1)
  assert(corrected.includes("$1,860"), `should contain $1,860, got: ${corrected}`)
  assertEquals(details[0].claimed, 2000)
  assertEquals(details[0].actual, 1860)
})

Deno.test("wrong total inside markdown bold is corrected, bold preserved", () => {
  const input = "Subtotal: $100 + $200 = **$400**"
  const { corrected, mistakes } = verifyArithmetic(input)
  assertEquals(mistakes, 1)
  assert(corrected.includes("**$300**"), `bold should wrap correct total: ${corrected}`)
})

Deno.test("multiple wrong sums in one response are all corrected", () => {
  const input = "Pipes: $100 + $100 = $300. Wire: $50 + $50 = $200."
  const { corrected, mistakes } = verifyArithmetic(input)
  assertEquals(mistakes, 2)
  assert(corrected.includes("$200"))
  assert(corrected.includes("$100"))
})

// ─── Edge cases ──────────────────────────────────────────────────────────────

Deno.test("single number with no addition is left alone", () => {
  const input = "Total: $1,860"
  const { corrected, mistakes } = verifyArithmetic(input)
  assertEquals(mistakes, 0)
  assertEquals(corrected, input)
})

Deno.test("empty string returns empty", () => {
  const { corrected, mistakes } = verifyArithmetic("")
  assertEquals(mistakes, 0)
  assertEquals(corrected, "")
})

Deno.test("addends with decimals and commas mixed", () => {
  const input = "$1,500 + $250.50 + $49.50 = $1,800"
  const { corrected, mistakes } = verifyArithmetic(input)
  assertEquals(mistakes, 0)
  assertEquals(corrected, input)
})

Deno.test("preserves text around the corrected expression", () => {
  const input = "You spent: $915 + $915 = $1,860 on pipes last month."
  const { corrected } = verifyArithmetic(input)
  assert(corrected.startsWith("You spent: "))
  assert(corrected.endsWith(" on pipes last month."))
  assert(corrected.includes("$1,830"))
})

// ─── Bulleted-list total verifier ────────────────────────────────────────────
//
// Q19-class bug: model lists items with $ amounts then claims a wrong
// "Total: $X". This verifier catches the conservative case where every
// bullet line in a contiguous block has a single $amount AND a "Total" /
// "Totaling" / "in total" claim follows within a few lines.

Deno.test("bullet list with wrong 'Total: $X' is corrected", () => {
  const input = `You have 3 overdue:
- TestClient — $115
- Joel Paulraj — $100
- Mike — $235.52
Total: $999`
  const { corrected, mistakes } = verifyArithmetic(input)
  assertEquals(mistakes, 1)
  // 115 + 100 + 235.52 = $450.52
  assert(corrected.includes("$450.52"), `expected $450.52 in: ${corrected}`)
  assert(!corrected.includes("$999"))
})

Deno.test("bullet list with correct 'Total: $X' is unchanged", () => {
  const input = `Three invoices:
- Steven — $535
- Hikaru — $274.04
- Mike — $235.52
Total: $1,044.56`
  const { mistakes } = verifyArithmetic(input)
  assertEquals(mistakes, 0)
})

Deno.test("bullet list with 'Totaling' phrasing is corrected", () => {
  const input = `Late invoices:
- TestClient — $115
- Joel Paulraj — $100
- Mike — $235.52
Totaling **$2,042.96**.`
  const { corrected, mistakes } = verifyArithmetic(input)
  assertEquals(mistakes, 1)
  // 115 + 100 + 235.52 = $450.52
  assert(corrected.includes("$450.52"), `expected $450.52 in: ${corrected}`)
  // Bold preserved
  assert(corrected.includes("**$450.52**"), `expected bold preserved: ${corrected}`)
})

Deno.test("bullet list with 'Total outstanding: $X' is corrected", () => {
  const input = `- A — $100
- B — $200
- C — $300
Total outstanding: $999`
  const { corrected, mistakes } = verifyArithmetic(input)
  assertEquals(mistakes, 1)
  assert(corrected.includes("$600"))
})

Deno.test("does NOT fire when a bullet line lacks a $ amount", () => {
  // Conservative: if any bullet line has no $amount, we abort.
  // This protects against partial enumerations like Q19's compound:
  // "totaling $2,042" but only 3 of 6 items shown inline.
  const input = `You have 6 overdue:
- TestClient — $115
- Joel Paulraj — $100
- (and 4 more)
Total: $2,042.96`
  const { mistakes } = verifyArithmetic(input)
  assertEquals(mistakes, 0) // bullet "(and 4 more)" lacks $amount → abort
})

Deno.test("does NOT fire on single-bullet 'list' (not enough to verify)", () => {
  const input = `You have one overdue:
- Steven — $535
Total: $999`
  const { mistakes } = verifyArithmetic(input)
  // Single-line is not a list; we don't try to verify.
  assertEquals(mistakes, 0)
})

Deno.test("works with asterisk bullets", () => {
  const input = `* A — $100
* B — $200
Total: $999`
  const { corrected, mistakes } = verifyArithmetic(input)
  assertEquals(mistakes, 1)
  assert(corrected.includes("$300"))
})

// ─── Headline-then-list pattern ──────────────────────────────────────────────
//
// Sonnet often LEADS with the answer ("You have 6 unpaid totaling $X")
// then enumerates. The verifier needs to catch wrong totals in that
// shape too, not just the trailing "Total: $X" form.

Deno.test("'totaling $X:' before list with wrong total is corrected", () => {
  const input = `You have **6 invoices awaiting payment**, totaling **$2,042.96**:
- Steven — $535
- Mike — $235.52
- Reed residence — $83.40
- Joel Paulraj — $100
- TestClient — $115
- Hikaru Nakamura — $274.04`
  const { corrected, mistakes } = verifyArithmetic(input)
  assertEquals(mistakes, 1)
  // 535 + 235.52 + 83.40 + 100 + 115 + 274.04 = 1342.96
  assert(corrected.includes("$1,342.96"), `expected $1,342.96 in: ${corrected}`)
  assert(!corrected.includes("$2,042.96"))
  // Bold preserved
  assert(corrected.includes("**$1,342.96**"))
})

Deno.test("'totaling $X:' before list with correct total is unchanged", () => {
  const input = `Three invoices totaling $1,044.56:
- Steven — $535
- Hikaru — $274.04
- Mike — $235.52`
  const { mistakes } = verifyArithmetic(input)
  assertEquals(mistakes, 0)
})

Deno.test("'a total of $X' phrasing is also caught", () => {
  const input = `Late invoices come to a total of $999:
- A — $100
- B — $200`
  const { corrected, mistakes } = verifyArithmetic(input)
  assertEquals(mistakes, 1)
  assert(corrected.includes("$300"))
})

Deno.test("headline pattern aborts when any bullet lacks $", () => {
  const input = `You have 6 unpaid totaling $2,042.96:
- A — $100
- (and several more)
- B — $200`
  const { mistakes } = verifyArithmetic(input)
  assertEquals(mistakes, 0) // partial enumeration → don't fix
})
