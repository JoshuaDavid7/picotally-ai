// deno test supabase/functions/_shared/usage_cost.test.ts
//
// Pins the guard that keeps the cost ledger clean: only positive, finite,
// whole-unit counts are ever persisted. A regression here would either corrupt
// per-user cost totals (negatives/NaN) or spam no-op RPC writes (zeros).
import { assertEquals } from "https://deno.land/std@0.224.0/assert/mod.ts"
import { costableEntries } from "./usage_cost.ts"

Deno.test("keeps positive counts, rounded to whole units", () => {
  assertEquals(
    costableEntries({ input_tokens: 5200, output_tokens: 183.6 }),
    [["input_tokens", 5200], ["output_tokens", 184]],
  )
})

Deno.test("drops zero, negative, NaN and Infinity", () => {
  assertEquals(
    costableEntries({
      a: 0, // zero → no-op write, skip
      b: -5, // negative → would corrupt totals, skip
      c: Number.NaN, // garbage, skip
      d: Number.POSITIVE_INFINITY, // garbage, skip
      e: 12, // the only real one
    }),
    [["e", 12]],
  )
})

Deno.test("empty map yields no entries", () => {
  assertEquals(costableEntries({}), [])
})

Deno.test("rounds first, so a sub-half count drops instead of writing a 0", () => {
  assertEquals(costableEntries({ x: 0.4 }), []) // 0.4 → 0 → dropped
  assertEquals(costableEntries({ y: 0.5 }), [["y", 1]]) // 0.5 → 1 → kept
})
