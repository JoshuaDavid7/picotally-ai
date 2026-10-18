// Tests that the business-context block stays prompt-cache-stable.
//
// Run with:
//   deno test supabase/functions/process_voice_command/context_cache.test.ts
//
// WHY THIS MATTERS (cost, not correctness):
// The Anthropic request places a `cache_control` breakpoint on the context
// block — the big (~12K-token) business snapshot. Anthropic prompt caching
// is PREFIX-based: a cached block only hits if it is byte-identical to a
// prior request. If any per-command volatile value (the current timestamp,
// the current screen) sits INSIDE that block, the block changes every
// command and the cache misses every time — silently paying ~10x on the
// single largest input on the dominant cost line (Claude).
//
// So volatile session state must live in the UNCACHED suffix next to the
// transcript, never in buildContextBlock's output. These tests pin that:
//   1. Two contexts that differ ONLY in `now` / `currentScreen` must
//      produce identical context-block text (i.e. those values don't leak).
//   2. The block must not contain the "Current time" / "Current screen"
//      labels at all.
//
// If someone re-adds a timestamp to the snapshot, these fail loudly.

import { assert, assertEquals } from "https://deno.land/std@0.224.0/assert/mod.ts"

import { buildContextBlock } from "./index.ts"

// A representative snapshot. The data is identical between the two cases
// below; only the volatile session fields differ.
const baseContext: Record<string, unknown> = {
  profile: {
    businessName: "Pico Plumbing",
    currency: "$",
    hourlyRate: 90,
    taxRate: 8,
  },
  stats: {
    totalOutstanding: 1200,
    outstandingCount: 3,
    sentCount: 3,
    draftCount: 1,
    monthlyRevenue: 5000,
    monthlyExpenses: 1200,
    monthlyProfit: 3800,
    totalCustomers: 2,
  },
  recentCustomers: ["Mike Johnson", "Vishy Anand"],
  customers: [
    { name: "Mike Johnson", balance: 320, phone: "555-0001", email: "m@x.com", jobCount: 2, totalBilled: 1820, totalPaid: 1500 },
    { name: "Vishy Anand", balance: 0, phone: "555-0002", email: "v@x.com", jobCount: 1, totalBilled: 600, totalPaid: 600 },
  ],
  jobs: [
    {
      id: "j1",
      title: "Bathroom remodel",
      clientName: "Mike Johnson",
      status: "sent",
      type: "invoice",
      totalAmount: 1500,
      amountPaid: 0,
      amountDue: 1500,
      createdAt: "2026-03-13T10:00:00Z",
      materials: [{ item: "Pipes", quantity: 1, cost: 1098 }],
    },
  ],
  recentExpenses: [
    { date: "2026-03-15T10:00:00Z", description: "Pipes", vendor: "Home Depot", category: "materials", amount: 87.5 },
  ],
  recentPayments: [
    { date: "2026-04-01T10:00:00Z", amount: 600, method: "cash", clientName: "Vishy Anand" },
  ],
  recentNotes: [],
}

Deno.test("buildContextBlock is cache-stable: identical when only now/currentScreen differ", () => {
  const morningOnHome = {
    ...baseContext,
    now: "2026-05-31T08:00:00Z",
    currentScreen: "home",
  }
  const nightOnExpenses = {
    ...baseContext,
    now: "2026-05-31T23:59:59Z",
    currentScreen: "expenses",
  }
  // Same data, different session state → MUST be byte-identical so the
  // prompt cache hits across commands.
  assertEquals(buildContextBlock(morningOnHome), buildContextBlock(nightOnExpenses))
})

Deno.test("buildContextBlock omits volatile session state (it belongs in the uncached suffix)", () => {
  const out = buildContextBlock({
    ...baseContext,
    now: "2026-05-31T08:00:00Z",
    currentScreen: "jobs",
  })
  assert(!out.includes("Current time"), "context block must not embed the current timestamp")
  assert(!out.includes("Current screen"), "context block must not embed the current screen")
  // Sanity: it still contains the stable business data.
  assert(out.includes("Pico Plumbing"))
  assert(out.includes("Mike Johnson"))
})
