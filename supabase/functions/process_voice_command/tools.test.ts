// Tests for the query tool executors.
//
// Run with:
//   deno test supabase/functions/process_voice_command/tools.test.ts
//
// These pin the contract the model relies on:
//   - query_materials excludes drafts by default, sums actualSupplierCost
//   - query_expenses returns deduplicated rows, flags partial at the cap
//   - query_jobs filters by status and date correctly
//   - query_payments filters by date / method / client
//   - query_customer_balance is a single-row lookup, no fabrication
//   - lookup_material_prices returns a real historical price or asks —
//     never fabricates; distinguishes sell vs cost basis; flags conflicts
//
// The fixtures are intentionally small but representative of the real
// snapshot shape — same field names the production payload uses.

import {
  assertEquals,
  assert,
} from "https://deno.land/std@0.224.0/assert/mod.ts"

import {
  executeQueryMaterials,
  executeQueryExpenses,
  executeQueryJobs,
  executeQueryPayments,
  executeQueryCustomerBalance,
  executeLookupMaterialPrices,
  dispatchQueryTool,
  dispatchQueryToolWithSemantics,
  semanticPhraseFor,
  isQueryToolName,
} from "./tools.ts"

// ─── Shared fixture: a realistic-shape business context ─────────────────────

const fixture = {
  jobs: [
    {
      id: "job1",
      title: "Bathroom remodel",
      clientName: "Vishy Anand",
      status: "sent",
      type: "invoice",
      totalAmount: 1500,
      amountPaid: 0,
      amountDue: 1500,
      createdAt: "2026-03-13T10:00:00Z",
      invoiceNumber: "INV-0001",
      materials: [
        { item: "Pipes", quantity: 1, cost: 1098, actualSupplierCost: 915 },
        { item: "Wire", quantity: 5, cost: 60, actualSupplierCost: 50 },
      ],
    },
    {
      id: "job2",
      title: "Tile remodel",
      clientName: "Daniel Mowery",
      status: "draft",
      type: "invoice",
      totalAmount: 8800,
      amountPaid: 0,
      amountDue: 8800,
      createdAt: "2026-04-12T10:00:00Z",
      invoiceNumber: "INV-0048",
      materials: [
        { item: "Pipe", quantity: 2, cost: 172.80, actualSupplierCost: 144 },
      ],
    },
    {
      id: "job3",
      title: "Heater install",
      clientName: "Mike Johnson",
      status: "paid",
      type: "invoice",
      totalAmount: 600,
      amountPaid: 600,
      amountDue: 0,
      createdAt: "2026-04-01T10:00:00Z",
      invoiceNumber: "INV-0002",
      materials: [
        { item: "Heater", quantity: 1, cost: 600, actualSupplierCost: 500 },
      ],
    },
  ],
  recentExpenses: [
    {
      date: "2026-03-15T10:00:00Z",
      description: "Pipes from Home Depot",
      vendor: "Home Depot",
      category: "materials",
      amount: 87.50,
      taxDeductible: true,
      source: "logged",
      hasReceipt: true,
    },
    {
      date: "2026-03-20T10:00:00Z",
      description: "Lunch with client",
      vendor: "Cafe",
      category: "meals",
      amount: 45,
      taxDeductible: false,
      source: "logged",
      hasReceipt: false,
    },
  ],
  recentPayments: [
    { date: "2026-04-01T10:00:00Z", amount: 600, method: "cash", clientName: "Mike Johnson", jobTitle: "Heater install", reference: "" },
    { date: "2026-03-15T10:00:00Z", amount: 1000, method: "bank_transfer", clientName: "Steven", jobTitle: "Plumbing", reference: "" },
  ],
  customers: [
    { name: "Vishy Anand", phone: "555-0001", email: "v@x.com", jobCount: 1, totalBilled: 1500, totalPaid: 0, balance: 1500 },
    { name: "Mike Johnson", phone: "555-0002", email: "m@x.com", jobCount: 1, totalBilled: 600, totalPaid: 600, balance: 0 },
    { name: "Daniel Mowery", phone: "555-0003", email: "d@x.com", jobCount: 1, totalBilled: 8800, totalPaid: 0, balance: 8800 },
  ],
  stats: {},
}

// ─── query_materials ─────────────────────────────────────────────────────────

Deno.test("query_materials with keyword 'pipe' excludes drafts by default", () => {
  const result = executeQueryMaterials(fixture, { keyword: "pipe" })
  // Default invoiceStatus is sent_or_paid → only Vishy's pipes (sent)
  assertEquals(result.count, 1)
  assertEquals(result.total, 915)
  assertEquals(result.items[0].clientName, "Vishy Anand")
  assert(result.note?.includes("Drafts excluded"))
})

Deno.test("query_materials with invoiceStatus 'any' includes drafts", () => {
  const result = executeQueryMaterials(fixture, { keyword: "pipe", invoiceStatus: "any" })
  // Vishy's pipes (915) + Daniel Mowery's draft pipe (144)
  assertEquals(result.count, 2)
  assertEquals(result.total, 1059)
})

Deno.test("query_materials with invoiceStatus 'draft_only' returns just drafts", () => {
  const result = executeQueryMaterials(fixture, { keyword: "pipe", invoiceStatus: "draft_only" })
  assertEquals(result.count, 1)
  assertEquals(result.total, 144)
  assert(result.note?.includes("draft"))
})

Deno.test("query_materials uses actualSupplierCost (pre-markup) for total", () => {
  // Daniel Mowery's pipe has cost=172.80 (post-markup) and actualSupplierCost=144 (pre-markup).
  // The tool MUST return 144, not 172.80.
  const result = executeQueryMaterials(fixture, { keyword: "pipe", invoiceStatus: "draft_only" })
  assertEquals(result.total, 144)
  assertEquals(result.items[0].actualSupplierCost, 144)
})

Deno.test("query_materials filters by date range", () => {
  const result = executeQueryMaterials(fixture, {
    invoiceStatus: "sent_or_paid",
    dateFrom: "2026-04-01",
    dateTo: "2026-04-30",
  })
  // April only — Mike's heater
  assertEquals(result.count, 1)
  assertEquals(result.items[0].item, "Heater")
})

Deno.test("query_materials filters by clientName", () => {
  const result = executeQueryMaterials(fixture, { clientName: "Vishy" })
  assertEquals(result.count, 2) // pipes + wire
  assertEquals(result.total, 965)
})

Deno.test("query_materials with no keyword returns all materials", () => {
  const result = executeQueryMaterials(fixture, { invoiceStatus: "sent_or_paid" })
  // Vishy: 915 + 50, Mike: 500 — Daniel excluded as draft
  assertEquals(result.total, 1465)
  assertEquals(result.count, 3)
})

// ─── query_expenses ──────────────────────────────────────────────────────────

Deno.test("query_expenses filters by category", () => {
  const result = executeQueryExpenses(fixture, { category: "materials" })
  assertEquals(result.count, 1)
  assertEquals(result.total, 87.50)
})

Deno.test("query_expenses filters by vendor", () => {
  const result = executeQueryExpenses(fixture, { vendor: "Home Depot" })
  assertEquals(result.count, 1)
  assertEquals(result.total, 87.50)
})

Deno.test("query_expenses filters by taxDeductible", () => {
  const result = executeQueryExpenses(fixture, { taxDeductible: true })
  assertEquals(result.count, 1)
  assertEquals(result.total, 87.50)
})

Deno.test("query_expenses respects date range", () => {
  const result = executeQueryExpenses(fixture, {
    dateFrom: "2026-03-16",
    dateTo: "2026-03-31",
  })
  assertEquals(result.count, 1) // Just the lunch, March 15 excluded
  assertEquals(result.items[0].vendor, "Cafe")
})

// ─── query_jobs ──────────────────────────────────────────────────────────────

Deno.test("query_jobs status='unpaid' returns sent + partially_paid", () => {
  const result = executeQueryJobs(fixture, { status: "unpaid" })
  assertEquals(result.count, 1)
  assertEquals(result.items[0].id, "job1")
})

Deno.test("query_jobs status='unpaid' excludes sent jobs with amountDue=0 (status drift)", () => {
  // Regression for the dashboard-vs-AI count mismatch: a fully paid invoice
  // whose status column never flipped to 'paid' must not count as unpaid.
  const driftedFixture = {
    ...fixture,
    jobs: [
      ...(fixture.jobs as Array<Record<string, unknown>>),
      {
        id: "drifted",
        title: "Old job, paid but status='sent'",
        clientName: "Stale",
        status: "sent",
        type: "invoice",
        totalAmount: 500,
        amountPaid: 500,
        amountDue: 0,
        createdAt: "2026-02-01T10:00:00Z",
        invoiceNumber: "INV-0099",
        materials: [],
      },
    ],
  }
  const result = executeQueryJobs(driftedFixture, { status: "unpaid" })
  // Still just job1 — the drifted row is genuinely paid (total == paid), so the
  // derived owed amount is 0 and it's correctly excluded.
  assertEquals(result.count, 1)
  assertEquals(result.items[0].id, "job1")
})

Deno.test("query_jobs status='unpaid' INCLUDES a sent invoice whose STORED amountDue is 0 but total>paid", () => {
  // The financial-truth bug (2026-06-20): a genuinely unpaid $850 invoice had
  // its stored amountDue left at 0, so it was silently dropped from 'unpaid'
  // and the AI told the user a customer owed $41 when they owed $891. Owed MUST
  // be derived from total - paid, never the unreliable stored field.
  const f = {
    ...fixture,
    jobs: [
      {
        id: "owed850",
        title: "Unpaid invoice, stored amountDue wrongly 0",
        clientName: "Greg Anderson",
        status: "sent",
        type: "invoice",
        totalAmount: 850,
        amountPaid: 0,
        amountDue: 0, // stored field is WRONG
        createdAt: "2026-03-01T10:00:00Z",
        invoiceNumber: "INV-0100",
        materials: [],
      },
    ],
  }
  const result = executeQueryJobs(f, { status: "unpaid" })
  assertEquals(result.count, 1)
  assertEquals(result.items[0].id, "owed850")
  // amountDue is derived (total - paid), not the stored 0.
  assertEquals(result.items[0].amountDue, 850)
})

Deno.test("query_jobs status='unpaid' excludes sent quotes (a quote isn't 'paid')", () => {
  // Regression for the "11 invoices awaiting payment" answer that should
  // have been 6: a sent quote with money owing was counting as unpaid,
  // inflating the invoice count by however many open quotes the user has.
  const fixtureWithQuote = {
    ...fixture,
    jobs: [
      ...(fixture.jobs as Array<Record<string, unknown>>),
      {
        id: "quote_open",
        title: "Quote awaiting decision",
        clientName: "Godwin",
        status: "sent",
        type: "quote",
        totalAmount: 850,
        amountPaid: 0,
        amountDue: 850,
        createdAt: "2026-04-01T10:00:00Z",
        invoiceNumber: "QUO-0010",
        materials: [],
      },
    ],
  }
  const result = executeQueryJobs(fixtureWithQuote, { status: "unpaid" })
  // Still just job1 — the open quote is filtered by type check.
  assertEquals(result.count, 1)
  assertEquals(result.items[0].id, "job1")
})

Deno.test("query_jobs status='paid' returns paid only", () => {
  const result = executeQueryJobs(fixture, { status: "paid" })
  assertEquals(result.count, 1)
  assertEquals(result.items[0].id, "job3")
})

Deno.test("query_jobs status='draft' returns drafts only", () => {
  const result = executeQueryJobs(fixture, { status: "draft" })
  assertEquals(result.count, 1)
  assertEquals(result.items[0].id, "job2")
})

Deno.test("query_jobs status='any' is the default and returns all", () => {
  const result = executeQueryJobs(fixture, {})
  assertEquals(result.count, 3)
})

// ─── query_jobs status='late' ────────────────────────────────────────────────
//
// "Late" is a synthetic status — sent invoices with balance owing AND a
// due date in the past. The fixture's snapshot `now` pins the comparison
// date so the test is deterministic across days.

const lateFixture = {
  ...fixture,
  now: "2026-04-26T15:00:00Z",
  jobs: [
    // Sent, fully unpaid, due date 13 days ago → late
    {
      id: "late1",
      title: "Bathroom remodel",
      clientName: "Mike Johnson",
      status: "sent",
      type: "invoice",
      totalAmount: 1500,
      amountPaid: 0,
      amountDue: 1500,
      createdAt: "2026-03-25T10:00:00Z",
      dueDate: "2026-04-13T10:00:00Z",
      invoiceNumber: "INV-0001",
      materials: [],
    },
    // Sent, partially paid, due date 5 days ago → late (amountDue > 0)
    {
      id: "late2",
      title: "Kitchen plumbing",
      clientName: "Steven",
      status: "sent",
      type: "invoice",
      totalAmount: 800,
      amountPaid: 300,
      amountDue: 500,
      createdAt: "2026-04-01T10:00:00Z",
      dueDate: "2026-04-21T10:00:00Z",
      invoiceNumber: "INV-0002",
      materials: [],
    },
    // Sent, due date 5 days from now → NOT late
    {
      id: "future",
      title: "HVAC service",
      clientName: "Vishy Anand",
      status: "sent",
      type: "invoice",
      totalAmount: 600,
      amountPaid: 0,
      amountDue: 600,
      createdAt: "2026-04-20T10:00:00Z",
      dueDate: "2026-05-01T10:00:00Z",
      invoiceNumber: "INV-0003",
      materials: [],
    },
    // Sent, FULLY paid, due date 10 days ago → NOT late (no balance owing)
    {
      id: "paid_in_past",
      title: "Heater install",
      clientName: "Daniel Mowery",
      status: "sent",
      type: "invoice",
      totalAmount: 600,
      amountPaid: 600,
      amountDue: 0,
      createdAt: "2026-04-01T10:00:00Z",
      dueDate: "2026-04-16T10:00:00Z",
      invoiceNumber: "INV-0004",
      materials: [],
    },
    // Quote with overdue due date → NOT late (quotes don't have due dates)
    {
      id: "quote",
      title: "Quote for fence",
      clientName: "Joel",
      status: "sent",
      type: "quote",
      totalAmount: 2000,
      amountPaid: 0,
      amountDue: 2000,
      createdAt: "2026-03-01T10:00:00Z",
      dueDate: "2026-04-01T10:00:00Z",
      invoiceNumber: "QUO-0001",
      materials: [],
    },
    // Sent, no due date → excluded but counted in note
    {
      id: "no_due",
      title: "Paint job",
      clientName: "Reed",
      status: "sent",
      type: "invoice",
      totalAmount: 400,
      amountPaid: 0,
      amountDue: 400,
      createdAt: "2026-04-10T10:00:00Z",
      dueDate: null,
      invoiceNumber: "INV-0005",
      materials: [],
    },
    // Draft with overdue due date (somehow) → NOT late (drafts not sent)
    {
      id: "draft_overdue",
      title: "Draft only",
      clientName: "Hikaru",
      status: "draft",
      type: "invoice",
      totalAmount: 100,
      amountPaid: 0,
      amountDue: 100,
      createdAt: "2026-03-01T10:00:00Z",
      dueDate: "2026-04-01T10:00:00Z",
      invoiceNumber: "INV-0006",
      materials: [],
    },
  ],
}

Deno.test("query_jobs status='late' returns only sent invoices with balance and past due date", () => {
  const result = executeQueryJobs(lateFixture, { status: "late" })
  // 3: late1 + late2 (stored due dates) + no_due, which has NO stored due date
  // and is now assessed via net-14 terms (created 2026-04-10 + 14 = 2026-04-24,
  // before now 2026-04-26) instead of being silently dropped.
  assertEquals(result.count, 3)
  // Most overdue first
  assertEquals(result.items[0].id, "late1") // 13 days late
  assertEquals(result.items[1].id, "late2") // 5 days late
  assertEquals(result.items[2].id, "no_due") // 2 days late (net-14 fallback)
  assertEquals(result.items[0].daysLate, 13)
  assertEquals(result.items[1].daysLate, 5)
  assertEquals(result.items[2].daysLate, 2)
})

Deno.test("query_jobs status='late' discloses the net-14 fallback in the note", () => {
  const result = executeQueryJobs(lateFixture, { status: "late" })
  assert(result.note?.includes("Late ="))
  // The "no_due" sent invoice has no stored due date, so it's assessed with
  // net-14 terms — disclosed in the note rather than silently dropped.
  assert(result.note?.includes("net-14 terms"))
  assert(result.note?.includes("1 had no due date set"))
})

Deno.test("query_jobs status='late' excludes paid invoices, drafts, quotes, and future-dated", () => {
  const result = executeQueryJobs(lateFixture, { status: "late" })
  const ids = result.items.map((i) => i.id)
  assert(!ids.includes("future"))
  assert(!ids.includes("paid_in_past"))
  assert(!ids.includes("quote"))
  assert(!ids.includes("draft_overdue"))
})

Deno.test("query_jobs status='late' total sums totalAmount of late invoices", () => {
  const result = executeQueryJobs(lateFixture, { status: "late" })
  // late1: 1500 + late2: 800 + no_due: 400 (net-14 fallback) = 2700
  assertEquals(result.total, 2700)
})

Deno.test("query_jobs status='late' uses snapshot.now over server clock", () => {
  // Pin a different `now` and verify the lateness comparison shifts.
  const yesterday = { ...lateFixture, now: "2026-04-12T15:00:00Z" }
  const result = executeQueryJobs(yesterday, { status: "late" })
  // On 2026-04-12, only late1 (due 2026-04-13) is NOT yet late.
  // late2 (due 2026-04-21) is NOT yet late either.
  // → 0 late invoices
  assertEquals(result.count, 0)
})

Deno.test("query_jobs surfaces daysLate on non-late queries when applicable", () => {
  // The future invoice is due 2026-05-01, today is 2026-04-26.
  // → Not late, no daysLate field expected.
  const result = executeQueryJobs(lateFixture, { status: "sent" })
  const future = result.items.find((i) => i.id === "future")
  assert(future != null)
  assertEquals((future as Record<string, unknown>).daysLate, undefined)
  // late1 is 13 days past due → daysLate=13 even on a non-late query.
  const late1 = result.items.find((i) => i.id === "late1")
  assertEquals((late1 as Record<string, unknown>).daysLate, 13)
})

// ─── query_payments ──────────────────────────────────────────────────────────

Deno.test("query_payments unfiltered total = authoritative sum(jobs.amountPaid)", () => {
  // "How much have I collected?" (no filter) must come from jobs.amountPaid —
  // every dollar received, INCLUDING jobs paid without a recorded payment_events
  // row — not the (capped) payment list. Here the recorded payment rows sum to
  // only 1000, but two invoices were paid totalling 1500; the quote is excluded.
  // This is what makes the text (local) and voice (cloud) brains agree.
  const ctx = {
    jobs: [
      { id: "j1", type: "invoice", amountPaid: 1000 },
      { id: "j2", type: "invoice", amountPaid: 500 },
      { id: "q1", type: "quote", amountPaid: 0 },
    ],
    recentPayments: [{ amount: 1000, method: "cash", jobId: "j1" }],
  }
  const result = executeQueryPayments(ctx, {})
  assertEquals(result.total, 1500)
})

Deno.test("query_payments filters by method", () => {
  const result = executeQueryPayments(fixture, { method: "cash" })
  assertEquals(result.count, 1)
  assertEquals(result.total, 600)
})

Deno.test("query_payments filters by clientName", () => {
  const result = executeQueryPayments(fixture, { clientName: "Mike" })
  assertEquals(result.count, 1)
  assertEquals(result.total, 600)
})

// ─── query_customer_balance ─────────────────────────────────────────────────

Deno.test("query_customer_balance returns the right customer", () => {
  const result = executeQueryCustomerBalance(fixture, { clientName: "Mike" })
  assertEquals(result.count, 1)
  assertEquals(result.items[0].name, "Mike Johnson")
  assertEquals(result.total, 0)
})

Deno.test("query_customer_balance returns count=0 for unknown customer", () => {
  const result = executeQueryCustomerBalance(fixture, { clientName: "Nobody" })
  assertEquals(result.count, 0)
  assert(result.note?.includes("No customer found"))
})

Deno.test("query_customer_balance is case-insensitive", () => {
  const result = executeQueryCustomerBalance(fixture, { clientName: "VISHY" })
  assertEquals(result.count, 1)
  assertEquals(result.items[0].name, "Vishy Anand")
  assertEquals(result.total, 1500)
})

Deno.test("query_customer_balance prefers shorter (closer) match on collision", () => {
  // If we had both "Mike" and "Mike Johnson", searching "Mike" should
  // pick "Mike" (length 4) over "Mike Johnson" (length 12). The fixture
  // doesn't have that case but we can synthesize it.
  const c = {
    customers: [
      { name: "Mike Johnson", balance: 100 },
      { name: "Mike", balance: 50 },
    ],
    // Balance is now DERIVED from the customer's invoices (not the stored
    // aggregate), so give "Mike" a $50-owed invoice. The stored `balance: 50`
    // above is intentionally ignored by the tool now.
    jobs: [
      {
        clientName: "Mike",
        type: "invoice",
        status: "sent",
        totalAmount: 50,
        amountPaid: 0,
      },
    ],
  }
  const result = executeQueryCustomerBalance(c, { clientName: "Mike" })
  assertEquals(result.items[0].name, "Mike")
  assertEquals(result.total, 50)
})

Deno.test("query_customer_balance EXCLUDES quotes (a quote is not a receivable)", () => {
  // Regression for the truth bug: an $850 QUOTE was counted in total_billed,
  // making a customer who owed $41 on his one unpaid invoice read as owing
  // $891. Quotes must never count toward an outstanding balance.
  const c = {
    customers: [{ name: "Greg", balance: 891 }],
    jobs: [
      { clientName: "Greg", type: "invoice", status: "sent", totalAmount: 40.77, amountPaid: 0 },
      { clientName: "Greg", type: "quote", status: "sent", totalAmount: 850, amountPaid: 0 },
      { clientName: "Greg", type: "invoice", status: "sent", totalAmount: 100, amountPaid: 100 },
    ],
  }
  const result = executeQueryCustomerBalance(c, { clientName: "Greg" })
  assertEquals(result.total, 40.77) // the quote is excluded; paid invoice nets to 0
  assertEquals(result.items[0].totalBilled, 140.77)
})

// ─── lookup_material_prices ──────────────────────────────────────────────────
//
// The auto-pricing source for new quotes/invoices. Contract under test:
//   - returns a per-unit SELL price from past invoices/quotes (cost/qty),
//     preferring sent/paid invoices (authoritative)
//   - falls back to a COST basis from receipt ocrItems / supplier cost,
//     clearly labelled so the model asks for margin instead of billing cost
//   - returns status "not_found" (never a fabricated number) when there's
//     no history
//   - flags a WIDE priceRange so the model asks instead of guessing
//
// `cost` on a material is the post-markup LINE total; dividing by quantity
// yields the per-unit sell price. `actualSupplierCost` is the pre-markup
// line total (a cost basis). Receipt `ocrItems` carry a per-unit unitPrice.

const pricingFixture = {
  now: "2026-05-01T10:00:00Z",
  jobs: [
    // Sent invoice → authoritative sell prices.
    // PEX: cost 90 over qty 100ft → 0.90/ft sell; supplier 70 → 0.70/ft cost.
    // Water heater: cost 600 → 600 sell; supplier 500.
    {
      id: "p_job1",
      clientName: "Vishy Anand",
      status: "sent",
      type: "invoice",
      createdAt: "2026-03-13T10:00:00Z",
      materials: [
        { item: "1/2in PEX pipe 100ft", quantity: 100, cost: 90, actualSupplierCost: 70 },
        { item: "Water heater 50gal", quantity: 1, cost: 600, actualSupplierCost: 500 },
      ],
    },
    // A more recent PAID invoice with a much higher heater price → conflict.
    {
      id: "p_job2",
      clientName: "Mike Johnson",
      status: "paid",
      type: "invoice",
      createdAt: "2026-04-20T10:00:00Z",
      materials: [
        { item: "Water heater 50gal", quantity: 1, cost: 900, actualSupplierCost: 720 },
      ],
    },
    // A DRAFT quote → non-authoritative sell price for shutoff valves.
    // cost 60 over qty 4 → 15/ea sell; supplier 40 → 10/ea cost.
    {
      id: "p_job3",
      clientName: "Daniel Mowery",
      status: "draft",
      type: "quote",
      createdAt: "2026-04-25T10:00:00Z",
      materials: [
        { item: "Brass shutoff valve", quantity: 4, cost: 60, actualSupplierCost: 40 },
      ],
    },
  ],
  recentExpenses: [
    // Scanned receipt → COST basis only (no sell-price history for these).
    {
      date: "2026-04-10T10:00:00Z",
      vendor: "Home Depot",
      category: "materials",
      amount: 12.5,
      source: "logged",
      hasReceipt: true,
      ocrItems: [
        { name: "Wax ring", quantity: 2, unitPrice: 3.25, totalPrice: 6.5 },
        // No unitPrice → derive per-unit from totalPrice / quantity = 4.00.
        { name: "Teflon tape", quantity: 1, totalPrice: 4.0 },
      ],
    },
  ],
  customers: [],
  stats: {},
}

Deno.test("lookup_material_prices suggests a sell price from a sent invoice", () => {
  const r = executeLookupMaterialPrices(pricingFixture, { items: ["PEX pipe"] })
  assertEquals(r.count, 1)
  assertEquals(r.items.length, 1)
  const item = r.items[0] as Record<string, unknown>
  assertEquals(item.status, "found")
  const sugg = item.suggestion as Record<string, unknown>
  assertEquals(sugg.basis, "sell")
  assertEquals(sugg.unitPrice, 0.9) // 90 / 100ft
  const range = item.priceRange as Record<string, number>
  assertEquals(range.min, 0.9)
  assertEquals(range.max, 0.9) // single source → tight
})

Deno.test("lookup_material_prices flags conflicting prices with a wide range", () => {
  const r = executeLookupMaterialPrices(pricingFixture, { items: ["water heater"] })
  const item = r.items[0] as Record<string, unknown>
  assertEquals(item.status, "found")
  const sugg = item.suggestion as Record<string, unknown>
  // Picks the most-recent authoritative sell price (900, from the paid invoice).
  assertEquals(sugg.unitPrice, 900)
  const range = item.priceRange as Record<string, number>
  assertEquals(range.min, 600)
  assertEquals(range.max, 900)
  assert(r.note?.includes("WIDE priceRange"))
})

Deno.test("lookup_material_prices returns not_found instead of inventing a price", () => {
  const r = executeLookupMaterialPrices(pricingFixture, { items: ["circuit breaker"] })
  assertEquals(r.count, 0) // nothing found
  const item = r.items[0] as Record<string, unknown>
  assertEquals(item.status, "not_found")
  assertEquals(item.suggestion, null)
  assertEquals(item.priceRange, null)
  assert(r.note?.includes("not_found"))
})

Deno.test("lookup_material_prices labels a receipt-only price as cost basis", () => {
  const r = executeLookupMaterialPrices(pricingFixture, { items: ["wax ring"] })
  const item = r.items[0] as Record<string, unknown>
  assertEquals(item.status, "found")
  const sugg = item.suggestion as Record<string, unknown>
  assertEquals(sugg.basis, "cost")
  assertEquals(sugg.source, "receipt")
  assertEquals(sugg.unitPrice, 3.25)
  assert(r.note?.includes("COST basis"))
})

Deno.test("lookup_material_prices derives unit price from totalPrice when unitPrice missing", () => {
  const r = executeLookupMaterialPrices(pricingFixture, { items: ["teflon tape"] })
  const item = r.items[0] as Record<string, unknown>
  const sugg = item.suggestion as Record<string, unknown>
  assertEquals(sugg.unitPrice, 4.0) // 4.00 total / qty 1
})

Deno.test("lookup_material_prices uses a draft quote's sell price (non-authoritative)", () => {
  const r = executeLookupMaterialPrices(pricingFixture, { items: ["shutoff valve"] })
  const item = r.items[0] as Record<string, unknown>
  assertEquals(item.status, "found")
  const sugg = item.suggestion as Record<string, unknown>
  assertEquals(sugg.basis, "sell")
  assertEquals(sugg.unitPrice, 15) // 60 / 4
})

Deno.test("lookup_material_prices handles a mixed batch in one call", () => {
  const r = executeLookupMaterialPrices(pricingFixture, {
    items: ["PEX pipe", "circuit breaker", "shutoff valve"],
  })
  assertEquals(r.items.length, 3) // one row per requested item
  assertEquals(r.count, 2) // PEX + shutoff found, breaker not
  const byQuery = Object.fromEntries(
    (r.items as Array<Record<string, unknown>>).map((i) => [i.query, i.status]),
  )
  assertEquals(byQuery["PEX pipe"], "found")
  assertEquals(byQuery["circuit breaker"], "not_found")
  assertEquals(byQuery["shutoff valve"], "found")
  assert(r.note?.includes("not_found"))
})

Deno.test("lookup_material_prices coerces a single string to an array", () => {
  const r = executeLookupMaterialPrices(pricingFixture, { items: "PEX pipe" })
  assertEquals(r.count, 1)
  assertEquals((r.items[0] as Record<string, unknown>).status, "found")
})

Deno.test("lookup_material_prices returns empty + prompt when no items given", () => {
  const r = executeLookupMaterialPrices(pricingFixture, { items: [] })
  assertEquals(r.count, 0)
  assertEquals(r.items.length, 0)
  assert(r.note?.includes("No item names"))
})

// ─── Dispatcher ──────────────────────────────────────────────────────────────

Deno.test("dispatchQueryTool routes to the right executor", () => {
  const r = dispatchQueryTool("query_materials", { keyword: "pipe" }, fixture)
  assertEquals(r.total, 915) // matches the query_materials default
})

Deno.test("dispatchQueryTool routes lookup_material_prices", () => {
  const r = dispatchQueryTool("lookup_material_prices", { items: ["PEX pipe"] }, pricingFixture)
  assertEquals(r.count, 1)
  assertEquals((r.items[0] as Record<string, unknown>).status, "found")
})

Deno.test("dispatchQueryTool returns helpful note on unknown name", () => {
  const r = dispatchQueryTool("not_a_tool", {}, fixture)
  assertEquals(r.count, 0)
  assert(r.note?.includes("Unknown tool"))
})

Deno.test("isQueryToolName recognizes all six tools", () => {
  assert(isQueryToolName("query_materials"))
  assert(isQueryToolName("query_expenses"))
  assert(isQueryToolName("query_jobs"))
  assert(isQueryToolName("query_payments"))
  assert(isQueryToolName("query_customer_balance"))
  assert(isQueryToolName("lookup_material_prices"))
  assert(!isQueryToolName("respond"))
  assert(!isQueryToolName("create_invoice"))
})


// ─── Semantic vendor fallback (dispatchQueryToolWithSemantics) ───────────────

const semanticFixture = {
  now: "2026-06-30T12:00:00Z",
  recentExpenses: [
    { date: "2026-06-05", description: "breakers and wire", vendor: "Elliott Electric Supply", category: "materials", amount: 120.5, taxDeductible: true, source: "logged" },
    { date: "2026-06-18", description: "conduit", vendor: "Elliott Electric Supply", category: "materials", amount: 79.5, taxDeductible: true, source: "logged" },
    { date: "2026-05-02", description: "lumber", vendor: "Home Depot", category: "materials", amount: 300, taxDeductible: true, source: "logged" },
  ],
}

Deno.test("query_expenses: `vendors` keeps only exact (case-insensitive) vendor names", () => {
  const r = executeQueryExpenses(semanticFixture, { vendors: ["elliott electric supply"] })
  assertEquals(r.count, 2)
  assertEquals(r.total, 200)
  const none = executeQueryExpenses(semanticFixture, { vendors: ["Elliott"] })
  assertEquals(none.count, 0)
})

Deno.test("semanticPhraseFor: vendor first, keyword fallback, nothing when too short", () => {
  assertEquals(semanticPhraseFor({ vendor: " the electrical supplier " }), "the electrical supplier")
  assertEquals(semanticPhraseFor({ keyword: "wire" }), "wire")
  assertEquals(semanticPhraseFor({ vendor: "x" }), undefined)
  assertEquals(semanticPhraseFor({}), undefined)
})

Deno.test("semantics: literal hit → resolver NOT called, result unchanged", async () => {
  let calls = 0
  const resolver = () => { calls += 1; return Promise.resolve([{ vendor: "Home Depot", similarity: 0.9 }]) }
  const r = await dispatchQueryToolWithSemantics("query_expenses", { vendor: "Elliott" }, semanticFixture, resolver)
  assertEquals(calls, 0)
  assertEquals(r.count, 2)
  assertEquals(r.semanticMatch, undefined)
})

Deno.test("semantics: literal miss → resolver maps phrase to vendors, executor re-runs, note names the vendor", async () => {
  const seen: string[] = []
  const resolver = (phrase: string) => {
    seen.push(phrase)
    return Promise.resolve([{ vendor: "Elliott Electric Supply", similarity: 0.71 }])
  }
  const r = await dispatchQueryToolWithSemantics(
    "query_expenses",
    { vendor: "the electrical supplier", dateFrom: "2026-06-01", dateTo: "2026-06-30" },
    semanticFixture,
    resolver,
  )
  assertEquals(seen, ["the electrical supplier"])
  assertEquals(r.count, 2)
  assertEquals(r.total, 200) // summed in code, not by the model
  assertEquals(r.semanticMatch?.phrase, "the electrical supplier")
  assertEquals(r.semanticMatch?.vendors[0].vendor, "Elliott Electric Supply")
  assert(String(r.note).includes("Elliott Electric Supply (71% match)"))
  assert(String(r.note).includes("Tell the user which vendor you used"))
})

Deno.test("semantics: date filter still applies after the vendor re-run", async () => {
  const resolver = () => Promise.resolve([{ vendor: "Elliott Electric Supply", similarity: 0.7 }])
  const r = await dispatchQueryToolWithSemantics(
    "query_expenses",
    { vendor: "electrical supplier", dateFrom: "2026-06-10", dateTo: "2026-06-30" },
    semanticFixture,
    resolver,
  )
  assertEquals(r.count, 1)
  assertEquals(r.total, 79.5)
})

Deno.test("semantics: resolver returns nothing → empty result with an explicit do-not-guess note", async () => {
  const r = await dispatchQueryToolWithSemantics("query_expenses", { vendor: "plumbing place" }, semanticFixture, () => Promise.resolve([]))
  assertEquals(r.count, 0)
  assert(String(r.note).includes("Do NOT guess a vendor"))
  assertEquals(r.semanticMatch, undefined)
})

Deno.test("semantics: resolver throws → literal result survives with a note, never an exception", async () => {
  const r = await dispatchQueryToolWithSemantics("query_expenses", { vendor: "plumbing place" }, semanticFixture, () => Promise.reject(new Error("boom")))
  assertEquals(r.count, 0)
  assert(String(r.note).includes("unavailable"))
})

Deno.test("semantics: vendor matched but outside the date window → says so, suggests Expenses screen", async () => {
  const resolver = () => Promise.resolve([{ vendor: "Home Depot", similarity: 0.66 }])
  const r = await dispatchQueryToolWithSemantics(
    "query_expenses",
    { vendor: "hardware store", dateFrom: "2026-06-01", dateTo: "2026-06-30" },
    semanticFixture,
    resolver,
  )
  assertEquals(r.count, 0)
  assertEquals(r.semanticMatch?.vendors[0].vendor, "Home Depot")
  assert(String(r.note).includes("no expenses from that vendor"))
})

Deno.test("semantics: other tools and missing resolver pass straight through", async () => {
  let calls = 0
  const resolver = () => { calls += 1; return Promise.resolve([]) }
  const jobs = await dispatchQueryToolWithSemantics("query_jobs", { clientName: "Nobody" }, semanticFixture, resolver)
  assertEquals(calls, 0)
  assertEquals(jobs.count, 0)
  const noResolver = await dispatchQueryToolWithSemantics("query_expenses", { vendor: "plumbing place" }, semanticFixture, undefined)
  assertEquals(noResolver.count, 0)
  assertEquals(noResolver.semanticMatch, undefined)
})
