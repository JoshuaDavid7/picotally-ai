// Tests for custom-category preservation in the voice-command normalizer
// (beta bug #17 / the #30 orElse data-loss trap, server side).
//
// Why this matters: the user's requirement is that a created category is
// "tracked through the app … AI should know it as a category and should treat
// it like any other category". The failure mode this suite pins is the edge
// function silently collapsing a custom category the AI correctly named —
// e.g. turning "equipment_rental" into "materials"/"other" before the client
// ever sees it, which permanently mis-buckets the expense.
//
// The contract, exercised through the exported `normalizeAssistantResult`
// (which runs the private normalizeExpenseCategory + the tax-deductible
// default), is:
//   * an explicit custom-category KEY is preserved verbatim;
//   * a keyword in the description/transcript ("drill" → built-in 'tools')
//     can NOT hijack an explicit custom category;
//   * a custom category's OWN tax-deductible flag drives the default when the
//     AI didn't specify one (not the built-in 'true');
//   * a spoken/display form ("Equipment Rental") slugifies onto its key;
//   * a genuinely unknown category still falls back; and
//   * with no custom categories sent (older client), built-in behaviour is
//     unchanged — backward compatible.
//
// Run with:
//   deno test custom_category.test.ts

import {
  assertEquals,
} from "https://deno.land/std@0.224.0/assert/mod.ts"

import {
  normalizeAssistantResult,
} from "./index.ts"

// Mirrors the `availableCategories` block the Flutter client now sends in the
// business snapshot: a few built-ins (builtIn: true) plus the user's own
// custom categories (builtIn: false), each with the key the AI must emit and
// the user's tax-deductible choice.
const contextWithCustoms = {
  recentCustomers: [],
  jobs: [],
  availableCategories: [
    { key: "materials", name: "Materials", taxDeductible: true, builtIn: true },
    { key: "fuel", name: "Fuel & Vehicle", taxDeductible: true, builtIn: true },
    {
      key: "equipment_rental",
      name: "Equipment Rental",
      taxDeductible: true,
      builtIn: false,
    },
    {
      key: "tool_rental",
      name: "Tool Rental",
      taxDeductible: true,
      builtIn: false,
    },
    {
      key: "disposal",
      name: "Dump & Disposal",
      taxDeductible: false,
      builtIn: false,
    },
  ],
}

function createExpense(
  params: Record<string, unknown>,
  context: Record<string, unknown>,
  transcript: string,
): Record<string, unknown> {
  const result = normalizeAssistantResult(
    {
      action: "create_expense",
      params,
      response: "",
      suggestionChips: [],
    },
    context,
    transcript,
  )
  return result.params as Record<string, unknown>
}

Deno.test("create_expense preserves an explicit custom category key", () => {
  // The core #17/#30 fix: pre-fix this collapsed to 'other'/'materials'.
  const params = createExpense(
    {
      amount: 120,
      description: "Scaffold hire",
      vendor: "United Rentals",
      category: "equipment_rental",
    },
    contextWithCustoms,
    "Log a \$120 equipment rental expense from United Rentals",
  )
  assertEquals(params.category, "equipment_rental")
})

Deno.test("create_expense does not let a keyword hijack an explicit custom category", () => {
  // "drill" matches the built-in 'tools' heuristic — but the AI explicitly
  // picked the user's custom 'tool_rental', which must win outright.
  const params = createExpense(
    {
      amount: 40,
      description: "Drill rental",
      vendor: "Acme",
      category: "tool_rental",
    },
    contextWithCustoms,
    "Forty bucks to rent a drill",
  )
  assertEquals(params.category, "tool_rental")
})

Deno.test("create_expense applies a custom category's own non-deductible flag", () => {
  // 'disposal' is non-deductible and the AI didn't set taxDeductible, so the
  // default must come from the custom category — NOT the built-in 'true'. The
  // 'fees' keyword ("Dump fees") would also fire, proving the custom key wins.
  const params = createExpense(
    {
      amount: 75,
      description: "Dump fees",
      vendor: "City Landfill",
      category: "disposal",
    },
    contextWithCustoms,
    "Log \$75 dump and disposal",
  )
  assertEquals(params.category, "disposal")
  assertEquals(params.taxDeductible, false)
})

Deno.test("create_expense slugifies a spoken category name onto its custom key", () => {
  // The AI returned the display form "Equipment Rental"; the slug's underscore
  // form "equipment_rental" must still be recovered.
  const params = createExpense(
    {
      amount: 200,
      description: "Lift hire",
      vendor: "Sunbelt",
      category: "Equipment Rental",
    },
    contextWithCustoms,
    "Two hundred for equipment rental",
  )
  assertEquals(params.category, "equipment_rental")
})

Deno.test("create_expense still falls back for a genuinely unknown category", () => {
  // Nothing in the user's vocabulary matches → must not invent a key.
  const params = createExpense(
    {
      amount: 30,
      description: "Mystery",
      vendor: "Acme",
      category: "crypto_mining",
    },
    contextWithCustoms,
    "Thirty dollars crypto mining",
  )
  assertEquals(params.category, "other")
})

Deno.test("create_expense keeps built-in behaviour when no custom categories are sent", () => {
  // Backward-compat: an older client sends no availableCategories at all. The
  // built-in heuristic + deductible default must be unchanged.
  const params = createExpense(
    {
      amount: 60,
      description: "Diesel",
      vendor: "Shell",
      category: "fuel",
    },
    { recentCustomers: [], jobs: [] },
    "Sixty bucks diesel at Shell",
  )
  assertEquals(params.category, "fuel")
  assertEquals(params.taxDeductible, true)
})
