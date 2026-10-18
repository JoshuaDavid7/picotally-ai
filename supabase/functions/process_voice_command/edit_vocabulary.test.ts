// Tests for the Hey Pico edit-vocabulary expansion (2026-07-25 audit).
//
// Why this matters: "Invoice Greg for $400 and do not tax him" left tax ON
// because the pipeline had no tax op anywhere — the 7-op edit vocabulary
// couldn't express it, and create_invoice dropped tax/markup params on the
// floor. This suite pins the server half of the fix:
//   * the 7 NEW edit ops survive normalization (set_tax, set_markup,
//     set_labor_type, set_description, set_due_date, set_invoice_number,
//     set_client_field);
//   * their params are sanitized (rate 0-100, percent 0-1000, ISO date,
//     field enum, laborType enum) — bad values are DROPPED, not passed;
//   * hallucinated keys are stripped by the allow-list;
//   * an unknown op still degrades to a plain answer;
//   * create_invoice carries taxEnabled / taxRate / markupPercent through
//     (numbers coerced, out-of-range dropped, non-boolean enabled dropped).
//
// Run with:
//   deno test edit_vocabulary.test.ts

import {
  assertEquals,
} from "https://deno.land/std@0.224.0/assert/mod.ts"

import {
  normalizeAssistantResult,
} from "./index.ts"

const context = { recentCustomers: [], jobs: [] }

function normalizeEdit(params: Record<string, unknown>) {
  return normalizeAssistantResult(
    { action: "edit_current_invoice", params, response: "Done." },
    context,
    "test transcript",
  ) as { action: string; params: Record<string, unknown> }
}

Deno.test("set_tax enabled=false survives — the 'do not tax him' op", () => {
  const result = normalizeEdit({ op: "set_tax", enabled: false })
  assertEquals(result.action, "edit_current_invoice")
  assertEquals(result.params, { op: "set_tax", enabled: false })
})

Deno.test("set_tax coerces a string rate and keeps it in 0-100", () => {
  const result = normalizeEdit({ op: "set_tax", rate: "8.25" })
  assertEquals(result.params, { op: "set_tax", rate: 8.25 })
})

Deno.test("set_tax drops an impossible rate instead of passing it", () => {
  const result = normalizeEdit({ op: "set_tax", rate: 150, enabled: true })
  assertEquals(result.params, { op: "set_tax", enabled: true })
})

Deno.test("set_markup keeps percent; non-boolean enabled is dropped", () => {
  const result = normalizeEdit({ op: "set_markup", percent: 25, enabled: "yes" })
  assertEquals(result.params, { op: "set_markup", percent: 25 })
})

Deno.test("set_labor_type keeps flat + laborAmount; bad type dropped", () => {
  const flat = normalizeEdit({ op: "set_labor_type", laborType: "flat", laborAmount: 500 })
  assertEquals(flat.params, { op: "set_labor_type", laborType: "flat", laborAmount: 500 })
  const bad = normalizeEdit({ op: "set_labor_type", laborType: "salaried" })
  assertEquals(bad.params, { op: "set_labor_type" })
})

Deno.test("set_due_date requires ISO yyyy-mm-dd", () => {
  const good = normalizeEdit({ op: "set_due_date", date: "2026-07-30" })
  assertEquals(good.params, { op: "set_due_date", date: "2026-07-30" })
  const bad = normalizeEdit({ op: "set_due_date", date: "next Tuesday" })
  assertEquals(bad.params, { op: "set_due_date" })
})

Deno.test("set_client_field enforces the field enum", () => {
  const good = normalizeEdit({ op: "set_client_field", field: "email", value: "bob@x.com" })
  assertEquals(good.params, { op: "set_client_field", field: "email", value: "bob@x.com" })
  const bad = normalizeEdit({ op: "set_client_field", field: "ssn", value: "nope" })
  assertEquals(bad.params, { op: "set_client_field", value: "nope" })
})

Deno.test("set_invoice_number keeps value; set_description keeps text", () => {
  const num = normalizeEdit({ op: "set_invoice_number", value: " INV-0042 " })
  assertEquals(num.params, { op: "set_invoice_number", value: "INV-0042" })
  const desc = normalizeEdit({ op: "set_description", text: "Water heater swap" })
  assertEquals(desc.params, { op: "set_description", text: "Water heater swap" })
})

Deno.test("hallucinated keys are stripped by the allow-list", () => {
  const result = normalizeEdit({ op: "set_tax", enabled: false, isPro: true, sql: "drop" })
  assertEquals(result.params, { op: "set_tax", enabled: false })
})

Deno.test("an unknown op still degrades to a plain answer", () => {
  const result = normalizeEdit({ op: "set_discount", percent: 10 })
  assertEquals(result.action, "answer")
  assertEquals(result.params, {})
})

Deno.test("create_invoice carries taxEnabled/taxRate/markupPercent through", () => {
  const result = normalizeAssistantResult(
    {
      action: "create_invoice",
      params: {
        clientName: "Greg",
        laborType: "flat",
        laborAmount: 400,
        taxEnabled: false,
        markupPercent: "25",
      },
      response: "Opening it.",
    },
    context,
    "invoice Greg for 400 and do not tax him",
  ) as { params: Record<string, unknown> }
  assertEquals(result.params.taxEnabled, false)
  assertEquals(result.params.markupPercent, 25)
})

Deno.test("create_invoice drops out-of-range tax and non-boolean enabled", () => {
  const result = normalizeAssistantResult(
    {
      action: "create_invoice",
      params: { clientName: "Greg", taxRate: 400, taxEnabled: "no", markupPercent: -5 },
      response: "Opening it.",
    },
    context,
    "invoice Greg",
  ) as { params: Record<string, unknown> }
  assertEquals(result.params.taxRate, undefined)
  assertEquals(result.params.taxEnabled, undefined)
  assertEquals(result.params.markupPercent, undefined)
})
