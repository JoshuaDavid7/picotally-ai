// Builds the `BusinessContext` snapshot the process_voice_command tool
// executors consume — but from Supabase (Postgres), scoped to a userId, instead
// of from a client-supplied payload. This lets the LiveKit voice agent reuse the
// EXACT same deterministic query tools (single source of truth) with no app in
// the loop.
//
// The pure row→snapshot shapers are split out so a Deno test can exercise them
// (and feed the real executors) without spinning up a database.

import type { SupabaseClient } from 'https://esm.sh/@supabase/supabase-js@2'

export interface BusinessContextSnapshot {
  now: string
  customers: Array<Record<string, unknown>>
  jobs: Array<Record<string, unknown>>
  recentExpenses: Array<Record<string, unknown>>
  recentPayments: Array<Record<string, unknown>>
}

type Row = Record<string, unknown>

function toIso(value: unknown): string {
  if (typeof value === 'string') return value
  if (value == null) return ''
  try {
    return new Date(value as string).toISOString()
  } catch {
    return ''
  }
}

// ─── Pure row → snapshot-element shapers (unit-tested without a DB) ───────────

function round2(n: number): number {
  return Math.round((Number(n) + Number.EPSILON) * 100) / 100
}

/** Normalize a customer/client name for matching customers <-> jobs. */
export function nameKey(name: unknown): string {
  return String(name ?? '').trim().toLowerCase().replace(/\s+/g, ' ')
}

export interface CustomerLedger {
  billed: number
  paid: number
  count: number
}

/**
 * Aggregate each customer's ledger from their jobs. The cloud `customers` table
 * summary fields (balance/total_billed/total_paid/job_count) are maintained by
 * the app on-device and are often stale or zero in Supabase — the jobs carry
 * the truth, so we derive the ledger from them: billed = sum(total_amount),
 * paid = sum(amount_paid), outstanding = billed - paid (matches the text
 * assistant, which sums the same fields from the client snapshot).
 */
export function computeCustomerLedger(jobRows: Row[]): Map<string, CustomerLedger> {
  const m = new Map<string, CustomerLedger>()
  for (const j of jobRows) {
    const key = nameKey(j.client_name)
    if (!key) continue
    const cur = m.get(key) ?? { billed: 0, paid: 0, count: 0 }
    cur.billed += Number(j.total_amount ?? 0)
    cur.paid += Number(j.amount_paid ?? 0)
    cur.count += 1
    m.set(key, cur)
  }
  return m
}

export function shapeCustomer(row: Row, ledger?: CustomerLedger): Row {
  // Prefer the jobs-derived ledger (truthful); fall back to the row's stored
  // summary fields only when no jobs matched this customer.
  const billed = ledger ? round2(ledger.billed) : Number(row.total_billed ?? 0)
  const paid = ledger ? round2(ledger.paid) : Number(row.total_paid ?? 0)
  return {
    name: String(row.name ?? ''),
    phone: String(row.phone ?? ''),
    email: String(row.email ?? ''),
    jobCount: ledger ? ledger.count : Number(row.job_count ?? 0),
    totalBilled: billed,
    totalPaid: paid,
    balance: round2(billed - paid),
  }
}

export function shapeMaterial(m: Row): Row {
  // jobs.materials JSONB already carries the app's camelCase keys
  // (item, quantity, cost, originalCost, markupEnabled, fromReceipt). The
  // executors prefer `actualSupplierCost` (pre-markup) and fall back to
  // originalCost → cost; the app aliases actualSupplierCost := originalCost, so
  // we mirror that. The raw object is passed through so no key is ever lost.
  const out: Row = { ...m }
  if (out.actualSupplierCost == null && m.originalCost != null) {
    out.actualSupplierCost = m.originalCost
  }
  return out
}

export function shapeJob(row: Row): Row {
  const rawMaterials = Array.isArray(row.materials) ? (row.materials as Row[]) : []
  return {
    id: String(row.id ?? ''),
    title: String(row.title ?? ''),
    clientName: String(row.client_name ?? ''),
    status: String(row.status ?? 'draft'),
    type: String(row.type ?? 'invoice'),
    totalAmount: Number(row.total_amount ?? 0),
    amountPaid: Number(row.amount_paid ?? 0),
    amountDue: Number(row.amount_due ?? 0),
    invoiceNumber: String(row.invoice_number ?? ''),
    createdAt: toIso(row.created_at),
    // The cloud `jobs` table has no due_date column, so the query_jobs
    // "late" filter (which needs it) just won't fire server-side. Every other
    // job/material query is unaffected.
    dueDate: null,
    paidAt: row.paid_at != null ? toIso(row.paid_at) : null,
    materials: rawMaterials.map(shapeMaterial),
  }
}

export function parseOcrItems(ocrText: unknown): Row[] {
  if (typeof ocrText !== 'string' || !ocrText.trim()) return []
  try {
    const parsed = JSON.parse(ocrText) as { items?: unknown }
    const items = parsed?.items
    if (!Array.isArray(items)) return []
    return items.map((it) => {
      const o = (it ?? {}) as Row
      return {
        name: String(o.name ?? ''),
        quantity: Number(o.quantity ?? 1),
        unitPrice: o.unitPrice != null ? Number(o.unitPrice) : undefined,
        totalPrice: o.totalPrice != null ? Number(o.totalPrice) : undefined,
      }
    })
  } catch {
    return []
  }
}

export function shapeExpense(row: Row): Row {
  return {
    date: toIso(row.expense_date),
    description: String(row.description ?? ''),
    vendor: String(row.vendor ?? ''),
    category: String(row.category ?? 'other'),
    amount: Number(row.amount ?? 0),
    taxDeductible: row.tax_deductible === true,
    source: 'logged',
    hasReceipt: Boolean(row.receipt_path || row.receipt_url),
    ocrItems: parseOcrItems(row.ocr_text),
  }
}

export function shapePayment(
  row: Row,
  jobsById: Map<string, { title: string; clientName: string }>,
): Row {
  const job = jobsById.get(String(row.job_id ?? ''))
  return {
    date: toIso(row.received_at),
    amount: Number(row.amount ?? 0),
    method: String(row.payment_method ?? 'other'),
    jobId: row.job_id ?? null,
    jobTitle: job?.title,
    clientName: job?.clientName,
    reference: row.transaction_reference ?? undefined,
  }
}

/** Index jobs by id so payments can be enriched with title + client name. */
export function indexJobsById(
  jobRows: Row[],
): Map<string, { title: string; clientName: string }> {
  const map = new Map<string, { title: string; clientName: string }>()
  for (const j of jobRows) {
    map.set(String(j.id ?? ''), {
      title: String(j.title ?? ''),
      clientName: String(j.client_name ?? ''),
    })
  }
  return map
}

// ─── DB-backed builder ───────────────────────────────────────────────────────

export async function buildBusinessContext(
  supabase: SupabaseClient,
  userId: string,
): Promise<BusinessContextSnapshot> {
  // The service role bypasses RLS, so EVERY query is explicitly scoped to
  // user_id — the same isolation RLS would enforce, made explicit here.
  const [customersRes, jobsRes, expensesRes, paymentsRes] = await Promise.all([
    supabase
      .from('customers')
      .select('name, phone, email, job_count, total_billed, total_paid, balance')
      .eq('user_id', userId)
      .order('updated_at', { ascending: false })
      .limit(200),
    supabase
      .from('jobs')
      .select(
        'id, title, client_name, status, type, total_amount, amount_paid, amount_due, invoice_number, materials, created_at, paid_at',
      )
      .eq('user_id', userId)
      .order('created_at', { ascending: false })
      .limit(200),
    supabase
      .from('expenses')
      .select(
        'description, vendor, category, amount, expense_date, tax_deductible, receipt_path, receipt_url, ocr_text',
      )
      .eq('user_id', userId)
      .order('expense_date', { ascending: false })
      .limit(100),
    supabase
      .from('payment_events')
      .select('amount, payment_method, transaction_reference, received_at, job_id')
      .eq('user_id', userId)
      .order('received_at', { ascending: false })
      .limit(60),
  ])

  const jobRows = (jobsRes.data ?? []) as Row[]
  const jobsById = indexJobsById(jobRows)
  const ledger = computeCustomerLedger(jobRows)

  return {
    now: new Date().toISOString(),
    customers: ((customersRes.data ?? []) as Row[]).map((c) =>
      shapeCustomer(c, ledger.get(nameKey(c.name)))
    ),
    jobs: jobRows.map(shapeJob),
    recentExpenses: ((expensesRes.data ?? []) as Row[]).map(shapeExpense),
    recentPayments: ((paymentsRes.data ?? []) as Row[]).map((r) => shapePayment(r, jobsById)),
  }
}
