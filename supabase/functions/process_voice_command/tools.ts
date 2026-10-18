// Picotally — query tool definitions and executors for the AI assistant.
//
// These are the deterministic "calculator" the model uses instead of doing
// filter+sum in its head. The architectural rationale lives in
// process_voice_command/index.ts header: large LLMs are unreliable when
// asked to scan 800+ rows and pick exactly the right field, especially on
// shapes like Material where four similar numeric columns sit next to each
// other (cost, unitPrice, originalCost, actualSupplierCost). Tools shift
// that work into TypeScript, where it's exact and reproducible.
//
// Every tool here:
//   - operates on the in-memory `business_context` snapshot the client
//     already shipped (no DB reads — the snapshot IS the source of truth
//     for the turn);
//   - returns `{ items, total, count, partial }` shaped consistently
//     across all six tools, so the model learns one result shape;
//   - filters case-insensitively where keyword matches are involved;
//   - is pure — same inputs produce same outputs, no side effects.
//
// The model is told (in the system prompt) to call exactly ONE tool, read
// the result, and then call `respond` with the final user-facing answer.
// The multi-turn loop in index.ts enforces that contract by capping at
// 4 turns of tool use before forcing a respond.
//
// Tool descriptions are written for Claude Haiku 4.5 — examples baked in,
// enums preferred over free-form strings, mandatory shape on the result.
// Sonnet would tolerate looser interfaces but Haiku does not, and we want
// the same prompt to work cross-model.

// ─── Types ───────────────────────────────────────────────────────────────────

export interface ToolResult {
  items: Array<Record<string, unknown>>
  total: number
  count: number
  /**
   * True when the search may have been truncated. Currently only true
   * when the underlying snapshot itself is partial (e.g. recentExpenses
   * is capped at 60). The model is told to disclose `partial: true` to
   * the user.
   */
  partial: boolean
  /**
   * Optional human-readable note about how the result was computed —
   * e.g. "drafts excluded", "linked-receipt materials counted as the
   * receipt amount, not the invoice line". Surfaced to the model so it
   * can pass nuance through to the user without re-deriving it.
   */
  note?: string
  /**
   * Present only when `query_expenses` found no literal vendor/keyword match
   * and the result was produced by resolving the user's phrase to vendor
   * names by meaning (see dispatchQueryToolWithSemantics). The model is told
   * to name the vendor it actually used.
   */
  semanticMatch?: {
    phrase: string
    vendors: Array<{ vendor: string; similarity: number }>
  }
}

interface BusinessContext {
  jobs?: unknown
  recentExpenses?: unknown
  recentPayments?: unknown
  customers?: unknown
  stats?: unknown
  /**
   * ISO timestamp captured by the client when the snapshot was built.
   * Used by `query_jobs(status='late')` to determine "is this past due"
   * deterministically — using the snapshot's clock instead of the edge
   * function's wall clock keeps test fixtures stable and prevents the
   * "answers shift between turns" flake from server vs. device clock
   * drift. Falls back to `new Date()` when missing (older clients).
   */
  now?: unknown
}

// ─── Tool schemas (sent to Anthropic) ────────────────────────────────────────
//
// Every tool description follows the same shape:
//   1. One-line summary of what the tool returns.
//   2. WHEN TO USE — the trigger phrases / question types.
//   3. EXAMPLES — concrete `tool_call({...})` literals with realistic
//      parameters. Haiku follows examples better than abstract specs.
//
// `additionalProperties: false` on the input schema is intentional: it
// forces the model to use the exact parameter names listed, instead of
// inventing shaped-similar keys (e.g. `material_keyword` vs `keyword`).

export const QUERY_MATERIALS_TOOL = {
  name: 'query_materials',
  description: `Sum and list materials across the user's invoices and jobs by keyword/date/client/status. Returns the actual supplier cost (pre-markup), which is what the user really paid.

USE THIS TOOL for ANY question about material spending or purchases:
- "How much did I spend on pipes / wire / lumber / X?"
- "What did I pay for materials on Sam's bathroom?"
- "How many heaters did I install last year?"
- "Total materials cost this month / Q1 / since January"
- Any question that requires summing or counting material line items.

DO NOT try to filter and sum the recentExpenses or jobs lists yourself — they are summaries, not the source of truth, and you will undercount or pick the wrong field. Always call this tool.

Examples:
- User: "How much did I spend on pipes last 2 months?" → query_materials({ keyword: "pipe", dateFrom: "2026-03-01", dateTo: "2026-04-30", invoiceStatus: "sent_or_paid" })
- User: "Materials on Steven's bathroom job" → query_materials({ clientName: "Steven", invoiceStatus: "sent_or_paid" })
- User: "Total I've spent on copper this year" → query_materials({ keyword: "copper", dateFrom: "2026-01-01", dateTo: "2026-12-31", invoiceStatus: "sent_or_paid" })`,
  input_schema: {
    type: 'object',
    properties: {
      keyword: {
        type: 'string',
        description: 'Case-insensitive substring to match against material item names. Use a short stem ("pipe" matches "Pipe", "Pipes", "Pipe x2", "PVC pipe"). Omit to match all materials.',
      },
      dateFrom: {
        type: 'string',
        description: 'Inclusive lower bound on the parent invoice/job date, ISO YYYY-MM-DD. Omit for no lower bound.',
      },
      dateTo: {
        type: 'string',
        description: 'Inclusive upper bound, ISO YYYY-MM-DD. Omit for no upper bound.',
      },
      invoiceStatus: {
        type: 'string',
        enum: ['sent_or_paid', 'draft_only', 'any'],
        description: 'Which invoice statuses count. "sent_or_paid" = real spend the user has committed to (DEFAULT for spend questions). "draft_only" = unsent drafts (the user has NOT paid these yet). "any" = both. ALWAYS use "sent_or_paid" unless the user explicitly asks about drafts.',
      },
      clientName: {
        type: 'string',
        description: 'Filter to materials on a specific client\'s jobs. Match a name from the customer list — do not invent. Case-insensitive.',
      },
      jobId: {
        type: 'string',
        description: 'Filter to materials on one specific job, by ID. Use this when you already know the job (e.g. from a prior turn or a navigate action).',
      },
    },
    required: [],
    additionalProperties: false,
  },
} as const

export const QUERY_EXPENSES_TOOL = {
  name: 'query_expenses',
  description: `Sum and list logged expenses (receipts and manually-entered costs) by vendor / category / keyword / date. Use this for receipt-tracked spend, NOT for invoice line items — query_materials handles invoice materials.

USE THIS TOOL for:
- "How much did I spend at Home Depot?"
- "Total fuel costs this quarter"
- "All my deductible expenses last month"
- "What receipts have I scanned for Mike?"

If the vendor or keyword has no literal match, the tool automatically retries by MEANING (semantic vendor match over the user's own expenses) and the result's note names the vendor(s) it used — repeat that vendor name to the user so they know what was matched. Pass the user's phrase as spoken (e.g. "electrical supplier") when they do not give an exact vendor name.

Examples:
- User: "Spend at Home Depot last 3 months" → query_expenses({ vendor: "Home Depot", dateFrom: "2026-02-01", dateTo: "2026-04-30" })
- User: "What did I spend at the electrical supplier in June" → query_expenses({ vendor: "electrical supplier", dateFrom: "2026-06-01", dateTo: "2026-06-30" })
- User: "Total fuel this year" → query_expenses({ category: "fuel", dateFrom: "2026-01-01", dateTo: "2026-12-31" })
- User: "Deductible expenses this month" → query_expenses({ taxDeductible: true, dateFrom: "2026-04-01", dateTo: "2026-04-30" })`,
  input_schema: {
    type: 'object',
    properties: {
      keyword: {
        type: 'string',
        description: 'Case-insensitive substring matched against the expense description. Omit to skip keyword filter.',
      },
      vendor: {
        type: 'string',
        description: 'Vendor name or the user\'s description of the vendor (e.g. "Home Depot", "the electrical supplier"). Literal substring first; if nothing matches, resolved by meaning. Omit to skip vendor filter.',
      },
      category: {
        type: 'string',
        enum: ['materials', 'labor', 'fuel', 'tools', 'supplies', 'insurance', 'utilities', 'marketing', 'fees', 'meals', 'other'],
        description: 'Expense category. Omit for all categories.',
      },
      dateFrom: {
        type: 'string',
        description: 'Inclusive lower bound, ISO YYYY-MM-DD.',
      },
      dateTo: {
        type: 'string',
        description: 'Inclusive upper bound, ISO YYYY-MM-DD.',
      },
      taxDeductible: {
        type: 'boolean',
        description: 'If true, include only deductible expenses. Omit for both deductible and non-deductible.',
      },
      clientName: {
        type: 'string',
        description: 'Filter to expenses linked to a specific customer\'s jobs. Match the customer list.',
      },
    },
    required: [],
    additionalProperties: false,
  },
} as const

export const QUERY_JOBS_TOOL = {
  name: 'query_jobs',
  description: `List and aggregate jobs (invoices and quotes) by client / status / date / type. Returns matching jobs with totals.

USE THIS TOOL for:
- "Show me unpaid invoices"
- "Who is late on payment?" / "Which invoices are overdue?"
- "Drafts I haven't sent"
- "All Mike's jobs"
- "Quotes from this quarter"
- "Total billed to Steven"

Examples:
- User: "Show unpaid invoices" → query_jobs({ status: "unpaid" })
- User: "Who is late on payment?" → query_jobs({ status: "late" })
- User: "All Mike's jobs" → query_jobs({ clientName: "Mike" })
- User: "Drafts older than a week" → query_jobs({ status: "draft", dateTo: "2026-04-19" })`,
  input_schema: {
    type: 'object',
    properties: {
      clientName: {
        type: 'string',
        description: 'Filter to one client. Match the customer list, case-insensitive.',
      },
      status: {
        type: 'string',
        enum: ['draft', 'sent', 'unpaid', 'late', 'paid', 'partially_paid', 'any'],
        description: 'Job status. "unpaid" = sent but not yet paid in full. "late" = sent invoices with money still owed AND a due date in the past (use this for "overdue" / "late" / "past due" questions). "any" = no status filter (DEFAULT).',
      },
      type: {
        type: 'string',
        enum: ['invoice', 'quote', 'any'],
        description: 'Filter by invoice vs quote. Default "any".',
      },
      dateFrom: {
        type: 'string',
        description: 'Inclusive lower bound on createdAt, ISO YYYY-MM-DD.',
      },
      dateTo: {
        type: 'string',
        description: 'Inclusive upper bound on createdAt, ISO YYYY-MM-DD.',
      },
    },
    required: [],
    additionalProperties: false,
  },
} as const

export const QUERY_PAYMENTS_TOOL = {
  name: 'query_payments',
  description: `List and sum payments received by date / client / method.

USE THIS TOOL for:
- "Total collected this month"
- "Payments from Mike"
- "Cash payments this year"
- "How did Steven pay last invoice?"

Examples:
- User: "Total collected this month" → query_payments({ dateFrom: "2026-04-01", dateTo: "2026-04-30" })
- User: "Cash payments this year" → query_payments({ method: "cash", dateFrom: "2026-01-01", dateTo: "2026-12-31" })`,
  input_schema: {
    type: 'object',
    properties: {
      clientName: {
        type: 'string',
        description: 'Filter to payments from one client.',
      },
      method: {
        type: 'string',
        enum: ['cash', 'card', 'check', 'bank_transfer', 'any'],
        description: 'Payment method. Default "any".',
      },
      dateFrom: {
        type: 'string',
        description: 'Inclusive lower bound, ISO YYYY-MM-DD.',
      },
      dateTo: {
        type: 'string',
        description: 'Inclusive upper bound, ISO YYYY-MM-DD.',
      },
    },
    required: [],
    additionalProperties: false,
  },
} as const

export const QUERY_CUSTOMER_BALANCE_TOOL = {
  name: 'query_customer_balance',
  description: `Look up one customer's outstanding balance, total billed, total paid, and job count. Use this whenever the user asks about what someone owes.

USE THIS TOOL for:
- "How much does Mike owe?"
- "What's Steven's balance?"
- "Has Hikaru paid yet?"

Match the name against the customer list — do not invent. If you cannot find a match, the tool returns count=0 so you can tell the user.

Example:
- User: "How much does Mike owe me?" → query_customer_balance({ clientName: "Mike" })`,
  input_schema: {
    type: 'object',
    properties: {
      clientName: {
        type: 'string',
        description: 'The customer name to look up. Required.',
      },
    },
    required: ['clientName'],
    additionalProperties: false,
  },
} as const

export const LOOKUP_MATERIAL_PRICES_TOOL = {
  name: 'lookup_material_prices',
  description: `Look up the user's OWN historical prices for named materials, so a new quote/invoice can be PRE-FILLED instead of guessed. Per item it returns a suggested unit price drawn from the user's past invoices/quotes (what they CHARGED — a "sell" price) and scanned receipts (what they PAID — a "cost"), plus the spread of prices seen.

USE THIS TOOL when the user asks to create a quote/invoice and names materials WITHOUT stating a price — e.g. "quote Mike for 50ft of PEX and two shutoff valves". Call it ONCE with all the un-priced item names. Do NOT call it for an item the user already priced ("...at $6 each") — use their number verbatim.

How to read each result item, and when to ASK instead of fill:
- status "found", suggestion.basis "sell", priceRange tight → this is a price the user has charged before; fill suggestion.unitPrice into the draft.
- status "found", suggestion.basis "sell", priceRange WIDE → past prices disagree; ASK which to use, don't pick silently.
- status "found", suggestion.basis "cost" → the only history is what the user PAID a supplier, NOT a past sell price. Do NOT bill cost as the price; tell the user it's their cost and ASK what to charge / what margin to add.
- status "not_found" → no history. Do NOT invent a price; ASK the user what to charge.

Over-communicate when unsure: a clarifying question is always better than a fabricated or mis-applied price.

Examples:
- User: "Quote Sarah for 50ft of PEX pipe and 2 shutoff valves" → lookup_material_prices({ items: ["PEX pipe", "shutoff valve"] })
- User: "Invoice Mike: 3 hours labor and a water heater" → lookup_material_prices({ items: ["water heater"] })  (labor is not a material — omit it)`,
  input_schema: {
    type: 'object',
    properties: {
      items: {
        type: 'array',
        items: { type: 'string' },
        description: 'The material names to look up, exactly as the user named them (e.g. ["PEX pipe", "shutoff valve"]). One call covers all un-priced items in the request.',
      },
    },
    required: ['items'],
    additionalProperties: false,
  },
} as const

/**
 * All query tools, in the order they appear in the request body. Order
 * doesn't affect the model's choice — it picks by description match —
 * but stable order keeps prompt-cache keys stable.
 */
export const QUERY_TOOLS = [
  QUERY_MATERIALS_TOOL,
  QUERY_EXPENSES_TOOL,
  QUERY_JOBS_TOOL,
  QUERY_PAYMENTS_TOOL,
  QUERY_CUSTOMER_BALANCE_TOOL,
  LOOKUP_MATERIAL_PRICES_TOOL,
] as const

export type QueryToolName =
  | 'query_materials'
  | 'query_expenses'
  | 'query_jobs'
  | 'query_payments'
  | 'query_customer_balance'
  | 'lookup_material_prices'

// ─── Helpers ─────────────────────────────────────────────────────────────────

function lower(value: unknown): string {
  return String(value ?? '').trim().toLowerCase()
}

function asArray(value: unknown): Array<Record<string, unknown>> {
  if (!Array.isArray(value)) return []
  return value.filter(
    (item): item is Record<string, unknown> =>
      !!item && typeof item === 'object' && !Array.isArray(item)
  )
}

function inDateRange(
  isoDate: unknown,
  dateFrom: string | undefined,
  dateTo: string | undefined
): boolean {
  if (!dateFrom && !dateTo) return true
  const raw = String(isoDate ?? '')
  if (!raw) return false
  // Compare on the date prefix (YYYY-MM-DD). String comparison works
  // because the format is lexicographically ordered.
  const datePart = raw.slice(0, 10)
  if (dateFrom && datePart < dateFrom) return false
  if (dateTo && datePart > dateTo) return false
  return true
}

function nameMatches(candidate: unknown, target: string | undefined): boolean {
  if (!target) return true
  const a = lower(candidate)
  const b = lower(target)
  if (!a) return false
  return a.includes(b) || b.includes(a)
}

function round2(n: number): number {
  return Math.round(n * 100) / 100
}

// ─── Tool executors ──────────────────────────────────────────────────────────
//
// Each takes the raw business_context map and the tool's `input` map, and
// returns a ToolResult. They never throw — bad input yields an empty
// result with `note` explaining why.

interface MaterialsInput {
  keyword?: string
  dateFrom?: string
  dateTo?: string
  invoiceStatus?: 'sent_or_paid' | 'draft_only' | 'any'
  clientName?: string
  jobId?: string
}

/**
 * Materials are sourced from `jobs[].materials`. Each material on a job
 * carries `actualSupplierCost` (pre-markup total — the real spend basis).
 *
 * Status filter semantics:
 *   - sent_or_paid: include only jobs whose status is sent / paid /
 *     partially_paid. This is "real spend" — the user has committed to
 *     the materials by sending the invoice.
 *   - draft_only: include only draft jobs. These are projections, NOT
 *     yet paid for; the user typically does NOT want them counted as
 *     spend. The model is told to use this only when the user explicitly
 *     asks about drafts.
 *   - any: include both.
 *
 * The result's `total` uses `actualSupplierCost` when present, falling
 * back to `cost` (post-markup) only when `actualSupplierCost` is missing
 * — which happens for legacy materials predating the markup feature.
 */
export function executeQueryMaterials(
  context: BusinessContext,
  input: MaterialsInput
): ToolResult {
  const jobs = asArray(context.jobs)
  const keyword = lower(input.keyword)
  const status = (input.invoiceStatus || 'sent_or_paid') as MaterialsInput['invoiceStatus']

  const sentLikeStatuses = new Set(['sent', 'paid', 'partially_paid'])
  const draftStatuses = new Set(['draft'])

  const items: Array<Record<string, unknown>> = []
  let total = 0

  for (const job of jobs) {
    if (input.jobId && lower(job.id) !== lower(input.jobId)) continue
    if (input.clientName && !nameMatches(job.clientName, input.clientName)) continue
    if (!inDateRange(job.createdAt, input.dateFrom, input.dateTo)) continue

    const jobStatus = lower(job.status)
    if (status === 'sent_or_paid' && !sentLikeStatuses.has(jobStatus)) continue
    if (status === 'draft_only' && !draftStatuses.has(jobStatus)) continue

    const materials = asArray(job.materials)
    for (const m of materials) {
      const itemName = String(m.item ?? '')
      if (keyword && !lower(itemName).includes(keyword)) continue

      const actualSupplierCost = (m.actualSupplierCost as number | undefined)
        ?? (m.originalCost as number | undefined)
        ?? (m.cost as number | undefined)
        ?? 0
      const numeric = Number(actualSupplierCost)
      if (!Number.isFinite(numeric)) continue
      total += numeric

      items.push({
        date: String(job.createdAt ?? '').slice(0, 10),
        item: itemName,
        quantity: Number(m.quantity ?? 1),
        actualSupplierCost: round2(numeric),
        clientName: String(job.clientName ?? ''),
        jobTitle: String(job.title ?? ''),
        jobId: String(job.id ?? ''),
        jobStatus,
        invoiceNumber: String(job.invoiceNumber ?? ''),
      })
    }
  }

  // Sort by date desc so the model sees most recent first.
  items.sort((a, b) => String(b.date).localeCompare(String(a.date)))

  const note =
    status === 'sent_or_paid'
      ? 'Drafts excluded. These are pre-markup actual supplier costs from sent/paid invoices.'
      : status === 'draft_only'
        ? 'Only unsent draft invoices. The user has NOT paid these yet — disclose this in the response.'
        : 'Includes both drafts and sent invoices. Drafts are projections, not real spend.'

  return {
    items,
    total: round2(total),
    count: items.length,
    partial: false,
    note,
  }
}

interface ExpensesInput {
  keyword?: string
  vendor?: string
  category?: string
  dateFrom?: string
  dateTo?: string
  taxDeductible?: boolean
  clientName?: string
  /**
   * Exact vendor names (case-insensitive) to keep. Set by the semantic
   * fallback after it resolved the user's phrase to real vendor names; the
   * model never fills this in directly.
   */
  vendors?: string[]
}

/**
 * Logged expenses come from the `recentExpenses` array. The client caps
 * this at 60 rows (most recent), so deep history queries may be partial
 * — the result flags `partial: true` to remind the model.
 *
 * `recentExpenses` is already deduplicated against invoice materials
 * upstream, so summing here will not double-count receipts that were
 * linked to an invoice line.
 */
export function executeQueryExpenses(
  context: BusinessContext,
  input: ExpensesInput
): ToolResult {
  const expenses = asArray(context.recentExpenses)

  const keyword = lower(input.keyword)
  const vendor = lower(input.vendor)
  const category = lower(input.category)
  const vendorSet = Array.isArray(input.vendors)
    ? input.vendors.map((v) => lower(v)).filter(Boolean)
    : []

  const items: Array<Record<string, unknown>> = []
  let total = 0

  for (const e of expenses) {
    if (vendor && !lower(e.vendor).includes(vendor)) continue
    if (vendorSet.length > 0 && !vendorSet.includes(lower(e.vendor))) continue
    if (category && lower(e.category) !== category) continue
    if (keyword && !lower(e.description).includes(keyword)) continue
    if (input.taxDeductible === true && e.taxDeductible !== true) continue
    if (!inDateRange(e.date, input.dateFrom, input.dateTo)) continue
    if (input.clientName) {
      const link = (e.linkedToInvoice && typeof e.linkedToInvoice === 'object')
        ? e.linkedToInvoice as Record<string, unknown>
        : null
      const linkClient = link ? String(link.clientName ?? '') : ''
      const directClient = String(e.clientName ?? '')
      if (
        !nameMatches(linkClient, input.clientName) &&
        !nameMatches(directClient, input.clientName)
      ) {
        continue
      }
    }

    const amount = Number(e.amount ?? 0)
    if (!Number.isFinite(amount)) continue
    total += amount

    items.push({
      date: String(e.date ?? '').slice(0, 10),
      description: String(e.description ?? ''),
      vendor: String(e.vendor ?? ''),
      category: String(e.category ?? ''),
      amount: round2(amount),
      taxDeductible: e.taxDeductible === true,
      source: String(e.source ?? 'logged'),
      hasReceipt: e.hasReceipt === true,
    })
  }

  items.sort((a, b) => String(b.date).localeCompare(String(a.date)))

  return {
    items,
    total: round2(total),
    count: items.length,
    // Snapshot caps at 60. Flag partial so the model can disclose if
    // the result is at the cap.
    partial: items.length >= 60,
    note: items.length >= 60
      ? 'Hit the 60-expense cap. Older results may be missing — recommend the Expenses or Analytics screen for older windows.'
      : undefined,
  }
}

interface JobsInput {
  clientName?: string
  status?: 'draft' | 'sent' | 'unpaid' | 'late' | 'paid' | 'partially_paid' | 'any'
  type?: 'invoice' | 'quote' | 'any'
  dateFrom?: string
  dateTo?: string
}

/**
 * Resolves "today" for lateness comparisons. Uses the snapshot's `now` —
 * captured by the client when it built the context — so the answer
 * doesn't drift between the client's clock and the edge function's
 * wall clock. Falls back to server time when the field is missing
 * (older clients pre-2026-04-26 or malformed contexts).
 *
 * Returned as `YYYY-MM-DD` so it can be compared lexicographically
 * against ISO date prefixes — same pattern `inDateRange` uses.
 */
function todayPrefix(context: BusinessContext): string {
  const raw = String(context.now ?? '').trim()
  if (raw) {
    const slice = raw.slice(0, 10)
    if (/^\d{4}-\d{2}-\d{2}$/.test(slice)) return slice
  }
  return new Date().toISOString().slice(0, 10)
}

/**
 * Whole-day diff between two `YYYY-MM-DD` strings (`later - earlier`).
 * Used to surface `daysLate` on overdue invoices so the AI can say
 * "Mike's $236 invoice is 9 days late" rather than just "is late".
 *
 * Returns null on malformed input rather than 0/NaN — the AI knows to
 * suppress the day count when null.
 */
function daysBetween(later: string, earlier: string): number | null {
  if (!/^\d{4}-\d{2}-\d{2}$/.test(later)) return null
  if (!/^\d{4}-\d{2}-\d{2}$/.test(earlier)) return null
  const a = Date.parse(later + 'T00:00:00Z')
  const b = Date.parse(earlier + 'T00:00:00Z')
  if (!Number.isFinite(a) || !Number.isFinite(b)) return null
  return Math.round((a - b) / 86400000)
}

/** Add `days` to a YYYY-MM-DD prefix, returning a YYYY-MM-DD prefix (UTC). */
function addDaysIso(prefix: string, days: number): string {
  if (!/^\d{4}-\d{2}-\d{2}$/.test(prefix)) return ''
  const t = Date.parse(prefix + 'T00:00:00Z')
  if (!Number.isFinite(t)) return ''
  return new Date(t + days * 86400000).toISOString().slice(0, 10)
}

export function executeQueryJobs(
  context: BusinessContext,
  input: JobsInput
): ToolResult {
  const jobs = asArray(context.jobs)
  const status = input.status || 'any'
  const type = input.type || 'any'
  const today = todayPrefix(context)

  const items: Array<Record<string, unknown>> = []
  let total = 0
  let missingDueDateCount = 0
  let fallbackDueCount = 0

  for (const job of jobs) {
    if (input.clientName && !nameMatches(job.clientName, input.clientName)) continue
    if (!inDateRange(job.createdAt, input.dateFrom, input.dateTo)) continue
    if (type !== 'any' && lower(job.type) !== type) continue

    const jobStatus = lower(job.status)
    const jobType = lower(job.type)
    // The stored amountDue is UNRELIABLE — it's frequently left at 0 on
    // genuinely unpaid invoices, which made "unpaid"/"late" silently drop real
    // receivables (an $850 owed invoice read as owing $0, so a customer who
    // owed $891 was reported as owing $41). Derive what's owed from
    // total − paid — always correct, and consistent with how
    // executeQueryCustomerBalance computes a balance (billed − paid).
    const totalAmt = Number(job.totalAmount ?? 0)
    const paidAmt = Number(job.amountPaid ?? 0)
    const safeTotal = Number.isFinite(totalAmt) ? totalAmt : 0
    const safePaid = Number.isFinite(paidAmt) ? paidAmt : 0
    const amountDue = round2(Math.max(0, safeTotal - safePaid))
    const dueRaw = String(job.dueDate ?? '').trim()
    const duePrefix = dueRaw ? dueRaw.slice(0, 10) : ''
    // The effective due date used for lateness + daysLate. Defaults to the
    // stored due date; the 'late' branch fills it from net-14 terms when none
    // is stored (legacy invoices). Stays = stored for every other query.
    let effectiveDue = duePrefix

    if (status !== 'any') {
      if (status === 'unpaid') {
        // "unpaid" is a synthetic status: sent INVOICES that still have
        // money owed. Three gates, all required to match the dashboard's
        // "X invoices awaiting payment" count:
        //  1. Status is 'sent' or 'partially_paid' (not draft/cancelled/paid).
        //  2. amountDue > 0.01 — guards against legacy sent rows where
        //     payment_status never flipped to 'paid' but the balance is 0.
        //  3. type is 'invoice' (not 'quote'). A quote is accepted or
        //     declined, never "paid"; counting an open quote as an
        //     unpaid invoice double-counted pipeline alongside real
        //     receivables and produced "11" answers when the dashboard
        //     said "6". Callers who genuinely want both invoices AND
        //     open quotes can pass `type: 'any'` explicitly.
        if (jobType === 'quote') continue
        if (!['sent', 'partially_paid'].includes(jobStatus)) continue
        if (!Number.isFinite(amountDue) || amountDue <= 0.01) continue
      } else if (status === 'late') {
        // "late" is a synthetic status: an invoice the user has SENT,
        // still has money owing on, AND whose due date is in the past.
        // Quotes are excluded — they don't carry a payment due date.
        // Cancelled / draft invoices are excluded — no payment expected.
        if (jobType === 'quote') continue
        if (jobStatus !== 'sent' && jobStatus !== 'partially_paid') continue
        if (!Number.isFinite(amountDue) || amountDue <= 0.01) continue
        // Legacy invoices sent before due-date tracking carry no stored due
        // date. Rather than silently dropping every such receivable (which
        // made "late" return 0 in the cloud while the text assistant computed
        // overdue from invoice-date + terms), fall back to net-14 terms —
        // created_at + 14 days — the same fallback the PDF renderer and the
        // due_date backfill use. Now both brains assess lateness identically.
        if (!duePrefix) {
          const createdPrefix = String(job.createdAt ?? '').slice(0, 10)
          const computed =
              createdPrefix ? addDaysIso(createdPrefix, 14) : ''
          if (!computed) {
            // No due date AND no usable invoice date — truly can't tell.
            missingDueDateCount++
            continue
          }
          effectiveDue = computed
          fallbackDueCount++
        }
        if (effectiveDue >= today) continue // Not yet past due
      } else if (jobStatus !== status) {
        continue
      }
    }

    total += safeTotal

    const item: Record<string, unknown> = {
      id: String(job.id ?? ''),
      title: String(job.title ?? ''),
      clientName: String(job.clientName ?? ''),
      status: jobStatus,
      type: jobType,
      totalAmount: round2(safeTotal),
      amountPaid: round2(safePaid),
      amountDue: amountDue,
      createdAt: String(job.createdAt ?? '').slice(0, 10),
      dueDate: effectiveDue || null,
      invoiceNumber: String(job.invoiceNumber ?? ''),
    }

    // Surface daysLate on every result so the AI can say "9 days late"
    // without needing to compute it. Uses effectiveDue (= stored due date for
    // most queries; net-14 fallback for legacy invoices in a 'late' query).
    if (effectiveDue) {
      const days = daysBetween(today, effectiveDue)
      if (days != null && days > 0) {
        item.daysLate = days
      }
    }

    items.push(item)
  }

  // Sort late items by daysLate desc (most overdue first); other queries
  // sort by createdAt desc as before. Most-overdue-first matches what the
  // user almost always wants to act on first.
  if (status === 'late') {
    items.sort((a, b) => Number(b.daysLate ?? 0) - Number(a.daysLate ?? 0))
  } else {
    items.sort((a, b) => String(b.createdAt).localeCompare(String(a.createdAt)))
  }

  let note: string | undefined
  if (status === 'late') {
    const parts = [
      `Late = sent invoices with a balance owing AND a due date (or net-14 terms) before ${today}.`,
    ]
    if (fallbackDueCount > 0) {
      parts.push(
        `${fallbackDueCount} had no due date set, so net-14 terms (invoice date + 14 days) were used to assess lateness.`,
      )
    }
    if (missingDueDateCount > 0) {
      parts.push(
        `${missingDueDateCount} sent invoice(s) had neither a due date nor a usable invoice date and were excluded.`,
      )
    }
    note = parts.join(' ')
  }

  return {
    items,
    total: round2(total),
    count: items.length,
    partial: false,
    ...(note ? { note } : {}),
  }
}

interface PaymentsInput {
  clientName?: string
  method?: 'cash' | 'card' | 'check' | 'bank_transfer' | 'any'
  dateFrom?: string
  dateTo?: string
}

export function executeQueryPayments(
  context: BusinessContext,
  input: PaymentsInput
): ToolResult {
  const payments = asArray(context.recentPayments)
  const method = input.method || 'any'

  const items: Array<Record<string, unknown>> = []
  let total = 0

  for (const p of payments) {
    if (input.clientName && !nameMatches(p.clientName, input.clientName)) continue
    if (method !== 'any' && lower(p.method) !== method) continue
    if (!inDateRange(p.date, input.dateFrom, input.dateTo)) continue

    const amount = Number(p.amount ?? 0)
    if (!Number.isFinite(amount)) continue
    total += amount

    items.push({
      date: String(p.date ?? '').slice(0, 10),
      amount: round2(amount),
      method: String(p.method ?? ''),
      clientName: String(p.clientName ?? ''),
      jobTitle: String(p.jobTitle ?? ''),
      reference: String(p.reference ?? ''),
    })
  }

  items.sort((a, b) => String(b.date).localeCompare(String(a.date)))

  // For the UNFILTERED "how much have I collected (all-time)?" question, the
  // authoritative total is sum(jobs.amountPaid) — every dollar received across
  // all invoices — NOT the sum of explicit payment rows. payment_events misses
  // money collected on jobs marked paid without a recorded payment (the local
  // app fills those with auto-reconciled rows; the cloud has none), and the
  // payment list is capped at 30. Jobs carry the truth and are identical on
  // both the text (local) and voice (cloud) brains, so this makes them agree.
  // Filtered queries (by method / client / date) still sum the matching
  // payment records — jobs carry no per-payment method or date.
  const unfiltered = method === 'any' &&
      !input.clientName &&
      !input.dateFrom &&
      !input.dateTo
  let resolvedTotal = total
  if (unfiltered) {
    let collected = 0
    for (const j of asArray(context.jobs)) {
      if (lower(j.type) === 'quote') continue
      const paid = Number(j.amountPaid ?? 0)
      if (Number.isFinite(paid)) collected += paid
    }
    resolvedTotal = collected
  }

  return {
    items,
    total: round2(resolvedTotal),
    count: items.length,
    // When unfiltered the total is complete (from jobs); the capped list is
    // just a recent sample, so don't flag the total as partial.
    partial: !unfiltered && items.length >= 30,
    note: unfiltered
      ? 'Total is all money received across every invoice (authoritative); the list below shows recent payments only.'
      : items.length >= 30
          ? 'Hit the 30-payment cap. Recommend the Analytics screen for older windows.'
          : undefined,
  }
}

interface CustomerBalanceInput {
  clientName?: string
}

export function executeQueryCustomerBalance(
  context: BusinessContext,
  input: CustomerBalanceInput
): ToolResult {
  const customers = asArray(context.customers)
  const target = input.clientName

  if (!target) {
    return {
      items: [],
      total: 0,
      count: 0,
      partial: false,
      note: 'No clientName provided.',
    }
  }

  // Find the best name match — case-insensitive substring is the same
  // semantic the rest of the tools use; for one-customer lookup we
  // additionally accept a contains-or-contained match.
  const targetLower = lower(target)
  let match: Record<string, unknown> | null = null
  let matchScore = Infinity

  for (const c of customers) {
    const candidate = lower(c.name)
    if (!candidate) continue
    if (candidate === targetLower) {
      match = c
      matchScore = 0
      break
    }
    if (candidate.includes(targetLower) || targetLower.includes(candidate)) {
      // Prefer shorter candidate (closer match) on ties.
      const score = Math.abs(candidate.length - targetLower.length)
      if (score < matchScore) {
        match = c
        matchScore = score
      }
    }
  }

  if (!match) {
    return {
      items: [],
      total: 0,
      count: 0,
      partial: false,
      note: `No customer found matching "${target}". Customer list does not contain that name — do not invent.`,
    }
  }

  // Derive the balance from the customer's INVOICES, not the stored customer
  // aggregate. The stored total_billed counts QUOTES as billed — but a quote is
  // a proposal, not a receivable — which inflated balances (an $850 quote made
  // a customer read as owing $891 when he owed $41 on his one unpaid invoice).
  // Single source of truth = the jobs: sum total/paid across the customer's
  // committed invoices (type 'invoice', not drafts/cancelled). Matches how
  // executeQueryJobs now derives owed (total - paid), so the two never disagree.
  const matchName = String(match.name ?? '')
  let totalBilled = 0
  let totalPaid = 0
  let invoiceCount = 0
  for (const job of asArray(context.jobs)) {
    if (lower(job.type) !== 'invoice') continue // exclude quotes
    if (!['sent', 'partially_paid', 'paid'].includes(lower(job.status))) continue // exclude drafts/cancelled
    if (!nameMatches(job.clientName, matchName)) continue
    const t = Number(job.totalAmount ?? 0)
    const p = Number(job.amountPaid ?? 0)
    totalBilled += Number.isFinite(t) ? t : 0
    totalPaid += Number.isFinite(p) ? p : 0
    invoiceCount++
  }
  const balance = round2(Math.max(0, totalBilled - totalPaid))
  return {
    items: [
      {
        name: matchName,
        phone: String(match.phone ?? ''),
        email: String(match.email ?? ''),
        jobCount: invoiceCount,
        totalBilled: round2(totalBilled),
        totalPaid: round2(totalPaid),
        balance: balance,
      },
    ],
    total: balance,
    count: 1,
    partial: false,
  }
}

interface LookupMaterialPricesInput {
  items?: unknown
}

/**
 * Fuzzy score (0–100) between a user-named material and a historical
 * material / receipt-line name. lookup_material_prices uses it to decide
 * which past line items refer to the same thing being priced now.
 *
 * Deliberately conservative: it returns 0 (no match) rather than a weak
 * guess, because a wrong match would seed a WRONG price — and this
 * feature's contract is "a real price the user used, or ASK", never a
 * fabricated number.
 *
 *   100   exact, after normalisation
 *   70    one string contains the other ("PEX pipe" ⊂ "1/2in PEX pipe 100ft")
 *   20–60 share of meaningful tokens (≥3 chars, minus tiny stopwords):
 *         20 points per shared token, capped at 60
 *   0     nothing in common
 */
function materialNameScore(query: string, candidate: string): number {
  const norm = (s: string) =>
    String(s ?? '')
      .toLowerCase()
      .replace(/[^a-z0-9]+/g, ' ')
      .trim()
  const a = norm(query)
  const b = norm(candidate)
  if (!a || !b) return 0
  if (a === b) return 100
  if (a.includes(b) || b.includes(a)) return 70

  const stop = new Set(['the', 'and', 'for', 'with', 'per', 'pack', 'set', 'box', 'each'])
  const tokenSet = (s: string) =>
    new Set(s.split(' ').filter((t) => t.length >= 3 && !stop.has(t)))
  const at = tokenSet(a)
  const bt = tokenSet(b)
  if (at.size === 0 || bt.size === 0) return 0
  let shared = 0
  for (const t of at) if (bt.has(t)) shared++
  if (shared === 0) return 0
  return Math.min(60, shared * 20)
}

interface PriceCandidate {
  name: string
  unitPrice: number
  basis: 'sell' | 'cost'
  source: 'invoice' | 'quote' | 'receipt'
  status: string
  date: string
  from: string
  /** sell price on a sent/paid/partially_paid invoice — the most authoritative. */
  authoritative: boolean
}

/**
 * Auto-pricing source for new quotes/invoices (AI_FEATURES_PLAN.md Feature 1).
 * For each material the user named WITHOUT a price, find what they've charged
 * or paid for the same thing before, so the draft can be pre-filled instead
 * of guessed.
 *
 * Two price sources, both already in the snapshot:
 *   1. jobs[].materials — past invoices/quotes the user built. `cost` is the
 *      post-markup LINE total (what they billed); dividing by quantity gives a
 *      per-unit SELL price. `actualSupplierCost` is the pre-markup line total
 *      (what they paid the supplier) — a per-unit COST basis.
 *   2. recentExpenses[].ocrItems — scanned-receipt line items, each with a
 *      per-unit `unitPrice` (or totalPrice/quantity): what the user PAID at
 *      the store. A COST basis, never a sell price.
 *
 * Why basis matters: building a quote needs the SELL price (what to charge).
 * A receipt only tells us COST, so the suggestion carries `basis` and the
 * model is told to ASK for a margin rather than billing cost as price. When
 * there is no history at all, the model ASKS rather than inventing — this
 * tool never fabricates a number. That "real price or ask" contract is the
 * whole point of the feature.
 *
 * Per-item suggestion ranking:
 *   1. most-recent SELL price from a sent/paid/partially_paid invoice
 *   2. most-recent SELL price from any other job (quote / draft invoice)
 *   3. most-recent COST price (receipt or supplier cost)
 *   4. none → status "not_found"
 */
export function executeLookupMaterialPrices(
  context: BusinessContext,
  input: LookupMaterialPricesInput
): ToolResult {
  // Coerce `items` to string[]; tolerate a single bare string or stray nulls.
  const rawItems = Array.isArray(input.items)
    ? input.items
    : input.items != null
      ? [input.items]
      : []
  const queries = rawItems
    .map((v) => String(v ?? '').trim())
    .filter((s) => s.length > 0)

  if (queries.length === 0) {
    return {
      items: [],
      total: 0,
      count: 0,
      partial: false,
      note: 'No item names provided. Ask the user which materials to price.',
    }
  }

  // Build the candidate pool ONCE, then score it against each query item.
  const pool: PriceCandidate[] = []
  const sentLike = new Set(['sent', 'paid', 'partially_paid'])

  for (const job of asArray(context.jobs)) {
    const source: 'invoice' | 'quote' = lower(job.type) === 'quote' ? 'quote' : 'invoice'
    const status = lower(job.status)
    const date = String(job.createdAt ?? '').slice(0, 10)
    const from = String(job.clientName ?? '')
    for (const m of asArray(job.materials)) {
      const name = String(m.item ?? '').trim()
      if (!name) continue
      const qtyRaw = Number(m.quantity ?? 1)
      const qty = Number.isFinite(qtyRaw) && qtyRaw > 0 ? qtyRaw : 1

      const cost = Number(m.cost ?? NaN) // post-markup line total = sell
      if (Number.isFinite(cost) && cost > 0) {
        pool.push({
          name,
          unitPrice: round2(cost / qty),
          basis: 'sell',
          source,
          status,
          date,
          from,
          authoritative: source === 'invoice' && sentLike.has(status),
        })
      }

      const supplier = Number(m.actualSupplierCost ?? NaN) // pre-markup line total = cost
      if (Number.isFinite(supplier) && supplier > 0) {
        pool.push({
          name,
          unitPrice: round2(supplier / qty),
          basis: 'cost',
          source,
          status,
          date,
          from,
          authoritative: false,
        })
      }
    }
  }

  for (const e of asArray(context.recentExpenses)) {
    const ocrItems = asArray(e.ocrItems)
    if (ocrItems.length === 0) continue
    const date = String(e.date ?? '').slice(0, 10)
    const vendor = String(e.vendor ?? '')
    for (const oi of ocrItems) {
      const name = String(oi.name ?? '').trim()
      if (!name) continue
      const qtyRaw = Number(oi.quantity ?? 1)
      const qty = Number.isFinite(qtyRaw) && qtyRaw > 0 ? qtyRaw : 1
      let unit = Number(oi.unitPrice ?? NaN)
      if (!Number.isFinite(unit) || unit <= 0) {
        const tp = Number(oi.totalPrice ?? NaN)
        unit = Number.isFinite(tp) && tp > 0 ? tp / qty : NaN
      }
      if (!Number.isFinite(unit) || unit <= 0) continue
      pool.push({
        name,
        unitPrice: round2(unit),
        basis: 'cost',
        source: 'receipt',
        status: '',
        date,
        from: vendor,
        authoritative: false,
      })
    }
  }

  const byRecency = (a: PriceCandidate, b: PriceCandidate) =>
    String(b.date).localeCompare(String(a.date))
  // Rank tier for the `matches` preview: authoritative sell, then sell, then cost.
  const rankTier = (c: PriceCandidate) =>
    c.authoritative ? 0 : c.basis === 'sell' ? 1 : 2

  const resultItems: Array<Record<string, unknown>> = []
  let foundCount = 0
  let anyCostOnly = false
  let anyConflict = false
  let anyNotFound = false

  for (const query of queries) {
    const scored = pool
      .map((c) => ({ c, score: materialNameScore(query, c.name) }))
      .filter((x) => x.score > 0)

    if (scored.length === 0) {
      anyNotFound = true
      resultItems.push({
        query,
        status: 'not_found',
        suggestion: null,
        priceRange: null,
        matches: [],
      })
      continue
    }

    // Order the preview: best name match, then most authoritative, then recent.
    scored.sort((x, y) => {
      if (y.score !== x.score) return y.score - x.score
      const tx = rankTier(x.c)
      const ty = rankTier(y.c)
      if (tx !== ty) return tx - ty
      return byRecency(x.c, y.c)
    })

    const matches = scored.slice(0, 6).map(({ c }) => ({
      unitPrice: c.unitPrice,
      basis: c.basis,
      source: c.source,
      ...(c.status ? { status: c.status } : {}),
      date: c.date,
      from: c.from,
      item: c.name,
    }))

    // Suggestion: authoritative-sell → any-sell → cost, each most recent.
    const cands = scored.map((s) => s.c)
    const authSell = cands.filter((c) => c.authoritative).sort(byRecency)
    const anySell = cands.filter((c) => c.basis === 'sell').sort(byRecency)
    const costOnly = cands.filter((c) => c.basis === 'cost').sort(byRecency)
    const chosen = authSell[0] ?? anySell[0] ?? costOnly[0] ?? null

    let suggestion: Record<string, unknown> | null = null
    let priceRange: { min: number; max: number } | null = null
    if (chosen) {
      suggestion = {
        unitPrice: chosen.unitPrice,
        basis: chosen.basis,
        source: chosen.source,
        date: chosen.date,
        from: chosen.from,
      }
      // Range over the SAME basis as the suggestion, so a "conflict" means
      // disagreement among comparable prices — not the natural sell-vs-cost gap.
      const prices = cands
        .filter((c) => c.basis === chosen.basis)
        .map((c) => c.unitPrice)
      const min = round2(Math.min(...prices))
      const max = round2(Math.max(...prices))
      priceRange = { min, max }
      if (chosen.basis === 'cost') anyCostOnly = true
      // Materials prices that disagree by >25% of the top price are a
      // conflict the model should resolve by asking, not by picking one.
      if (max > 0 && (max - min) / max > 0.25) anyConflict = true
    }

    foundCount++
    resultItems.push({
      query,
      status: 'found',
      suggestion,
      priceRange,
      matches,
    })
  }

  const noteParts: string[] = [
    'Per item: fill the draft from suggestion.unitPrice ONLY when status is "found", basis is "sell", and priceRange is tight.',
  ]
  if (anyConflict) {
    noteParts.push(
      'Some items have a WIDE priceRange (past prices disagree) — ASK the user which price to use rather than guessing.',
    )
  }
  if (anyCostOnly) {
    noteParts.push(
      'Some items only have a COST basis (what the user paid a supplier, not a past sell price) — do NOT bill cost as the price; tell the user and ASK what to charge / what margin to add.',
    )
  }
  if (anyNotFound) {
    noteParts.push(
      'Some items are "not_found" — the user has no price history for them. Do NOT invent a price; ASK the user what to charge.',
    )
  }

  return {
    // One row per requested item (found or not), so the model can see and
    // ASK about misses. `count` reports how many had usable history.
    items: resultItems,
    total: 0,
    count: foundCount,
    partial: false,
    note: noteParts.join(' '),
  }
}

// ─── Dispatcher ──────────────────────────────────────────────────────────────

/**
 * Route a tool call to the right executor. Unknown tool names produce
 * an error result that the model will see and (hopefully) recover from
 * by calling `respond` with an apology.
 */
export function dispatchQueryTool(
  name: string,
  input: Record<string, unknown>,
  context: BusinessContext
): ToolResult {
  switch (name) {
    case 'query_materials':
      return executeQueryMaterials(context, input as MaterialsInput)
    case 'query_expenses':
      return executeQueryExpenses(context, input as ExpensesInput)
    case 'query_jobs':
      return executeQueryJobs(context, input as JobsInput)
    case 'query_payments':
      return executeQueryPayments(context, input as PaymentsInput)
    case 'query_customer_balance':
      return executeQueryCustomerBalance(context, input as CustomerBalanceInput)
    case 'lookup_material_prices':
      return executeLookupMaterialPrices(context, input as LookupMaterialPricesInput)
    default:
      return {
        items: [],
        total: 0,
        count: 0,
        partial: false,
        note: `Unknown tool: ${name}. Call respond with what you know.`,
      }
  }
}

// ─── Semantic fallback for query_expenses ─────────────────────────────────────
//
// The literal vendor/keyword filter is exact-substring. Users describe vendors
// by what they are ("the electrical supplier"), not by their stored name
// ("Elliott Electric Supply"). When the literal filter finds nothing, the
// dispatcher below asks an injected resolver to map the phrase to real vendor
// names (by embedding similarity over the user's own expenses) and re-runs the
// SAME deterministic executor restricted to those vendors. The resolver only
// ever returns names; it never returns amounts, so the arithmetic stays in code.
//
// The resolver is injected so this file stays free of network and database
// concerns and the behaviour is unit-tested with a fake.

export interface SemanticVendorMatch {
  vendor: string
  similarity: number
}

export type SemanticVendorResolver = (
  phrase: string,
) => Promise<SemanticVendorMatch[]>

function joinNotes(...parts: Array<string | undefined>): string | undefined {
  const kept = parts.map((p) => (p ?? '').trim()).filter(Boolean)
  return kept.length ? kept.join(' ') : undefined
}

/** The phrase the user gave for the vendor, or the keyword as a fallback. */
export function semanticPhraseFor(input: ExpensesInput): string | undefined {
  const vendor = String(input.vendor ?? '').trim()
  if (vendor.length >= 2) return vendor
  const keyword = String(input.keyword ?? '').trim()
  if (keyword.length >= 2) return keyword
  return undefined
}

function describeMatches(matches: SemanticVendorMatch[]): string {
  return matches
    .map((m) => `${m.vendor} (${Math.round(m.similarity * 100)}% match)`)
    .join(', ')
}

/**
 * Like `dispatchQueryTool`, plus the meaning-based vendor fallback for
 * `query_expenses`. Every other tool is passed straight through. Never throws:
 * a resolver failure degrades to the literal (empty) result with a note.
 */
export async function dispatchQueryToolWithSemantics(
  name: string,
  input: Record<string, unknown>,
  context: BusinessContext,
  resolver?: SemanticVendorResolver,
): Promise<ToolResult> {
  const literal = dispatchQueryTool(name, input, context)
  if (name !== 'query_expenses' || !resolver) return literal
  if (literal.count > 0) return literal

  const expensesInput = input as ExpensesInput
  // Already a semantic re-run, or nothing to resolve.
  if (Array.isArray(expensesInput.vendors) && expensesInput.vendors.length > 0) return literal
  const phrase = semanticPhraseFor(expensesInput)
  if (!phrase) return literal

  let matches: SemanticVendorMatch[]
  try {
    matches = (await resolver(phrase)).filter(
      (m) => m && typeof m.vendor === 'string' && m.vendor.trim().length > 0,
    )
  } catch (err) {
    console.error('query_expenses semantic fallback failed:', err)
    return {
      ...literal,
      note: joinNotes(
        literal.note,
        `No literal match for "${phrase}" and the meaning-based vendor match was unavailable. Tell the user nothing matched and suggest the Expenses screen.`,
      ),
    }
  }

  if (matches.length === 0) {
    return {
      ...literal,
      note: joinNotes(
        literal.note,
        `No vendor or description matched "${phrase}", literally or by meaning. Do NOT guess a vendor — tell the user nothing matched and suggest the Expenses screen.`,
      ),
    }
  }

  const rerun = executeQueryExpenses(context, {
    ...expensesInput,
    vendor: undefined,
    keyword: undefined,
    vendors: matches.map((m) => m.vendor),
  })
  const summary = describeMatches(matches)
  const note = rerun.count > 0
    ? `No literal match for "${phrase}"; matched by meaning to ${summary}. Tell the user which vendor you used.`
    : `"${phrase}" matched by meaning to ${summary}, but no expenses from that vendor are in the recent snapshot for the requested filters. Say which vendor was matched and suggest the Expenses screen for older history.`
  return {
    ...rerun,
    semanticMatch: { phrase, vendors: matches },
    note: joinNotes(rerun.note, note),
  }
}

export function isQueryToolName(name: string): name is QueryToolName {
  return (
    name === 'query_materials' ||
    name === 'query_expenses' ||
    name === 'query_jobs' ||
    name === 'query_payments' ||
    name === 'query_customer_balance' ||
    name === 'lookup_material_prices'
  )
}
