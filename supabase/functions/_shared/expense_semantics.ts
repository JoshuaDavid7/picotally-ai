// Meaning-based vendor retrieval for `query_expenses`.
//
// Problem: the literal vendor filter is an exact substring. A user asks "what
// did I spend at the electrical supplier in June?" and the stored vendor is
// "Elliott Electric Supply" — zero rows, and the model has nothing to work
// with. This module turns the user's phrase into vendor NAMES by embedding
// similarity over the user's own expenses. The deterministic executor in
// process_voice_command/tools.ts then filters and sums as it always has. The
// model never sees an amount that was not computed in code.
//
// Pipeline (all per user, all server-side with the service role):
//   1. ensureExpenseEmbeddings — embed every expense whose (vendor,
//      description, category) text is new or changed, in batches, with
//      Gemini `gemini-embedding-001` at 768 dims, L2-normalised, stored in
//      public.expense_embeddings (pgvector). Idempotent: unchanged rows cost
//      nothing.
//   2. resolveVendorsSemantically — embed the phrase as a RETRIEVAL_QUERY,
//      ask Postgres for the nearest expenses (cosine), aggregate by vendor,
//      keep those above a similarity floor, return at most MAX_VENDORS.
//
// Storage and network are behind the `EmbeddingStore` interface and the
// injectable `fetch`, so every decision in here is unit-tested with fakes
// (see expense_semantics.test.ts). Failure is always soft: callers get an
// empty match list and a logged error, never an exception on the user path.

import { callGemini } from './gemini.ts'
import { recordUsage as recordUsageDefault } from './usage_cost.ts'

export const EMBEDDING_MODEL_DEFAULT = 'gemini-embedding-001'
export const EMBEDDING_DIMS = 768
/** Cosine similarity floor below which a vendor is not considered a match. */
export const DEFAULT_MIN_SIMILARITY = 0.5
/** Most vendors a single phrase may resolve to. */
export const MAX_VENDORS = 5
/** Most recent expenses kept embedded per user (bounds cost and latency). */
export const MAX_EXPENSES_TO_EMBED = 400
/** Gemini batchEmbedContents accepts up to 100 texts per request. */
export const EMBED_BATCH_SIZE = 100
/** Nearest neighbours requested from Postgres before vendor aggregation. */
const MATCH_LIMIT = 20

export type EmbeddingTaskType = 'RETRIEVAL_DOCUMENT' | 'RETRIEVAL_QUERY'

export interface ExpenseEmbeddingSource {
  id: string
  vendor: string | null
  description: string | null
  category: string | null
}

export interface StoredEmbeddingMeta {
  expense_id: string
  source_text: string
  model: string
}

export interface EmbeddingUpsertRow {
  expense_id: string
  user_id: string
  model: string
  dims: number
  source_text: string
  embedding: number[]
}

export interface VendorNeighbour {
  expense_id: string
  vendor: string
  description: string
  similarity: number
}

/** Storage seam. The real implementation wraps supabase-js; tests use a fake. */
export interface EmbeddingStore {
  listExpenses(userId: string, limit: number): Promise<ExpenseEmbeddingSource[]>
  listEmbeddings(userId: string): Promise<StoredEmbeddingMeta[]>
  upsertEmbeddings(rows: EmbeddingUpsertRow[]): Promise<void>
  matchVendors(userId: string, query: number[], limit: number): Promise<VendorNeighbour[]>
}

export interface EmbedDeps {
  apiKey: string
  model?: string
  fetchImpl?: typeof fetch
  sleepImpl?: (ms: number) => Promise<void>
}

export interface VendorMatch {
  vendor: string
  similarity: number
  /** How many of the nearest expenses carried this vendor. */
  expenseCount: number
}

// ─── Pure helpers ────────────────────────────────────────────────────────────

function clean(s: unknown): string {
  return String(s ?? '').replace(/\s+/g, ' ').trim()
}

/**
 * The text that gets embedded for one expense. Vendor first because that is
 * what the user's phrase is really about; description and category add the
 * "what kind of place is this" signal that makes "electrical supplier" land on
 * the right row even when the vendor name alone would not.
 */
export function buildEmbeddingSourceText(row: ExpenseEmbeddingSource): string {
  const parts: string[] = []
  const vendor = clean(row.vendor)
  const description = clean(row.description)
  const category = clean(row.category)
  if (vendor) parts.push(`Vendor: ${vendor}.`)
  if (description) parts.push(`Expense: ${description}.`)
  if (category) parts.push(`Category: ${category}.`)
  return parts.join(' ')
}

/** L2-normalise so cosine distance in Postgres is exact. Zero vectors pass through. */
export function normalizeVector(v: number[]): number[] {
  let sum = 0
  for (const x of v) sum += x * x
  const norm = Math.sqrt(sum)
  if (!Number.isFinite(norm) || norm === 0) return v.slice()
  return v.map((x) => x / norm)
}

/** The embeddings endpoint returns no usage metadata; ~4 chars per token. */
export function estimateTokens(text: string): number {
  return Math.ceil(text.length / 4)
}

/** Case/space-insensitive vendor key used for aggregation. */
export function vendorKey(vendor: string): string {
  return clean(vendor).toLowerCase()
}

/**
 * Aggregate nearest-neighbour rows into vendor matches: best similarity per
 * vendor, count of supporting rows, floor applied, strongest first, capped.
 * Rows with an empty vendor are skipped — there is nothing to filter on.
 */
export function aggregateVendorMatches(
  rows: VendorNeighbour[],
  minSimilarity: number = DEFAULT_MIN_SIMILARITY,
  maxVendors: number = MAX_VENDORS,
): VendorMatch[] {
  const byVendor = new Map<string, VendorMatch>()
  for (const r of rows) {
    const key = vendorKey(r.vendor)
    if (!key) continue
    const sim = Number(r.similarity)
    if (!Number.isFinite(sim)) continue
    const cur = byVendor.get(key)
    if (!cur) {
      byVendor.set(key, { vendor: clean(r.vendor), similarity: sim, expenseCount: 1 })
    } else {
      cur.expenseCount += 1
      if (sim > cur.similarity) cur.similarity = sim
    }
  }
  return [...byVendor.values()]
    .filter((m) => m.similarity >= minSimilarity)
    .sort((a, b) => b.similarity - a.similarity)
    .slice(0, Math.max(1, maxVendors))
}

/** Which expenses need (re)embedding: new rows, changed text, or a different model. */
export function planEmbeddingWork(
  expenses: ExpenseEmbeddingSource[],
  existing: StoredEmbeddingMeta[],
  model: string,
): Array<{ id: string; sourceText: string }> {
  const have = new Map(existing.map((e) => [e.expense_id, e]))
  const work: Array<{ id: string; sourceText: string }> = []
  for (const e of expenses) {
    const sourceText = buildEmbeddingSourceText(e)
    if (!sourceText) continue
    const prior = have.get(e.id)
    if (prior && prior.source_text === sourceText && prior.model === model) continue
    work.push({ id: e.id, sourceText })
  }
  return work
}

// ─── Gemini embeddings ───────────────────────────────────────────────────────

interface BatchEmbedResponse {
  embeddings?: Array<{ values?: unknown }>
}

/**
 * Embed texts with Gemini in batches of EMBED_BATCH_SIZE. Returns one
 * normalised vector per input, in order. Throws if the API returns the wrong
 * number or shape of vectors — a silent misalignment would attach the wrong
 * embedding to the wrong expense, which is worse than no embedding.
 */
export async function embedTexts(
  texts: string[],
  taskType: EmbeddingTaskType,
  deps: EmbedDeps,
): Promise<number[][]> {
  if (texts.length === 0) return []
  const model = deps.model ?? EMBEDDING_MODEL_DEFAULT
  const out: number[][] = []
  for (let i = 0; i < texts.length; i += EMBED_BATCH_SIZE) {
    const batch = texts.slice(i, i + EMBED_BATCH_SIZE)
    const body = {
      requests: batch.map((text) => ({
        model: `models/${model}`,
        content: { parts: [{ text }] },
        taskType,
        outputDimensionality: EMBEDDING_DIMS,
      })),
    }
    const res = (await callGemini({
      apiKey: deps.apiKey,
      model,
      endpoint: 'batchEmbedContents',
      body,
      timeoutMs: 15000,
      maxRetries: 1,
      fetchImpl: deps.fetchImpl,
      sleepImpl: deps.sleepImpl,
    })) as BatchEmbedResponse
    const embeddings = res?.embeddings
    if (!Array.isArray(embeddings) || embeddings.length !== batch.length) {
      throw new Error(
        `embedTexts: expected ${batch.length} embeddings, got ${Array.isArray(embeddings) ? embeddings.length : 'none'}`,
      )
    }
    for (const e of embeddings) {
      const values = e?.values
      if (!Array.isArray(values) || values.length !== EMBEDDING_DIMS) {
        throw new Error(
          `embedTexts: expected ${EMBEDDING_DIMS}-dim vector, got ${Array.isArray(values) ? values.length : 'none'}`,
        )
      }
      out.push(normalizeVector(values.map(Number)))
    }
  }
  return out
}

// ─── Orchestration ───────────────────────────────────────────────────────────

export interface EnsureOptions {
  maxExpenses?: number
  recordUsageImpl?: typeof recordUsageDefault
}

/**
 * Bring the user's embeddings up to date. Idempotent and cheap when nothing
 * changed: one read of expenses, one read of embedding metadata, zero Gemini
 * calls. Returns how many rows were (re)embedded.
 */
export async function ensureExpenseEmbeddings(
  store: EmbeddingStore,
  userId: string,
  deps: EmbedDeps,
  opts: EnsureOptions = {},
): Promise<{ embedded: number; considered: number }> {
  const model = deps.model ?? EMBEDDING_MODEL_DEFAULT
  const limit = opts.maxExpenses ?? MAX_EXPENSES_TO_EMBED
  const [expenses, existing] = await Promise.all([
    store.listExpenses(userId, limit),
    store.listEmbeddings(userId),
  ])
  const work = planEmbeddingWork(expenses, existing, model)
  if (work.length === 0) return { embedded: 0, considered: expenses.length }

  const vectors = await embedTexts(work.map((w) => w.sourceText), 'RETRIEVAL_DOCUMENT', deps)
  const rows: EmbeddingUpsertRow[] = work.map((w, i) => ({
    expense_id: w.id,
    user_id: userId,
    model,
    dims: EMBEDDING_DIMS,
    source_text: w.sourceText,
    embedding: vectors[i],
  }))
  await store.upsertEmbeddings(rows)

  const tokens = work.reduce((n, w) => n + estimateTokens(w.sourceText), 0)
  ;(opts.recordUsageImpl ?? recordUsageDefault)(userId, 'gemini', { embedding_tokens: tokens })
  return { embedded: rows.length, considered: expenses.length }
}

export interface ResolveOptions extends EnsureOptions {
  minSimilarity?: number
  maxVendors?: number
}

/**
 * Resolve a user's phrase ("the electrical supplier") to vendor names from
 * their own expenses. Returns [] when nothing clears the similarity floor.
 */
export async function resolveVendorsSemantically(
  store: EmbeddingStore,
  userId: string,
  phrase: string,
  deps: EmbedDeps,
  opts: ResolveOptions = {},
): Promise<VendorMatch[]> {
  const text = clean(phrase)
  if (text.length < 2) return []
  await ensureExpenseEmbeddings(store, userId, deps, opts)
  const [query] = await embedTexts([text], 'RETRIEVAL_QUERY', deps)
  ;(opts.recordUsageImpl ?? recordUsageDefault)(userId, 'gemini', {
    embedding_tokens: estimateTokens(text),
  })
  const rows = await store.matchVendors(userId, query, MATCH_LIMIT)
  return aggregateVendorMatches(rows, opts.minSimilarity, opts.maxVendors)
}

/**
 * The resolver `dispatchQueryToolWithSemantics` expects. Undefined when the
 * feature is not configured (no Gemini key), so callers can pass it straight
 * through and get literal-only behaviour. Never throws: errors are logged and
 * reported as "no matches".
 */
export function createSemanticVendorResolver(args: {
  apiKey: string | undefined | null
  userId: string | undefined | null
  store: EmbeddingStore
  model?: string
  minSimilarity?: number
  fetchImpl?: typeof fetch
  sleepImpl?: (ms: number) => Promise<void>
  recordUsageImpl?: typeof recordUsageDefault
  logger?: (message: string, err: unknown) => void
}): ((phrase: string) => Promise<Array<{ vendor: string; similarity: number }>>) | undefined {
  const apiKey = (args.apiKey ?? '').trim()
  const userId = (args.userId ?? '').trim()
  if (!apiKey || !userId) return undefined
  const log = args.logger ?? ((m, e) => console.error(m, e))
  const deps: EmbedDeps = {
    apiKey,
    model: args.model,
    fetchImpl: args.fetchImpl,
    sleepImpl: args.sleepImpl,
  }
  return async (phrase: string) => {
    try {
      const matches = await resolveVendorsSemantically(args.store, userId, phrase, deps, {
        minSimilarity: args.minSimilarity,
        recordUsageImpl: args.recordUsageImpl,
      })
      return matches.map((m) => ({ vendor: m.vendor, similarity: m.similarity }))
    } catch (err) {
      log('semantic vendor resolver failed', err)
      return []
    }
  }
}

// ─── Supabase-backed store ───────────────────────────────────────────────────

// Minimal structural type so this module does not pin a supabase-js version.
// deno-lint-ignore no-explicit-any
type SupabaseLike = any

export function supabaseEmbeddingStore(client: SupabaseLike): EmbeddingStore {
  return {
    async listExpenses(userId, limit) {
      const { data, error } = await client
        .from('expenses')
        .select('id, vendor, description, category')
        .eq('user_id', userId)
        .order('expense_date', { ascending: false })
        .limit(limit)
      if (error) throw new Error(`listExpenses: ${error.message ?? error}`)
      return (data ?? []) as ExpenseEmbeddingSource[]
    },
    async listEmbeddings(userId) {
      const { data, error } = await client
        .from('expense_embeddings')
        .select('expense_id, source_text, model')
        .eq('user_id', userId)
      if (error) throw new Error(`listEmbeddings: ${error.message ?? error}`)
      return (data ?? []) as StoredEmbeddingMeta[]
    },
    async upsertEmbeddings(rows) {
      if (rows.length === 0) return
      const payload = rows.map((r) => ({ ...r, updated_at: new Date().toISOString() }))
      const { error } = await client
        .from('expense_embeddings')
        .upsert(payload, { onConflict: 'expense_id' })
      if (error) throw new Error(`upsertEmbeddings: ${error.message ?? error}`)
    },
    async matchVendors(userId, query, limit) {
      const { data, error } = await client.rpc('match_expense_vendors', {
        p_user_id: userId,
        p_query: query,
        p_limit: limit,
      })
      if (error) throw new Error(`match_expense_vendors: ${error.message ?? error}`)
      return (data ?? []) as VendorNeighbour[]
    },
  }
}
