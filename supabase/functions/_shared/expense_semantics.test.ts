// Tests for the meaning-based vendor resolver behind query_expenses.
//
// Run with:
//   deno test supabase/functions/_shared/expense_semantics.test.ts
//
// No network and no database: Gemini is a fetch stub that records request
// bodies and returns vectors of the right shape; the store is an in-memory
// fake. What these pin:
//   - source text composition and vector normalisation
//   - batching at 100 and the exact request shape Gemini expects
//   - idempotency: unchanged rows are not re-embedded, changed/new rows are
//   - vendor aggregation: best score per vendor, floor, cap, empty vendors skipped
//   - the resolver is undefined without a key and NEVER throws on failure
//   - misaligned API responses are rejected, not silently attached to rows

import { assert, assertEquals, assertRejects } from 'https://deno.land/std@0.224.0/assert/mod.ts'

import {
  aggregateVendorMatches,
  buildEmbeddingSourceText,
  createSemanticVendorResolver,
  EMBEDDING_DIMS,
  embedTexts,
  ensureExpenseEmbeddings,
  estimateTokens,
  normalizeVector,
  planEmbeddingWork,
  resolveVendorsSemantically,
  type EmbeddingStore,
  type EmbeddingUpsertRow,
  type VendorNeighbour,
} from './expense_semantics.ts'

// ─── Fakes ───────────────────────────────────────────────────────────────────

function vec(seed: number): number[] {
  // Deterministic, non-normalised vector so normalisation is observable.
  return Array.from({ length: EMBEDDING_DIMS }, (_, i) => ((i * 7 + seed) % 13) - 6)
}

/** A fetch stub for batchEmbedContents that echoes one vector per request. */
function embedFetch(opts: { dimsOverride?: number; shortBy?: number } = {}) {
  const calls: Array<{ url: string; body: Record<string, unknown> }> = []
  const impl = ((url: string | URL | Request, init?: RequestInit) => {
    const body = JSON.parse(String(init?.body ?? '{}'))
    calls.push({ url: String(url), body })
    const n = (body.requests?.length ?? 0) - (opts.shortBy ?? 0)
    const embeddings = Array.from({ length: Math.max(0, n) }, (_, i) => ({
      values: opts.dimsOverride
        ? Array.from({ length: opts.dimsOverride }, () => 1)
        : vec(i + 1),
    }))
    return Promise.resolve(new Response(JSON.stringify({ embeddings }), { status: 200 }))
  }) as unknown as typeof fetch
  return { impl, calls }
}

function fakeStore(init: {
  expenses?: Array<{ id: string; vendor: string | null; description: string | null; category: string | null }>
  existing?: Array<{ expense_id: string; source_text: string; model: string }>
  neighbours?: VendorNeighbour[]
} = {}) {
  const upserts: EmbeddingUpsertRow[][] = []
  const matchCalls: Array<{ userId: string; limit: number; queryNorm: number }> = []
  const store: EmbeddingStore = {
    listExpenses: (_u, limit) => Promise.resolve((init.expenses ?? []).slice(0, limit)),
    listEmbeddings: () => Promise.resolve(init.existing ?? []),
    upsertEmbeddings: (rows) => {
      upserts.push(rows)
      return Promise.resolve()
    },
    matchVendors: (userId, query, limit) => {
      matchCalls.push({
        userId,
        limit,
        queryNorm: Math.sqrt(query.reduce((s, x) => s + x * x, 0)),
      })
      return Promise.resolve(init.neighbours ?? [])
    },
  }
  return { store, upserts, matchCalls }
}

const noUsage = () => {}
const DEPS = (fetchImpl: typeof fetch) => ({ apiKey: 'k', fetchImpl, sleepImpl: () => Promise.resolve() })

// ─── Pure helpers ────────────────────────────────────────────────────────────

Deno.test('buildEmbeddingSourceText: vendor first, parts optional, whitespace collapsed', () => {
  assertEquals(
    buildEmbeddingSourceText({ id: '1', vendor: '  Elliott   Electric Supply ', description: 'wire\n and  breakers', category: 'materials' }),
    'Vendor: Elliott Electric Supply. Expense: wire and breakers. Category: materials.',
  )
  assertEquals(buildEmbeddingSourceText({ id: '2', vendor: null, description: 'fuel', category: null }), 'Expense: fuel.')
  assertEquals(buildEmbeddingSourceText({ id: '3', vendor: '', description: '', category: '' }), '')
})

Deno.test('normalizeVector: unit length; zero vector passes through', () => {
  const v = normalizeVector([3, 4])
  assertEquals(v, [0.6, 0.8])
  assertEquals(normalizeVector([0, 0]), [0, 0])
})

Deno.test('estimateTokens: ~4 chars per token, rounded up', () => {
  assertEquals(estimateTokens(''), 0)
  assertEquals(estimateTokens('abcd'), 1)
  assertEquals(estimateTokens('abcde'), 2)
})

Deno.test('aggregateVendorMatches: best score per vendor, floor, cap, empty vendor skipped', () => {
  const rows: VendorNeighbour[] = [
    { expense_id: 'a', vendor: 'Elliott Electric Supply', description: '', similarity: 0.71 },
    { expense_id: 'b', vendor: 'elliott electric supply', description: '', similarity: 0.66 },
    { expense_id: 'c', vendor: 'Home Depot', description: '', similarity: 0.58 },
    { expense_id: 'd', vendor: '', description: 'no vendor', similarity: 0.99 },
    { expense_id: 'e', vendor: 'Shell', description: '', similarity: 0.31 },
  ]
  const out = aggregateVendorMatches(rows, 0.5, 5)
  assertEquals(out.map((m) => m.vendor), ['Elliott Electric Supply', 'Home Depot'])
  assertEquals(out[0].similarity, 0.71)
  assertEquals(out[0].expenseCount, 2)
  assertEquals(aggregateVendorMatches(rows, 0.5, 1).length, 1)
})

Deno.test('planEmbeddingWork: skips unchanged rows, includes new/changed/model-changed, drops empty text', () => {
  const expenses = [
    { id: '1', vendor: 'Home Depot', description: 'lumber', category: 'materials' },
    { id: '2', vendor: 'Shell', description: 'gas', category: 'fuel' },
    { id: '3', vendor: 'Shell', description: 'gas', category: 'fuel' },
    { id: '4', vendor: null, description: null, category: null },
  ]
  const existing = [
    { expense_id: '1', source_text: buildEmbeddingSourceText(expenses[0]), model: 'm' },
    { expense_id: '2', source_text: 'Vendor: Shell. Expense: OLD. Category: fuel.', model: 'm' },
    { expense_id: '3', source_text: buildEmbeddingSourceText(expenses[2]), model: 'other-model' },
  ]
  const work = planEmbeddingWork(expenses, existing, 'm')
  assertEquals(work.map((w) => w.id), ['2', '3'])
})

// ─── Gemini embedding call ───────────────────────────────────────────────────

Deno.test('embedTexts: batches at 100, hits batchEmbedContents, sends taskType + 768 dims, normalises', async () => {
  const f = embedFetch()
  const texts = Array.from({ length: 150 }, (_, i) => `t${i}`)
  const out = await embedTexts(texts, 'RETRIEVAL_DOCUMENT', DEPS(f.impl))
  assertEquals(out.length, 150)
  assertEquals(f.calls.length, 2)
  assert(f.calls[0].url.includes(':batchEmbedContents?key=k'))
  const reqs = f.calls[0].body.requests as Array<Record<string, unknown>>
  assertEquals(reqs.length, 100)
  assertEquals(reqs[0].model, 'models/gemini-embedding-001')
  assertEquals(reqs[0].taskType, 'RETRIEVAL_DOCUMENT')
  assertEquals(reqs[0].outputDimensionality, 768)
  const norm = Math.sqrt(out[0].reduce((s, x) => s + x * x, 0))
  assert(Math.abs(norm - 1) < 1e-9, `expected unit vector, got ${norm}`)
})

Deno.test('embedTexts: rejects a misaligned response instead of attaching wrong vectors', async () => {
  await assertRejects(() => embedTexts(['a', 'b'], 'RETRIEVAL_DOCUMENT', DEPS(embedFetch({ shortBy: 1 }).impl)))
  await assertRejects(() => embedTexts(['a'], 'RETRIEVAL_DOCUMENT', DEPS(embedFetch({ dimsOverride: 10 }).impl)))
})

Deno.test('embedTexts: empty input makes no calls', async () => {
  const f = embedFetch()
  assertEquals(await embedTexts([], 'RETRIEVAL_QUERY', DEPS(f.impl)), [])
  assertEquals(f.calls.length, 0)
})

// ─── Orchestration ───────────────────────────────────────────────────────────

Deno.test('ensureExpenseEmbeddings: embeds only missing/changed rows, upserts correct shape, meters tokens', async () => {
  const expenses = [
    { id: '1', vendor: 'Home Depot', description: 'lumber', category: 'materials' },
    { id: '2', vendor: 'Elliott Electric Supply', description: 'breakers', category: 'materials' },
  ]
  const { store, upserts } = fakeStore({
    expenses,
    existing: [{ expense_id: '1', source_text: buildEmbeddingSourceText(expenses[0]), model: 'gemini-embedding-001' }],
  })
  const f = embedFetch()
  const metered: Array<[string, string, Record<string, number>]> = []
  const res = await ensureExpenseEmbeddings(store, 'user-1', DEPS(f.impl), {
    recordUsageImpl: (u, s, units) => metered.push([String(u), s, units]),
  })
  assertEquals(res, { embedded: 1, considered: 2 })
  assertEquals(upserts.length, 1)
  assertEquals(upserts[0].length, 1)
  const row = upserts[0][0]
  assertEquals(row.expense_id, '2')
  assertEquals(row.user_id, 'user-1')
  assertEquals(row.model, 'gemini-embedding-001')
  assertEquals(row.dims, 768)
  assertEquals(row.embedding.length, 768)
  assertEquals(metered.length, 1)
  assertEquals(metered[0][1], 'gemini')
  assert(metered[0][2].embedding_tokens > 0)
})

Deno.test('ensureExpenseEmbeddings: nothing to do → zero Gemini calls, zero upserts', async () => {
  const expenses = [{ id: '1', vendor: 'Home Depot', description: 'lumber', category: 'materials' }]
  const { store, upserts } = fakeStore({
    expenses,
    existing: [{ expense_id: '1', source_text: buildEmbeddingSourceText(expenses[0]), model: 'gemini-embedding-001' }],
  })
  const f = embedFetch()
  const res = await ensureExpenseEmbeddings(store, 'u', DEPS(f.impl), { recordUsageImpl: noUsage })
  assertEquals(res.embedded, 0)
  assertEquals(f.calls.length, 0)
  assertEquals(upserts.length, 0)
})

Deno.test('resolveVendorsSemantically: query is RETRIEVAL_QUERY, unit-norm, scoped to the user; aggregates', async () => {
  const { store, matchCalls } = fakeStore({
    expenses: [],
    neighbours: [
      { expense_id: 'a', vendor: 'Elliott Electric Supply', description: '', similarity: 0.72 },
      { expense_id: 'b', vendor: 'Home Depot', description: '', similarity: 0.41 },
    ],
  })
  const f = embedFetch()
  const out = await resolveVendorsSemantically(store, 'user-9', 'the electrical supplier', DEPS(f.impl), {
    recordUsageImpl: noUsage,
  })
  assertEquals(out.map((m) => m.vendor), ['Elliott Electric Supply'])
  const reqs = f.calls[0].body.requests as Array<Record<string, unknown>>
  assertEquals(reqs[0].taskType, 'RETRIEVAL_QUERY')
  assertEquals(matchCalls[0].userId, 'user-9')
  assert(Math.abs(matchCalls[0].queryNorm - 1) < 1e-9)
})

Deno.test('resolveVendorsSemantically: phrase too short → [] without any calls', async () => {
  const { store, matchCalls } = fakeStore()
  const f = embedFetch()
  assertEquals(await resolveVendorsSemantically(store, 'u', ' x ', DEPS(f.impl), { recordUsageImpl: noUsage }), [])
  assertEquals(f.calls.length, 0)
  assertEquals(matchCalls.length, 0)
})

// ─── Resolver factory ────────────────────────────────────────────────────────

Deno.test('createSemanticVendorResolver: undefined without key or user', () => {
  const { store } = fakeStore()
  assertEquals(createSemanticVendorResolver({ apiKey: '', userId: 'u', store }), undefined)
  assertEquals(createSemanticVendorResolver({ apiKey: 'k', userId: '', store }), undefined)
  assert(createSemanticVendorResolver({ apiKey: 'k', userId: 'u', store }) !== undefined)
})

Deno.test('createSemanticVendorResolver: never throws — failures log and return []', async () => {
  const { store } = fakeStore()
  const failingFetch = (() => Promise.reject(new Error('network down'))) as unknown as typeof fetch
  const logged: unknown[] = []
  const resolver = createSemanticVendorResolver({
    apiKey: 'k',
    userId: 'u',
    store,
    fetchImpl: failingFetch,
    sleepImpl: () => Promise.resolve(),
    recordUsageImpl: noUsage,
    logger: (_m, e) => logged.push(e),
  })!
  assertEquals(await resolver('the electrical supplier'), [])
  assertEquals(logged.length, 1)
})

Deno.test('createSemanticVendorResolver: returns vendor+similarity pairs only', async () => {
  const { store } = fakeStore({
    neighbours: [{ expense_id: 'a', vendor: 'Elliott Electric Supply', description: '', similarity: 0.8 }],
  })
  const resolver = createSemanticVendorResolver({
    apiKey: 'k',
    userId: 'u',
    store,
    fetchImpl: embedFetch().impl,
    sleepImpl: () => Promise.resolve(),
    recordUsageImpl: noUsage,
  })!
  assertEquals(await resolver('electrical supplier'), [{ vendor: 'Elliott Electric Supply', similarity: 0.8 }])
})
