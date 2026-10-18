import test from 'node:test'
import assert from 'node:assert/strict'
import { PicoTallyApi } from '../src/api.js'

function fakeFetch(handler: (url: string, init: RequestInit) => { status: number; body: unknown }): typeof fetch {
  return (async (input: string | URL | Request, init?: RequestInit) => {
    const r = handler(String(input), init ?? {})
    return new Response(JSON.stringify(r.body), { status: r.status, headers: { 'Content-Type': 'application/json' } })
  }) as typeof fetch
}

test('catalog() sends the bearer token with a GET and returns the tools array', async () => {
  let seen: RequestInit | undefined
  const api = new PicoTallyApi('https://x.test/functions/v1', async () => 'tok', fakeFetch((url, init) => {
    seen = init
    assert.equal(url, 'https://x.test/functions/v1/mcp_tools')
    return { status: 200, body: { tools: [{ name: 'query_jobs', description: 'd', input_schema: { type: 'object' } }] } }
  }))
  const tools = await api.catalog()
  assert.equal(tools[0].name, 'query_jobs')
  assert.equal(seen?.method, 'GET')
  assert.equal((seen?.headers as Record<string, string>).Authorization, 'Bearer tok')
})

test('call() posts { tool, input } and surfaces non-2xx responses as errors', async () => {
  const api = new PicoTallyApi('https://x.test/functions/v1', async () => 'tok', fakeFetch((_url, init) => {
    const body = JSON.parse(String(init.body))
    assert.deepEqual(body, { tool: 'query_expenses', input: { vendor: 'Home Depot' } })
    return { status: 200, body: { count: 1, total: 42 } }
  }))
  assert.deepEqual(await api.call('query_expenses', { vendor: 'Home Depot' }), { count: 1, total: 42 })

  const failing = new PicoTallyApi('https://x.test/functions/v1', async () => 'tok', fakeFetch(() => ({ status: 401, body: { error: 'unauthorized' } })))
  await assert.rejects(() => failing.call('query_jobs', {}), /401/)
})
