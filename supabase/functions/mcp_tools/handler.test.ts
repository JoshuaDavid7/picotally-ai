import { assert, assertEquals } from 'https://deno.land/std@0.224.0/assert/mod.ts'
import { parseToolRequest, toolCatalog, unknownCustomerRefusal } from './handler.ts'
import { QUERY_TOOLS } from '../process_voice_command/tools.ts'

Deno.test('toolCatalog exposes exactly the read-only query tools, each with a JSON schema', () => {
  const catalog = toolCatalog()
  assertEquals(catalog.map((t) => t.name), QUERY_TOOLS.map((t) => t.name))
  assertEquals(catalog.length, 6)
  for (const t of catalog) {
    assert(t.description.length > 20, `${t.name} needs a description the MCP client can show`)
    assertEquals((t.input_schema as { type: string }).type, 'object')
  }
  // Never a write action: the MCP surface reads books, it does not change them.
  for (const t of catalog) assert(!/create|record|send|delete|update/.test(t.name))
})

Deno.test('parseToolRequest accepts a known tool with an object input and defaults input to {}', () => {
  const ok = parseToolRequest({ tool: 'query_expenses', input: { vendor: 'Home Depot' } })
  assert(ok.ok)
  assertEquals(ok.tool, 'query_expenses')
  assertEquals(ok.input, { vendor: 'Home Depot' })
  const bare = parseToolRequest({ tool: 'query_jobs' })
  assert(bare.ok)
  assertEquals(bare.input, {})
})

Deno.test('parseToolRequest refuses unknown tools, non-object bodies and non-object inputs', () => {
  for (const body of [null, 'x', [], 42]) {
    const r = parseToolRequest(body)
    assert(!r.ok && r.status === 400 && r.error === 'invalid_body', `body ${JSON.stringify(body)}`)
  }
  const unknown = parseToolRequest({ tool: 'create_invoice', input: {} })
  assert(!unknown.ok && unknown.error === 'unknown_tool')
  const respond = parseToolRequest({ tool: 'respond', input: {} })
  assert(!respond.ok && respond.error === 'unknown_tool')
  const badInput = parseToolRequest({ tool: 'query_jobs', input: ['x'] })
  assert(!badInput.ok && badInput.error === 'invalid_input')
})

const context = { customers: [{ name: 'Garcia' }, { name: 'Tom Nguyen' }] }

Deno.test('unknownCustomerRefusal lets calls without a clientName through', () => {
  assertEquals(unknownCustomerRefusal({ vendor: 'Home Depot' }, context), null)
})

Deno.test('unknownCustomerRefusal tolerates a typo in a real customer name', () => {
  assertEquals(unknownCustomerRefusal({ clientName: 'Garcai' }, context), null)
})

Deno.test('unknownCustomerRefusal answers an unknown customer with an explicit empty result', () => {
  const r = unknownCustomerRefusal({ clientName: 'Martinez' }, context)
  assert(r !== null)
  assertEquals(r.count, 0)
  assertEquals(r.total, 0)
  assert(r.note?.includes('Martinez'))
  assert(r.note?.includes('do not invent'))
})

Deno.test('unknownCustomerRefusal never blocks when the user has no customers yet', () => {
  assertEquals(unknownCustomerRefusal({ clientName: 'Anyone' }, { customers: [] }), null)
})
