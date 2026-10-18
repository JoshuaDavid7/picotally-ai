import test from 'node:test'
import assert from 'node:assert/strict'
import { formatResult, normalizeSchema, toMcpTools } from '../src/catalog.js'

test('toMcpTools carries name, description and the object schema through', () => {
  const tools = toMcpTools([
    { name: 'query_expenses', description: 'Sum and list expenses', input_schema: { type: 'object', properties: { vendor: { type: 'string' } } } },
  ])
  assert.equal(tools.length, 1)
  assert.equal(tools[0].name, 'query_expenses')
  assert.equal(tools[0].description, 'Sum and list expenses')
  assert.equal(tools[0].inputSchema.type, 'object')
  assert.deepEqual((tools[0].inputSchema as unknown as { properties: unknown }).properties, { vendor: { type: 'string' } })
})

test('normalizeSchema forces an object schema and tolerates garbage from the wire', () => {
  assert.deepEqual(normalizeSchema(null), { type: 'object', properties: {} })
  assert.deepEqual(normalizeSchema('x'), { type: 'object', properties: {} })
  assert.equal(normalizeSchema({ type: 'string' }).type, 'object')
})

test('formatResult renders a ToolResult as one JSON text block', () => {
  const out = formatResult({ count: 2, total: 587.4, items: [] })
  assert.equal(out.content.length, 1)
  assert.equal(out.content[0].type, 'text')
  assert.ok(out.content[0].text.includes('"total": 587.4'))
  assert.equal(out.isError, undefined)
})

test('formatResult flags an error payload so the client does not treat it as data', () => {
  const out = formatResult({ error: 'unknown_tool' })
  assert.equal(out.isError, true)
})
