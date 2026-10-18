// End-to-end over stdio: spawn the real CLI, speak MCP JSON-RPC to it, and
// point it at a fake mcp_tools endpoint. Catches SDK wiring mistakes the unit
// tests cannot (handler registration, transport framing, result shapes).
import test from 'node:test'
import assert from 'node:assert/strict'
import http from 'node:http'
import os from 'node:os'
import path from 'node:path'
import fs from 'node:fs/promises'
import { spawn } from 'node:child_process'
import { fileURLToPath } from 'node:url'

async function waitFor(pred: () => boolean, ms = 8000): Promise<void> {
  const start = Date.now()
  while (!pred()) {
    if (Date.now() - start > ms) throw new Error('timed out waiting for a response')
    await new Promise((r) => setTimeout(r, 25))
  }
}

test('the stdio server initializes, lists tools from the endpoint, and forwards a call', async () => {
  const catalog = [{
    name: 'query_jobs',
    description: 'List jobs',
    input_schema: { type: 'object', properties: { clientName: { type: 'string' } } },
  }]
  const endpoint = http.createServer((req, res) => {
    let body = ''
    req.on('data', (c) => (body += c))
    req.on('end', () => {
      if (req.headers.authorization !== 'Bearer test-token') {
        res.writeHead(401, { 'Content-Type': 'application/json' })
        res.end(JSON.stringify({ error: 'unauthorized' }))
        return
      }
      res.writeHead(200, { 'Content-Type': 'application/json' })
      if (req.method === 'GET') res.end(JSON.stringify({ tools: catalog }))
      else res.end(JSON.stringify({ count: 1, total: 450, items: [{ title: 'Bathroom', received: JSON.parse(body) }] }))
    })
  })
  await new Promise<void>((r) => endpoint.listen(0, '127.0.0.1', r))
  const port = (endpoint.address() as { port: number }).port

  const dir = await fs.mkdtemp(path.join(os.tmpdir(), 'picotally-mcp-stdio-'))
  const sessionFile = path.join(dir, 'session.json')
  await fs.writeFile(sessionFile, JSON.stringify({
    access_token: 'test-token', refresh_token: 'r', expires_at: Math.floor(Date.now() / 1000) + 3600,
  }))

  const cli = fileURLToPath(new URL('../src/cli.js', import.meta.url))
  const child = spawn(process.execPath, [cli, 'serve'], {
    env: { ...process.env, PICOTALLY_SUPABASE_URL: `http://127.0.0.1:${port}`, PICOTALLY_SESSION_FILE: sessionFile },
    stdio: ['pipe', 'pipe', 'pipe'],
  })
  const responses: Array<{ id?: number; result?: any }> = []
  let buf = ''
  child.stdout.on('data', (c) => {
    buf += c.toString()
    let i: number
    while ((i = buf.indexOf('\n')) >= 0) {
      const line = buf.slice(0, i).trim()
      buf = buf.slice(i + 1)
      if (line) responses.push(JSON.parse(line))
    }
  })
  const send = (m: unknown) => child.stdin.write(JSON.stringify(m) + '\n')

  try {
    send({ jsonrpc: '2.0', id: 1, method: 'initialize', params: { protocolVersion: '2025-06-18', capabilities: {}, clientInfo: { name: 'test', version: '0' } } })
    await waitFor(() => responses.some((r) => r.id === 1))
    send({ jsonrpc: '2.0', method: 'notifications/initialized' })
    send({ jsonrpc: '2.0', id: 2, method: 'tools/list' })
    await waitFor(() => responses.some((r) => r.id === 2))
    send({ jsonrpc: '2.0', id: 3, method: 'tools/call', params: { name: 'query_jobs', arguments: { clientName: 'Garcia' } } })
    await waitFor(() => responses.some((r) => r.id === 3))
    send({ jsonrpc: '2.0', id: 4, method: 'tools/call', params: { name: 'create_invoice', arguments: {} } })
    await waitFor(() => responses.some((r) => r.id === 4))
  } finally {
    child.kill()
    endpoint.close()
    await fs.rm(dir, { recursive: true, force: true })
  }

  const init = responses.find((r) => r.id === 1)!
  assert.equal(init.result.serverInfo.name, 'picotally')
  const list = responses.find((r) => r.id === 2)!
  assert.deepEqual(list.result.tools.map((t: { name: string }) => t.name), ['query_jobs'])
  assert.equal(list.result.tools[0].inputSchema.type, 'object')
  const call = responses.find((r) => r.id === 3)!
  const text: string = call.result.content[0].text
  assert.ok(text.includes('"total": 450'))
  assert.ok(text.includes('Garcia'), 'arguments reach the endpoint')
  const unknown = responses.find((r) => r.id === 4)!
  assert.equal(unknown.result.isError, true, 'a write-looking tool name is refused client-side too')
})
