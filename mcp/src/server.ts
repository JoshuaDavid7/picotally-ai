// The MCP server over stdio. Tools are discovered from the mcp_tools endpoint
// at first use, so the catalog on the server side is the single source of
// truth; nothing about the tools is duplicated here.
//
// Rule for this file: stdout belongs to the MCP protocol. Log to stderr only.

import { Server } from '@modelcontextprotocol/sdk/server/index.js'
import { StdioServerTransport } from '@modelcontextprotocol/sdk/server/stdio.js'
import { CallToolRequestSchema, ListToolsRequestSchema } from '@modelcontextprotocol/sdk/types.js'
import type { PicoTallyApi } from './api.js'
import { formatResult, toMcpTools, type McpToolDef } from './catalog.js'

export const SERVER_INFO = { name: 'picotally', version: '0.1.0' }

export function buildServer(api: PicoTallyApi): Server {
  const server = new Server(SERVER_INFO, { capabilities: { tools: {} } })
  let cached: McpToolDef[] | null = null
  const tools = async (): Promise<McpToolDef[]> => {
    if (!cached) cached = toMcpTools(await api.catalog())
    return cached
  }

  server.setRequestHandler(ListToolsRequestSchema, async () => ({ tools: await tools() }))

  server.setRequestHandler(CallToolRequestSchema, async (req) => {
    const name = req.params.name
    const args = (req.params.arguments ?? {}) as Record<string, unknown>
    if (!(await tools()).some((t) => t.name === name)) {
      return { content: [{ type: 'text', text: `Unknown tool: ${name}` }], isError: true }
    }
    try {
      return formatResult(await api.call(name, args))
    } catch (err) {
      return { content: [{ type: 'text', text: (err as Error).message }], isError: true }
    }
  })
  return server
}

export async function serveStdio(api: PicoTallyApi): Promise<void> {
  const server = buildServer(api)
  await server.connect(new StdioServerTransport())
  process.stderr.write('picotally-mcp: serving over stdio\n')
}
