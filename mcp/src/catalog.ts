// Pure mapping between the mcp_tools catalog and what an MCP client expects.
// Kept free of I/O so it can be unit-tested with node:test.

export interface ToolCatalogEntry {
  name: string
  description: string
  input_schema: unknown
}

export interface McpToolDef {
  name: string
  description: string
  inputSchema: { type: 'object'; [key: string]: unknown }
}

/** MCP requires an object schema; the server's schemas already are, but never trust the wire. */
export function normalizeSchema(schema: unknown): McpToolDef['inputSchema'] {
  if (schema && typeof schema === 'object' && !Array.isArray(schema)) {
    return { ...(schema as Record<string, unknown>), type: 'object' }
  }
  return { type: 'object', properties: {} }
}

export function toMcpTools(catalog: ToolCatalogEntry[]): McpToolDef[] {
  return catalog.map((t) => ({
    name: t.name,
    description: t.description,
    inputSchema: normalizeSchema(t.input_schema),
  }))
}

// A type alias (not an interface) so it is assignable to the SDK's result
// shapes, which carry an index signature.
export type McpToolResult = {
  content: Array<{ type: 'text'; text: string }>
  isError?: boolean
}

/**
 * A ToolResult becomes one JSON text block. The numbers inside were computed
 * by the server in integer cents; the client model only reads them. An
 * `{ error }` payload is flagged so the client knows not to treat it as data.
 */
export function formatResult(result: unknown): McpToolResult {
  const isError = !!(result && typeof result === 'object' && 'error' in (result as Record<string, unknown>))
  const text = JSON.stringify(result, null, 2)
  return isError ? { content: [{ type: 'text', text }], isError: true } : { content: [{ type: 'text', text }] }
}
