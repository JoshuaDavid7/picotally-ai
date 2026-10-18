// mcp_tools — pure request handling for the PicoTally MCP server endpoint.
//
// The MCP server (see /mcp in the repo root) runs on the user's own machine
// and talks to this edge function with the user's Supabase session. The
// function exposes the SAME read-only query tools the chat and voice agents
// use (`QUERY_TOOLS` in process_voice_command/tools.ts), so a question asked
// through Claude Desktop or Claude Code is answered by the same deterministic
// executors, in integer cents, with the same grounding guard. No write
// actions are exposed here: an MCP client can read a user's books, never
// change them.
//
// Everything in this file is pure so a Deno test can exercise it without a
// database or a network. The HTTP handler lives in index.ts.

import {
  isQueryToolName,
  QUERY_TOOLS,
  type QueryToolName,
  type ToolResult,
} from '../process_voice_command/tools.ts'
import { isKnownCustomer } from '../process_voice_command/grounding.ts'
import type { BusinessContextSnapshot } from '../voice_agent_tools/context.ts'

export interface ToolCatalogEntry {
  name: string
  description: string
  input_schema: unknown
}

/** The tool definitions, serialized for the MCP server's `tools/list`. */
export function toolCatalog(): ToolCatalogEntry[] {
  return QUERY_TOOLS.map((t) => ({
    name: t.name,
    description: t.description,
    input_schema: t.input_schema,
  }))
}

export type ParsedToolRequest =
  | { ok: true; tool: QueryToolName; input: Record<string, unknown> }
  | { ok: false; error: string; status: number }

/** Validates a `{ tool, input }` body. Unknown tools and non-object inputs are refused. */
export function parseToolRequest(body: unknown): ParsedToolRequest {
  if (!body || typeof body !== 'object' || Array.isArray(body)) {
    return { ok: false, error: 'invalid_body', status: 400 }
  }
  const { tool, input } = body as Record<string, unknown>
  if (typeof tool !== 'string' || !isQueryToolName(tool)) {
    return { ok: false, error: 'unknown_tool', status: 400 }
  }
  if (input !== undefined && (input === null || typeof input !== 'object' || Array.isArray(input))) {
    return { ok: false, error: 'invalid_input', status: 400 }
  }
  return { ok: true, tool, input: (input ?? {}) as Record<string, unknown> }
}

/**
 * Grounding guard, mirrored from voice_agent_tools: a tool call scoped to a
 * customer who is not in the user's records is answered with an explicit
 * empty result instead of letting the caller treat "nothing found" as real.
 * `isKnownCustomer` is deliberately generous (typo-tolerant), so real
 * customers are never wrongly blocked. Returns null when the call may proceed.
 */
export function unknownCustomerRefusal(
  input: Record<string, unknown>,
  context: Pick<BusinessContextSnapshot, 'customers'>,
): ToolResult | null {
  const clientName = typeof input.clientName === 'string' ? input.clientName.trim() : ''
  if (!clientName) return null
  const names = context.customers
    .map((c) => String((c as Record<string, unknown>).name ?? ''))
    .filter(Boolean)
  if (names.length > 0 && !isKnownCustomer(clientName, names)) {
    return {
      items: [],
      total: 0,
      count: 0,
      partial: false,
      note: `No customer named "${clientName}" in your records — do not invent one.`,
    }
  }
  return null
}
