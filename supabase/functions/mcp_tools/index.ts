// mcp_tools — the endpoint behind the PicoTally MCP server (repo root /mcp).
//
// Auth: verify_jwt = false at the gateway (ES256 limitation, see
// _shared/auth.ts); the caller's Supabase JWT is validated IN-FUNCTION with
// requireUser. This is a user-facing endpoint, unlike voice_agent_tools, which
// serves the trusted voice worker. Two consequences:
//   * The business snapshot is read with an RLS-SCOPED client built from the
//     caller's own token, so the database enforces per-user isolation on top
//     of the explicit user_id filters in buildBusinessContext.
//   * Only read-only query tools are reachable. Writes stay in the app, behind
//     the user's own taps.
//
// GET  → { tools: [...] }   the tool catalog for the MCP server's tools/list
// POST → { tool, input }    runs one query tool and returns its ToolResult
//
// Semantic vendor matching (query_expenses fallback) needs the embeddings
// table, which is service-role only, so that one helper uses the service
// role, scoped to the caller's user id, exactly as process_voice_command does.

import { serve } from 'https://deno.land/std@0.168.0/http/server.ts'
import { createClient } from 'https://esm.sh/@supabase/supabase-js@2'
import { requireUser } from '../_shared/auth.ts'
import { createSemanticVendorResolver, supabaseEmbeddingStore } from '../_shared/expense_semantics.ts'
import { dispatchQueryToolWithSemantics } from '../process_voice_command/tools.ts'
import { buildBusinessContext } from '../voice_agent_tools/context.ts'
import { parseToolRequest, toolCatalog, unknownCustomerRefusal } from './handler.ts'

const corsHeaders = {
  'Access-Control-Allow-Origin': '*',
  'Access-Control-Allow-Headers': 'authorization, x-client-info, apikey, content-type',
}

function json(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { ...corsHeaders, 'Content-Type': 'application/json' },
  })
}

function getRequiredEnv(key: string): string {
  const v = Deno.env.get(key)?.trim()
  if (!v) throw new Error(`Missing required environment variable: ${key}`)
  return v
}

serve(async (req) => {
  if (req.method === 'OPTIONS') return new Response('ok', { headers: corsHeaders })

  const auth = await requireUser(req)
  if (auth instanceof Response) return auth

  if (req.method === 'GET') return json({ tools: toolCatalog() })
  if (req.method !== 'POST') return json({ error: 'method_not_allowed' }, 405)

  let body: unknown
  try {
    body = await req.json()
  } catch {
    return json({ error: 'invalid_json' }, 400)
  }
  const parsed = parseToolRequest(body)
  if (!parsed.ok) return json({ error: parsed.error }, parsed.status)

  try {
    const url = getRequiredEnv('SUPABASE_URL')
    const userClient = createClient(url, getRequiredEnv('SUPABASE_ANON_KEY'), {
      global: { headers: { Authorization: `Bearer ${auth.accessToken}` } },
      auth: { persistSession: false, autoRefreshToken: false },
    })
    const context = await buildBusinessContext(userClient, auth.userId)

    const refusal = unknownCustomerRefusal(parsed.input, context)
    if (refusal) return json(refusal)

    const geminiKey = Deno.env.get('GOOGLE_GEMINI_KEY')
    const serviceKey = Deno.env.get('SUPABASE_SERVICE_ROLE_KEY')
    const resolver = geminiKey && serviceKey
      ? createSemanticVendorResolver({
        apiKey: geminiKey,
        userId: auth.userId,
        store: supabaseEmbeddingStore(
          createClient(url, serviceKey, { auth: { persistSession: false, autoRefreshToken: false } }),
        ),
      })
      : undefined

    const result = await dispatchQueryToolWithSemantics(parsed.tool, parsed.input, context, resolver)
    return json(result)
  } catch (err) {
    console.error('mcp_tools failed', err)
    return json({ error: 'internal_error' }, 500)
  }
})
