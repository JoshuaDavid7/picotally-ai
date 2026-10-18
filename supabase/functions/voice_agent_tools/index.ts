// voice_agent_tools — the brain bridge for the real-time voice agent.
//
// The LiveKit agent worker (agent/main.py — trusted infra holding the service-
// role key) calls this with { userId, tool, input } to run the SAME deterministic
// query tools that Claude uses in process_voice_command. Single source of truth:
// the tool logic + grounding guard stay in Deno and are never re-implemented in
// Python. We build the BusinessContext snapshot from Supabase (scoped to userId),
// apply the grounding guard, dispatch to the existing executor, and return its
// ToolResult.
//
// READ-ONLY query tools only (balances, materials, prices, jobs, payments,
// expenses). The create/record WRITE actions are a separate, carefully gated
// follow-up and are deliberately NOT exposed here.
//
// Auth: this is a server-to-server endpoint (like stripe_webhook /
// app_store_notifications_v2), so it runs verify_jwt=false at the gateway and
// authenticates the caller IN-BODY by requiring the Supabase SERVICE ROLE key as
// the bearer token (only trusted infra holds it). There is no end-user JWT here —
// the worker passes the participant identity (= user id) it got from the
// LiveKit room token minted by voice_room_token.

import { serve } from 'https://deno.land/std@0.168.0/http/server.ts'
import { createClient } from 'https://esm.sh/@supabase/supabase-js@2'
import { dispatchQueryToolWithSemantics, isQueryToolName } from '../process_voice_command/tools.ts'
import { createSemanticVendorResolver, supabaseEmbeddingStore } from '../_shared/expense_semantics.ts'
import { isKnownCustomer } from '../process_voice_command/grounding.ts'
import { buildBusinessContext } from './context.ts'

const corsHeaders = {
  'Access-Control-Allow-Origin': '*',
  'Access-Control-Allow-Headers': 'authorization, x-client-info, apikey, content-type',
}

function getRequiredEnv(key: string): string {
  const v = Deno.env.get(key)?.trim()
  if (!v) throw new Error(`Missing required environment variable: ${key}`)
  return v
}

function json(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { ...corsHeaders, 'Content-Type': 'application/json' },
  })
}

/** Constant-time string compare (don't leak the service key via timing). */
function safeEqual(a: string, b: string): boolean {
  if (a.length !== b.length) return false
  let diff = 0
  for (let i = 0; i < a.length; i += 1) diff |= a.charCodeAt(i) ^ b.charCodeAt(i)
  return diff === 0
}

function bearerToken(req: Request): string | null {
  const h = req.headers.get('Authorization') ?? req.headers.get('authorization')
  if (!h) return null
  const m = h.match(/^Bearer\s+(.+)$/i)
  return m ? m[1].trim() : null
}

serve(async (req) => {
  if (req.method === 'OPTIONS') return new Response('ok', { headers: corsHeaders })
  if (req.method !== 'POST') return json({ error: 'method_not_allowed' }, 405)

  // ── Authenticate the trusted worker (dedicated shared secret) ──
  // We use a dedicated VOICE_AGENT_SECRET (set on both the worker and here)
  // rather than the service-role key: the worker and the edge runtime can hold
  // different representations of the service-role key (Supabase's legacy JWT vs
  // the new secret-key format), so an exact-match on it is brittle. A shared
  // secret we control is both robust and cleaner (don't reuse the god-key as an
  // API token).
  const expectedSecret = getRequiredEnv('VOICE_AGENT_SECRET')
  const token = bearerToken(req)
  if (!token || !safeEqual(token, expectedSecret)) {
    return json({ error: 'unauthorized' }, 401)
  }

  // ── Parse + validate ──
  let payload: { userId?: unknown; tool?: unknown; input?: unknown; roomName?: unknown }
  try {
    payload = await req.json()
  } catch {
    return json({ error: 'bad_json' }, 400)
  }
  const userId = typeof payload.userId === 'string' ? payload.userId.trim() : ''
  const roomName = typeof payload.roomName === 'string' ? payload.roomName.trim() : ''
  const tool = typeof payload.tool === 'string' ? payload.tool : ''
  const input = (payload.input && typeof payload.input === 'object'
    ? (payload.input as Record<string, unknown>)
    : {})
  if (!userId) return json({ error: 'missing_userId' }, 400)
  if (!roomName) return json({ error: 'missing_roomName' }, 400)
  if (!isQueryToolName(tool)) return json({ error: `unknown_tool:${tool}` }, 400)

  // ── Session binding (1.0.1 hardening 3/4) ──
  // The shared secret proves the CALLER is our worker; this proves the CALL
  // is inside a live voice session of the exact user it queries. A leaked
  // secret alone can no longer query arbitrary users: the (userId, roomName)
  // pair must match a session voice_room_token minted and voice_room_close
  // has not revoked, no older than 24h (the backstop for sessions that died
  // unclosed). Cheap syntactic gate first — room names embed the user id.
  if (!roomName.startsWith(`voice-${userId}-`)) {
    return json({ error: 'session_not_bound' }, 403)
  }

  try {
    const supabase = createClient(
      getRequiredEnv('SUPABASE_URL'),
      getRequiredEnv('SUPABASE_SERVICE_ROLE_KEY'),
    )

    const cutoff = new Date(Date.now() - 24 * 60 * 60 * 1000).toISOString()
    const { data: session, error: sessionError } = await supabase
      .from('voice_sessions')
      .select('room_name')
      .eq('room_name', roomName)
      .eq('user_id', userId)
      .is('ended_at', null)
      .gt('started_at', cutoff)
      .maybeSingle()
    if (sessionError) {
      console.error('voice_agent_tools: session lookup failed', sessionError)
      return json({ error: 'session_lookup_failed' }, 500)
    }
    if (!session) {
      return json({ error: 'session_not_bound' }, 403)
    }
    // Touch the session (observability; never blocks the tool call).
    void supabase
      .from('voice_sessions')
      .update({ last_tool_call_at: new Date().toISOString() })
      .eq('room_name', roomName)
      .then(() => {}, () => {})

    const context = await buildBusinessContext(supabase, userId)

    // ── Grounding guard (defense-in-depth) ──
    // If a tool is scoped to a named client who isn't in the user's records,
    // refuse rather than let the model treat empty results as real. Mirrors the
    // pre-LLM guard in process_voice_command. `isKnownCustomer` is deliberately
    // generous, so real customers are never wrongly blocked.
    const clientName = typeof input.clientName === 'string' ? input.clientName : ''
    if (clientName) {
      const names = context.customers
        .map((c) => String((c as Record<string, unknown>).name ?? ''))
        .filter(Boolean)
      if (names.length > 0 && !isKnownCustomer(clientName, names)) {
        return json({
          items: [],
          total: 0,
          count: 0,
          partial: false,
          note: `No customer named "${clientName}" in your records — do not invent one.`,
        })
      }
    }

    // Meaning-based vendor fallback for query_expenses. Same service-role
    // client (this function already runs as service role); every lookup inside
    // is scoped to `userId`. Undefined when the Gemini key is absent.
    const resolver = createSemanticVendorResolver({
      apiKey: Deno.env.get('GOOGLE_GEMINI_KEY'),
      userId,
      store: supabaseEmbeddingStore(supabase),
    })
    const result = await dispatchQueryToolWithSemantics(tool, input, context, resolver)
    return json(result)
  } catch (err) {
    console.error('voice_agent_tools failed', err)
    return json({ error: 'internal_error' }, 500)
  }
})
