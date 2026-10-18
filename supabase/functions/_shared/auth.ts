// Shared auth helper for Picotally edge functions.
//
// Why this exists:
//   In April 2026 we migrated to a new Supabase project that ships with
//   the asymmetric JWT signing-keys system enabled by default. GoTrue
//   issues ES256 (ECC P-256) access tokens. The Supabase Edge Function
//   gateway's `verify_jwt = true` flag only knows how to validate the
//   legacy HS256 JWTs and rejects ES256 with
//   `UNAUTHORIZED_UNSUPPORTED_TOKEN_ALGORITHM`. Per the canonical
//   Supabase 2026 guidance (issue #41691, closed not-planned):
//     1. Set `verify_jwt = false` on every user-facing function
//     2. Validate the caller's JWT *inside* the function using the
//        server-side GoTrue `auth.getUser(token)` call, which works
//        regardless of signing algorithm
//
// If you skip step 2, the function is wide open. There is no longer
// any platform-level safety net.
//
// Performance note: getUser() makes a network call to GoTrue per
// invocation (~30-100ms). For our AI pipeline functions that already
// take multiple seconds, that's negligible. If we ever need to drive
// this latency down, swap to `auth.getClaims(token)` which verifies
// locally via cached JWKS — but that requires supabase-js >= 2.50 and
// adds complexity around JWKS rotation handling.

import { createClient } from "https://esm.sh/@supabase/supabase-js@2";

const SUPABASE_URL = Deno.env.get("SUPABASE_URL") ?? "";
const ANON_KEY = Deno.env.get("SUPABASE_ANON_KEY") ?? "";

// Minimal CORS surface — every function should already export its own
// corsHeaders, but we re-emit on the 401 response so the SDK's fetch
// CORS preflight is satisfied without the function having to wire us
// through.
const corsHeaders = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Headers": "authorization, x-client-info, apikey, content-type",
};

export interface AuthedUser {
  /** auth.users.id — UUID. Use this for every per-user row lookup. */
  userId: string;
  /** Caller's verified email. May be undefined for some auth providers. */
  email?: string;
  /** Raw JWT — pass to createClient() if you need an RLS-scoped client. */
  accessToken: string;
}

/**
 * Validates the caller's Authorization Bearer JWT against GoTrue.
 *
 * Returns either:
 *   - `AuthedUser` on success, OR
 *   - a `Response` (401) that the caller should return verbatim.
 *
 * Usage:
 *   const auth = await requireUser(req);
 *   if (auth instanceof Response) return auth;
 *   // auth.userId is now safe to trust
 */
export async function requireUser(req: Request): Promise<AuthedUser | Response> {
  const authHeader = req.headers.get("Authorization") ?? req.headers.get("authorization") ?? "";
  const token = authHeader.startsWith("Bearer ")
    ? authHeader.slice("Bearer ".length).trim()
    : "";

  if (!token) {
    return new Response(
      JSON.stringify({ error: "Missing authorization" }),
      { status: 401, headers: { ...corsHeaders, "Content-Type": "application/json" } },
    );
  }

  if (!SUPABASE_URL || !ANON_KEY) {
    // This is a deploy/config bug, not a user error — log loudly so it
    // shows up in function logs.
    console.error("requireUser: SUPABASE_URL or SUPABASE_ANON_KEY missing from function secrets");
    return new Response(
      JSON.stringify({ error: "Server misconfigured" }),
      { status: 500, headers: { ...corsHeaders, "Content-Type": "application/json" } },
    );
  }

  const client = createClient(SUPABASE_URL, ANON_KEY, {
    global: { headers: { Authorization: `Bearer ${token}` } },
    auth: { persistSession: false, autoRefreshToken: false },
  });

  const { data, error } = await client.auth.getUser(token);
  if (error || !data?.user) {
    console.warn(`requireUser: auth.getUser failed: ${error?.message ?? "no user"}`);
    return new Response(
      JSON.stringify({ error: "Invalid session" }),
      { status: 401, headers: { ...corsHeaders, "Content-Type": "application/json" } },
    );
  }

  return {
    userId: data.user.id,
    email: data.user.email,
    accessToken: token,
  };
}
