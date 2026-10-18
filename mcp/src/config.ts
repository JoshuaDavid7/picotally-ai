// Configuration for the PicoTally MCP server.
//
// The Supabase URL is public. The publishable key is public by design too (it
// ships in every PicoTally client and the database enforces per-user isolation
// with row-level security), but this repository keeps every key out of source
// as a rule, so it is read from the environment. The user's own session token
// is what grants access; it is stored in a 0600 file under the home directory.

import os from 'node:os'
import path from 'node:path'

export const DEFAULT_SUPABASE_URL = 'https://txtgzycaopcmnfamwuam.supabase.co'

export interface Config {
  supabaseUrl: string
  /** Empty when PICOTALLY_PUBLISHABLE_KEY is unset; only sign-in needs it. */
  publishableKey: string
  functionsUrl: string
  sessionFile: string
}

export const PUBLISHABLE_KEY_HELP =
  'PICOTALLY_PUBLISHABLE_KEY is not set. It is the public client key that ships with the ' +
  'PicoTally app (not a secret); export it in your shell or add it to the MCP client config. See mcp/README.md.'

export function loadConfig(env: NodeJS.ProcessEnv = process.env): Config {
  const supabaseUrl = (env.PICOTALLY_SUPABASE_URL ?? DEFAULT_SUPABASE_URL).replace(/\/+$/, '')
  return {
    supabaseUrl,
    publishableKey: (env.PICOTALLY_PUBLISHABLE_KEY ?? '').trim(),
    functionsUrl: `${supabaseUrl}/functions/v1`,
    sessionFile: env.PICOTALLY_SESSION_FILE ?? path.join(os.homedir(), '.picotally', 'mcp-session.json'),
  }
}
