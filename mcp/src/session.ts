// Session storage and refresh for the user's Supabase login.
//
// The file holds the user's access and refresh tokens, mode 0600, in
// ~/.picotally/mcp-session.json by default. Access tokens live about an hour;
// the refresh token renews them. Nothing here is a PicoTally secret: it is the
// user's own session, the same one the app holds on their phone.

import fs from 'node:fs/promises'
import path from 'node:path'
import { createClient, type SupabaseClient } from '@supabase/supabase-js'
import { PUBLISHABLE_KEY_HELP, type Config } from './config.js'

export interface StoredSession {
  access_token: string
  refresh_token: string
  /** Unix seconds. */
  expires_at: number
  email?: string
}

export function needsRefresh(expiresAt: number, nowSeconds = Math.floor(Date.now() / 1000), leewaySeconds = 60): boolean {
  return expiresAt - leewaySeconds <= nowSeconds
}

export function toStored(
  session: { access_token: string; refresh_token: string; expires_at?: number; expires_in?: number },
  email?: string,
  nowSeconds = Math.floor(Date.now() / 1000),
): StoredSession {
  const expires_at = session.expires_at ?? nowSeconds + (session.expires_in ?? 3600)
  return { access_token: session.access_token, refresh_token: session.refresh_token, expires_at, email }
}

export async function readSession(file: string): Promise<StoredSession | null> {
  try {
    const raw = JSON.parse(await fs.readFile(file, 'utf8')) as Partial<StoredSession>
    if (typeof raw.access_token === 'string' && typeof raw.refresh_token === 'string' && typeof raw.expires_at === 'number') {
      return raw as StoredSession
    }
    return null
  } catch {
    return null
  }
}

export async function writeSession(file: string, session: StoredSession): Promise<void> {
  await fs.mkdir(path.dirname(file), { recursive: true, mode: 0o700 })
  await fs.writeFile(file, JSON.stringify(session, null, 2), { mode: 0o600 })
  await fs.chmod(file, 0o600)
}

export async function clearSession(file: string): Promise<void> {
  try {
    await fs.unlink(file)
  } catch {
    // already gone
  }
}

export class SessionManager {
  constructor(private readonly cfg: Config) {}

  private client(): SupabaseClient {
    if (!this.cfg.publishableKey) throw new Error(PUBLISHABLE_KEY_HELP)
    return createClient(this.cfg.supabaseUrl, this.cfg.publishableKey, {
      auth: { persistSession: false, autoRefreshToken: false, detectSessionInUrl: false },
    })
  }

  /** A valid access token, refreshing first when it is about to expire. */
  async accessToken(): Promise<string> {
    const stored = await readSession(this.cfg.sessionFile)
    if (!stored) throw new Error('Not logged in. Run: picotally-mcp login')
    if (!needsRefresh(stored.expires_at)) return stored.access_token
    return (await this.refresh(stored)).access_token
  }

  async refresh(stored: StoredSession): Promise<StoredSession> {
    const { data, error } = await this.client().auth.refreshSession({ refresh_token: stored.refresh_token })
    if (error || !data.session) {
      throw new Error(`Session expired; run: picotally-mcp login (${error?.message ?? 'no session returned'})`)
    }
    const next = toStored(data.session, stored.email)
    await writeSession(this.cfg.sessionFile, next)
    return next
  }

  /** Email + password sign-in (the app's email accounts). */
  async loginWithPassword(email: string, password: string): Promise<void> {
    const { data, error } = await this.client().auth.signInWithPassword({ email, password })
    if (error || !data.session) throw new Error(`Sign-in failed: ${error?.message ?? 'no session returned'}`)
    await writeSession(this.cfg.sessionFile, toStored(data.session, email))
  }

  /** Email one-time code: a code is emailed, the user types it back. */
  async loginWithEmailCode(email: string, promptCode: () => Promise<string>): Promise<void> {
    const sb = this.client()
    const { error } = await sb.auth.signInWithOtp({ email, options: { shouldCreateUser: false } })
    if (error) throw new Error(`Could not send a code: ${error.message}`)
    const token = (await promptCode()).trim()
    const { data, error: verifyError } = await sb.auth.verifyOtp({ email, token, type: 'email' })
    if (verifyError || !data.session) throw new Error(`Code rejected: ${verifyError?.message ?? 'no session returned'}`)
    await writeSession(this.cfg.sessionFile, toStored(data.session, email))
  }
}
