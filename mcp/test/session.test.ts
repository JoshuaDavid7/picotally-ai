import test from 'node:test'
import assert from 'node:assert/strict'
import os from 'node:os'
import path from 'node:path'
import fs from 'node:fs/promises'
import { needsRefresh, readSession, toStored, writeSession, clearSession } from '../src/session.js'

test('needsRefresh is true inside the leeway window and false well before expiry', () => {
  assert.equal(needsRefresh(1000, 900, 60), false)
  assert.equal(needsRefresh(1000, 950, 60), true)
  assert.equal(needsRefresh(1000, 1200, 60), true)
})

test('toStored derives expires_at from expires_in when the server omits it', () => {
  const s = toStored({ access_token: 'a', refresh_token: 'r', expires_in: 3600 }, 'me@example.com', 1_000_000)
  assert.equal(s.expires_at, 1_003_600)
  assert.equal(s.email, 'me@example.com')
  assert.equal(toStored({ access_token: 'a', refresh_token: 'r', expires_at: 42 }).expires_at, 42)
})

test('session file round-trips, is private, and malformed files read as logged out', async () => {
  const dir = await fs.mkdtemp(path.join(os.tmpdir(), 'picotally-mcp-'))
  const file = path.join(dir, 'session.json')
  await writeSession(file, { access_token: 'a', refresh_token: 'r', expires_at: 1 })
  const mode = (await fs.stat(file)).mode & 0o777
  assert.equal(mode, 0o600)
  assert.deepEqual(await readSession(file), { access_token: 'a', refresh_token: 'r', expires_at: 1 })
  await fs.writeFile(file, '{"access_token": 1}')
  assert.equal(await readSession(file), null)
  await clearSession(file)
  assert.equal(await readSession(file), null)
  await fs.rm(dir, { recursive: true, force: true })
})
