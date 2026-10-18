#!/usr/bin/env node
// picotally-mcp: login | logout | doctor | serve (default)

import readline from 'node:readline/promises'
import { stdin, stdout, stderr } from 'node:process'
import { loadConfig } from './config.js'
import { SessionManager, clearSession, readSession } from './session.js'
import { PicoTallyApi } from './api.js'
import { serveStdio } from './server.js'

async function ask(question: string): Promise<string> {
  const rl = readline.createInterface({ input: stdin, output: stderr })
  try {
    return (await rl.question(question)).trim()
  } finally {
    rl.close()
  }
}

async function askHidden(question: string): Promise<string> {
  stderr.write(question)
  return new Promise((resolve) => {
    let buf = ''
    const onData = (chunk: Buffer) => {
      for (const ch of chunk.toString('utf8')) {
        if (ch === '\n' || ch === '\r') {
          stdin.setRawMode?.(false)
          stdin.pause()
          stdin.off('data', onData)
          stderr.write('\n')
          resolve(buf)
          return
        }
        if (ch === '\u0003') process.exit(130)
        if (ch === '\u007f' || ch === '\b') buf = buf.slice(0, -1)
        else buf += ch
      }
    }
    stdin.setRawMode?.(true)
    stdin.resume()
    stdin.on('data', onData)
  })
}

function usage(): void {
  stderr.write(
    [
      'Usage: picotally-mcp <command>',
      '  login [--password]   sign in with an emailed code (default) or your password',
      '  logout               forget the stored session',
      '  doctor               check the session and list the tools the server will expose',
      '  serve                run the MCP server over stdio (default; used by MCP clients)',
      '',
    ].join('\n'),
  )
}

async function main(): Promise<void> {
  const cfg = loadConfig()
  const sessions = new SessionManager(cfg)
  const [command = 'serve', ...flags] = process.argv.slice(2)

  switch (command) {
    case 'login': {
      const email = await ask('PicoTally account email: ')
      if (!email) throw new Error('An email address is required.')
      if (flags.includes('--password')) {
        const password = await askHidden('Password: ')
        await sessions.loginWithPassword(email, password)
      } else {
        await sessions.loginWithEmailCode(email, () => ask('Enter the code from the email: '))
      }
      stderr.write(`Signed in as ${email}. Session stored at ${cfg.sessionFile}\n`)
      return
    }
    case 'logout': {
      await clearSession(cfg.sessionFile)
      stderr.write('Session removed.\n')
      return
    }
    case 'doctor': {
      const stored = await readSession(cfg.sessionFile)
      stderr.write(stored ? `Session: ${stored.email ?? 'unknown email'} (${cfg.sessionFile})\n` : 'Session: not logged in\n')
      const api = new PicoTallyApi(cfg.functionsUrl, () => sessions.accessToken())
      const tools = await api.catalog()
      stderr.write(`Endpoint: ${cfg.functionsUrl}/mcp_tools\nTools (${tools.length}):\n`)
      for (const t of tools) stderr.write(`  - ${t.name}\n`)
      return
    }
    case 'serve': {
      const api = new PicoTallyApi(cfg.functionsUrl, () => sessions.accessToken())
      await serveStdio(api)
      return
    }
    default:
      usage()
      process.exit(2)
  }
}

main().catch((err: Error) => {
  stderr.write(`picotally-mcp: ${err.message}\n`)
  process.exit(1)
})
