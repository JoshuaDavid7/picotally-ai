# picotally-mcp

An MCP server that lets Claude Desktop, Claude Code, or any MCP client ask questions
about your PicoTally books: expenses, jobs, payments, materials, customer balances, and
price history. It exposes the same six read-only query tools the in-app assistant uses,
answered by the same server-side executors in integer cents, with the same guard against
questions about customers you do not have. Nothing can be changed through it.

## How it works

```
MCP client ──stdio──▶ picotally-mcp ──HTTPS (your session)──▶ mcp_tools edge function
                                                                  │
                                       RLS-scoped read of your books, deterministic tools
```

The server runs on your machine. It signs you in with your PicoTally account, stores the
session in `~/.picotally/mcp-session.json` (mode 0600), and forwards each tool call to the
`mcp_tools` endpoint with your token. The tool list is fetched from the server, so this
package never has to be updated when a tool's schema changes.

## Install

```bash
cd mcp
npm install
npm run build
npm link            # puts `picotally-mcp` on your PATH
```

## Sign in

Signing in needs the app's public client key. It is not a secret (every PicoTally client
ships with it), but this repository keeps all keys out of source, so set it once:

```bash
export PICOTALLY_PUBLISHABLE_KEY=sb_publishable_...   # from the app's .env.production
```

```bash
picotally-mcp login             # emails you a one-time code
picotally-mcp login --password  # or use your email password
picotally-mcp doctor            # confirms the session and lists the tools
```

Accounts that sign in with Google or Apple: set a password in the app first
(Settings, Change password), or use the emailed code.

## Connect a client

Claude Code:

```bash
claude mcp add picotally -- picotally-mcp
```

Claude Desktop (`claude_desktop_config.json`):

```json
{ "mcpServers": { "picotally": { "command": "picotally-mcp",
    "env": { "PICOTALLY_PUBLISHABLE_KEY": "sb_publishable_..." } } } }
```

(The key is only needed when a session has to be refreshed; `login` already stored one.)

Then ask: "What did I spend at the electrical supplier in June?" or "Does Garcia owe me
anything?" The answer comes from your books, not from the model's memory.

## Tools

`query_expenses`, `query_jobs`, `query_payments`, `query_materials`,
`query_customer_balance`, `lookup_material_prices`. Each takes the filters documented in
its schema (vendor, category, dates, client name, job, status).

## Security notes

- Read-only. There is no tool that creates, edits, sends, or deletes anything.
- Your token, your data: the endpoint reads your books through a client scoped by your own
  session, so the database's row-level security applies on top of the explicit filters.
- The Supabase URL in `src/config.ts` and the publishable key you set are public values
  that ship in every PicoTally client; they grant nothing by themselves.
- `picotally-mcp logout` deletes the stored session.

## Tests

```bash
npm test
```
