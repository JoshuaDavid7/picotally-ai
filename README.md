# PicoTally AI layer

The AI "office manager" inside [PicoTally](https://picotally.com), an AI bookkeeping and
invoicing app for tradespeople, live on iOS, Android, and the web. This repository is the
server-side AI code extracted verbatim from the private product repository at commit
`72062d0` on 2026-10-17. The Flutter client, billing, sync, and operations code are not included.
No secrets have ever been committed to the product repository; every vendor key lives in
server-side environment variables, and `gitleaks` over the full history reports zero leaks.

Built and run by one engineer, directing coding agents, since January 2026.

## What is here

| Path | What it is |
|---|---|
| `supabase/functions/process_voice_command/` | The text brain: a tool-calling agent loop on Claude with a schema-enforced exit, a pre-model grounding guard, a post-model arithmetic verifier, per-action sanitizers, and prompt caching. |
| `supabase/functions/voice_agent_tools/` | The bridge the voice worker calls. Builds the user's data snapshot from Postgres and dispatches to the **same** tool executors as the text brain, so voice and text always give the same answer. |
| `supabase/functions/_shared/` | Gemini client with retries, semantic vendor retrieval (embeddings in pgvector), per-user usage and cost metering, the auth helper, voice access checks. |
| `supabase/migrations/20261003120000_expense_embeddings.sql` | The pgvector table and similarity function behind semantic retrieval. Service-role only; deny-all row-level security. |
| `agent/main.py` | The voice worker on LiveKit Agents and Gemini Live: 26 tools, pre-dispatch into the room before the user speaks, an action-result loop so it only speaks success it received. |
| `supabase/functions/mcp_tools/` + `mcp/` | An MCP server (`picotally-mcp`) that lets Claude Desktop or Claude Code ask questions about a user's books through the same six query tools, via a user-authenticated, RLS-scoped endpoint. Read-only by construction. |
| `docs/ARCHITECTURE.md` | How the whole product is built, written from the code: client, sync, receipts, both AI brains, guards, security, billing, monitoring, deployment, and a glossary. |

## The design in five rules

1. **Tools do the math; the model only reads.** Every number a user sees was computed by
   deterministic code in integer cents. The model chooses tools and phrases results.
2. **One exit door.** The model finishes every turn by calling `respond`, whose schema fixes
   the action (an enum of nineteen), the parameters, the text, and two to four follow-up
   chips. The server validates the call before the phone sees it, and on the sixth round it
   is the only tool the model may call.
3. **Guards before and after the model.** A deterministic guard refuses questions about
   customers who do not exist before the model runs; a verifier re-adds any arithmetic in
   the reply after it runs; a sanitizer strips invented fields from actions.
4. **Honest about success.** For actions, the phone reports back whether the edit applied.
   The agent says "done" only after that confirmation, and says so when nothing came back.
5. **Voice and text share one tool layer.** The Python worker never re-implements a tool;
   it calls the same executors through an authenticated, session-bound endpoint.

## Numbers

- 6 query tools, 19 actions, 26 voice tools, agent loop capped at 6 rounds.
- Wake-word-to-ready voice latency cut from 5–9 s to about 3.4 s by pre-dispatching the
  agent, measured on a physical phone.
- Semantic vendor matching: 768-dimensional Gemini embeddings, cosine similarity floor 0.5,
  at most five vendors, re-run through the deterministic filter.
- The tests in this repository are the server-side suite for these modules; the full product
  carries about 2,100 automated tests on every change.

## Running the tests

```bash
deno test --allow-net --allow-env --allow-read supabase/functions/
cd mcp && npm ci && npm test
```

The edge functions target the Supabase Deno runtime; the voice worker is Python (see
`agent/DEPLOY.md`). Nothing here runs without the product's database and the vendor
keys, which is intentional: this repository exists to be read.

## Author

Joshua David John Rathinaraj. PhD, MIT 2024, on hybrid mechanistic and machine-learning
models ([PNAS Nexus 2026](https://doi.org/10.1093/pnasnexus/pgag346), code at
[NN-FIKH](https://github.com/JoshuaDavid7/NN-FIKH)).

## License

Source-available, all rights reserved. See [LICENSE](LICENSE).
