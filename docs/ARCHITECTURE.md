# How PicoTally is built, step by step

Written 2026-10-14 from the actual code, for the PostEra technical interview. Every number below was measured in the repository today.

## 0. The one-paragraph version

PicoTally is an AI bookkeeping and invoicing app for tradespeople, live on iOS, Android and the web. A Flutter app keeps an encrypted local database on the device and syncs it to a Postgres database in the cloud. Thirty-seven small server functions handle everything that needs a secret or a vendor: receipt reading with Gemini, the chat agent on Claude, the voice agent on Gemini Live, payments on Stripe, app-store subscriptions, email, monitoring. Two "brains," text and voice, can do everything a user can do by hand, through one validated action vocabulary. Every number the AI reports is computed by code, and the system is held together by about 2,100 automated tests that run on every change.

## 1. Size and shape

| Part | Lines | What it is |
|---|---|---|
| Flutter app (`lib/`) | ~125,000 Dart | Screens, state, local database, sync, PDF, voice client |
| Flutter tests (`test/`) | ~24,000 Dart, 1,620 test cases in 161 files | Widget and unit tests |
| Server functions (`supabase/functions/`) | ~24,000 TypeScript, 37 functions, 510 test cases in 37 files | Deno edge functions |
| Voice agent (`agent/`) | ~1,400 Python | LiveKit worker with 26 tools |
| Database migrations | 108 SQL files, 136 row-level-security policies | Schema and permissions |

About 800 commits since March 2026. One engineer, directing coding agents.

## 2. The stack, layer by layer

**Client.** Flutter with Material 3, written in Dart. State management is Riverpod. The code follows a clean-architecture split: `data/` (repositories, services, database access), `domain/` (models and pure business rules), `presentation/` (providers, screens, widgets).

**Local database.** Drift, a typed SQL layer over SQLite, schema version 15. On phones the file is encrypted with AES-256 through SQLite3MultipleCiphers, and the app refuses to open an unencrypted database. On the web the same Drift schema runs on SQLite compiled to WebAssembly, so the web app is the same app as another synced device, not a thin console.

**Cloud.** Supabase, which is hosted Postgres plus authentication, storage buckets and a runtime for serverless functions. Row-level security is enabled on every table: a user's session can only ever read or write that user's rows, enforced by the database, not by application code. A lint script fails the build if a migration creates a table without a policy.

**Server functions.** Thirty-seven Deno (TypeScript) edge functions, grouped by job:

- AI: `extract_receipt`, `process_voice_command` (the text brain), `voice_agent_tools` (the voice brain's tool bridge), `summarize_job_title`, `transcribe_audio`, `voice_room_token`, `voice_room_close`, `voice_usage`.
- Money: `create_stripe_account_link`, `create_stripe_checkout`, `stripe_webhook`, `get_stripe_account_status`, `disconnect_stripe_account`, `create_billing_checkout`, `stripe_billing_webhook`, `billing_portal`, `validate_app_store_receipt`, `app_store_notifications_v2`, `validate_play_purchase`, `play_rtdn_webhook`.
- Email: `send_invoice_email`, `send_receipt_email` (through Resend).
- Account: `delete_account` (real GDPR erasure), `join_waitlist`.
- Operations: `service_health`, `ops_usage_report`, `owner_cost_report`, referral endpoints, and a few one-shot installers.

Shared modules hold the pieces several functions need: the auth helper, the Gemini client with retries, the embedding and semantic-vendor module, the usage-cost ledger, Stripe helpers, Play Billing helpers, the LiveKit token minter.

**AI vendors.** Claude Sonnet 4.6 is the text brain. Gemini 2.5 Flash reads receipts and does cheap classification. Gemini 3.1 Flash Live is the speech-to-speech model for voice. Gemini embedding model (768 dimensions) for semantic vendor matching. Claude Haiku 4.5 summarizes job titles. Deepgram transcribes voice memos.

**Voice infrastructure.** LiveKit for real-time audio rooms. The agent worker runs on Fly.io, one always-warm machine, because LiveKit's hosted agents scale to zero and cold-start in 10 to 20 seconds.

**Payments.** Stripe Connect (Standard accounts) lets a tradesperson get paid by their customers through a payment link or QR code on the invoice; PicoTally takes no fee on that. PicoTally's own subscription is sold three ways: Apple in-app purchase, Google Play Billing, and Stripe Checkout on the web.

**Monitoring.** Sentry for crashes, with a scrubber that strips emails, phone numbers, tokens and keys before anything leaves the device. TelemetryDeck for pseudonymous usage analytics. An hourly server health check that emails the owner. A per-user cost ledger for every AI call.

**Continuous integration.** GitHub Actions on every pull request: `flutter analyze`, the full Flutter test suite, the full Deno test suite. A separate security workflow runs gitleaks and detect-secrets, verifies no `.env` file is tracked, and checks that every migration has row-level security. Merges are blocked on any failure.

## 3. Offline-first and sync

The phone writes to its local database first, always, so the app works with no signal on a job site. A sync service then pushes unsynced rows to Supabase and pulls the server's copy back.

The pull side has one pure decision function, exhaustively unit-tested, applied to every table:

1. No local row: insert it.
2. Local row has unpushed edits: skip it, the next push reconciles.
3. Local row is synced and the remote is not newer: skip, avoid a wasted write. "Newer" is compared in whole milliseconds because Postgres stores microseconds and SQLite stores milliseconds, and without that tolerance every row re-synced every time.
4. Otherwise: the server's version wins.

Deletes use tombstones. A delete is queued, drained to the server before any pull, skipped on pull if still pending, and reconciled after. A synced row that is absent from the server's snapshot is deleted locally, which is what makes a deletion converge on every device.

## 4. The receipt pipeline

The feature users value most: photograph a receipt, get an expense with line items, categories and tax deductibility.

1. **Capture.** The camera takes the photo; the client compresses it to a JPEG under about a megabyte.
2. **OCR on the device.** Google ML Kit text recognition runs on the phone and produces the receipt's text. No image leaves the device in this mode.
3. **Server extraction.** The text, plus the user's category vocabulary, goes to `extract_receipt`, which calls Gemini 2.5 Flash with a fixed prompt and asks for structured JSON: vendor, date, line items with printed prices, subtotal, tax, total, the printed item count, and a category per item. The Gemini key lives only on the server; it used to be in the app bundle, which was the April 2026 security incident that drove the hardening.
4. **Defensive parsing.** Gemini's JSON is not strictly typed; the same field can be a number one call and a string the next. Each field is parsed with a fallback so one bad field never discards the whole receipt.
5. **Money rules, in code.** The printed total beats any arithmetic we do. The model is asked to associate a discount with its item and report both printed numbers; the subtraction happens in integer cents on our side. A guard that fires must name the gap and its direction ("items add up to $51.36 but the receipt says $45.46, $5.90 missing"). The review screen and the save use one plan object so the preview is exactly what gets saved.
6. **Storage.** The expense is saved with the structured result attached, so the line items can be shown and linked to invoice materials later.

A vision mode that sends the photo itself to Gemini exists behind a flag, off until it passes a battery of real receipts.

## 5. The text brain: the agent loop

Covered in detail in our conversation; the essentials:

- The phone sends the user's message plus a snapshot of their data (customers, jobs, last 60 expenses, settings).
- A deterministic grounding guard runs first: a question about a customer who does not exist is answered without calling the model.
- Claude receives the system prompt, the snapshot, six query tools and the message, and must reply with a tool call. Query tools (`query_expenses`, `query_jobs`, `query_payments`, `query_materials`, `query_customer_balance`, `lookup_material_prices`) are executed by code against the snapshot, in integer cents. The model exits only through the `respond` tool, whose schema fixes the action (an enum of nineteen), the parameters, the text and two to four suggestion chips.
- One tool per round, hard cap of six rounds; on the last round the model's only allowed tool is `respond`.
- A per-action sanitizer validates the parameters, strips invented fields, and downgrades to a plain answer if anything essential is wrong. Confirmation text for actions is rebuilt from the validated parameters. An arithmetic verifier re-adds any sums in the reply text.
- The long stable prefix (system prompt, tools, snapshot) carries prompt-cache breakpoints; a test keeps the snapshot block byte-stable so the cache keeps hitting. Volatile items like the current screen and time sit outside the cached block.
- Semantic vendor matching: when a vendor filter finds no literal match, the tool embeds the user's phrase, finds the nearest stored vendor names by cosine similarity, keeps those above 0.5 (at most five), reruns the deterministic filter, and names the matched vendor in its note.

## 6. The voice brain

- **Wake word.** "Hey Pico" is detected on the device with openWakeWord models running on ONNX, two of three recent frames above a threshold. A status line under the settings toggle always explains why the listener is or is not running, because silent failures were the original bug.
- **Pre-dispatch.** While the app is open and armed, the client mints a LiveKit room token with a prewarm flag; the server creates the room and the Fly worker joins and waits, with Gemini not yet started, costing only participant-minutes. When the user speaks, connection is near-instant. Total time from wake word to mic-ready went from 5 to 9 seconds to about 3.4, measured on a physical iPhone; the remainder is the client's room connect and iOS microphone activation.
- **The agent.** A Python worker on LiveKit Agents, using Gemini 3.1 Flash Live for speech in and speech out, with the instructions set at model construction because that model ignores mid-session instruction changes. Twenty-six function tools: the same six query tools as text, every write action, screen context, and session control.
- **One source of truth for tools.** Voice tools do not reimplement anything in Python. They call `voice_agent_tools`, a server function that builds the user's snapshot from the database, applies the same grounding guard, and dispatches to the same executor the text brain uses. Voice and text must give the same answer; that is a standing rule.
- **Security of that bridge.** The worker authenticates with a dedicated shared secret, not the database's master key, and every call must match a live session: the user id and room name pair must have been minted by the token service and not revoked, no older than 24 hours. A leaked secret alone cannot query arbitrary users.
- **Actions.** The agent never writes to the database. It publishes an action over the LiveKit data channel to the phone, which opens the pre-filled screen for the user to review and commit through the app's own tested write paths. For edits, the phone replies with an action result, applied true or false, and the agent only speaks success it received.
- **Caps.** Voice minutes are metered per user per month and capped by plan, because real-time speech is the most expensive thing the product does.

## 7. The single gate on the phone: AiActionCoordinator

Both brains end at one class on the client, the coordinator, which is the only place AI actions execute. It re-validates every parameter even though the server already did, because voice bypasses the server sanitizer. It resolves names against the user's live vocabulary, including custom categories. Safety tiers: tier-one edits apply instantly with an undo snackbar; deletes require an in-app confirmation; sending an invoice always requires the user's own tap on the preview screen. The AI never sends anything itself. Drafts may be deleted by AI; sent or paid documents are financial history and are refused. The confirmation reports what was actually recorded (if the AI heard $500 but the balance was $300, the user sees $300).

## 8. Money correctness beyond receipts

- **The spend rule** (`lib/domain/services/spend_rule.dart`): every logged expense counts in full on the day it was paid; an invoice material counts only while no receipt backs it; a receipt-backed material counts for nothing because its money is the receipt's. Linking never changes totals. Every reader of money on the dashboard, analytics and AI context goes through one function, so card and list agree by construction.
- **Document snapshots.** An invoice stores a snapshot of the business profile at render time; changing settings later never rewrites history.
- **Invoice engine.** One function parses line money; a golden-image matrix of rendered PDFs is checked by eye before the PDF service changes.
- **Tax guardrail.** The AI gives general information only, never tax or legal advice; a mutation-tested test pins the guardrail text.

## 9. Authentication and security

- Sign-in: email and password, native Sign in with Apple, native Google Sign-In, password reset, email change.
- Supabase moved to asymmetric ES256 tokens in 2026 and the gateway's built-in check only understood HS256, so every user-facing function calls `requireUser` at the top of its handler and validates the token server-side. A pre-commit hook blocks the old gateway flag so nobody can half-secure a function.
- Secrets live only in Supabase secrets, Fly secrets and the GitHub Actions store. The client bundle contains only the publishable key, which is designed to be public and is protected by row-level security.
- Server-to-server endpoints (Stripe webhooks, Apple and Google notifications, the voice bridge) authenticate with signatures or dedicated shared secrets compared in constant time.
- Account deletion is a real right-to-erasure: it deletes the auth user and sweeps every storage bucket in seconds.
- Local data is encrypted at rest; crash reports are scrubbed of personal data; analytics use a hashed identifier.

## 10. Subscriptions and entitlements

One ledger table, `subscription_receipts`, records every store transaction: platform, original transaction id, product, tier, status, expiry. A unique constraint on (platform, transaction id) means one store transaction can entitle at most one account, ever; a replay gets an explicit 409. Status uses a platform-neutral vocabulary (active, billing retry, grace period, expired, revoked) so Apple, Google and Stripe paths cannot disagree. The profile row keeps a summary that every client reads.

Webhook rules, learned the hard way: record the event only after processing succeeds; return a 5xx on transient failures so the vendor retries; derive the profile from all receipts, never from the latest event alone; always create the profile row if missing. Manual grants (for testers or goodwill) use dedicated columns and a trigger, because store validations overwrite the pro flag at every launch.

## 11. Observability and cost

- **Usage ledger.** Every AI call records raw units (tokens, seconds) per user into a monthly table through a fire-and-forget call that never throws, with prices in a separate `cost_rates` book, so dollar math is a query. The owner sees cost per user and per service in-app.
- **Service health.** An hourly job counts PicoTally's own consumption of each vendor (emails sent, database size, AI minutes) rather than asking vendors, and emails the owner when a line is crossed. A GitHub heartbeat fails if the hourly check goes quiet.
- **Crash and performance.** Sentry with local symbolication. One App Hang was traced to resizing a logo on the UI thread and moved off it.
- **Measured totals.** Recorded vendor spend for the whole product was under ten dollars a month at the time it was measured; the old $80 figure was an estimate, not a measurement.

## 12. Deployment

- iOS through App Store Connect (manual release), Android through Play Console (Play Billing Library 8 required), web through a single build script that exists because of three incidents: it fails on compile errors hidden by pipes, uses the web entrypoint without `dart:ffi`, verifies the SQLite WASM asset is served with the right content type, and stamps asset URLs so Cloudflare's cache does not hide a release for four hours.
- Server functions deploy with the Supabase CLI; merging to main does not deploy. Database changes are SQL migrations pushed with the CLI, each required to carry row-level security.
- The voice worker deploys with `fly deploy`.

## 13. Numbers to have ready

- 3 platforms, 1 engineer, about 800 commits since March 2026.
- 125k lines of Dart, 24k of TypeScript, 1.4k of Python.
- 1,620 Flutter tests, 510 server tests, both suites on every pull request.
- 37 server functions, 108 migrations, 136 row-level-security policies.
- 19 AI actions, 6 query tools, 26 voice tools, agent loop capped at 6 rounds.
- Voice ready time 5–9 s to about 3.4 s. Embedding similarity floor 0.5, top 5 vendors. Expense snapshot capped at 60 rows.

## 14. Twenty technical questions and short answers

1. **Why Flutter?** One codebase for iOS, Android and web; a tradesperson's phone first, desk second. The web build reuses the same database layer through WebAssembly.
2. **Why Drift plus Supabase rather than Supabase alone?** Job sites have no signal. Local-first with sync is the only honest architecture for that user.
3. **How do you resolve sync conflicts?** Server wins on strictly newer rows; local unpushed edits are never overwritten; deletes converge through tombstones. One pure function, unit-tested.
4. **Why edge functions for every AI call?** Secrets stay off the device. The Gemini key once shipped in the app bundle; that was the incident that set the rule.
5. **Why does the model never do arithmetic?** It is a reader, not a calculator. Tools compute in integer cents; a verifier backstops the rare hand-sum.
6. **How do you stop hallucinated data?** Three layers: block questions about nonexistent entities before the model, validate every structured reply after it, and confirm actions only when the phone reports success.
7. **Why is the final answer a tool call?** A single structured exit the server can detect, validate and force. The action enum means the model cannot invent capabilities.
8. **What happens if the model loops?** Six rounds, then the only allowed tool is `respond`. Logged when it happens; one to three rounds is normal.
9. **How do you keep prompt caching effective?** Stable prefix with cache breakpoints, volatile context outside it, and a test that the snapshot block is byte-stable.
10. **What is the retrieval component?** Vendor names embedded once into pgvector; a user's phrase embedded at query time; cosine nearest neighbours above 0.5; the deterministic filter reruns on the matched names. The embedding picks names, never numbers.
11. **Why one executor for voice and text?** Same answer by voice and text is non-negotiable for trust. Voice tools are thin schemas over the same server code.
12. **How do you secure a server-to-server bridge?** Dedicated shared secret compared in constant time, plus session binding: the user and room pair must have been minted and not revoked within 24 hours.
13. **Why Fly instead of LiveKit's hosted agents?** Hosted agents scale to zero and cold-start 10 to 20 seconds; one warm machine costs a fraction of the always-warm tier.
14. **Where did the voice latency go?** Pre-dispatch: the agent joins the room while the app is open, before the user speaks. Measured on a physical phone, not estimated.
15. **How do you test an LLM feature?** Pin the contract, not the prose: tool filters and date math, the guard, the verifier, the schema sanitizer, the cache stability. The model's exact wording is not asserted.
16. **How did you handle receipt money errors?** Printed total beats computed; the model associates, code subtracts; a guard names the gap and direction; preview and save share one plan. Each rule came from a real receipt that produced a wrong number.
17. **What is row-level security and why everywhere?** Postgres policies that make the database itself refuse cross-user reads and writes. A lint fails the build if a migration lacks one.
18. **How do subscriptions stay consistent across three stores?** One receipts ledger, a uniqueness rule per transaction, a neutral status vocabulary, and webhook rules that derive state from all receipts.
19. **How do you know what the product costs to run?** A per-call usage ledger with a separate price book, plus an hourly self-count of vendor consumption with alerts.
20. **What would you change?** Vision-mode receipts once validated; an offline evaluation set for the agent built from real anonymized commands; moving the web app onto the same encrypted-at-rest guarantees as mobile.

## 15. Glossary

- **Snapshot:** the user's data the phone sends with each AI request.
- **System prompt:** the standing rules the model reads first.
- **Tool definition:** a name, description and parameter schema the model can call.
- **Tool call / tool result:** the model's request to run a tool, and what code returns.
- **respond:** the single exit tool whose schema is the structured answer.
- **Action enum:** the nineteen allowed action names.
- **Sanitizer:** per-action server code that validates parameters and strips invented fields.
- **Grounding guard:** deterministic pre-model check for questions about nonexistent entities.
- **Verifier:** post-model re-addition of any arithmetic in the reply.
- **Action result:** the phone's report that an edit applied, which gates "done."
- **Coordinator:** the one class on the phone where AI actions execute.
- **Embedding:** a 768-number vector representing a text's meaning; cosine similarity measures closeness.
- **pgvector:** the Postgres extension that stores vectors and finds nearest neighbours.
- **Prompt cache breakpoint:** a marker that lets the stable prefix of a prompt be reused at a tenth of the price.
- **Row-level security:** database policies that restrict rows per user.
- **Tombstone:** a queued record of a deletion so it converges across devices.
- **Edge function:** a small serverless function on Supabase's Deno runtime.
- **LiveKit room / data channel:** the real-time audio session, and its side channel for structured messages.
- **Pre-dispatch:** the agent joining a room before the user speaks.
