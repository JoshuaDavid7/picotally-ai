// Tests for the shared Gemini call helper.
//
// Run with:
//   deno test supabase/functions/_shared/gemini.test.ts
//
// The retry/backoff logic is the whole point of this module, so these pin it
// precisely: which statuses retry, which fail fast, how many attempts happen,
// and that truncated (MAX_TOKENS) output is flagged. fetch and sleep are
// injected so the tests are instant and deterministic — no network, no waits.

import { assertEquals, assertRejects, assert } from "https://deno.land/std@0.224.0/assert/mod.ts"

import { callGemini, extractGeminiText, GeminiHttpError } from "./gemini.ts"

/** Build a fetch stub that returns the given responses in order, one per call. */
function sequenceFetch(responses: Array<Response | Error>) {
  let calls = 0
  const impl = ((_url: string | URL | Request, _init?: RequestInit) => {
    const next = responses[calls] ?? responses[responses.length - 1]
    calls += 1
    if (next instanceof Error) return Promise.reject(next)
    return Promise.resolve(next)
  }) as unknown as typeof fetch
  return { impl, get calls() { return calls } }
}

function okJson(body: unknown): Response {
  return new Response(JSON.stringify(body), { status: 200 })
}

/** A sleep stub that records each requested delay and never actually waits. */
function recordingSleep() {
  const delays: number[] = []
  const impl = (ms: number) => {
    delays.push(ms)
    return Promise.resolve()
  }
  return { impl, delays }
}

const BASE = { apiKey: "k", model: "gemini-2.5-flash", body: { contents: [] } }

Deno.test("returns parsed body on first success — no retry, no sleep", async () => {
  const fetchStub = sequenceFetch([okJson({ candidates: [{ ok: true }] })])
  const sleep = recordingSleep()

  const result = await callGemini({
    ...BASE,
    fetchImpl: fetchStub.impl,
    sleepImpl: sleep.impl,
  }) as { candidates: unknown[] }

  assertEquals(result.candidates.length, 1)
  assertEquals(fetchStub.calls, 1)
  assertEquals(sleep.delays.length, 0)
})

Deno.test("retries a transient 503 then succeeds", async () => {
  const fetchStub = sequenceFetch([
    new Response("overloaded", { status: 503 }),
    okJson({ candidates: [{ ok: true }] }),
  ])
  const sleep = recordingSleep()

  const result = await callGemini({
    ...BASE,
    fetchImpl: fetchStub.impl,
    sleepImpl: sleep.impl,
  }) as { candidates: unknown[] }

  assertEquals(result.candidates.length, 1)
  assertEquals(fetchStub.calls, 2)
  assertEquals(sleep.delays.length, 1) // one backoff before the retry
})

Deno.test("retries a network error then succeeds", async () => {
  const fetchStub = sequenceFetch([
    new Error("connection reset"),
    okJson({ candidates: [] }),
  ])
  const sleep = recordingSleep()

  await callGemini({ ...BASE, fetchImpl: fetchStub.impl, sleepImpl: sleep.impl })

  assertEquals(fetchStub.calls, 2)
  assertEquals(sleep.delays.length, 1)
})

Deno.test("retries an abort/timeout then succeeds", async () => {
  const abort = new DOMException("the operation was aborted", "AbortError")
  const fetchStub = sequenceFetch([abort, okJson({ candidates: [] })])
  const sleep = recordingSleep()

  await callGemini({ ...BASE, fetchImpl: fetchStub.impl, sleepImpl: sleep.impl })

  assertEquals(fetchStub.calls, 2)
})

Deno.test("throws GeminiHttpError after exhausting retries on persistent 503", async () => {
  const fetchStub = sequenceFetch([new Response("down", { status: 503 })])
  const sleep = recordingSleep()

  const err = await assertRejects(
    () => callGemini({ ...BASE, maxRetries: 2, fetchImpl: fetchStub.impl, sleepImpl: sleep.impl }),
    GeminiHttpError,
  )
  assertEquals(err.status, 503)
  assertEquals(fetchStub.calls, 3) // first attempt + 2 retries
  assertEquals(sleep.delays.length, 2)
})

Deno.test("does NOT retry a permanent 400", async () => {
  const fetchStub = sequenceFetch([new Response("bad request", { status: 400 })])
  const sleep = recordingSleep()

  const err = await assertRejects(
    () => callGemini({ ...BASE, fetchImpl: fetchStub.impl, sleepImpl: sleep.impl }),
    GeminiHttpError,
  )
  assertEquals(err.status, 400)
  assertEquals(fetchStub.calls, 1) // no retry on a non-retryable status
  assertEquals(sleep.delays.length, 0)
})

Deno.test("does NOT retry a permanent 404 (unknown model)", async () => {
  const fetchStub = sequenceFetch([new Response("model not found", { status: 404 })])
  const err = await assertRejects(
    () => callGemini({ ...BASE, fetchImpl: fetchStub.impl, sleepImpl: recordingSleep().impl }),
    GeminiHttpError,
  )
  assertEquals(err.status, 404)
  assertEquals(fetchStub.calls, 1)
})

Deno.test("maxRetries: 0 makes exactly one attempt", async () => {
  const fetchStub = sequenceFetch([new Response("down", { status: 500 })])
  await assertRejects(
    () => callGemini({ ...BASE, maxRetries: 0, fetchImpl: fetchStub.impl, sleepImpl: recordingSleep().impl }),
    GeminiHttpError,
  )
  assertEquals(fetchStub.calls, 1)
})

Deno.test("backoff grows and is capped at ~2s", async () => {
  const fetchStub = sequenceFetch([new Response("down", { status: 503 })])
  const sleep = recordingSleep()
  await assertRejects(() =>
    callGemini({ ...BASE, maxRetries: 4, fetchImpl: fetchStub.impl, sleepImpl: sleep.impl }))
  // 4 retries → 4 backoffs. Base sequence 250, 500, 1000, 2000 (+<100 jitter).
  assertEquals(sleep.delays.length, 4)
  assert(sleep.delays[0] >= 250 && sleep.delays[0] < 350)
  assert(sleep.delays[1] >= 500 && sleep.delays[1] < 600)
  assert(sleep.delays[2] >= 1000 && sleep.delays[2] < 1100)
  assert(sleep.delays[3] >= 2000 && sleep.delays[3] < 2100) // capped
})

// ─── extractGeminiText ───────────────────────────────────────────────────────

Deno.test("extractGeminiText pulls text and marks not-truncated", () => {
  const out = extractGeminiText({
    candidates: [{ finishReason: "STOP", content: { parts: [{ text: '{"ok":true}' }] } }],
  })
  assertEquals(out.text, '{"ok":true}')
  assertEquals(out.finishReason, "STOP")
  assertEquals(out.truncated, false)
})

Deno.test("extractGeminiText flags MAX_TOKENS truncation", () => {
  const out = extractGeminiText({
    candidates: [{ finishReason: "MAX_TOKENS", content: { parts: [{ text: '{"clipped":' }] } }],
  })
  assertEquals(out.truncated, true)
})

Deno.test("extractGeminiText concatenates multiple parts", () => {
  const out = extractGeminiText({
    candidates: [{ content: { parts: [{ text: "a" }, { text: "b" }] } }],
  })
  assertEquals(out.text, "ab")
})

Deno.test("extractGeminiText tolerates a missing/empty candidate list", () => {
  assertEquals(extractGeminiText({}).text, "")
  assertEquals(extractGeminiText({ candidates: [] }).text, "")
  assertEquals(extractGeminiText(null).text, "")
})


Deno.test("endpoint option targets a different model method; default stays generateContent", async () => {
  const urls: string[] = []
  const capture = ((url: string | URL | Request) => {
    urls.push(String(url))
    return Promise.resolve(okJson({ ok: true }))
  }) as unknown as typeof fetch
  await callGemini({ ...BASE, fetchImpl: capture, sleepImpl: recordingSleep().impl })
  await callGemini({ ...BASE, endpoint: "batchEmbedContents", fetchImpl: capture, sleepImpl: recordingSleep().impl })
  assert(urls[0].includes("gemini-2.5-flash:generateContent?key=k"))
  assert(urls[1].includes("gemini-2.5-flash:batchEmbedContents?key=k"))
})
