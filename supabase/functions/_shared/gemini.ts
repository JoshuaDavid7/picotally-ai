// Shared Gemini REST helper for Picotally edge functions.
//
// Why this exists:
//   `extract_receipt` and `process_job` both POST to the Gemini
//   generateContent endpoint. Each used to inline its own single-attempt
//   fetch with no retry, so any transient blip — a 429 rate-limit, a brief
//   503, a dropped connection, a response that grazed the abort timeout —
//   became a hard user-facing failure (a blank receipt, or a "couldn't build
//   your invoice" error) even though an immediate retry almost always
//   succeeds. That is the kind of "the AI randomly doesn't work" flakiness
//   beta users reported (#22).
//
//   This module centralises the three things both callers need:
//     1. callGemini()        — fetch with a per-attempt timeout and bounded
//                              exponential-backoff retry on transient errors
//     2. extractGeminiText() — pull the model's text out of the response and
//                              flag MAX_TOKENS truncation (which otherwise
//                              silently yields clipped, unparseable JSON)
//     3. GeminiHttpError     — a typed non-2xx error so each caller can choose
//                              fail-soft vs fail-hard per its own contract
//
//   `fetch` and `sleep` are injectable so the retry/backoff logic is unit
//   tested deterministically — no real network, no real waiting. See
//   gemini.test.ts.

export class GeminiHttpError extends Error {
  constructor(readonly status: number, readonly bodyText: string) {
    super(`Gemini API returned ${status}`)
    this.name = "GeminiHttpError"
  }
}

// 408 Request Timeout, 429 Too Many Requests, and the 5xx family are
// transient: the request was well-formed, so re-sending the identical body
// can succeed. Any other 4xx (400 bad request, 404 unknown model, 403 bad
// key) is permanent — retrying just burns time and quota, so we surface it
// immediately rather than after three pointless attempts.
const RETRYABLE_STATUS = new Set([408, 429, 500, 502, 503, 504])

export interface CallGeminiOptions {
  apiKey: string
  model: string
  /** Full request body for the endpoint: { contents, generationConfig, ... }. */
  body: unknown
  /**
   * Model method to call. Defaults to `generateContent`. The embeddings
   * helper passes `batchEmbedContents`; everything else about the call
   * (timeout, retry, backoff, error typing) is identical.
   */
  endpoint?: string
  /** Per-attempt abort timeout in ms. Default 30000. */
  timeoutMs?: number
  /** Retries AFTER the first attempt. Default 2 (→ 3 attempts total). */
  maxRetries?: number
  /** Injectable for tests. Defaults to global fetch. */
  fetchImpl?: typeof fetch
  /** Injectable for tests. Defaults to a real setTimeout-based sleep. */
  sleepImpl?: (ms: number) => Promise<void>
}

const defaultSleep = (ms: number) => new Promise<void>((resolve) => setTimeout(resolve, ms))

/**
 * POST a generateContent request to Gemini with a per-attempt timeout and
 * bounded exponential backoff. Resolves with the parsed JSON response.
 *
 * Throws:
 *   - `GeminiHttpError` for a non-2xx that is either non-retryable or has
 *     exhausted every retry, or
 *   - the underlying network / AbortError if every attempt failed to connect
 *     or timed out.
 */
export async function callGemini(opts: CallGeminiOptions): Promise<unknown> {
  const {
    apiKey,
    model,
    body,
    timeoutMs = 30000,
    maxRetries = 2,
    fetchImpl = fetch,
    sleepImpl = defaultSleep,
    endpoint = "generateContent",
  } = opts

  const url =
    `https://generativelanguage.googleapis.com/v1beta/models/${model}:${endpoint}?key=${apiKey}`
  const payload = JSON.stringify(body)

  let lastError: unknown
  for (let attempt = 0; attempt <= maxRetries; attempt++) {
    if (attempt > 0) {
      // 250ms, 500ms, 1s … capped at 2s, plus up to 100ms of jitter so
      // concurrent callers don't synchronise into a thundering-herd retry.
      const backoff =
        Math.min(2000, 250 * 2 ** (attempt - 1)) + Math.floor(Math.random() * 100)
      await sleepImpl(backoff)
    }

    const controller = new AbortController()
    const timer = setTimeout(() => controller.abort(), timeoutMs)
    let response: Response
    try {
      response = await fetchImpl(url, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: payload,
        signal: controller.signal,
      })
    } catch (err) {
      // Network failure or abort (timeout). Both are transient — note it and
      // retry until we run out of attempts, then surface the last one.
      clearTimeout(timer)
      lastError = err
      continue
    }
    clearTimeout(timer)

    if (response.ok) {
      return await response.json()
    }

    const bodyText = await response.text().catch(() => "")
    const httpError = new GeminiHttpError(response.status, bodyText)
    if (!RETRYABLE_STATUS.has(response.status)) {
      throw httpError // permanent — don't waste a retry
    }
    lastError = httpError // transient — loop and try again
  }

  throw lastError
}

export interface GeminiText {
  /** The first candidate's concatenated text parts ("" if none). */
  text: string
  /** Raw finishReason, e.g. "STOP", "MAX_TOKENS", "SAFETY", or null. */
  finishReason: string | null
  /** True when the model hit maxOutputTokens and the text is clipped. */
  truncated: boolean
}

/**
 * Pull the text out of a generateContent response and flag truncation.
 *
 * A "MAX_TOKENS" finishReason means the model ran into maxOutputTokens and the
 * returned text is almost certainly a clipped, unparseable JSON fragment.
 * Callers should treat truncated output as a failure rather than parse it.
 */
export function extractGeminiText(result: unknown): GeminiText {
  const candidates =
    (result as { candidates?: Array<Record<string, unknown>> } | null)?.candidates
  const first = candidates?.[0]
  const finishReason = (first?.finishReason as string | undefined) ?? null
  const content = first?.content as { parts?: Array<{ text?: string }> } | undefined
  const text = (content?.parts ?? []).map((p) => p?.text ?? "").join("")
  return {
    text,
    finishReason,
    truncated: finishReason === "MAX_TOKENS",
  }
}
