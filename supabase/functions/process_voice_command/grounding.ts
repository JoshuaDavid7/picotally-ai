// Pure functions that make up the pre-model grounding guard for
// process_voice_command. Extracted so a Deno test file can exercise them
// without needing to spin up the full HTTP handler.
//
// The guard answers one question: "Does the transcript ask about a
// customer we don't have?" If so, short-circuit with a safe reply
// BEFORE calling the LLM. This prevents the model from being asked to
// invent data (balances, invoice dates, payment history) about people
// who aren't in the user's snapshot.
//
// See /supabase/functions/process_voice_command/README.md or index.ts
// call-site comments for the full policy rationale.

// ─── Text normalization helpers ──────────────────────────────────────────────

/**
 * Normalize a free-text string for fuzzy matching: lowercase, strip
 * punctuation, collapse whitespace, expand "&" to "and". Used by both
 * the grounding guard and the broader customer-name resolver.
 */
export function normalizeText(value: string): string {
  return value
    .trim()
    .toLowerCase()
    .replace(/&/g, ' and ')
    .replace(/[^a-z0-9\s]/g, ' ')
    .replace(/\s+/g, ' ')
    .trim()
}

/**
 * Classic Levenshtein (edit) distance. O(|a| * |b|) time, O(min(|a|,
 * |b|)) extra space. Used to allow minor typos in user-spoken names.
 */
export function levenshteinDistance(a: string, b: string): number {
  if (a === b) return 0
  if (!a) return b.length
  if (!b) return a.length

  const costs = Array.from({ length: b.length + 1 }, (_, i) => i)
  for (let i = 1; i <= a.length; i += 1) {
    let previous = costs[0]
    costs[0] = i
    for (let j = 1; j <= b.length; j += 1) {
      const current = costs[j]
      const substitutionCost = a[i - 1] === b[j - 1] ? 0 : 1
      costs[j] = Math.min(costs[j] + 1, costs[j - 1] + 1, previous + substitutionCost)
      previous = current
    }
  }
  return costs[b.length]
}

// ─── Grounding guard patterns ────────────────────────────────────────────────

/**
 * Regexes that recognize "data-access" questions — ones whose honest
 * answer requires looking up a specific customer's records. If we
 * detect one of these AND the named customer isn't in the context, we
 * refuse to hand the question off to the LLM.
 */
export const DATA_QUESTION_PATTERNS: RegExp[] = [
  /how much does ([\w .'-]+?) owe/i,
  /how much do(?:es)? ([\w .'-]+?) owe/i,
  /what does ([\w .'-]+?) owe/i,
  /what(?:'s| is) ([\w .'-]+?)(?:'s)? balance/i,
  /what(?:'s| is) the balance (?:for|on) ([\w .'-]+)/i,
  /has ([\w .'-]+?) paid/i,
  /did ([\w .'-]+?) pay/i,
  /when did i (?:last )?invoice ([\w .'-]+)/i,
  /show (?:me )?([\w .'-]+?)'s (?:invoice|balance|jobs|account)/i,
]

/**
 * Words that a regex capture group might pick up but that are NOT
 * customer names. Without this filter, "how much does the customer
 * owe?" would extract "the customer" and try to match it.
 */
export const GROUNDING_STOP_WORDS = new Set([
  'me', 'my', 'our', 'the', 'a', 'an', 'this', 'that', 'him', 'her', 'them',
  'customer', 'client', 'person', 'someone', 'anybody', 'anyone',
  'everyone', 'everybody', 'we', 'us', 'i',
  'unpaid', 'paid', 'all', 'any', 'some', 'no', 'open', 'closed', 'outstanding',
  'overdue', 'recent', 'latest', 'new', 'old', 'draft', 'sent', 'due',
  'pending', 'active', 'inactive', 'current', 'last', 'first',
])

// ─── Guard entry points ──────────────────────────────────────────────────────

/**
 * Extract a customer name from a data-access question. Returns null
 * if the transcript doesn't look like one of [DATA_QUESTION_PATTERNS]
 * or if the captured token is a stop-word / too short to be a name.
 *
 * Stop-word rejection is token-aware: "the customer" is rejected even
 * though the full phrase isn't in [GROUNDING_STOP_WORDS], because every
 * token in it is. Otherwise "how much does the customer owe?" would
 * leak through and falsely short-circuit against real customer names.
 */
export function extractCustomerNameFromDataQuestion(transcript: string): string | null {
  const trimmed = transcript.trim()
  for (const pattern of DATA_QUESTION_PATTERNS) {
    const match = trimmed.match(pattern)
    if (!match) continue
    const name = (match[1] || '').trim().replace(/[?.!,]+$/, '').trim()
    if (!name) continue
    if (name.length < 2) continue
    const tokens = name.toLowerCase().split(/\s+/).filter((t) => t.length > 0)
    if (tokens.length === 0) continue
    if (tokens.every((t) => GROUNDING_STOP_WORDS.has(t))) continue
    return name
  }
  return null
}

/**
 * True if [name] matches any of [knownNames] after normalization. The
 * match is generous on purpose — a false positive here means "let the
 * LLM handle it" (safe, the LLM sees the full customer list); a false
 * negative means "short-circuit with 'not found'" for a customer who
 * actually exists (bad UX). When in doubt, return true.
 *
 * Match levels, in order:
 *   1. Exact normalized match — "Mike Jones" vs "mike jones."
 *   2. Token-level exact match (either direction) — "Mike" vs "Mike Jones"
 *   3. Full-string fuzzy — "Johnsen" vs "Johnson" (threshold len/5)
 *   4. Token-level fuzzy — "Allise" vs "Alice Smith"
 *      Handles the common case where the user says just a first name
 *      (or last name) and speech-to-text garbles it. Threshold is
 *      floored at 2 edits so short names tolerate a typo/transposition.
 */
export function isKnownCustomer(name: string, knownNames: string[]): boolean {
  if (knownNames.length === 0) return false
  const normalized = normalizeText(name)
  if (!normalized) return false
  const nameTokens = normalized.split(' ').filter((t) => t.length > 0)
  for (const known of knownNames) {
    const knownNorm = normalizeText(known)
    if (!knownNorm) continue
    // Level 1: exact normalized match.
    if (knownNorm === normalized) return true
    const knownTokens = knownNorm.split(' ').filter((t) => t.length > 0)
    // Level 2: token-level exact match (either direction).
    if (knownTokens.some((tok) => tok === normalized)) return true
    if (nameTokens.some((tok) => tok === knownNorm)) return true
    // Level 3: full-string fuzzy.
    const distance = levenshteinDistance(knownNorm, normalized)
    const maxLen = Math.max(knownNorm.length, normalized.length)
    if (distance <= Math.max(1, Math.round(maxLen / 5))) return true
    // Level 4: token-level fuzzy. Skip tokens under 3 chars to avoid
    // trivial collisions on initials ("J" matches "K" at 1 edit).
    for (const nameTok of nameTokens) {
      if (nameTok.length < 3) continue
      for (const knownTok of knownTokens) {
        if (knownTok.length < 3) continue
        const tokDist = levenshteinDistance(knownTok, nameTok)
        const tokMax = Math.max(knownTok.length, nameTok.length)
        if (tokDist <= Math.max(2, Math.round(tokMax / 5))) return true
      }
    }
  }
  return false
}

/**
 * The short-circuit reply the user gets when the guard fires.
 * Structured like the action payload that would come back from the
 * LLM, so the client's routing code doesn't need to special-case it.
 */
export function buildUnknownCustomerResponse(name: string): {
  action: 'answer'
  params: Record<string, unknown>
  response: string
  suggestionChips: string[]
} {
  return {
    action: 'answer',
    params: {},
    response: `I don't see a customer named ${name} in your records. Want me to open your clients list, or add them as a new customer?`,
    suggestionChips: ['Open clients', 'Create an invoice', 'Show unpaid invoices'],
  }
}
