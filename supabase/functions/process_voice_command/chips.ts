// Picotally — suggestion-chip sanitizer.
//
// Suggestion chips appear under each AI response and the user taps them
// to send a follow-up question. The system prompt + tool schema both
// instruct the model to never suggest actions the app physically cannot
// perform (send texts/emails/SMS, schedule, export, post). In pre-launch
// testing the model still violated that rule in ~30% of overdue/payment
// turns, surfacing chips like "Send Mike a reminder" or "Email Steven".
//
// Suggesting an action and then having tap → silence is worse UX than
// suggesting nothing — the user assumes the feature exists, taps, and
// gets a confused chat turn explaining "actually I can't do that". So
// this sanitizer is the deterministic backstop: chips starting with a
// forbidden verb are dropped, with safe generic chips used to top up to
// a minimum of two so the UI never renders an empty chip row.
//
// FORBIDDEN_PREFIXES is a denylist by design: enumerating allowed verbs
// would suppress legitimate chips we hadn't anticipated. The denylist
// catches the specific verbs that imply outbound communication or
// scheduling — the only real footgun.

export const FORBIDDEN_CHIP_PREFIXES = [
  'send', 'message', 'text ', 'email', 'call ', 'remind', 'notify',
  'schedule', 'export', 'share', 'post', 'tweet', 'dm ', 'sms',
]

export const SAFE_DEFAULT_CHIPS = [
  'How is my business doing?',
  'Show unpaid invoices',
  'Who owes me the most?',
]

/**
 * Filter banned chips, then top up to ≥2 with safe generics so the UI
 * never renders empty. Returns up to 4 chips. Logs a warning for every
 * dropped chip so that drift can be monitored in production.
 */
export function sanitizeSuggestionChips(rawChips: unknown): string[] {
  const input = Array.isArray(rawChips) ? rawChips : []
  const accepted: string[] = []

  for (const item of input) {
    if (typeof item !== 'string') continue
    const chip = item.trim()
    if (!chip) continue
    const lower = chip.toLowerCase()
    const isForbidden = FORBIDDEN_CHIP_PREFIXES.some(
      (prefix) => lower === prefix.trim() || lower.startsWith(prefix)
    )
    if (isForbidden) {
      console.warn(`Sanitized forbidden suggestion chip: "${chip}"`)
      continue
    }
    accepted.push(chip)
    if (accepted.length === 4) break
  }

  // Top up to a minimum of 2 chips. Case-insensitive dedupe so a model
  // chip "Show unpaid invoices" doesn't render twice if a default also
  // says it.
  const seen = new Set(accepted.map((c) => c.toLowerCase()))
  for (const fallback of SAFE_DEFAULT_CHIPS) {
    if (accepted.length >= 2) break
    if (seen.has(fallback.toLowerCase())) continue
    accepted.push(fallback)
    seen.add(fallback.toLowerCase())
  }
  return accepted
}
