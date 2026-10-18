// Picotally — server-side arithmetic verifier.
//
// The post-architecture-fix story is: query tools do the math; the
// model just reads the result. This verifier is the BACKSTOP for the
// rare case where the model still hand-sums numbers (against the
// system prompt's instructions) and gets the arithmetic wrong.
//
// What it catches:
//   "$915 + $915 = $1,830"  — passes (correct)
//   "$915 + $915 = $1,860"  — caught (wrong by $30)
//   "$915 + $915 + $15 + $15 = $1,830"  — caught (wrong by $30)
//   "$1,234.56 + $789.10 = $2,023.66"  — passes
//
// What it doesn't catch (yet):
//   - Bullet-list totals with no "+" signs:
//       "• $915 (Vishy)\n• $915 (Taylor)\nTotal: $1,830"
//     No expression to validate. Add later if it becomes a problem.
//   - Subtotals across paragraphs.
//   - Multiplication ("12 × $50 = $600"). Add when needed.
//
// Behavior on a mismatch:
//   - Log a warning with the original expression and the correct sum.
//   - Replace the wrong total in-place with the correct one.
//   - Return the count of corrections so the caller can log it.
//
// Why correct in-place rather than re-prompt the model:
//   - Re-prompting doubles latency on a path that should be rare.
//   - The user gets a correct number even on the first turn.
//   - The wrongness is logged for monitoring; if it happens often,
//     fix the prompt or the tool, not the verifier.
//
// Why correct rather than warn user:
//   - We're confident the addends are right (the model just listed
//     them) and the sum is wrong. Returning a known-wrong sum
//     because it's "more honest" makes the product worse.
//   - The log is the audit trail.

export interface VerifyResult {
  /** The response text, with any wrong totals replaced by correct ones. */
  corrected: string
  /** Number of arithmetic mistakes caught and corrected. */
  mistakes: number
  /** Detail about each mistake — useful for logging / Sentry. */
  details: Array<{
    expression: string
    claimed: number
    actual: number
  }>
}

const NUMBER_PATTERN = /\$?[\d,]+(?:\.\d+)?/.source

/**
 * Match patterns like "$X + $Y [+ $Z ...] = $T" with at least 2 addends.
 * Allows: optional $ prefix, commas, decimals, optional **bold** wrap
 * around the total, optional "Total:" or "= " prefix on the result.
 *
 * The regex is intentionally permissive on whitespace and conservative
 * on what counts as a "sum expression" — we only want to fire on text
 * that LOOKS like the model is showing arithmetic.
 */
const SUM_EXPR = new RegExp(
  // Group 1: addends side, e.g. "$915 + $915 + $15"
  `(${NUMBER_PATTERN}(?:\\s*\\+\\s*${NUMBER_PATTERN}){1,})` +
  // Equals or "Total:"
  `\\s*=\\s*` +
  // Optional bold opener
  `(\\*\\*)?` +
  // Group 3: the claimed total
  `(${NUMBER_PATTERN})` +
  // Optional bold closer
  `(\\*\\*)?`,
  'g'
)

function parseNumber(raw: string): number | null {
  const cleaned = raw.replace(/[$,\s]/g, '')
  const n = parseFloat(cleaned)
  return Number.isFinite(n) ? n : null
}

/**
 * Format a number the way the rest of the response renders dollars.
 * Heuristic: if the input looked like an integer, render integer; if
 * it had cents, render two decimals. The verifier sees only the
 * claimed total here, so we infer formatting from that.
 */
function formatTotal(n: number, claimedRaw: string): string {
  const hadDollar = claimedRaw.includes('$')
  const hadCents = /\.\d{1,2}$/.test(claimedRaw.trim())
  const fixed = hadCents ? n.toFixed(2) : Number.isInteger(n) ? n.toString() : n.toFixed(2)
  // Re-introduce thousands separators when the claimed had them.
  const hadComma = claimedRaw.includes(',')
  const withCommas = hadComma
    ? fixed.replace(/\B(?=(\d{3})+(?!\d))/g, ',').replace(/(\d+,\d+)\.(\d{2})/, '$1.$2')
    : fixed
  return hadDollar ? `$${withCommas}` : withCommas
}

/**
 * Walk every "X + Y [+ Z] = T" pattern in [text]. For each, sum the
 * addends and compare to the claimed total. On mismatch (> 1¢), log
 * and replace the claimed total with the correct sum.
 *
 * Tolerance: 0.01 — accounts for rounding noise on materials with
 * fractional unit prices (e.g. 3 × $33.34 = $100.02 due to per-unit
 * rounding; not actually wrong arithmetic).
 */
export function verifyArithmetic(text: string): VerifyResult {
  if (!text || typeof text !== 'string') {
    return { corrected: text, mistakes: 0, details: [] }
  }

  const details: VerifyResult['details'] = []

  let corrected = text.replace(
    SUM_EXPR,
    (full, addendsExpr: string, _open: string | undefined, claimed: string, _close: string | undefined) => {
      const addends = addendsExpr.split('+').map(parseNumber)
      const claimedNum = parseNumber(claimed)
      if (claimedNum == null || addends.some((a) => a == null)) {
        return full
      }
      const actual = (addends as number[]).reduce((a, b) => a + b, 0)
      if (Math.abs(actual - claimedNum) <= 0.011) {
        return full
      }
      details.push({
        expression: full,
        claimed: claimedNum,
        actual,
      })
      // Reconstruct the full expression with the correct total. Preserve
      // the original addends side verbatim and bold wrappers if any.
      const corrected = formatTotal(actual, claimed)
      const open = _open ?? ''
      const close = _close ?? ''
      return `${addendsExpr.trim()} = ${open}${corrected}${close}`
    }
  )

  // ── Bulleted-list total verifier ────────────────────────────────────────
  //
  // Catches the Q19-class mistake where the model lists items as a
  // bullet/dash list with $amounts and claims a "Total: $X" that does not
  // match the sum. Two patterns:
  //   1. List → "Total: $X"            (model adds total at end)
  //   2. "totaling $X:" → list          (model leads with the answer)
  // Both run conservatively: if any bullet line in the matched range
  // lacks a $amount, abort, since we can't enumerate the addends with
  // confidence.
  const listResult = verifyListTotals(corrected)
  corrected = listResult.corrected
  details.push(...listResult.details)

  const headlineResult = verifyHeadlineTotalThenList(corrected)
  corrected = headlineResult.corrected
  details.push(...headlineResult.details)

  return {
    corrected,
    mistakes: details.length,
    details,
  }
}

/**
 * Match a contiguous block of "- $X ..." or "* $X ..." lines (markdown
 * bullets / dashes), followed within a few lines by a claim like
 * "Total: $T" / "Totaling $T" / "Total outstanding: $T" / "in total: $T".
 *
 * Group 1: the bulleted block (one or more lines).
 * Group 2: optional "**" before claimed total.
 * Group 3: the claimed total.
 * Group 4: optional "**" after claimed total.
 *
 * Each line must start with a "-" or "*" then optional "**...**" name then a
 * "$amount" — if any line in the block lacks a $amount, the regex won't
 * match and the verifier won't fire (conservative-by-design).
 */
const LIST_TOTAL_EXPR = new RegExp(
  // Group 1: one or more lines that start with "- " or "* " and contain a $amount.
  // We anchor each line on a newline (or BOL) and require the dash/star at the
  // very start of the line. The line content is non-greedy until end-of-line.
  `((?:^|\\n)(?:[-*]\\s+[^\\n]*\\$[\\d,]+(?:\\.\\d+)?[^\\n]*\\n?)+)` +
  // Up to 200 chars / a few lines of buffer for a "Total" / "Totaling" claim.
  `[\\s\\S]{0,200}?` +
  // The claim phrase. Match common forms.
  `(?:Total(?:\\s+outstanding|\\s+overdue|\\s+billed|\\s+owed)?:?|Totaling|in\\s+total:?)\\s*` +
  // Optional bold opener
  `(\\*\\*)?` +
  // Group 3: the claimed total
  `(\\$?[\\d,]+(?:\\.\\d+)?)` +
  // Optional bold closer
  `(\\*\\*)?`,
  'gi'
)

/**
 * Reverse pattern: a "totaling $T" / "total of $T" / "= $T" claim that
 * appears BEFORE the bullet list, e.g. "You have 6 unpaid totaling $X:
 * - Item $A …". This is the shape Sonnet emits when it leads with the
 * answer (the prompt asks it to). Same conservative gate as
 * LIST_TOTAL_EXPR — every bullet in the matched range must carry a
 * $amount, otherwise the function aborts.
 *
 * Group 1: optional "**" before claimed total.
 * Group 2: the claimed total.
 * Group 3: optional "**" after claimed total.
 * Group 4: the bulleted block.
 */
const TOTAL_THEN_LIST_EXPR = new RegExp(
  // Claim phrase. Slightly looser than LIST_TOTAL_EXPR — accepts "of"
  // ("a total of $X"), "is" ("the total is $X"), and bare "totaling".
  `(?:Total(?:ing)?(?:\\s+(?:of|is))?|in\\s+total)\\s*[:=]?\\s*` +
  // Optional bold opener
  `(\\*\\*)?` +
  // Group 2: the claimed total
  `(\\$?[\\d,]+(?:\\.\\d+)?)` +
  // Optional bold closer
  `(\\*\\*)?` +
  // Up to 200 chars / a few lines of buffer between the claim and list.
  `[\\s\\S]{0,200}?` +
  // Group 4: one or more bullet lines, each with a $amount.
  `((?:^|\\n)(?:[-*]\\s+[^\\n]*\\$[\\d,]+(?:\\.\\d+)?[^\\n]*\\n?){2,})`,
  'gi'
)

interface ListVerifyResult {
  corrected: string
  details: VerifyResult['details']
}

function verifyListTotals(text: string): ListVerifyResult {
  const details: VerifyResult['details'] = []
  // Each list line ($amount within the line). We grab the FIRST $amount
  // per line — additional $X tokens within a line (e.g. "due April 12,
  // $X overdue since") would introduce noise.
  const LINE_AMOUNT = /\$([\d,]+(?:\.\d+)?)/

  const corrected = text.replace(
    LIST_TOTAL_EXPR,
    (full, listBlock: string, openBold: string | undefined, claimed: string, closeBold: string | undefined) => {
      // Conservative-by-design: if ANY bullet line in the entire matched
      // span (listBlock + buffer up to the "Total" claim) lacks a $amount,
      // we abort. This protects against partial enumerations like
      // "- TestClient — \$115\n- Joel — \$100\n- (and 4 more)\nTotal: \$X"
      // where the model only fully listed some of the items. Re-summing
      // just the visible amounts would produce a confidently-wrong
      // correction. Better to leave the message untouched and fall back
      // to the prompt-level guidance ("quote tool totals verbatim").
      const allLines = full.split(/\n/).map((l) => l.trim())
      for (const line of allLines) {
        const isBullet = line.startsWith('-') || line.startsWith('*')
        if (!isBullet) continue
        if (!LINE_AMOUNT.test(line)) {
          return full
        }
      }

      // Extract one $amount per line.
      const lines = listBlock
        .split(/\n/)
        .map((l) => l.trim())
        .filter((l) => l.startsWith('-') || l.startsWith('*'))
      if (lines.length < 2) return full // Single-item "list" isn't worth verifying.

      const addends: number[] = []
      for (const line of lines) {
        const m = line.match(LINE_AMOUNT)
        if (!m) {
          // A bullet line without a $amount → can't enumerate confidently.
          return full
        }
        const value = parseNumber(m[1])
        if (value == null) return full
        addends.push(value)
      }

      const claimedNum = parseNumber(claimed)
      if (claimedNum == null) return full

      const actual = addends.reduce((a, b) => a + b, 0)
      if (Math.abs(actual - claimedNum) <= 0.011) {
        return full // Math is right; leave the message untouched.
      }

      details.push({
        expression: full.trim().replace(/\s+/g, ' ').slice(0, 200),
        claimed: claimedNum,
        actual,
      })

      // Replace just the claimed total within the matched block. We can't
      // easily reconstruct the full string, so use lastIndexOf on the
      // claimed substring inside `full` — there's only one match per block.
      const correctedTotal = formatTotal(actual, claimed)
      const open = openBold ?? ''
      const close = closeBold ?? ''
      const targetClaim = `${open}${claimed}${close}`
      const replaced = full.replace(targetClaim, `${open}${correctedTotal}${close}`)
      // Defensive: if the simple replace didn't work (unusual whitespace
      // around bold markers), fall back to swapping just the claimed
      // numeric token.
      if (replaced === full) {
        return full.replace(claimed, correctedTotal)
      }
      return replaced
    }
  )

  return { corrected, details }
}

/// Mirror of [verifyListTotals] for the case where the claim ("totaling
/// $X") appears BEFORE the bullet list. Same gate (every bullet must have
/// a $amount), same correction strategy (replace the claimed total with
/// the actual sum). The two functions can't be merged into one regex
/// without complicating the abort logic — we want to abort cleanly when
/// any bullet lacks a $amount, and that check operates on the full match.
function verifyHeadlineTotalThenList(text: string): ListVerifyResult {
  const details: VerifyResult['details'] = []
  const LINE_AMOUNT = /\$([\d,]+(?:\.\d+)?)/

  const corrected = text.replace(
    TOTAL_THEN_LIST_EXPR,
    (full, openBold: string | undefined, claimed: string, closeBold: string | undefined, listBlock: string) => {
      // Conservative-by-design: any bullet in the range must carry $.
      const allLines = full.split(/\n/).map((l) => l.trim())
      for (const line of allLines) {
        const isBullet = line.startsWith('-') || line.startsWith('*')
        if (!isBullet) continue
        if (!LINE_AMOUNT.test(line)) return full
      }

      const lines = listBlock
        .split(/\n/)
        .map((l) => l.trim())
        .filter((l) => l.startsWith('-') || l.startsWith('*'))
      if (lines.length < 2) return full

      const addends: number[] = []
      for (const line of lines) {
        const m = line.match(LINE_AMOUNT)
        if (!m) return full
        const value = parseNumber(m[1])
        if (value == null) return full
        addends.push(value)
      }

      const claimedNum = parseNumber(claimed)
      if (claimedNum == null) return full

      const actual = addends.reduce((a, b) => a + b, 0)
      if (Math.abs(actual - claimedNum) <= 0.011) return full

      details.push({
        expression: full.trim().replace(/\s+/g, ' ').slice(0, 200),
        claimed: claimedNum,
        actual,
      })

      const correctedTotal = formatTotal(actual, claimed)
      const open = openBold ?? ''
      const close = closeBold ?? ''
      const targetClaim = `${open}${claimed}${close}`
      const replaced = full.replace(targetClaim, `${open}${correctedTotal}${close}`)
      if (replaced === full) {
        return full.replace(claimed, correctedTotal)
      }
      return replaced
    }
  )

  return { corrected, details }
}
