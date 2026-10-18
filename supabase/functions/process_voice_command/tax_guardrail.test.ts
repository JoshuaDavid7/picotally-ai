// Tests that the tax/legal/financial-advice guardrail is present in BOTH
// brains.
//
// Run locally with:
//   deno test --allow-read supabase/functions/process_voice_command/tax_guardrail.test.ts
//
// Why this test exists: the guardrail lives in prose inside two system
// prompts, and prose has no compiler. Both copies were well written and
// neither was pinned by anything — a prompt refactor could have deleted
// either one and CI would have stayed green. For a bookkeeping app that
// talks to sole proprietors about deductions, "the model quietly stopped
// saying it isn't a tax professional" is a liability event, not a
// regression.
//
// The two brains, and why both must carry it:
//   - supabase/functions/process_voice_command/index.ts — the Claude text
//     brain. Handles typed chat and transcribed voice notes.
//   - agent/main.py — the real-time "Hey Pico" voice agent on Gemini Live
//     (LiveKit → Fly.io). A different model, a different vendor, a
//     different file, and no shared prompt module between them.
//
// A user can ask "is my new van deductible?" down either path and must get
// the same boundary, so the invariant is symmetry, not merely presence.
//
// These assertions deliberately match SEMANTIC anchors rather than exact
// sentences. The wording of either prompt should stay free to improve;
// what must not change is that all four promises survive. Whitespace is
// normalised first because the Python prompt hard-wraps mid-sentence.

import { assert } from "https://deno.land/std@0.224.0/assert/mod.ts"

/** Collapse all runs of whitespace so hard-wrapped prose matches. */
function flatten(source: string): string {
  return source.replace(/\s+/g, " ")
}

async function loadPrompt(relativePath: string): Promise<string> {
  const url = new URL(relativePath, import.meta.url)
  return flatten(await Deno.readTextFile(url))
}

const BRAINS: ReadonlyArray<{ name: string; path: string }> = [
  { name: "text brain (Claude)", path: "./index.ts" },
  { name: "live voice brain (Gemini)", path: "../../../agent/main.py" },
]

/**
 * The four promises the guardrail makes. Each is expressed as a pair of
 * patterns so a partial deletion still fails: the role denial, the hard
 * prohibition on determinations, the referral to a professional, and the
 * say-it-once rule that keeps the caveat from degrading into disclaimer
 * spam (which is its own product failure — see the "not hedgy" rule).
 */
const PROMISES: ReadonlyArray<{ promise: string; patterns: RegExp[] }> = [
  {
    promise: "denies being a tax preparer / accountant / lawyer / advisor",
    patterns: [/not a tax preparer, accountant, lawyer, or financial advisor/i],
  },
  {
    promise: "refuses to rule on whether a specific expense is deductible",
    patterns: [/must not/i, /deductible/i],
  },
  {
    promise: "refers the user to a qualified professional before filing",
    patterns: [/before you file/i],
  },
  {
    promise: "says the caveat exactly once (no stacked disclaimers)",
    patterns: [/say it (once|a single time)/i],
  },
]

for (const brain of BRAINS) {
  for (const { promise, patterns } of PROMISES) {
    Deno.test(`tax guardrail — ${brain.name} ${promise}`, async () => {
      const prompt = await loadPrompt(brain.path)
      for (const pattern of patterns) {
        assert(
          pattern.test(prompt),
          `The ${brain.name} prompt (${brain.path}) no longer ${promise}.\n` +
            `Expected to match ${pattern}.\n\n` +
            `If you rewrote this guardrail, update the pattern here in the ` +
            `same change — do not delete the assertion. If you removed the ` +
            `guardrail, put it back: both brains must carry it, and the ` +
            `Terms (picotally.com/terms §8) promise users that they do.`,
        )
      }
    })
  }
}
