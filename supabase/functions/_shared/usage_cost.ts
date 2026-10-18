// Shared per-user COGS metering. Records RAW usage (tokens, seconds) into
// public.usage_events_monthly via the service-role `increment_usage_units`
// RPC; the dollar math lives entirely in the `cost_rates` price book + the
// cost rollup views (see migration 20260722000001_cost_ledger.sql).
//
// Contract: FIRE-AND-FORGET and NEVER THROWS. Cost metering is observability —
// it must never slow down, block, or break a product path. A dropped datapoint
// is always preferable to a failed AI command or a broken receipt scan.

import { createClient } from 'https://esm.sh/@supabase/supabase-js@2'
import { periodMonthKey } from './voice_access.ts'

/**
 * Pure: the subset of a usage map that is safe to persist — a positive, finite
 * count, rounded to a whole unit. Guards the ledger against garbage (NaN,
 * Infinity, negatives, zeros) that would otherwise corrupt cost totals or spam
 * no-op RPC calls. Exported for testing.
 */
export function costableEntries(
  units: Record<string, number>,
): Array<[string, number]> {
  return Object.entries(units)
    .map(([k, n]) =>
      [k, typeof n === 'number' && Number.isFinite(n) ? Math.round(n) : 0] as [
        string,
        number,
      ]
    )
    .filter(([, n]) => n > 0) // round FIRST, so 0.4 → 0 is dropped, not a no-op write
}

/**
 * Record usage for one service as a map of `{ unit_type: count }`
 * (e.g. `{ input_tokens: 5200, output_tokens: 180 }`). Zero/negative counts
 * are skipped. Returns immediately; the writes happen on a detached promise.
 *
 * @param userId  auth.users.id — required; a no-op if absent.
 * @param service e.g. 'claude', 'gemini', 'deepgram'. Must match cost_rates.
 * @param units   raw quantities keyed by unit_type (must match cost_rates).
 */
export function recordUsage(
  userId: string | undefined | null,
  service: string,
  units: Record<string, number>,
): void {
  if (!userId) return
  const entries = costableEntries(units)
  if (entries.length === 0) return

  // Detached: we deliberately do NOT await this from the caller.
  void (async () => {
    try {
      const url = Deno.env.get('SUPABASE_URL')
      const key = Deno.env.get('SUPABASE_SERVICE_ROLE_KEY')
      if (!url || !key) return
      const admin = createClient(url, key)
      const period = periodMonthKey(Date.now())
      await Promise.all(
        entries.map(([unitType, n]) =>
          admin.rpc('increment_usage_units', {
            p_user: userId,
            p_period: period,
            p_service: service,
            p_unit_type: unitType,
            p_units: n,
          }),
        ),
      )
    } catch (e) {
      // Non-fatal by design — log and move on.
      console.error('recordUsage failed (non-fatal):', e)
    }
  })()
}
