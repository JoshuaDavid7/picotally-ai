-- =============================================================================
-- Migration: Semantic vendor retrieval for expense questions (pgvector + Gemini embeddings)
-- Why: "What did I spend at the electrical supplier in June?" only worked when
--      those words were literally inside the stored vendor name. `query_expenses`
--      now falls back to meaning-based vendor matching over a per-user vector
--      index when the literal filter finds nothing.
--
--      The money rule is unchanged: the LLM never does the arithmetic. The
--      semantic step only resolves a phrase to vendor NAMES; the deterministic
--      executor (process_voice_command/tools.ts) still filters and sums.
--
-- Shape:
--   * public.expense_embeddings — one 768-dim Gemini embedding per expense,
--     written and read ONLY by edge functions with the service role. Deny-all
--     RLS for app users; nothing here is client-readable.
--   * public.match_expense_vendors(...) — nearest-neighbour lookup scoped to one
--     user_id, executable by service_role only.
--   * cost_rates row for embedding tokens so per-user COGS metering stays honest.
-- =============================================================================

create extension if not exists vector with schema extensions;

create table if not exists public.expense_embeddings (
  expense_id  text primary key references public.expenses(id) on delete cascade,
  user_id     uuid not null references auth.users(id) on delete cascade,
  model       text not null,
  dims        integer not null,
  source_text text not null,
  embedding   extensions.vector(768) not null,
  created_at  timestamptz not null default now(),
  updated_at  timestamptz not null default now()
);

-- RLS policies filter on user_id; keep that lookup indexed.
create index if not exists idx_expense_embeddings_user_id
  on public.expense_embeddings (user_id);

-- Cosine HNSW. Per-user sets are small today; the index keeps the RPC cheap as
-- the table grows without ever changing the query.
create index if not exists idx_expense_embeddings_hnsw
  on public.expense_embeddings using hnsw (embedding extensions.vector_cosine_ops);

-- ================================ RLS ===================================

alter table public.expense_embeddings enable row level security;

-- rls-exempt: server-only table. Written and read exclusively by edge functions
--             using the service role (which bypasses RLS). App users never touch
--             embeddings directly; the deny-all policy makes that explicit for
--             anon/authenticated, and the REVOKE below removes PostgREST's
--             default table grant so even a policy bug cannot expose rows.
create policy "expense_embeddings_deny_all"
  on public.expense_embeddings
  for all
  using (false)
  with check (false);

revoke all on table public.expense_embeddings from anon, authenticated;

-- ============================ Nearest-neighbour RPC ===========================
--
-- Returns the closest expenses to a query embedding for ONE user. Similarity is
-- cosine (1 - distance). Callers aggregate by vendor and apply a threshold; the
-- function itself never decides what "matches" — it just ranks.

create or replace function public.match_expense_vendors(
  p_user_id uuid,
  p_query   extensions.vector(768),
  p_limit   integer default 20
)
returns table (
  expense_id  text,
  vendor      text,
  description text,
  similarity  double precision
)
language sql
stable
security invoker
set search_path = public, extensions
as $$
  select e.id                                        as expense_id,
         coalesce(e.vendor, '')                      as vendor,
         e.description                               as description,
         1 - (emb.embedding operator(extensions.<=>) p_query) as similarity
  from public.expense_embeddings emb
  join public.expenses e on e.id = emb.expense_id
  where emb.user_id = p_user_id
    and e.user_id   = p_user_id
  order by emb.embedding operator(extensions.<=>) p_query
  limit greatest(1, least(coalesce(p_limit, 20), 100));
$$;

revoke execute on function public.match_expense_vendors(uuid, extensions.vector, integer)
  from public, anon, authenticated;
grant execute on function public.match_expense_vendors(uuid, extensions.vector, integer)
  to service_role;

-- ============================ COGS price book ================================
-- gemini-embedding-001 input is ~$0.15 per 1M tokens. The edge function
-- estimates tokens as ceil(chars / 4) because the embeddings endpoint does not
-- return usage metadata; the note records that so the ledger is not over-read.

insert into public.cost_rates (service, unit_type, usd_per_unit, note)
values ('gemini', 'embedding_tokens', 0.000000150000,
        'gemini-embedding-001 input ~$0.15/M; token count ESTIMATED as chars/4')
on conflict (service, unit_type) do nothing;
