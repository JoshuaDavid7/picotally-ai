"""
PicoTally real-time voice agent (LiveKit Agents + Gemini 3.1 Flash Live).

PHASE-A READY. Validated against livekit-agents 1.5.17 (imports, constructs,
tools register). Runnable for a voice test once agent/.env has LIVEKIT_* +
GOOGLE_API_KEY:
  - `python main.py console`  -> talk to it via your Mac mic/speaker (fastest)
  - `python main.py dev`      -> join the LiveKit room from a real iPhone, which
                                 tests iOS echo-cancellation + barge-in (#1 risk)
The data tools fail SOFT until the `voice_agent_tools` edge function exists
(Phase B), so the audio loop validates without it. See VOICE_PREMIUM_RUNBOOK.md.

What this gives you:
  - A LiveKit worker that joins the user's room and runs Gemini 3.1 Flash Live
    (native speech-to-speech: continuous listening, semantic turn detection,
    barge-in, affective voice — all built in).
  - Function tools that call BACK to Supabase so the deterministic business
    tools + grounding guard stay the single source of truth (in Deno), never
    re-implemented here.

Run (after provisioning):  python main.py dev
"""

import asyncio
import datetime
import json
import logging
import os

import httpx
from dotenv import load_dotenv
from google.genai import types as genai_types
from pydantic import BaseModel, Field
from livekit import rtc
from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    RunContext,
    WorkerOptions,
    cli,
    function_tool,
)
from livekit.plugins import google

# Load agent/.env regardless of the directory the worker is launched from.
load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

logger = logging.getLogger("picotally.agent")

SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
SUPABASE_ANON_KEY = os.environ.get("SUPABASE_ANON_KEY", "")

# How long a pre-dispatched agent waits in a prod (`voice-…`) room for the user
# to say "Hey Pico" before standing down. Must exceed the client's prewarm
# freshness window + refresh cadence (5 min / 4.5 min in
# pico_session_provider.dart) with slack, so a slot the client still considers
# fresh ALWAYS has its agent waiting. Idle cost while waiting: LiveKit
# participant-minutes only (~$0.0005/min); Gemini is NOT started until a
# participant actually joins.
PARTICIPANT_WAIT = 7 * 60.0
# The agent calls this Deno edge function to run the EXISTING deterministic
# query tools + grounding guard (single source of truth). Phase B builds it.
TOOLS_ENDPOINT = f"{SUPABASE_URL}/functions/v1/voice_agent_tools"

# Voice persona. Mirrors the text assistant's system prompt intent, tuned for
# SPOKEN delivery: short, conversational, never a wall of text; ask a
# clarifying question when unsure; confirm before any create/send action.
INSTRUCTIONS = """\
You are PicoTally's voice assistant for a tradesperson, talking hands-free.
Speak like a sharp, calm office manager who knows their books cold.

Voice rules:
- On the user's FIRST utterance of a session, lead with a 1-3 word warm
  acknowledgment ("Hey!" / "Sure thing —") before answering, so they know
  you're live. Skip it for everything after.
- Keep replies SHORT and conversational — a sentence or two. The user can ask
  for more. Never read a long list unless asked.
- If the user interrupts you, stop immediately and listen.
- USE the details the user already gave — never re-ask for something they just
  said. If they named the customer, use it. If they said "invoice" or "quote",
  use that; otherwise DEFAULT to an invoice. Only ask when a truly REQUIRED
  detail is missing (e.g. no customer named at all) or you genuinely couldn't
  make out a word.
- NEVER invent a customer, balance, amount, or date — only use what the tools
  return. (A server-side guard enforces this too.)
- Do NOT ask "are you sure?" / "want me to open it?" / "is this an invoice or a
  quote?". The moment you have the customer + the gist, just DO it: call the
  tool and say what you're doing in ONE line — "Opening an invoice for Daniel,
  3 hours labor and 50 PEX pipe at $5." The app opens a draft the user reviews
  and sends, so THAT screen is the confirmation; you never need a spoken "yes".
- Use the tools for any number, total, balance, or price — never do the math in
  your head.
- CONTEXT FIRST, always. Before acting on anything that could mean "the thing
  in front of me" — "this", "it", "him" right after a draft opened, "the
  markup", "the tax" — call get_current_screen. If the user is on an editable
  invoice and the request is a field on that invoice, edit IN PLACE with
  edit_current_invoice. Only reach for create/navigate/query when the request
  clearly is not about the open screen. Say which record you changed:
  "Took the tax off this invoice" beats "done".
- INTENT over phrasing. "Don't tax him", "no tax", "make it tax exempt",
  "remove the sales tax", "turn tax off" are ONE intent → set_tax
  enabled=false. Same for every op: map what they MEAN, not the exact words.
  Tax/markup on the OPEN invoice = edit_current_invoice, NEVER
  update_settings (that changes their global defaults — a different request).
- NEVER claim what you didn't see. Edit tools return the app's verdict:
  if the tool result says the change didn't take, SAY SO and ask what's on
  the screen — do not smooth it over. Success language only when the tool
  result confirms it. If the user asked for several changes in one breath
  ("$400, no tax, 25% markup"), make ALL the calls — one per change — and
  confirm each; never silently drop the second half of a sentence.
- Date questions: compute date_from/date_to from TODAY'S DATE (given below) and
  pass them to the tool. "Last 2 months" = from the 1st of (today minus 2
  months) through today. "This month" = the 1st through today. NEVER say a tool
  can't filter by date — the query tools all take date_from/date_to.
- Spend questions ("how much did I spend on/at X") = query_expenses FIRST:
  that's the expense ledger, the money that actually left the user's wallet
  (receipts + logged purchases, including things never billed to a job). Use
  query_materials only for job costing ("what did materials cost on my sent
  invoices / the Watson job"), or offer its number in one short clause when
  it tells a usefully different story than the ledger. Date bounds always
  when the user gave a period.
- "Who owes me money?", "who's late / overdue / past due", "what's
  outstanding", "who owes the most" = query_jobs with status 'late' (sent
  invoices that still have a balance AND whose due date has passed, sorted
  most-overdue first, each with a daysLate field). This is the SAME definition
  the text assistant uses, so both give the same answer — do NOT use 'unpaid'
  for these. Report the tool's `total` and `count` VERBATIM; never add up the
  names yourself. Name EVERY customer the tool returns, biggest first — don't
  stop after two or three. If the tool's note says some sent invoices have no
  due date, pass that caveat through ("a couple have no due date set, so I
  can't tell if those are late — want to open them?"). An UNSENT DRAFT is NOT
  money owed — it hasn't been billed yet; the tools already exclude drafts, so
  never count a customer's drafts as what they owe. If they have big drafts,
  mention them SEPARATELY as not-yet-sent pipeline, never as a debt.
- When the user points at something on screen instead of naming it — "this",
  "it", "the 3 pipes", "this invoice", "change it to two" — call
  get_current_screen FIRST to see what they mean, then act on that.
- If an invoice/quote draft is open and the user wants to change it ("make the
  pipes 2", "add a valve", "make it 4 hours", "change the rate to $95 an hour"),
  use edit_current_invoice to edit THAT draft in place. Never start a new draft
  for an edit. Keep the labor RATE ("an hour"), labor HOURS, and a material's
  PRICE distinct — they are three different numbers.

Tax, legal & money advice — a hard limit:
- You are a bookkeeping helper, NOT a tax preparer, accountant, lawyer, or
  financial advisor, and you must never imply you are one. You CAN read back the
  user's OWN recorded numbers ("you've logged about four thousand in deductible
  expenses this year") and explain in plain words how something generally works.
- You must NOT tell them what they OWE, what they SHOULD claim or elect, or
  whether a specific expense IS deductible — that's a professional call you
  can't be responsible for. If they ask, give a short general answer and add,
  once and plainly, "I'm not a tax pro though — double-check with one before you
  file." Say it a single time; keep it light and never repeat it.

Ending the conversation:
- YOU are responsible for ending the session — there is no button the user
  wants to press. Err on the side of bowing out; they can always say
  "Hey Pico" again.
- Call end_session (with a brief, warm one-line goodbye) when the user signals
  they're done in ANY of these ways:
  * explicit: "that's it", "that's all", "thanks, that's everything",
    "nothing else", "I'm good", "we're done", "bye", "see you", "later".
  * a bare acknowledgment AFTER you've completed what they asked — "okay",
    "ok thanks", "cool", "great", "perfect", "got it", "sounds good", "alright"
    with no new request attached. Task done + short thanks = done. End.
  * they answered your final question and there is plainly nothing left to do.
- After you finish a task, do NOT ask "anything else?" more than once. If the
  reply to it is any acknowledgment or thanks, end the session.
- Do not end mid-task, on a thinking pause, or when they're clearly composing
  a longer request. Pauses are fine; goodbyes and bare thanks are endings.
"""


def build_instructions() -> str:
    """INSTRUCTIONS + today's date. Computed per session (each entrypoint builds
    a fresh AgentSession) so 'last 2 months' style date math is always right —
    the same way the text brain gets `now` injected into its prompt. Must be
    passed to RealtimeModel at construction (the model ignores mid-session
    instruction updates)."""
    today = datetime.date.today()
    return (
        f"{INSTRUCTIONS}\n"
        f"TODAY'S DATE is {today.isoformat()} ({today.strftime('%A, %B %d, %Y')}). "
        "Use it for all relative-date math (last month, this year, two weeks ago)."
    )


class InvoiceMaterial(BaseModel):
    """One material line item the user named for an invoice or quote."""

    item: str = Field(
        description="material name ONLY, e.g. 'PEX pipe' or 'corn' — no quantity or price in this field",
    )
    quantity: float = Field(
        default=1,
        description="how many units the user said, e.g. 50 for '50 PEX pipes' or '50kg of corn'",
    )
    unit_price: float = Field(
        default=0,
        description=(
            "price PER UNIT in dollars if the user stated one — e.g. 5 for "
            "'$5 each', 10 for '$10 per kg'. Use 0 if the user did NOT say a "
            "price; the app then auto-fills it from their past prices."
        ),
    )


class PicoTallyAgent(Agent):
    def __init__(self, user_id: str, room: rtc.Room | None = None) -> None:
        super().__init__(instructions=build_instructions())
        # The participant identity (= Supabase user id) from their room token.
        # The tools endpoint trusts this (the worker is trusted infra) and runs
        # the deterministic tools scoped to this user via the service role.
        self._user_id = user_id
        # The room — so action tools can hand a finished invoice/payment to the
        # app over the data channel (the app commits it, not us).
        self._room = room
        # Latest page-context the app pushed (topic picotally.context) — what the
        # user is looking at. Lets get_current_screen resolve "this / it / the 3
        # pipes". Best-effort; never load-bearing for correctness.
        self._screen_context: dict | None = None
        if room is not None:
            room.on("data_received", self._on_data)

    async def _call_tool(self, name: str, args: dict) -> str:
        """Run a deterministic business tool server-side and return its JSON.

        Fails SOFT: in Phase A the voice_agent_tools endpoint isn't deployed yet,
        so on any error we return a JSON error the model can speak gracefully
        ("I can't look that up right now") instead of crashing the live session.
        """
        try:
            async with httpx.AsyncClient(timeout=20) as client:
                resp = await client.post(
                    TOOLS_ENDPOINT,
                    json={
                        "userId": self._user_id,
                        "tool": name,
                        "input": args,
                        # 1.0.1 binding: the server refuses tool calls that
                        # are not inside a live session of this exact user.
                        "roomName": self._room.name if self._room else "",
                    },
                    headers={
                        # Worker→edge auth: a dedicated shared secret (not the
                        # service-role key, whose representation can differ
                        # between the worker and the edge runtime).
                        "Authorization": f"Bearer {os.environ.get('VOICE_AGENT_SECRET', '')}",
                        "apikey": SUPABASE_ANON_KEY,
                    },
                )
                resp.raise_for_status()
                return resp.text  # JSON the model reads back
        except Exception as e:
            return json.dumps({"error": "lookup unavailable", "detail": str(e)})

    async def _publish_action(self, action: str, params: dict, spoken: str) -> str:
        """Hand a finished action (create invoice / record payment) to the app
        over the room data channel. The app opens the pre-filled screen and the
        user gives the final nod — the actual write happens in the app's tested
        paths, never here. The spoken read-back + 'yes' already happened in
        conversation (the instructions require it before these tools fire).

        HONESTY LOOP (2026-07-25 audit): for in-place edits the app reports
        back whether the change ACTUALLY applied (picotally.action_result).
        We wait briefly for that verdict and speak from it — success is never
        claimed on hope. Older app builds don't ack; the timeout falls back to
        the optimistic line so nothing regresses.
        """
        if self._room is None:
            return "I couldn't open that here — try it from the app."
        clean = {k: v for k, v in params.items() if v is not None}
        wants_ack = action == "edit_current_invoice"
        if wants_ack:
            self._pending_ack = asyncio.get_running_loop().create_future()
        try:
            await self._room.local_participant.publish_data(
                json.dumps({"action": action, "params": clean, "response": spoken}),
                reliable=True,
                topic="picotally.action",
            )
        except Exception as e:
            self._pending_ack = None
            return json.dumps({"error": "could not open", "detail": str(e)})
        if not wants_ack:
            return spoken
        try:
            result = await asyncio.wait_for(self._pending_ack, timeout=2.5)
        except asyncio.TimeoutError:
            return spoken  # pre-ack app build — behave exactly as before
        finally:
            self._pending_ack = None
        if result.get("applied") is True:
            return spoken
        return (
            "Hmm — that change didn't take on your screen. It may not apply "
            "here; tell me what you see and I'll try another way."
        )

    def _on_data(self, data: "rtc.DataPacket") -> None:
        """Store the latest screen context the app pushes (topic
        picotally.context). Fails soft — bad/edge packets just leave the last
        good context in place.
        """
        try:
            if data.topic == "picotally.context":
                self._screen_context = json.loads(bytes(data.data).decode("utf-8"))
            elif data.topic == "picotally.action_result":
                payload = json.loads(bytes(data.data).decode("utf-8"))
                fut = getattr(self, "_pending_ack", None)
                if fut is not None and not fut.done():
                    fut.set_result(payload)
        except Exception:
            pass

    async def _publish_control(self, action: str) -> None:
        """Send a lightweight control signal to the app (topic picotally.control).
        Today: 'end_session' — the user gave a closing cue, so the app should
        tear the session down once the agent's goodbye finishes playing. Soft-
        fails: if it doesn't arrive, the app's silence timeout still closes out,
        so there's never a runaway session.
        """
        if self._room is None:
            return
        try:
            await self._room.local_participant.publish_data(
                json.dumps({"action": action}),
                reliable=True,
                topic="picotally.control",
            )
        except Exception:
            pass

    # ── Tools (mirror the text assistant's query tools) ──────────────────────
    # NOTE: Phase B fills out the full set + finalizes arg names to match
    # supabase/functions/process_voice_command/tools.ts.

    @function_tool
    async def get_current_screen(self, ctx: RunContext) -> str:
        """What the user is looking at in the app RIGHT NOW. Call this to resolve
        anything that points at the screen instead of naming it — "this", "it",
        "the 3 pipes", "this invoice", "change it to…". Returns the screen, a
        one-line summary, and (if it's editable) its kind + id + detail.
        """
        if self._screen_context is None:
            return json.dumps(
                {"screen": "unknown", "note": "nothing on screen yet"}
            )
        return json.dumps(self._screen_context)

    @function_tool
    async def query_customer_balance(self, ctx: RunContext, client_name: str) -> str:
        """What a customer owes — outstanding balance, total billed and paid.
        Use whenever the user asks what someone owes or about a balance.

        Args:
            client_name: the customer's name, as the user said it.
        """
        return await self._call_tool("query_customer_balance", {"clientName": client_name})

    @function_tool
    async def query_materials(
        self,
        ctx: RunContext,
        keyword: str | None = None,
        client_name: str | None = None,
        date_from: str | None = None,
        date_to: str | None = None,
        invoice_status: str | None = None,
        job_id: str | None = None,
    ) -> str:
        """Sum/list material spend by keyword/date/client (pre-markup supplier
        cost — what the user really paid). Use for ANY material-spend question:
        'how much did I spend on pipes', '...in the last 2 months', 'materials
        on Sam's job', 'total copper this year'. For time periods, compute
        date_from/date_to from today's date.

        Args:
            keyword: material stem to match (e.g. 'pipe' matches Pipe/PVC pipe),
                or omit for all materials.
            client_name: limit to one client's jobs, or omit.
            date_from: ISO YYYY-MM-DD inclusive lower bound on the invoice/job
                date, or omit. E.g. 'last 2 months' → first day of (today − 2
                months).
            date_to: ISO YYYY-MM-DD inclusive upper bound, or omit (today for
                'so far' questions).
            invoice_status: 'sent_or_paid' (DEFAULT — real committed spend),
                'draft_only', or 'any'. Use sent_or_paid unless the user asks
                about drafts.
            job_id: limit to one specific job by id (e.g. from
                get_current_screen), or omit.
        """
        return await self._call_tool(
            "query_materials",
            {
                "keyword": keyword,
                "clientName": client_name,
                "dateFrom": date_from,
                "dateTo": date_to,
                "invoiceStatus": invoice_status or "sent_or_paid",
                "jobId": job_id,
            },
        )

    @function_tool
    async def lookup_material_prices(self, ctx: RunContext, items: list[str]) -> str:
        """The user's OWN past prices for named materials, to pre-fill a quote.
        Call when they name materials WITHOUT a price. Returns a suggested
        unit price per item (sell vs cost basis) — ASK if there's no history.

        Args:
            items: the material names the user said, e.g. ['PEX pipe', 'valve'].
        """
        return await self._call_tool("lookup_material_prices", {"items": items})

    @function_tool
    async def query_expenses(
        self,
        ctx: RunContext,
        keyword: str | None = None,
        vendor: str | None = None,
        category: str | None = None,
        client_name: str | None = None,
        date_from: str | None = None,
        date_to: str | None = None,
        tax_deductible: bool | None = None,
    ) -> str:
        """Sum/list logged expenses (receipts + manual entries) by vendor,
        category, or keyword. Use for 'how much did I spend at Home Depot',
        'total fuel this quarter', 'my deductible expenses'. For time periods,
        compute date_from/date_to from today's date. If the vendor has no
        literal match the tool retries by MEANING ('the electrical supplier')
        and the result's note names the vendor it used — say that vendor name
        to the user. Pass the user's phrase as spoken when they do not give an
        exact vendor name.

        Args:
            keyword: substring of the expense description, or omit.
            vendor: vendor name, e.g. 'Home Depot', or omit.
            category: one of materials/labor/fuel/tools/supplies/insurance/
                utilities/marketing/fees/meals/other, or omit.
            client_name: limit to one client, or omit.
            date_from: ISO YYYY-MM-DD lower bound, or omit.
            date_to: ISO YYYY-MM-DD upper bound, or omit.
            tax_deductible: true for only deductible expenses, false for only
                non-deductible, omit for all.
        """
        return await self._call_tool(
            "query_expenses",
            {
                "keyword": keyword,
                "vendor": vendor,
                "category": category,
                "clientName": client_name,
                "dateFrom": date_from,
                "dateTo": date_to,
                "taxDeductible": tax_deductible,
            },
        )

    @function_tool
    async def query_jobs(
        self,
        ctx: RunContext,
        client_name: str | None = None,
        status: str | None = None,
        kind: str | None = None,
        date_from: str | None = None,
        date_to: str | None = None,
    ) -> str:
        """List/aggregate invoices and quotes by client, status, or type. Use for
        'how many unpaid invoices', 'what's outstanding', 'show Dave's jobs'.

        Args:
            client_name: limit to one client, or omit.
            status: draft/sent/unpaid/late/paid/partially_paid/any, or omit.
            kind: invoice/quote/any, or omit.
            date_from: ISO YYYY-MM-DD lower bound, or omit.
            date_to: ISO YYYY-MM-DD upper bound, or omit.
        """
        return await self._call_tool(
            "query_jobs",
            {
                "clientName": client_name,
                "status": status,
                "type": kind,
                "dateFrom": date_from,
                "dateTo": date_to,
            },
        )

    @function_tool
    async def query_payments(
        self,
        ctx: RunContext,
        client_name: str | None = None,
        method: str | None = None,
        date_from: str | None = None,
        date_to: str | None = None,
    ) -> str:
        """Sum payments received by client, method, or date. Use for 'how much
        did I collect this month', 'what did Dave pay me', 'cash payments'.

        Args:
            client_name: limit to one client, or omit.
            method: cash/card/check/bank_transfer/any, or omit.
            date_from: ISO YYYY-MM-DD lower bound, or omit.
            date_to: ISO YYYY-MM-DD upper bound, or omit.
        """
        return await self._call_tool(
            "query_payments",
            {
                "clientName": client_name,
                "method": method,
                "dateFrom": date_from,
                "dateTo": date_to,
            },
        )

    # ── Actions (voice fills the draft; the APP commits after the user reviews) ─
    # Call these as soon as the customer + the gist are known — no spoken "yes"
    # needed. They hand the action to the app, which opens the pre-filled screen
    # for the user to review and commit via its own tested write paths. That
    # screen IS the confirmation — nothing is sent or saved here.

    @function_tool
    async def create_invoice(
        self,
        ctx: RunContext,
        client_name: str,
        kind: str = "invoice",
        labor_hours: float = 1.0,
        labor_type: str = "profile",
        labor_rate: float | None = None,
        labor_amount: float | None = None,
        description: str | None = None,
        materials: list[InvoiceMaterial] | None = None,
        tax_enabled: bool | None = None,
        tax_rate: float | None = None,
        markup_percent: float | None = None,
    ) -> str:
        """Start an invoice or quote for a customer. Call this as soon as you
        have the customer + the gist — no need to confirm first. Opens a
        pre-filled draft in the app for the user's review + send.

        Args:
            client_name: the customer.
            kind: 'invoice' or 'quote'.
            labor_hours: hours of labor (default 1).
            labor_type: 'profile' (their saved rate), 'hourly', or 'flat'.
            labor_rate: dollars/hour — only for 'hourly'.
            labor_amount: flat job total — only for 'flat'.
            description: short job description.
            materials: the materials the user named — extract each one's item,
                quantity, and unit_price from speech ("50 PEX pipes at $5 each"
                -> item 'PEX pipe', quantity 50, unit_price 5). Leave unit_price
                at 0 only when the user gave no price.
            tax_enabled: false when the user says any variant of "don't tax
                him / no tax / tax exempt" IN THE SAME BREATH as the create.
                Omit to use their default.
            tax_rate: a specific tax percent if the user named one.
            markup_percent: a markup percent if the user named one at create.
        """
        params: dict = {
            "clientName": client_name,
            "type": kind,
            "laborHours": labor_hours,
            "laborType": labor_type,
            "laborRate": labor_rate,
            "laborAmount": labor_amount,
            "description": description,
            "taxEnabled": tax_enabled,
            "taxRate": tax_rate,
            "markupPercent": markup_percent,
        }
        if materials:
            params["materials"] = [
                {
                    "item": m.item,
                    "quantity": m.quantity,
                    "unitPrice": m.unit_price,
                    "cost": round(m.quantity * m.unit_price, 2),
                }
                for m in materials
            ]
        return await self._publish_action(
            "create_invoice", params, f"Opening the {kind} for {client_name}."
        )

    @function_tool
    async def create_expense(
        self,
        ctx: RunContext,
        amount: float,
        description: str,
        vendor: str | None = None,
        category: str | None = None,
        client_name: str | None = None,
    ) -> str:
        """Log an expense — call it once you have the amount + what it was for. Opens the
        pre-filled expense screen in the app for review.

        Args:
            amount: the expense amount.
            description: what it was for.
            vendor: where it was bought, or omit.
            category: materials/fuel/tools/supplies/insurance/... or omit.
            client_name: link to a customer's job, or omit.
        """
        return await self._publish_action(
            "create_expense",
            {
                "amount": amount,
                "description": description,
                "vendor": vendor,
                "category": category,
                "clientName": client_name,
            },
            f"Opening a ${amount:.0f} expense for review.",
        )

    @function_tool
    async def record_payment(
        self,
        ctx: RunContext,
        client_name: str,
        amount: float,
        method: str | None = None,
    ) -> str:
        """Record a payment received from a customer — call it once you have the
        customer + amount. Opens the app's payment sheet on the matching invoice
        for the user to confirm.

        Args:
            client_name: who paid.
            amount: how much.
            method: cash/card/check/bank_transfer, or omit.
        """
        return await self._publish_action(
            "record_payment",
            {"clientName": client_name, "amount": amount, "method": method},
            f"Opening {client_name}'s payment for ${amount:.0f} to confirm.",
        )

    @function_tool
    async def edit_current_invoice(
        self,
        ctx: RunContext,
        op: str,
        item: str | None = None,
        quantity: float | None = None,
        unit_price: float | None = None,
        labor_hours: float | None = None,
        labor_rate: float | None = None,
        enabled: bool | None = None,
        rate: float | None = None,
        percent: float | None = None,
        labor_type: str | None = None,
        labor_amount: float | None = None,
        text: str | None = None,
        date: str | None = None,
        field: str | None = None,
        value: str | None = None,
    ) -> str:
        """Edit the invoice/quote draft the user is LOOKING AT right now, in
        place — do NOT open a new one. Call get_current_screen first to confirm
        an editable invoice is on screen. Use for "change the 3 pipes to 2",
        "drop the valve", "make the pipe $6 each", "add 2 elbows at $3 each",
        "make it 4 hours", "change the rate to $95 an hour", "add a payment link
        to this one".

        Three DIFFERENT numbers — do NOT confuse them. The labor HOURLY RATE
        (data.laborRate, shown as "× $80/hr") is NOT the hours and NOT a
        material's price:
          - "rate", "hourly rate", "per hour", "an hour", "charge X an hour"
            → set_labor_rate (labor_rate).
          - "hours", "make it N hours" → set_labor_hours (labor_hours).
          - a named material's price ("the pipe at $6 each") → set_unit_price.

        TAX AND MARKUP are per-invoice ops on THIS draft — never
        update_settings (that changes the user's global defaults, a different
        thing entirely):
          - "don't tax him / this", "no tax", "tax exempt", "remove the sales
            tax", "turn tax off" → op set_tax, enabled=false.
          - "add tax", "tax it at 8.25" → set_tax with rate (or enabled=true
            for their default rate).
          - "no markup", "drop the markup" → set_markup, enabled=false.
          - "make the markup 25%", "mark it up 30" → set_markup, percent.
          - "make it a flat 500" → set_labor_type, labor_type='flat',
            labor_amount=500. "use my normal rate" → labor_type='profile'.
          - "change the description to X" → set_description, text=X.
          - "due Friday / in 30 days" → set_due_date, date=YYYY-MM-DD (you
            compute the calendar date from TODAY'S DATE).
          - "change the invoice number to 1052" → set_invoice_number, value.
          - "his email is x@y.com" / "fix the client name to Gregg" →
            set_client_field, field one of name/email/phone/address, value.

        Args:
            op: one of set_quantity, set_unit_price, remove_material,
                add_material, set_labor_hours, set_labor_rate, set_stripe,
                set_tax, set_markup, set_labor_type, set_description,
                set_due_date, set_invoice_number, set_client_field.
            item: which material the user means (e.g. 'PEX pipe'). Omit for
                set_labor_hours/set_labor_rate/set_stripe, or when there's a
                single line item.
            quantity: new quantity (set_quantity) or quantity to add
                (add_material).
            unit_price: per-unit price of a MATERIAL (set_unit_price /
                add_material).
            labor_hours: new labor hours (set_labor_hours).
            labor_rate: new labor hourly rate in dollars/hour (set_labor_rate) —
                e.g. "$95 an hour".
            enabled: for set_stripe / set_tax / set_markup — turn the thing
                on or off for THIS invoice.
            rate: for set_tax — the tax percent (8.25 means 8.25%).
            percent: for set_markup — the markup percent.
            labor_type: for set_labor_type — 'profile', 'hourly', or 'flat'.
            labor_amount: for set_labor_type flat — the flat labor total.
            text: for set_description — the new job description.
            date: for set_due_date — ISO date YYYY-MM-DD (you do the math).
            field: for set_client_field — name, email, phone, or address.
            value: for set_client_field / set_invoice_number — the new value.
        """
        params: dict = {"op": op}
        if item is not None:
            params["item"] = item
        if quantity is not None:
            params["quantity"] = quantity
        if unit_price is not None:
            params["unitPrice"] = unit_price
        if labor_hours is not None:
            params["laborHours"] = labor_hours
        if labor_rate is not None:
            params["laborRate"] = labor_rate
        if enabled is not None:
            params["enabled"] = enabled
        if rate is not None:
            params["rate"] = rate
        if percent is not None:
            params["percent"] = percent
        if labor_type is not None:
            params["laborType"] = labor_type
        if labor_amount is not None:
            params["laborAmount"] = labor_amount
        if text is not None:
            params["text"] = text
        if date is not None:
            params["date"] = date
        if field is not None:
            params["field"] = field
        if value is not None:
            params["value"] = value

        spoken = "Done."
        if op == "set_quantity" and item and quantity is not None:
            spoken = f"Done — {item} set to {quantity:g}."
        elif op == "set_unit_price" and item and unit_price is not None:
            spoken = f"Done — {item} at ${unit_price:g} each."
        elif op == "remove_material" and item:
            spoken = f"Removed {item}."
        elif op == "add_material" and item:
            spoken = f"Added {item}."
        elif op == "set_labor_hours" and labor_hours is not None:
            spoken = f"Labor set to {labor_hours:g} hours."
        elif op == "set_labor_rate" and labor_rate is not None:
            spoken = f"Rate set to ${labor_rate:g} an hour."
        elif op == "set_stripe" and enabled is not None:
            spoken = (
                "Payment link added to this invoice."
                if enabled
                else "Payment link removed from this invoice."
            )
        elif op == "set_tax":
            if enabled is False:
                spoken = "Tax is off for this one."
            elif rate is not None:
                spoken = f"Tax set to {rate:g} percent."
            else:
                spoken = "Tax is back on."
        elif op == "set_markup":
            if enabled is False:
                spoken = "Markup removed."
            elif percent is not None:
                spoken = f"Markup set to {percent:g} percent."
            else:
                spoken = "Markup is back on."
        elif op == "set_labor_type" and labor_type == "flat":
            spoken = (
                f"Flat rate — ${labor_amount:g} for the job."
                if labor_amount is not None
                else "Switched to a flat rate."
            )
        elif op == "set_description":
            spoken = "Description updated."
        elif op == "set_due_date" and date:
            spoken = f"Due date set to {date}."
        elif op == "set_invoice_number" and value:
            spoken = f"Invoice number changed to {value}."
        elif op == "set_client_field" and field:
            spoken = f"Client {field} updated."
        return await self._publish_action("edit_current_invoice", params, spoken)

    @function_tool
    async def navigate(
        self,
        ctx: RunContext,
        screen: str,
        client_name: str | None = None,
        period: str | None = None,
    ) -> str:
        """Take the user to a screen in the app. Use when they ask to open/show/
        go to something: "open my expenses", "show me Daniel's profile",
        "go to settings".

        Args:
            screen: EXACTLY one of: home, jobs, expenses, clients, analytics,
                settings, notes, drafts, sent, paid. Never invent other values.
                To open one customer's profile use screen 'clients' + client_name.
            client_name: the customer to open (with screen 'clients'), or omit.
            period: for screen 'analytics' only — 'week', 'month',
                'six_months', or 'ytd' ("show my last 6 months").
        """
        params: dict = {"screen": screen}
        if client_name:
            params["clientName"] = client_name
        if period:
            params["period"] = period
        spoken = (
            f"Opening {client_name}." if client_name else f"Opening {screen}."
        )
        return await self._publish_action("navigate", params, spoken)

    @function_tool
    async def update_settings(
        self,
        ctx: RunContext,
        hourly_rate: float | None = None,
        tax_rate: float | None = None,
        markup_percent: float | None = None,
        default_due_days: int | None = None,
        business_name: str | None = None,
        business_address: str | None = None,
        business_phone: str | None = None,
        business_email: str | None = None,
        tax_id: str | None = None,
        currency_symbol: str | None = None,
        invoice_prefix: str | None = None,
        quote_prefix: str | None = None,
        theme: str | None = None,
        wake_word_enabled: bool | None = None,
    ) -> str:
        """Change the user's business settings/defaults. Pass ONLY what they
        asked to change. Applies instantly with an Undo. Use for: "set my hourly
        rate to 50", "change my business email to …", "make invoices due in 30
        days", "switch to dark mode", "turn off Hey Pico".

        Args:
            hourly_rate: default labor rate in dollars/hour.
            tax_rate: default tax percent.
            markup_percent: default materials markup percent.
            default_due_days: invoice due period in whole days (1-365).
            business_name: the business name shown on invoices.
            business_address: business address shown on invoices.
            business_phone: business phone shown on invoices.
            business_email: business email shown on invoices.
            tax_id: tax ID shown on invoices.
            currency_symbol: like $ or € or £.
            invoice_prefix: short invoice-number prefix like INV.
            quote_prefix: short quote-number prefix like QUO.
            theme: 'light', 'dark', or 'system'.
            wake_word_enabled: true/false for the "Hey Pico" wake word.
        """
        params: dict = {
            "hourlyRate": hourly_rate,
            "taxRate": tax_rate,
            "markupPercent": markup_percent,
            "defaultDueDays": default_due_days,
            "businessName": business_name,
            "businessAddress": business_address,
            "businessPhone": business_phone,
            "businessEmail": business_email,
            "taxId": tax_id,
            "currencySymbol": currency_symbol,
            "invoicePrefix": invoice_prefix,
            "quotePrefix": quote_prefix,
            "theme": theme,
            "wakeWordEnabled": wake_word_enabled,
        }
        return await self._publish_action(
            "update_settings", params, "Done — settings updated."
        )

    @function_tool
    async def update_customer(
        self,
        ctx: RunContext,
        client_name: str,
        email: str | None = None,
        phone: str | None = None,
        address: str | None = None,
        notes: str | None = None,
    ) -> str:
        """Update contact details on an EXISTING customer. Only email, phone,
        address, and notes can change. Applies instantly with an Undo.

        Args:
            client_name: which customer (must already exist).
            email: new email, or omit.
            phone: new phone, or omit.
            address: new address, or omit.
            notes: new notes text, or omit.
        """
        changes = {
            k: v
            for k, v in {
                "email": email,
                "phone": phone,
                "address": address,
                "notes": notes,
            }.items()
            if v is not None
        }
        return await self._publish_action(
            "update_customer",
            {"clientName": client_name, "changes": changes},
            f"Updated {client_name}.",
        )

    @function_tool
    async def open_invoice(
        self,
        ctx: RunContext,
        client_name: str | None = None,
        job_name: str | None = None,
    ) -> str:
        """Open an existing invoice/quote so the user can look at it. Use for
        "open Daniel's invoice", "show me the bathroom quote". Drafts open in
        the editor; sent/paid open in the preview.

        Args:
            client_name: whose invoice, or omit.
            job_name: job title/description words, or omit.
        """
        return await self._publish_action(
            "open_invoice",
            {"clientName": client_name, "jobName": job_name},
            f"Opening {client_name + chr(39) + 's' if client_name else 'the'} invoice.",
        )

    @function_tool
    async def send_invoice(
        self,
        ctx: RunContext,
        client_name: str | None = None,
        job_name: str | None = None,
    ) -> str:
        """Bring the user to the point of SENDING an invoice — the preview with
        its Share button. You never send anything yourself; the user taps
        Share. Use for "send Daniel his invoice", "share the quote with Mike".

        Args:
            client_name: whose invoice, or omit.
            job_name: job title words, or omit.
        """
        return await self._publish_action(
            "send_invoice",
            {"clientName": client_name, "jobName": job_name},
            "Here it is — tap Share to send it.",
        )

    @function_tool
    async def delete_draft(
        self,
        ctx: RunContext,
        client_name: str | None = None,
        job_name: str | None = None,
    ) -> str:
        """Delete an UNSENT draft invoice/quote. The app shows its own confirm
        dialog — don't ask twice. ONLY drafts: if the user asks to delete a
        SENT or PAID invoice, refuse — that's financial history.

        Args:
            client_name: whose draft, or omit.
            job_name: job title words, or omit.
        """
        return await self._publish_action(
            "delete_draft",
            {"clientName": client_name, "jobName": job_name},
            "Confirm on screen to delete the draft.",
        )

    @function_tool
    async def create_customer(
        self,
        ctx: RunContext,
        name: str,
        email: str | None = None,
        phone: str | None = None,
        address: str | None = None,
    ) -> str:
        """Add a brand-new customer. Use for "add a new customer called Bob,
        phone 555-1234". If they already exist the app says so.

        Args:
            name: the customer's name (required).
            email: email, or omit.
            phone: phone, or omit.
            address: address, or omit.
        """
        return await self._publish_action(
            "create_customer",
            {"name": name, "email": email, "phone": phone, "address": address},
            f"Adding {name}.",
        )

    @function_tool
    async def delete_customer(
        self,
        ctx: RunContext,
        name: str,
    ) -> str:
        """Delete a customer entirely. DESTRUCTIVE and permanent -- their
        invoices are kept (the name is just detached), but the customer record
        is gone for good. The app shows a confirm dialog, so don't ask twice.
        Only use on an explicit delete request; never guess.

        Args:
            name: the customer's name (required).
        """
        return await self._publish_action(
            "delete_customer",
            {"name": name},
            f"Confirm on screen to delete {name}.",
        )

    @function_tool
    async def duplicate_invoice(
        self,
        ctx: RunContext,
        client_name: str | None = None,
        job_name: str | None = None,
    ) -> str:
        """Create a new DRAFT as an exact copy of an existing invoice/quote
        (new draft, no invoice number, payments reset; client, materials, labor
        and totals carry over). Opens the copy for review. Use for "duplicate
        the invoice for Daniel", "copy the bathroom quote".

        Args:
            client_name: whose invoice to copy, or omit.
            job_name: job title/description words, or omit.
        """
        return await self._publish_action(
            "duplicate_invoice",
            {"clientName": client_name, "jobName": job_name},
            f"Duplicating {client_name + chr(39) + 's' if client_name else 'the'} invoice.",
        )

    @function_tool
    async def toggle_stripe_link(
        self,
        ctx: RunContext,
        enabled: bool | None = None,
        client_name: str | None = None,
        job_name: str | None = None,
    ) -> str:
        """Add or remove the Stripe online-payment link on a specific invoice
        DRAFT. Opens that draft positioned to flip the link (only drafts -- a
        sent invoice's link follows the saved template). Use for "add a payment
        link to Daniel's invoice", "turn off online payment on the fence quote".
        If the user is already LOOKING at the draft, use edit_current_invoice
        with op set_stripe instead.

        Args:
            enabled: true to add the link, false to remove it.
            client_name: whose invoice, or omit.
            job_name: job title words, or omit.
        """
        return await self._publish_action(
            "toggle_stripe_link",
            {"clientName": client_name, "jobName": job_name, "enabled": enabled},
            "Opening the draft to change the payment link.",
        )

    @function_tool
    async def update_expense(
        self,
        ctx: RunContext,
        description: str | None = None,
        vendor: str | None = None,
        amount: float | None = None,
        new_amount: float | None = None,
        new_description: str | None = None,
        new_vendor: str | None = None,
        new_category: str | None = None,
        new_tax_deductible: bool | None = None,
        new_date: str | None = None,
    ) -> str:
        """Change an existing logged expense. The description/vendor/amount
        IDENTIFY which expense ("the $40 Shell fuel one"); the new_* fields are
        what changes. Applies instantly with an Undo.

        Args:
            description: words from the expense being changed, or omit.
            vendor: its vendor, or omit.
            amount: its current amount (helps identify), or omit.
            new_amount: the new amount, or omit.
            new_description: the new description, or omit.
            new_vendor: the new vendor, or omit.
            new_category: the new category, or omit.
            new_tax_deductible: new deductible flag, or omit.
            new_date: corrected date as ISO "YYYY-MM-DD" — for fixing a
                wrong expense date ("that meal should be June 15 2026").
                Future dates are rejected by the app.
        """
        changes = {
            k: v
            for k, v in {
                "amount": new_amount,
                "description": new_description,
                "vendor": new_vendor,
                "category": new_category,
                "taxDeductible": new_tax_deductible,
                "date": new_date,
            }.items()
            if v is not None
        }
        return await self._publish_action(
            "update_expense",
            {
                "description": description,
                "vendor": vendor,
                "amount": amount,
                "changes": changes,
            },
            "Updating the expense.",
        )

    @function_tool
    async def delete_expense(
        self,
        ctx: RunContext,
        description: str | None = None,
        vendor: str | None = None,
        amount: float | None = None,
    ) -> str:
        """Delete a logged expense. The app shows its own confirm dialog —
        don't ask twice.

        Args:
            description: words from the expense, or omit.
            vendor: its vendor, or omit.
            amount: its amount (helps identify), or omit.
        """
        return await self._publish_action(
            "delete_expense",
            {"description": description, "vendor": vendor, "amount": amount},
            "Confirm on screen to delete it.",
        )

    @function_tool
    async def update_template(
        self,
        ctx: RunContext,
        primary_color: str | None = None,
        accent_color: str | None = None,
        font_family: str | None = None,
        header_style: str | None = None,
        logo_size: str | None = None,
        show_logo: bool | None = None,
        show_business_address: bool | None = None,
        show_business_phone: bool | None = None,
        show_business_email: bool | None = None,
        show_tax_id: bool | None = None,
        show_invoice_number: bool | None = None,
        show_due_date: bool | None = None,
        show_payment_terms: bool | None = None,
        header_tagline: str | None = None,
        footer_note: str | None = None,
        payment_terms: str | None = None,
        thank_you_message: str | None = None,
        stripe_enabled: bool | None = None,
    ) -> str:
        """Restyle the user's invoice/quote template. Pass ONLY what they asked
        to change. Applies instantly with an Undo. Use for: "make my invoices
        green", "center my logo", "just use my logo as the header", "hide
        my tax id", "change the footer to …", "add a payment link to all my
        invoices".

        Args:
            primary_color: hex like '#2E7D32' OR a palette name: 'midnight red',
                'navy blue', 'forest green', 'charcoal', 'deep red', 'purple',
                'teal', 'orange', 'black'. For a plain color word use the
                closest name ('green' → 'forest green', 'blue' → 'navy blue',
                'gray' → 'charcoal'). Naming just this also sets the matching
                accent automatically.
            accent_color: hex or palette name, or omit to pair automatically.
            font_family: 'helvetica', 'times', or 'courier'.
            header_style: the logo/header treatment — 'lockup' (logo beside the
                business name), 'centered' (logo centered on top), or
                'logo_only' (logo replaces the business-name text).
            logo_size: 'small', 'medium', or 'large' ("make my logo
                bigger" -> 'large').
            show_logo: show/hide the logo.
            show_business_address: show/hide the business address.
            show_business_phone: show/hide the business phone.
            show_business_email: show/hide the business email.
            show_tax_id: show/hide the tax ID.
            show_invoice_number: show/hide the invoice number.
            show_due_date: show/hide the due date.
            show_payment_terms: show/hide payment terms.
            header_tagline: short tagline under the business name.
            footer_note: footer text.
            payment_terms: payment-terms text.
            thank_you_message: closing thank-you line.
            stripe_enabled: include the Stripe payment link on ALL invoices
                (true/false).
        """
        params: dict = {
            "primaryColor": primary_color,
            "accentColor": accent_color,
            "fontFamily": font_family,
            "headerStyle": header_style,
            "logoSize": logo_size,
            "showLogo": show_logo,
            "showBusinessAddress": show_business_address,
            "showBusinessPhone": show_business_phone,
            "showBusinessEmail": show_business_email,
            "showTaxId": show_tax_id,
            "showInvoiceNumber": show_invoice_number,
            "showDueDate": show_due_date,
            "showPaymentTerms": show_payment_terms,
            "headerTagline": header_tagline,
            "footerNote": footer_note,
            "paymentTerms": payment_terms,
            "thankYouMessage": thank_you_message,
            "stripeEnabled": stripe_enabled,
        }
        return await self._publish_action(
            "update_template", params, "Done — template updated."
        )

    @function_tool
    async def create_note(
        self,
        ctx: RunContext,
        title: str | None = None,
        content: str | None = None,
        client_name: str | None = None,
    ) -> str:
        """Open the note editor with a draft note, optionally linked to a
        customer. Use for "make a note that…", "note for Daniel: …".

        Args:
            title: short note title, or omit.
            content: the note body, or omit.
            client_name: customer to attach it to, or omit for a general note.
        """
        return await self._publish_action(
            "create_note",
            {"title": title, "content": content, "clientName": client_name},
            "Opening the note.",
        )

    # ── Session control ──────────────────────────────────────────────────────

    @function_tool
    async def end_session(self, ctx: RunContext) -> str:
        """End the voice session because the user is finished. Call this ONLY on
        a clear closing cue ("that's it", "that's all", "thanks, bye", "nothing
        else", "I'm good"), never on a mere pause. Say a brief, warm one-line
        goodbye; the app closes the assistant once you've finished speaking.
        """
        await self._publish_control("end_session")
        return "Sounds good — I'm right here whenever you need me."


async def entrypoint(ctx: JobContext) -> None:
    await ctx.connect()

    # Per-user id = the participant identity (set by voice_room_token to the
    # Supabase user id). MUST WAIT for the participant to actually join: reading
    # remote_participants right after connect raced the join and silently fell
    # back to "local-console", so every books query scoped to a nonexistent
    # user — "I don't see any invoices for Greg." The race was usually won on
    # the local worker but lost from the cloud (more network distance), which is
    # why answers regressed after the cloud move.
    #
    # PRE-DISPATCH (latency cut): prod rooms (`voice-…`) may now be created
    # server-side by voice_room_token BEFORE the user speaks — the agent joins
    # while the app is merely open ("Hey Pico" armed) so the eventual connect is
    # ~1s instead of ~3s. Two consequences here:
    #   * Wait LONG (PARTICIPANT_WAIT) — the "Hey Pico" may come minutes later.
    #     Must exceed the client's prewarm freshness window + refresh cadence
    #     (5 min / 4.5 min, pico_session_provider.dart) so the client never
    #     reuses a slot whose agent has already given up.
    #   * On timeout EXIT CLEANLY — never fall back to local-console for a prod
    #     room: that would boot a billable Gemini session in an empty room
    #     (validated empirically 2026-07-11: server-side CreateRoom dispatches
    #     ~1.4s later; the old 15s fallback then started a ghost session).
    # Non-`voice-` rooms (lk console validation) keep the short-wait fallback.
    is_prod_room = ctx.room.name.startswith("voice-")
    participant_wait = PARTICIPANT_WAIT if is_prod_room else 15.0
    user_id = "local-console"
    try:
        participant = await asyncio.wait_for(
            ctx.wait_for_participant(), timeout=participant_wait
        )
        if participant and participant.identity:
            user_id = participant.identity
    except asyncio.TimeoutError:
        if is_prod_room:
            logger.info(
                "[diag] pre-dispatched agent: no participant within %ss in %s — "
                "standing down (no Gemini session started)",
                participant_wait, ctx.room.name,
            )
            ctx.shutdown(reason="prewarm expired, no participant")
            return
        logger.warning("[diag] no participant joined in 15s — user_id=local-console")
    except RuntimeError:
        # "room disconnected while waiting for participant" — the client stood
        # this prewarm down (voice_room_close deleted the room: app backgrounded,
        # wake word disarmed, or slot refreshed). Routine, NOT an error.
        logger.info(
            "[diag] pre-dispatched agent: room %s closed while waiting — "
            "standing down", ctx.room.name,
        )
        return
    logger.info("[diag] resolved user_id=%s", user_id)

    session = AgentSession(
        # Native speech-to-speech. Built-in semantic VAD turn detection + barge-in.
        llm=google.realtime.RealtimeModel(
            # Instructions MUST be set HERE at construction, not only on the Agent.
            # gemini-3.1-flash-live-preview ignores mid-session instruction updates
            # — and pushing the Agent's instructions is exactly such an update — so
            # the Agent's copy alone never reaches the model; it would run on
            # defaults (which is why it kept confirming + asking who it's for).
            # build_instructions() also injects TODAY'S DATE per session, so
            # "last 2 months" date math works (parity with the text brain).
            instructions=build_instructions(),
            model="gemini-3.1-flash-live-preview",
            voice="Charon",  # warm; A/B other Gemini voices (Puck, Kore, …) to taste
            # DIAGNOSTIC (2026-06-14, "silent agent" regression hunt): have
            # Gemini transcribe BOTH sides so the logs show what the model
            # actually heard from the user's mic (empty transcript ⇒ audio
            # isn't reaching the model) and what it tried to say back.
            input_audio_transcription=genai_types.AudioTranscriptionConfig(),
            output_audio_transcription=genai_types.AudioTranscriptionConfig(),
            # ⚠️ DO NOT add realtime_input_config(AutomaticActivityDetection…)
            # here. Tried 2026-06-12 (END_SENSITIVITY_LOW + 900ms silence, to
            # stop long sentences being clipped) — with it set, this model
            # joined the room, listened, and NEVER replied (no turn completion).
            # Reverted to the model's built-in defaults, which are validated.
            # Re-attempt only with a locally validated config (console mode).
            #
            # NOTE: enable_affective_dialog=True made Gemini Live reject the
            # session on first real connect ("1011 Internal error encountered" /
            # Failed to connect to Gemini Live). Removed until we confirm the
            # model + key support it; re-add behind a flag once verified.
        ),
        allow_interruptions=True,  # barge-in: user can talk over the agent
    )

    # ── DIAGNOSTIC INSTRUMENTATION (2026-06-14 silent-agent hunt) ──
    # Does the user's mic track ever reach the agent? (No audio track / no
    # frames ⇒ the client isn't publishing — e.g. the sim's WebRTC mic.)
    @ctx.room.on("track_subscribed")
    def _on_track_subscribed(track, publication, participant):  # noqa: ANN001
        logger.info(
            "[diag] track_subscribed kind=%s from=%s muted=%s",
            track.kind, participant.identity, getattr(publication, "muted", "?"),
        )

    @ctx.room.on("track_published")
    def _on_track_published(publication, participant):  # noqa: ANN001
        logger.info(
            "[diag] track_published kind=%s from=%s",
            publication.kind, participant.identity,
        )

    # What did Gemini actually HEAR (from the mic) and SAY (back)?
    @session.on("user_input_transcribed")
    def _on_user_transcribed(ev):  # noqa: ANN001
        logger.info("[diag] USER heard=%r final=%s",
                    getattr(ev, "transcript", None), getattr(ev, "is_final", None))

    @session.on("conversation_item_added")
    def _on_item_added(ev):  # noqa: ANN001
        try:
            item = ev.item
            logger.info("[diag] item role=%s text=%r",
                        getattr(item, "role", "?"), getattr(item, "text_content", None))
        except Exception as exc:  # pragma: no cover - diagnostic only
            logger.info("[diag] item (unparsed): %s", exc)

    await session.start(agent=PicoTallyAgent(user_id, ctx.room), room=ctx.room)
    logger.info("[diag] session started for user_id=%s room=%s",
                user_id, ctx.room.name)
    # NOTE: no spoken ready-greeting. session.generate_reply() is rejected by
    # gemini-3.1-flash-live-preview ("generate_reply is not compatible", logged
    # as an in-session ERROR). The client's "ready" cues are the haptic + the
    # strip flipping from "Connecting…" to "Listening…" once the agent joins.
    # The greeting instruction lives in INSTRUCTIONS instead (model greets on
    # the user's first words).


if __name__ == "__main__":
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            # Keep warm subprocesses ready so the agent joins a new voice room in
            # ~1-2s instead of cold-starting per call. `dev` mode defaults this to
            # 0 (every call cold-starts, which tripped the client join watchdog);
            # prod defaults to 4. Two covers back-to-back calls while a consumed
            # process's replacement re-warms in the background.
            num_idle_processes=2,
            # macOS `spawn` re-imports the (heavy) deps in each subprocess; the
            # 10s default isn't enough to warm one and it gets killed mid-init.
            initialize_process_timeout=30.0,
        )
    )
