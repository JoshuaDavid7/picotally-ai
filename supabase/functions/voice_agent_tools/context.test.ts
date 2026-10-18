import { assert, assertEquals } from 'https://deno.land/std@0.224.0/assert/mod.ts'

import {
  computeCustomerLedger,
  indexJobsById,
  parseOcrItems,
  shapeCustomer,
  shapeExpense,
  shapeJob,
  shapeMaterial,
  shapePayment,
} from './context.ts'

// The whole point of the snapshot is to be consumable by the REAL executors,
// so the integration tests below feed shaped rows straight into them.
import {
  executeQueryCustomerBalance,
  executeQueryMaterials,
} from '../process_voice_command/tools.ts'

// ─── Pure shapers ─────────────────────────────────────────────────────────────

Deno.test('shapeCustomer maps snake_case columns to the camelCase the executor reads', () => {
  const out = shapeCustomer({
    name: 'Dave Smith',
    phone: '555-1234',
    email: 'd@example.com',
    job_count: 3,
    total_billed: 1000,
    total_paid: 750,
    balance: 250,
  })
  assertEquals(out, {
    name: 'Dave Smith',
    phone: '555-1234',
    email: 'd@example.com',
    jobCount: 3,
    totalBilled: 1000,
    totalPaid: 750,
    balance: 250,
  })
})

Deno.test('shapeCustomer defaults missing numerics to 0 and strings to empty', () => {
  const out = shapeCustomer({ name: 'Solo' })
  assertEquals(out.balance, 0)
  assertEquals(out.totalBilled, 0)
  assertEquals(out.phone, '')
})

Deno.test('computeCustomerLedger derives balance from jobs (cloud summary is stale)', () => {
  // The customers row carries 0s (stale on the server); the jobs carry truth.
  const jobs = [
    { client_name: 'Daniel Smith', total_amount: 26190.14, amount_paid: 26190.14 },
    { client_name: 'Daniel Smith', total_amount: 500, amount_paid: 0 }, // unsent draft
    { client_name: 'Someone Else', total_amount: 99, amount_paid: 99 },
  ]
  const ledger = computeCustomerLedger(jobs)
  const out = shapeCustomer(
    { name: 'Daniel Smith', total_billed: 0, total_paid: 0, balance: 0, job_count: 0 },
    ledger.get('daniel smith'),
  )
  assertEquals(out.totalBilled, 26690.14)
  assertEquals(out.totalPaid, 26190.14)
  assertEquals(out.balance, 500) // billed - paid, not the stale 0
  assertEquals(out.jobCount, 2)
})

Deno.test('shapeMaterial aliases actualSupplierCost from originalCost without losing keys', () => {
  const out = shapeMaterial({ item: 'PEX pipe', quantity: 50, cost: 90, originalCost: 60 })
  assertEquals(out.actualSupplierCost, 60) // pre-markup alias the executor prefers
  assertEquals(out.originalCost, 60) // original key preserved
  assertEquals(out.cost, 90)
  assertEquals(out.item, 'PEX pipe')
})

Deno.test('shapeMaterial does not clobber an existing actualSupplierCost', () => {
  const out = shapeMaterial({ item: 'x', actualSupplierCost: 5, originalCost: 99 })
  assertEquals(out.actualSupplierCost, 5)
})

Deno.test('shapeJob maps fields, nulls dueDate (no cloud column), and shapes materials', () => {
  const out = shapeJob({
    id: 'j1',
    title: "Dave's bathroom",
    client_name: 'Dave',
    status: 'sent',
    type: 'invoice',
    total_amount: 500,
    amount_paid: 100,
    amount_due: 400,
    invoice_number: 'INV-1',
    created_at: '2026-05-01T00:00:00.000Z',
    paid_at: null,
    materials: [{ item: 'PEX pipe', quantity: 50, cost: 90, originalCost: 60 }],
  })
  assertEquals(out.clientName, 'Dave')
  assertEquals(out.status, 'sent')
  assertEquals(out.totalAmount, 500)
  assertEquals(out.amountDue, 400)
  assertEquals(out.dueDate, null)
  assertEquals(out.paidAt, null)
  const mats = out.materials as Array<Record<string, unknown>>
  assertEquals(mats.length, 1)
  assertEquals(mats[0].actualSupplierCost, 60)
})

Deno.test('shapeJob tolerates a missing/!array materials column', () => {
  const out = shapeJob({ id: 'j2', title: 't', status: 'draft' })
  assertEquals((out.materials as unknown[]).length, 0)
})

Deno.test('parseOcrItems parses the {items:[...]} JSON; tolerates junk', () => {
  const ok = parseOcrItems(JSON.stringify({
    items: [{ name: 'Valve', quantity: 2, unitPrice: 18, totalPrice: 36 }],
  }))
  assertEquals(ok.length, 1)
  assertEquals(ok[0].name, 'Valve')
  assertEquals(ok[0].unitPrice, 18)

  assertEquals(parseOcrItems('not json'), [])
  assertEquals(parseOcrItems(null), [])
  assertEquals(parseOcrItems(JSON.stringify({ items: 'nope' })), [])
})

Deno.test('shapeExpense maps columns + derives hasReceipt + source', () => {
  const out = shapeExpense({
    description: 'Pipe run',
    vendor: 'Home Depot',
    category: 'materials',
    amount: 42.5,
    expense_date: '2026-05-02T00:00:00.000Z',
    tax_deductible: true,
    receipt_url: 'https://x/r.jpg',
    ocr_text: null,
  })
  assertEquals(out.vendor, 'Home Depot')
  assertEquals(out.amount, 42.5)
  assertEquals(out.taxDeductible, true)
  assertEquals(out.hasReceipt, true)
  assertEquals(out.source, 'logged')
  assertEquals((out.ocrItems as unknown[]).length, 0)
})

Deno.test('shapePayment enriches method/amount and joins job title + client', () => {
  const jobsById = indexJobsById([{ id: 'j1', title: "Dave's bathroom", client_name: 'Dave' }])
  const out = shapePayment(
    {
      received_at: '2026-05-03T00:00:00.000Z',
      amount: 200,
      payment_method: 'cash',
      transaction_reference: 'CHK-9',
      job_id: 'j1',
    },
    jobsById,
  )
  assertEquals(out.amount, 200)
  assertEquals(out.method, 'cash')
  assertEquals(out.clientName, 'Dave')
  assertEquals(out.jobTitle, "Dave's bathroom")
  assertEquals(out.reference, 'CHK-9')
})

// ─── Integration: shaped rows must be consumable by the real executors ────────

Deno.test('INTEGRATION: shaped customer → executeQueryCustomerBalance finds the balance', () => {
  // The balance is derived from the customer's INVOICES, not from the stored
  // customer aggregate. That aggregate counts quotes as billed, and a quote is
  // a proposal rather than a receivable — which is how an $850 quote once made
  // a customer read as owing $891 when he actually owed $41. So the shaped
  // customer supplies identity, and the jobs supply the money.
  const context = {
    customers: [
      shapeCustomer({
        name: 'Dave Smith',
        phone: '555',
        email: 'd@x.com',
        job_count: 3,
        total_billed: 1000,
        total_paid: 750,
        balance: 250,
      }),
    ],
    jobs: [
      // Counts: a sent invoice, part paid. 1000 - 750 = 250 outstanding.
      {
        type: 'invoice',
        status: 'sent',
        clientName: 'Dave Smith',
        totalAmount: 1000,
        amountPaid: 750,
      },
      // Must NOT count: a quote is not money owed. This is the exact row shape
      // that caused the original inflated-balance bug.
      {
        type: 'quote',
        status: 'sent',
        clientName: 'Dave Smith',
        totalAmount: 5000,
        amountPaid: 0,
      },
      // Must NOT count: an unsent draft is not a receivable either.
      {
        type: 'invoice',
        status: 'draft',
        clientName: 'Dave Smith',
        totalAmount: 300,
        amountPaid: 0,
      },
    ],
  }
  // The user says just "Dave"; the executor's substring match should resolve it.
  const res = executeQueryCustomerBalance(context, { clientName: 'Dave' })
  assertEquals(res.count, 1)
  assertEquals(res.total, 250)
  const item = res.items[0] as Record<string, unknown>
  assertEquals(item.balance, 250)
  assertEquals(item.totalBilled, 1000)
  assertEquals(item.totalPaid, 750)
  // Only the one real invoice was counted — not the quote, not the draft.
  assertEquals(item.jobCount, 1)
})

Deno.test('INTEGRATION: shaped jobs+materials → executeQueryMaterials sums pre-markup cost', () => {
  const context = {
    now: '2026-06-05T00:00:00.000Z',
    jobs: [
      shapeJob({
        id: 'j1',
        title: "Dave's bathroom",
        client_name: 'Dave',
        status: 'sent', // counts under invoiceStatus 'sent_or_paid'
        type: 'invoice',
        created_at: '2026-05-01T00:00:00.000Z',
        materials: [
          { item: 'PEX pipe', quantity: 50, cost: 90, originalCost: 60 },
          { item: 'Copper fitting', quantity: 10, cost: 30, originalCost: 20 },
        ],
      }),
    ],
  }
  // "How much did I spend on pipe?" → pre-markup supplier cost of the PEX line.
  const res = executeQueryMaterials(context, {
    keyword: 'pipe',
    invoiceStatus: 'sent_or_paid',
  })
  assert(res.count >= 1)
  assertEquals(res.total, 60)
})

Deno.test('INTEGRATION: draft jobs are excluded from sent_or_paid material spend', () => {
  const context = {
    now: '2026-06-05T00:00:00.000Z',
    jobs: [
      shapeJob({
        id: 'j2',
        title: 'Unsent draft',
        client_name: 'Dave',
        status: 'draft',
        type: 'invoice',
        created_at: '2026-05-01T00:00:00.000Z',
        materials: [{ item: 'PEX pipe', quantity: 50, cost: 90, originalCost: 60 }],
      }),
    ],
  }
  const res = executeQueryMaterials(context, {
    keyword: 'pipe',
    invoiceStatus: 'sent_or_paid',
  })
  assertEquals(res.total, 0)
})
