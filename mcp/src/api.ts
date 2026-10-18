// HTTP client for the mcp_tools edge function.

import type { ToolCatalogEntry } from './catalog.js'

export class PicoTallyApi {
  constructor(
    private readonly functionsUrl: string,
    private readonly tokenProvider: () => Promise<string>,
    private readonly fetchImpl: typeof fetch = fetch,
  ) {}

  private async request(method: 'GET' | 'POST', body?: unknown): Promise<unknown> {
    const token = await this.tokenProvider()
    const res = await this.fetchImpl(`${this.functionsUrl}/mcp_tools`, {
      method,
      headers: { Authorization: `Bearer ${token}`, 'Content-Type': 'application/json' },
      body: body === undefined ? undefined : JSON.stringify(body),
    })
    const text = await res.text()
    let json: unknown
    try {
      json = JSON.parse(text)
    } catch {
      json = { error: text }
    }
    if (!res.ok) throw new Error(`mcp_tools ${method} failed (${res.status}): ${JSON.stringify(json)}`)
    return json
  }

  async catalog(): Promise<ToolCatalogEntry[]> {
    const json = (await this.request('GET')) as { tools?: ToolCatalogEntry[] }
    if (!Array.isArray(json.tools)) throw new Error('mcp_tools returned no tool catalog')
    return json.tools
  }

  call(tool: string, input: Record<string, unknown>): Promise<unknown> {
    return this.request('POST', { tool, input })
  }
}
