/**
 * API client for the Mira Memory Engine
 */

// Use environment variable for production, fallback to relative path for dev (Vite proxy)
export const API_BASE = 'https://mira-memory-engine.onrender.com/api/v1'

// Types
export interface Memory {
  id: string
  text: string
  score: number
  metadata: Record<string, unknown>
}

export interface IngestRequest {
  text: string
  source?: string
  session_id?: string
  metadata?: Record<string, unknown>
}

export interface IngestResponse {
  success: boolean
  memory_ids: string[]
  chunks_created: number
  processing_time_ms: number
  latency_breakdown: Record<string, number>
}

export interface QueryRequest {
  query: string
  top_k?: number
  score_threshold?: number
  include_context?: boolean
  stream?: boolean
}

export interface QueryResponse {
  answer: string
  memories: Memory[]
  query: string
  processing_time_ms: number
  latency_breakdown: Record<string, number>
}

export interface HealthResponse {
  status: 'healthy' | 'degraded' | 'unhealthy'
  version: string
  timestamp: string
  services: Record<string, boolean>
}

export interface MetricsResponse {
  uptime_seconds: number
  total_memories: number
  total_queries: number
  total_ingests: number
  latencies: Array<{
    operation: string
    p50_ms: number
    p95_ms: number
    p99_ms: number
    count: number
  }>
}

export interface RecentMemoriesResponse {
  memories: Memory[]
  total_count: number
  limit: number
}

// API Error class
export class ApiError extends Error {
  constructor(
    public status: number,
    message: string,
    public details?: unknown
  ) {
    super(message)
    this.name = 'ApiError'
  }
}

// Helper function for API requests
async function request<T>(
  endpoint: string,
  options: RequestInit = {}
): Promise<T> {
  const url = `${API_BASE}${endpoint}`
  
  const response = await fetch(url, {
    ...options,
    headers: {
      'Content-Type': 'application/json',
      ...options.headers,
    },
  })

  if (!response.ok) {
    let errorMessage = `Request failed: ${response.statusText}`
    let details: unknown = null
    
    try {
      details = await response.json()
      errorMessage = (details as { detail?: string }).detail || errorMessage
    } catch {
      // Ignore JSON parse errors
    }
    
    throw new ApiError(response.status, errorMessage, details)
  }

  // Handle 204 No Content
  if (response.status === 204) {
    return undefined as T
  }

  return response.json()
}

// API functions
export const api = {
  /**
   * Health check
   */
  health: (): Promise<HealthResponse> => {
    return request('/health')
  },

  /**
   * Get metrics
   */
  metrics: (): Promise<MetricsResponse> => {
    return request('/metrics')
  },

  /**
   * Ingest text into memory
   */
  ingest: (data: IngestRequest): Promise<IngestResponse> => {
    return request('/memory/ingest', {
      method: 'POST',
      body: JSON.stringify(data),
    })
  },

  /**
   * Query memories
   */
  query: (data: QueryRequest): Promise<QueryResponse> => {
    return request('/memory/query', {
      method: 'POST',
      body: JSON.stringify(data),
    })
  },

  /**
   * Search memories (without LLM generation)
   */
  search: (
    query: string,
    topK: number = 5,
    scoreThreshold: number = 0.7
  ): Promise<Memory[]> => {
    const params = new URLSearchParams({
      query,
      top_k: String(topK),
      score_threshold: String(scoreThreshold),
    })
    return request(`/memory/search?${params}`, { method: 'POST' })
  },

  /**
   * Get recent memories
   */
  recent: (limit: number = 10): Promise<RecentMemoriesResponse> => {
    return request(`/memory/recent?limit=${limit}`)
  },

  /**
   * Clear all memories
   */
  clear: (): Promise<void> => {
    return request('/memory/clear', { method: 'DELETE' })
  },

  /**
   * Stream query response (returns async generator)
   */
  queryStream: async function* (
    data: QueryRequest
  ): AsyncGenerator<{ type: string; content?: string; memories?: Memory[] }> {
    const response = await fetch(`${API_BASE}/memory/query`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ ...data, stream: true }),
    })

    if (!response.ok) {
      throw new ApiError(response.status, 'Stream request failed')
    }

    const reader = response.body?.getReader()
    if (!reader) throw new Error('No response body')

    const decoder = new TextDecoder()
    let buffer = ''

    while (true) {
      const { done, value } = await reader.read()
      if (done) break

      buffer += decoder.decode(value, { stream: true })
      const lines = buffer.split('\n')
      buffer = lines.pop() || ''

      for (const line of lines) {
        if (line.startsWith('data: ')) {
          try {
            const data = JSON.parse(line.slice(6))
            yield data
          } catch {
            // Ignore JSON parse errors
          }
        }
      }
    }
  },
}

export default api
