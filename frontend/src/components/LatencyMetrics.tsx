import { useState, useEffect, useCallback } from 'react'
import { RefreshCw, Clock, Zap, TrendingUp, Server, Loader2 } from 'lucide-react'

interface LatencyMetric {
  operation: string
  p50_ms: number
  p95_ms: number
  p99_ms: number
  count: number
}

interface QualityMetrics {
  hit_rate: number
  avg_retrieval_score: number
  avg_memories_per_query: number
  total_words_ingested: number
  total_chunks_created: number
}

interface Metrics {
  uptime_seconds: number
  total_memories: number
  total_queries: number
  total_ingests: number
  latencies: LatencyMetric[]
  quality?: QualityMetrics
}

const OPERATION_LABELS: Record<string, string> = {
  embedding_single: 'Embedding (single)',
  embedding_batch: 'Embedding (batch)',
  vectordb_add: 'Vector DB Write',
  vectordb_query: 'Vector DB Query',
  llm_generate: 'LLM Generation',
  llm_first_token: 'LLM First Token',
  chunking: 'Text Chunking',
  context_assembly: 'Context Assembly',
  ingest_chunking: 'Ingest: Chunking',
  ingest_embedding: 'Ingest: Embedding',
  ingest_storage: 'Ingest: Storage',
  query_embedding: 'Query: Embedding',
  query_search: 'Query: Search',
  query_context: 'Query: Context',
  query_llm: 'Query: LLM',
  stt_single: 'STT Transcription',
}

export function LatencyMetrics() {
  const [metrics, setMetrics] = useState<Metrics | null>(null)
  const [isLoading, setIsLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [autoRefresh, setAutoRefresh] = useState(false)

  const fetchMetrics = useCallback(async () => {
    try {
      const response = await fetch('/api/v1/metrics')
      if (!response.ok) throw new Error('Failed to fetch metrics')

      const data = await response.json()
      setMetrics(data)
      setError(null)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load metrics')
    } finally {
      setIsLoading(false)
    }
  }, [])

  useEffect(() => {
    fetchMetrics()

    if (autoRefresh) {
      const interval = setInterval(fetchMetrics, 5000)
      return () => clearInterval(interval)
    }
  }, [fetchMetrics, autoRefresh])

  const formatUptime = (seconds: number) => {
    const hours = Math.floor(seconds / 3600)
    const minutes = Math.floor((seconds % 3600) / 60)
    const secs = Math.floor(seconds % 60)

    if (hours > 0) return `${hours}h ${minutes}m ${secs}s`
    if (minutes > 0) return `${minutes}m ${secs}s`
    return `${secs}s`
  }

  const getLatencyColor = (ms: number) => {
    // Adjusted thresholds for realistic API latencies
    if (ms < 100) return 'text-accent-emerald'   // Excellent (internal ops)
    if (ms < 500) return 'text-accent-cyan'      // Good (fast API calls)
    if (ms < 2000) return 'text-accent-amber'    // Normal (embedding APIs)
    return 'text-mira-400'                       // Expected (LLM APIs - not red!)
  }

  const getLatencyBarWidth = (ms: number, max: number = 6000) => {
    // Adjusted max for LLM latencies
    return Math.min((ms / max) * 100, 100)
  }

  if (isLoading) {
    return (
      <div className="flex items-center justify-center py-12">
        <Loader2 className="w-8 h-8 text-mira-400 animate-spin" />
      </div>
    )
  }

  if (error) {
    return (
      <div className="glass rounded-2xl p-8 text-center">
        <p className="text-accent-rose mb-4">{error}</p>
        <button
          onClick={fetchMetrics}
          className="px-4 py-2 text-sm bg-surface-raised hover:bg-surface-muted border border-surface-muted rounded-lg transition-all"
        >
          Retry
        </button>
      </div>
    )
  }

  if (!metrics) return null

  return (
    <div className="space-y-6">
      {/* Controls */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-4">
          <label className="flex items-center gap-2 text-sm text-gray-400 cursor-pointer">
            <input
              type="checkbox"
              checked={autoRefresh}
              onChange={(e) => setAutoRefresh(e.target.checked)}
              className="w-4 h-4 rounded border-surface-muted bg-surface-raised text-mira-500 focus:ring-mira-500"
            />
            Auto-refresh (5s)
          </label>
        </div>
        <button
          onClick={fetchMetrics}
          className="flex items-center gap-2 px-3 py-2 text-sm text-gray-400 hover:text-white bg-surface-raised hover:bg-surface-muted border border-surface-muted rounded-lg transition-all"
        >
          <RefreshCw className="w-4 h-4" />
          <span>Refresh</span>
        </button>
      </div>

      {/* Overview stats */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
        <div className="glass rounded-xl p-4">
          <div className="flex items-center gap-2 text-gray-500 mb-2">
            <Server className="w-4 h-4" />
            <span className="text-xs">Uptime</span>
          </div>
          <p className="text-xl font-semibold text-white">
            {formatUptime(metrics.uptime_seconds)}
          </p>
        </div>

        <div className="glass rounded-xl p-4">
          <div className="flex items-center gap-2 text-gray-500 mb-2">
            <Clock className="w-4 h-4" />
            <span className="text-xs">Total Memories</span>
          </div>
          <p className="text-xl font-semibold text-white">
            {metrics.total_memories.toLocaleString()}
          </p>
        </div>

        <div className="glass rounded-xl p-4">
          <div className="flex items-center gap-2 text-gray-500 mb-2">
            <Zap className="w-4 h-4" />
            <span className="text-xs">Queries</span>
          </div>
          <p className="text-xl font-semibold text-white">
            {metrics.total_queries.toLocaleString()}
          </p>
        </div>

        <div className="glass rounded-xl p-4">
          <div className="flex items-center gap-2 text-gray-500 mb-2">
            <TrendingUp className="w-4 h-4" />
            <span className="text-xs">Ingests</span>
          </div>
          <p className="text-xl font-semibold text-white">
            {metrics.total_ingests.toLocaleString()}
          </p>
        </div>
      </div>

      {/* Quality Metrics */}
      {metrics.quality && (
        <div className="glass rounded-2xl overflow-hidden">
          <div className="px-6 py-4 border-b border-surface-muted">
            <h3 className="font-medium text-white">Retrieval Quality</h3>
            <p className="text-xs text-gray-500 mt-1">How well the system finds relevant memories</p>
          </div>
          <div className="grid grid-cols-2 lg:grid-cols-5 gap-4 p-6">
            <div className="text-center">
              <p className={`text-2xl font-bold ${metrics.quality.hit_rate >= 80 ? 'text-accent-emerald' : metrics.quality.hit_rate >= 50 ? 'text-accent-amber' : 'text-gray-400'}`}>
                {metrics.quality.hit_rate}%
              </p>
              <p className="text-xs text-gray-500 mt-1">Hit Rate</p>
              <p className="text-[10px] text-gray-600">Queries with results</p>
            </div>
            <div className="text-center">
              <p className={`text-2xl font-bold ${metrics.quality.avg_retrieval_score >= 40 ? 'text-accent-emerald' : metrics.quality.avg_retrieval_score >= 25 ? 'text-accent-amber' : 'text-gray-400'}`}>
                {metrics.quality.avg_retrieval_score}%
              </p>
              <p className="text-xs text-gray-500 mt-1">Avg Relevance</p>
              <p className="text-[10px] text-gray-600">Similarity score</p>
            </div>
            <div className="text-center">
              <p className="text-2xl font-bold text-mira-400">
                {metrics.quality.avg_memories_per_query}
              </p>
              <p className="text-xs text-gray-500 mt-1">Memories/Query</p>
              <p className="text-[10px] text-gray-600">Avg retrieved</p>
            </div>
            <div className="text-center">
              <p className="text-2xl font-bold text-white">
                {metrics.quality.total_words_ingested.toLocaleString()}
              </p>
              <p className="text-xs text-gray-500 mt-1">Words Processed</p>
              <p className="text-[10px] text-gray-600">Total ingested</p>
            </div>
            <div className="text-center">
              <p className="text-2xl font-bold text-white">
                {metrics.quality.total_chunks_created}
              </p>
              <p className="text-xs text-gray-500 mt-1">Chunks Created</p>
              <p className="text-[10px] text-gray-600">Memory segments</p>
            </div>
          </div>
        </div>
      )}

      {/* Latency breakdown */}
      <div className="glass rounded-2xl overflow-hidden">
        <div className="px-6 py-4 border-b border-surface-muted">
          <h3 className="font-medium text-white">Latency Breakdown</h3>
          <p className="text-xs text-gray-500 mt-1">Response time percentiles</p>
        </div>

        {metrics.latencies.length === 0 ? (
          <div className="px-6 py-8 text-center text-gray-500">
            No latency data yet. Make some queries to see metrics.
          </div>
        ) : (
          <div className="divide-y divide-surface-muted">
            {metrics.latencies.map((metric) => (
              <div key={metric.operation} className="px-6 py-4">
                <div className="flex items-center justify-between mb-2">
                  <span className="text-sm text-white">
                    {OPERATION_LABELS[metric.operation] || metric.operation}
                  </span>
                  <span className="text-xs text-gray-500">
                    {metric.count.toLocaleString()} calls
                  </span>
                </div>

                {/* Latency bars */}
                <div className="space-y-2">
                  {/* P50 */}
                  <div className="flex items-center gap-3">
                    <span className="text-xs text-gray-500 w-8">P50</span>
                    <div className="flex-1 h-2 bg-surface-muted rounded-full overflow-hidden">
                      <div
                        className="h-full bg-gradient-to-r from-mira-600 to-mira-500 rounded-full transition-all duration-500"
                        style={{ width: `${getLatencyBarWidth(metric.p50_ms)}%` }}
                      />
                    </div>
                    <span className={`text-xs font-mono w-16 text-right ${getLatencyColor(metric.p50_ms)}`}>
                      {metric.p50_ms.toFixed(1)}ms
                    </span>
                  </div>

                  {/* P95 */}
                  <div className="flex items-center gap-3">
                    <span className="text-xs text-gray-500 w-8">P95</span>
                    <div className="flex-1 h-2 bg-surface-muted rounded-full overflow-hidden">
                      <div
                        className="h-full bg-gradient-to-r from-accent-cyan to-accent-emerald rounded-full transition-all duration-500"
                        style={{ width: `${getLatencyBarWidth(metric.p95_ms)}%` }}
                      />
                    </div>
                    <span className={`text-xs font-mono w-16 text-right ${getLatencyColor(metric.p95_ms)}`}>
                      {metric.p95_ms.toFixed(1)}ms
                    </span>
                  </div>

                  {/* P99 */}
                  <div className="flex items-center gap-3">
                    <span className="text-xs text-gray-500 w-8">P99</span>
                    <div className="flex-1 h-2 bg-surface-muted rounded-full overflow-hidden">
                      <div
                        className="h-full bg-gradient-to-r from-accent-amber to-mira-500 rounded-full transition-all duration-500"
                        style={{ width: `${getLatencyBarWidth(metric.p99_ms)}%` }}
                      />
                    </div>
                    <span className={`text-xs font-mono w-16 text-right ${getLatencyColor(metric.p99_ms)}`}>
                      {metric.p99_ms.toFixed(1)}ms
                    </span>
                  </div>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Latency budget */}
      <div className="glass rounded-2xl p-6">
        <h3 className="font-medium text-white mb-4">Target Latency Budget</h3>
        <div className="grid grid-cols-2 gap-4 text-sm">
          <div className="p-3 bg-surface-raised rounded-lg">
            <p className="text-gray-500 mb-1">Ingest Pipeline</p>
            <p className="text-white font-medium">~200ms</p>
            <p className="text-xs text-gray-600 mt-1">STT + Chunk + Embed + Store</p>
          </div>
          <div className="p-3 bg-surface-raised rounded-lg">
            <p className="text-gray-500 mb-1">Query Pipeline</p>
            <p className="text-white font-medium">~400ms</p>
            <p className="text-xs text-gray-600 mt-1">Embed + Search + Context + LLM</p>
          </div>
        </div>
      </div>
    </div>
  )
}
