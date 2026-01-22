import { useState, useEffect, useCallback } from 'react'
import { Trash2, RefreshCw, Search, Calendar, Loader2, AlertCircle, Database } from 'lucide-react'

interface Memory {
  id: string
  text: string
  score: number
  metadata: {
    source?: string
    timestamp?: string
    session_id?: string
  }
}

export function MemoryList() {
  const [memories, setMemories] = useState<Memory[]>([])
  const [totalCount, setTotalCount] = useState(0)
  const [isLoading, setIsLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [searchQuery, setSearchQuery] = useState('')
  const [_isSearching, setIsSearching] = useState(false)

  const fetchMemories = useCallback(async () => {
    setIsLoading(true)
    setError(null)

    try {
      const response = await fetch('/api/v1/memory/recent?limit=50')
      if (!response.ok) throw new Error('Failed to fetch memories')

      const data = await response.json()
      setMemories(data.memories)
      setTotalCount(data.total_count)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load memories')
    } finally {
      setIsLoading(false)
    }
  }, [])

  const searchMemories = useCallback(async () => {
    if (!searchQuery.trim()) {
      fetchMemories()
      return
    }

    setIsSearching(true)
    setError(null)

    try {
      const response = await fetch(
        `/api/v1/memory/search?query=${encodeURIComponent(searchQuery)}&top_k=20`,
        { method: 'POST' }
      )
      if (!response.ok) throw new Error('Search failed')

      const data = await response.json()
      setMemories(data)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Search failed')
    } finally {
      setIsSearching(false)
    }
  }, [searchQuery, fetchMemories])

  const clearMemories = useCallback(async () => {
    if (!confirm('Are you sure you want to clear all memories? This cannot be undone.')) {
      return
    }

    try {
      const response = await fetch('/api/v1/memory/clear', { method: 'DELETE' })
      if (!response.ok) throw new Error('Failed to clear memories')

      setMemories([])
      setTotalCount(0)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to clear memories')
    }
  }, [])

  useEffect(() => {
    fetchMemories()
  }, [fetchMemories])

  const formatDate = (dateStr: string) => {
    try {
      return new Date(dateStr).toLocaleDateString('en-US', {
        month: 'short',
        day: 'numeric',
        hour: '2-digit',
        minute: '2-digit',
      })
    } catch {
      return dateStr
    }
  }

  return (
    <div className="space-y-6">
      {/* Header controls */}
      <div className="flex flex-col sm:flex-row gap-4 items-start sm:items-center justify-between">
        {/* Search */}
        <div className="relative flex-1 max-w-md">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-500" />
          <input
            type="text"
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            onKeyDown={(e) => e.key === 'Enter' && searchMemories()}
            placeholder="Search memories..."
            className="input-field pl-10 pr-4"
          />
        </div>

        {/* Actions */}
        <div className="flex items-center gap-2">
          <button
            onClick={fetchMemories}
            disabled={isLoading}
            className="flex items-center gap-2 px-3 py-2 text-sm text-gray-400 hover:text-white bg-surface-raised hover:bg-surface-muted border border-surface-muted rounded-lg transition-all"
          >
            <RefreshCw className={`w-4 h-4 ${isLoading ? 'animate-spin' : ''}`} />
            <span>Refresh</span>
          </button>
          <button
            onClick={clearMemories}
            className="flex items-center gap-2 px-3 py-2 text-sm text-accent-rose hover:bg-accent-rose/10 border border-accent-rose/20 rounded-lg transition-all"
          >
            <Trash2 className="w-4 h-4" />
            <span>Clear All</span>
          </button>
        </div>
      </div>

      {/* Stats */}
      <div className="flex items-center gap-4 text-sm text-gray-500">
        <div className="flex items-center gap-2">
          <Database className="w-4 h-4" />
          <span>{totalCount} total memories</span>
        </div>
        {memories.length > 0 && memories.length !== totalCount && (
          <span>• Showing {memories.length}</span>
        )}
      </div>

      {/* Error state */}
      {error && (
        <div className="flex items-center gap-3 px-4 py-3 bg-accent-rose/10 border border-accent-rose/20 rounded-xl">
          <AlertCircle className="w-5 h-5 text-accent-rose flex-shrink-0" />
          <p className="text-sm text-accent-rose">{error}</p>
        </div>
      )}

      {/* Loading state */}
      {isLoading && (
        <div className="flex items-center justify-center py-12">
          <Loader2 className="w-8 h-8 text-mira-400 animate-spin" />
        </div>
      )}

      {/* Empty state */}
      {!isLoading && memories.length === 0 && (
        <div className="glass rounded-2xl p-12 text-center">
          <div className="w-16 h-16 rounded-full bg-surface-muted flex items-center justify-center mx-auto mb-4">
            <Database className="w-8 h-8 text-gray-600" />
          </div>
          <h3 className="text-lg font-medium text-white mb-2">No memories yet</h3>
          <p className="text-gray-500 max-w-md mx-auto">
            Start recording or ingesting text to build your memory collection.
          </p>
        </div>
      )}

      {/* Memory list */}
      {!isLoading && memories.length > 0 && (
        <div className="grid gap-4">
          {memories.map((memory) => (
            <div key={memory.id} className="memory-card">
              <div className="flex items-start justify-between gap-4 mb-3">
                <div className="flex items-center gap-2 text-xs text-gray-500">
                  {memory.metadata?.timestamp && (
                    <>
                      <Calendar className="w-3.5 h-3.5" />
                      <span>{formatDate(memory.metadata.timestamp)}</span>
                    </>
                  )}
                  {memory.metadata?.source && (
                    <>
                      <span>•</span>
                      <span className="capitalize">{memory.metadata.source}</span>
                    </>
                  )}
                </div>
                {memory.score > 0 && (
                  <span
                    className={`
                      text-xs px-2 py-0.5 rounded-full
                      ${memory.score >= 0.8
                        ? 'bg-accent-emerald/20 text-accent-emerald'
                        : memory.score >= 0.6
                        ? 'bg-accent-amber/20 text-accent-amber'
                        : 'bg-gray-500/20 text-gray-400'
                      }
                    `}
                  >
                    {(memory.score * 100).toFixed(0)}%
                  </span>
                )}
              </div>
              <p className="text-sm text-gray-300 leading-relaxed">
                {memory.text}
              </p>
              <div className="mt-3 pt-3 border-t border-surface-muted">
                <span className="text-xs font-mono text-gray-600 truncate block">
                  ID: {memory.id}
                </span>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}
