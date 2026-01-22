import { Brain, Clock, Zap, ChevronDown, ChevronUp } from 'lucide-react'
import { useState } from 'react'

interface Memory {
  id: string
  text: string
  score: number
}

interface QueryResponse {
  answer: string
  memories: Memory[]
  processingTime: number
}

interface ResponseStreamProps {
  response: QueryResponse | null
  streamingText: string
  isLoading: boolean
}

export function ResponseStream({
  response,
  streamingText,
  isLoading,
}: ResponseStreamProps) {
  const [showMemories, setShowMemories] = useState(false)

  // Show loading state
  if (isLoading && !streamingText) {
    return (
      <div className="glass rounded-2xl p-8">
        <div className="flex items-start gap-4">
          <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-mira-500 to-accent-cyan flex items-center justify-center flex-shrink-0">
            <Brain className="w-5 h-5 text-white animate-pulse" />
          </div>
          <div className="flex-1 space-y-3">
            <div className="flex items-center gap-2">
              <span className="text-sm font-medium text-white">Mira</span>
              <span className="text-xs text-gray-500">is thinking...</span>
            </div>
            <div className="space-y-2">
              <div className="h-4 bg-surface-muted rounded animate-pulse w-3/4" />
              <div className="h-4 bg-surface-muted rounded animate-pulse w-1/2" />
              <div className="h-4 bg-surface-muted rounded animate-pulse w-2/3" />
            </div>
          </div>
        </div>
      </div>
    )
  }

  // Show nothing if no response
  if (!response && !streamingText) {
    return null
  }

  const displayText = streamingText || response?.answer || ''
  const memories = response?.memories || []
  const processingTime = response?.processingTime || 0

  return (
    <div className="space-y-4">
      {/* Main response */}
      <div className="glass rounded-2xl p-6">
        <div className="flex items-start gap-4">
          <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-mira-500 to-accent-cyan flex items-center justify-center flex-shrink-0">
            <Brain className="w-5 h-5 text-white" />
          </div>
          <div className="flex-1 min-w-0">
            <div className="flex items-center gap-2 mb-3">
              <span className="text-sm font-medium text-white">Mira</span>
              {processingTime > 0 && (
                <span className="flex items-center gap-1 text-xs text-gray-500">
                  <Zap className="w-3 h-3" />
                  {processingTime.toFixed(0)}ms
                </span>
              )}
            </div>
            <div className="prose prose-invert prose-sm max-w-none">
              <p className="text-gray-300 leading-relaxed whitespace-pre-wrap">
                {displayText}
                {isLoading && (
                  <span className="inline-block w-2 h-4 ml-1 bg-mira-400 animate-pulse" />
                )}
              </p>
            </div>
          </div>
        </div>
      </div>

      {/* Retrieved memories */}
      {memories.length > 0 && (
        <div className="glass rounded-xl overflow-hidden">
          <button
            onClick={() => setShowMemories(!showMemories)}
            className="w-full px-4 py-3 flex items-center justify-between text-sm hover:bg-surface-raised transition-colors"
          >
            <div className="flex items-center gap-2 text-gray-400">
              <Clock className="w-4 h-4" />
              <span>Retrieved {memories.length} memories</span>
            </div>
            {showMemories ? (
              <ChevronUp className="w-4 h-4 text-gray-500" />
            ) : (
              <ChevronDown className="w-4 h-4 text-gray-500" />
            )}
          </button>

          {showMemories && (
            <div className="border-t border-surface-muted divide-y divide-surface-muted">
              {memories.map((memory, index) => (
                <div key={memory.id} className="p-4">
                  <div className="flex items-center justify-between mb-2">
                    <span className="text-xs font-mono text-gray-500">
                      Memory #{index + 1}
                    </span>
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
                      {(memory.score * 100).toFixed(0)}% match
                    </span>
                  </div>
                  <p className="text-sm text-gray-400 line-clamp-3">
                    {memory.text}
                  </p>
                </div>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  )
}
