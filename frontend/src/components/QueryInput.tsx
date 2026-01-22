import { useState, useCallback, FormEvent, KeyboardEvent } from 'react'
import { Send, Loader2, Sparkles } from 'lucide-react'

interface QueryInputProps {
  onQuery: (query: string) => void
  isLoading: boolean
}

const EXAMPLE_QUERIES = [
  "What did I discuss in my last meeting?",
  "Summarize my thoughts about the project",
  "What were the key decisions made today?",
  "Remind me about the action items",
]

export function QueryInput({ onQuery, isLoading }: QueryInputProps) {
  const [query, setQuery] = useState('')

  const handleSubmit = useCallback((e: FormEvent) => {
    e.preventDefault()
    if (query.trim() && !isLoading) {
      onQuery(query.trim())
      setQuery('')
    }
  }, [query, isLoading, onQuery])

  const handleKeyDown = useCallback((e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSubmit(e as unknown as FormEvent)
    }
  }, [handleSubmit])

  const handleExampleClick = useCallback((example: string) => {
    setQuery(example)
  }, [])

  return (
    <div className="space-y-4">
      {/* Query input */}
      <form onSubmit={handleSubmit} className="relative">
        <div className="glass rounded-2xl overflow-hidden focus-within:ring-2 focus-within:ring-mira-500/30 transition-all duration-200">
          <textarea
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Ask a question about your memories..."
            disabled={isLoading}
            rows={3}
            className="
              w-full bg-transparent px-6 py-4 text-white placeholder-gray-500
              resize-none focus:outline-none
            "
          />
          
          {/* Bottom bar */}
          <div className="flex items-center justify-between px-4 py-3 border-t border-surface-muted bg-surface-raised/50">
            <div className="flex items-center gap-2 text-xs text-gray-500">
              <Sparkles className="w-3.5 h-3.5" />
              <span>Powered by Claude</span>
            </div>
            
            <button
              type="submit"
              disabled={!query.trim() || isLoading}
              className={`
                flex items-center gap-2 px-4 py-2 rounded-xl font-medium text-sm
                transition-all duration-200
                ${query.trim() && !isLoading
                  ? 'bg-gradient-to-r from-mira-600 to-mira-700 text-white hover:from-mira-500 hover:to-mira-600 shadow-glow-sm'
                  : 'bg-surface-muted text-gray-500 cursor-not-allowed'
                }
              `}
            >
              {isLoading ? (
                <>
                  <Loader2 className="w-4 h-4 animate-spin" />
                  <span>Thinking...</span>
                </>
              ) : (
                <>
                  <Send className="w-4 h-4" />
                  <span>Send</span>
                </>
              )}
            </button>
          </div>
        </div>
      </form>

      {/* Example queries */}
      <div className="space-y-2">
        <p className="text-xs text-gray-500 px-1">Try asking:</p>
        <div className="flex flex-wrap gap-2">
          {EXAMPLE_QUERIES.map((example, i) => (
            <button
              key={i}
              onClick={() => handleExampleClick(example)}
              disabled={isLoading}
              className="
                px-3 py-1.5 text-xs text-gray-400 
                bg-surface-raised hover:bg-surface-muted
                border border-surface-muted hover:border-mira-500/30
                rounded-full transition-all duration-200
                disabled:opacity-50 disabled:cursor-not-allowed
              "
            >
              {example}
            </button>
          ))}
        </div>
      </div>
    </div>
  )
}
