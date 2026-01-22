import { useState, useCallback, FormEvent } from 'react'
import { Send, Loader2, FileText, Check } from 'lucide-react'
import { API_BASE } from '../services/api'

interface TextIngestProps {
  onIngested?: (chunks: number) => void
}

export function TextIngest({ onIngested }: TextIngestProps) {
  const [text, setText] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const [success, setSuccess] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)

  const handleSubmit = useCallback(async (e: FormEvent) => {
    e.preventDefault()
    if (!text.trim() || isLoading) return

    setIsLoading(true)
    setError(null)
    setSuccess(null)

    try {
      const response = await fetch(`${API_BASE}/memory/ingest`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          text: text.trim(),
          source: 'manual',
        }),
      })

      if (!response.ok) throw new Error('Failed to ingest text')

      const data = await response.json()
      setSuccess(`Added ${data.chunks_created} memory chunk(s)`)
      setText('')
      onIngested?.(data.chunks_created)

      // Clear success message after 3 seconds
      setTimeout(() => setSuccess(null), 3000)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to add memory')
    } finally {
      setIsLoading(false)
    }
  }, [text, isLoading, onIngested])

  return (
    <div className="glass rounded-2xl p-6">
      <div className="flex items-center gap-3 mb-4">
        <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-mira-600 to-mira-800 flex items-center justify-center">
          <FileText className="w-5 h-5 text-white" />
        </div>
        <div>
          <h3 className="text-lg font-semibold text-white">Add Text Memory</h3>
          <p className="text-sm text-gray-500">Type or paste text to store as memory</p>
        </div>
      </div>

      <form onSubmit={handleSubmit} className="space-y-4">
        <textarea
          value={text}
          onChange={(e) => setText(e.target.value)}
          placeholder="Enter your thoughts, notes, or any text you want to remember..."
          disabled={isLoading}
          rows={4}
          className="
            w-full bg-surface-raised border border-surface-muted rounded-xl
            px-4 py-3 text-white placeholder-gray-500
            resize-none focus:outline-none focus:ring-2 focus:ring-mira-500/30
            disabled:opacity-50
          "
        />

        <div className="flex items-center justify-between">
          <div className="text-sm text-gray-500">
            {text.length > 0 && `${text.split(/\s+/).filter(Boolean).length} words`}
          </div>
          
          <button
            type="submit"
            disabled={!text.trim() || isLoading}
            className={`
              flex items-center gap-2 px-5 py-2.5 rounded-xl font-medium text-sm
              transition-all duration-200
              ${text.trim() && !isLoading
                ? 'bg-gradient-to-r from-mira-600 to-mira-700 text-white hover:from-mira-500 hover:to-mira-600 shadow-glow-sm'
                : 'bg-surface-muted text-gray-500 cursor-not-allowed'
              }
            `}
          >
            {isLoading ? (
              <>
                <Loader2 className="w-4 h-4 animate-spin" />
                <span>Adding...</span>
              </>
            ) : (
              <>
                <Send className="w-4 h-4" />
                <span>Add to Memory</span>
              </>
            )}
          </button>
        </div>

        {/* Success message */}
        {success && (
          <div className="flex items-center gap-2 px-4 py-2 bg-accent-emerald/10 border border-accent-emerald/20 rounded-lg">
            <Check className="w-4 h-4 text-accent-emerald" />
            <span className="text-sm text-accent-emerald">{success}</span>
          </div>
        )}

        {/* Error message */}
        {error && (
          <div className="px-4 py-2 bg-accent-rose/10 border border-accent-rose/20 rounded-lg">
            <p className="text-sm text-accent-rose">{error}</p>
          </div>
        )}
      </form>
    </div>
  )
}
