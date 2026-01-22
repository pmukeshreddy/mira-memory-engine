import { useEffect, useRef } from 'react'
import { Clock } from 'lucide-react'

interface TranscriptEntry {
  id: string
  text: string
  timestamp: Date
  isFinal: boolean
}

interface TranscriptViewProps {
  entries: TranscriptEntry[]
}

export function TranscriptView({ entries }: TranscriptViewProps) {
  const containerRef = useRef<HTMLDivElement>(null)

  // Auto-scroll to bottom on new entries
  useEffect(() => {
    if (containerRef.current) {
      containerRef.current.scrollTop = containerRef.current.scrollHeight
    }
  }, [entries])

  const formatTime = (date: Date) => {
    return date.toLocaleTimeString('en-US', {
      hour: '2-digit',
      minute: '2-digit',
      second: '2-digit',
    })
  }

  if (entries.length === 0) {
    return (
      <div className="glass rounded-2xl p-8 h-[400px] flex items-center justify-center">
        <div className="text-center">
          <div className="w-16 h-16 rounded-full bg-surface-muted flex items-center justify-center mx-auto mb-4">
            <Clock className="w-8 h-8 text-gray-600" />
          </div>
          <p className="text-gray-500">
            Transcripts will appear here as you speak
          </p>
        </div>
      </div>
    )
  }

  return (
    <div
      ref={containerRef}
      className="glass rounded-2xl p-4 h-[400px] overflow-y-auto space-y-3"
    >
      {entries.map((entry) => (
        <div
          key={entry.id}
          className={`
            p-4 rounded-xl transition-all duration-300
            ${entry.isFinal
              ? 'bg-surface-raised border border-surface-muted'
              : 'bg-mira-900/20 border border-mira-500/20'
            }
          `}
        >
          <div className="flex items-start justify-between gap-4 mb-2">
            <span className="text-xs text-gray-500 font-mono">
              {formatTime(entry.timestamp)}
            </span>
            {!entry.isFinal && (
              <span className="text-xs text-mira-400 flex items-center gap-1">
                <span className="w-1.5 h-1.5 rounded-full bg-mira-400 animate-pulse" />
                Listening...
              </span>
            )}
          </div>
          <p className={`text-sm leading-relaxed ${entry.isFinal ? 'text-white' : 'text-gray-400'}`}>
            {entry.text}
          </p>
        </div>
      ))}
      
      {/* Typing indicator */}
      {entries.length > 0 && !entries[entries.length - 1].isFinal && (
        <div className="flex items-center gap-1.5 px-4 py-2">
          <div className="typing-dot" />
          <div className="typing-dot" />
          <div className="typing-dot" />
        </div>
      )}
    </div>
  )
}
