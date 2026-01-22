import { useState, useCallback } from 'react'
import { AudioRecorder } from './components/AudioRecorder'
import { TranscriptView } from './components/TranscriptView'
import { QueryInput } from './components/QueryInput'
import { ResponseStream } from './components/ResponseStream'
import { LatencyMetrics } from './components/LatencyMetrics'
import { MemoryList } from './components/MemoryList'
import { TextIngest } from './components/TextIngest'
import { Brain, Waves, MessageSquare, Activity } from 'lucide-react'

type Tab = 'record' | 'query' | 'memories' | 'metrics'

interface TranscriptEntry {
  id: string
  text: string
  timestamp: Date
  isFinal: boolean
}

interface QueryResponse {
  answer: string
  memories: Array<{
    id: string
    text: string
    score: number
  }>
  processingTime: number
}

function App() {
  const [activeTab, setActiveTab] = useState<Tab>('record')
  const [isRecording, setIsRecording] = useState(false)
  const [transcripts, setTranscripts] = useState<TranscriptEntry[]>([])
  const [currentResponse, setCurrentResponse] = useState<QueryResponse | null>(null)
  const [isQuerying, setIsQuerying] = useState(false)
  const [streamingText, setStreamingText] = useState('')

  const handleTranscript = useCallback((text: string, isFinal: boolean) => {
    const entry: TranscriptEntry = {
      id: `${Date.now()}-${Math.random().toString(36).slice(2)}`,
      text,
      timestamp: new Date(),
      isFinal,
    }

    setTranscripts(prev => {
      // If not final, update the last non-final entry
      if (!isFinal) {
        const lastEntry = prev[prev.length - 1]
        if (lastEntry && !lastEntry.isFinal) {
          return [...prev.slice(0, -1), entry]
        }
      }
      return [...prev, entry]
    })
  }, [])

  const handleQuery = useCallback(async (query: string) => {
    setIsQuerying(true)
    setStreamingText('')
    setCurrentResponse(null)

    try {
      const response = await fetch('/api/v1/memory/query', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          query,
          top_k: 5,
          include_context: true,
          stream: false,
        }),
      })

      if (!response.ok) throw new Error('Query failed')

      const data = await response.json()
      setCurrentResponse({
        answer: data.answer,
        memories: data.memories,
        processingTime: data.processing_time_ms,
      })
    } catch (error) {
      console.error('Query error:', error)
      setCurrentResponse({
        answer: 'Sorry, I encountered an error processing your query.',
        memories: [],
        processingTime: 0,
      })
    } finally {
      setIsQuerying(false)
    }
  }, [])

  const tabs = [
    { id: 'record' as Tab, label: 'Record', icon: Waves },
    { id: 'query' as Tab, label: 'Query', icon: MessageSquare },
    { id: 'memories' as Tab, label: 'Memories', icon: Brain },
    { id: 'metrics' as Tab, label: 'Metrics', icon: Activity },
  ]

  return (
    <div className="min-h-screen bg-surface bg-mesh noise">
      {/* Header */}
      <header className="border-b border-surface-muted">
        <div className="max-w-6xl mx-auto px-6 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-mira-500 to-accent-cyan flex items-center justify-center shadow-glow-sm">
                <Brain className="w-5 h-5 text-white" />
              </div>
              <div>
                <h1 className="text-xl font-semibold tracking-tight">
                  <span className="gradient-text">Mira</span>
                </h1>
                <p className="text-xs text-gray-500">Memory Engine</p>
              </div>
            </div>

            {/* Status indicator */}
            <div className="flex items-center gap-2 text-sm">
              <span className={`w-2 h-2 rounded-full ${isRecording ? 'bg-accent-rose recording-pulse' : 'bg-accent-emerald'}`} />
              <span className="text-gray-400">
                {isRecording ? 'Recording' : 'Ready'}
              </span>
            </div>
          </div>
        </div>
      </header>

      {/* Navigation */}
      <nav className="border-b border-surface-muted sticky top-0 bg-surface/80 backdrop-blur-xl z-10">
        <div className="max-w-6xl mx-auto px-6">
          <div className="flex gap-1">
            {tabs.map(({ id, label, icon: Icon }) => (
              <button
                key={id}
                onClick={() => setActiveTab(id)}
                className={`
                  flex items-center gap-2 px-4 py-3 text-sm font-medium transition-all duration-200
                  border-b-2 -mb-px
                  ${activeTab === id
                    ? 'text-mira-400 border-mira-500'
                    : 'text-gray-500 border-transparent hover:text-gray-300 hover:border-gray-700'
                  }
                `}
              >
                <Icon className="w-4 h-4" />
                {label}
              </button>
            ))}
          </div>
        </div>
      </nav>

      {/* Main Content */}
      <main className="max-w-6xl mx-auto px-6 py-8">
        {activeTab === 'record' && (
          <div className="space-y-8">
            <div className="grid lg:grid-cols-2 gap-8">
              <div className="space-y-6">
                <div>
                  <h2 className="text-lg font-medium text-white mb-2">Voice Recording</h2>
                  <p className="text-sm text-gray-500">
                    Record your thoughts and they'll be automatically transcribed and stored as memories.
                  </p>
                </div>
                <AudioRecorder
                  isRecording={isRecording}
                  onRecordingChange={setIsRecording}
                  onTranscript={handleTranscript}
                />
              </div>
              <div>
                <h2 className="text-lg font-medium text-white mb-4">Live Transcript</h2>
                <TranscriptView entries={transcripts} />
              </div>
            </div>
            
            {/* Divider */}
            <div className="flex items-center gap-4">
              <div className="flex-1 h-px bg-surface-muted" />
              <span className="text-sm text-gray-500">or</span>
              <div className="flex-1 h-px bg-surface-muted" />
            </div>
            
            {/* Text Input */}
            <div className="max-w-2xl mx-auto">
              <TextIngest />
            </div>
          </div>
        )}

        {activeTab === 'query' && (
          <div className="max-w-3xl mx-auto space-y-8">
            <div className="text-center">
              <h2 className="text-2xl font-semibold text-white mb-2">Ask Your Memories</h2>
              <p className="text-gray-500">
                Query your stored memories using natural language
              </p>
            </div>
            <QueryInput onQuery={handleQuery} isLoading={isQuerying} />
            <ResponseStream
              response={currentResponse}
              streamingText={streamingText}
              isLoading={isQuerying}
            />
          </div>
        )}

        {activeTab === 'memories' && (
          <div>
            <div className="mb-6">
              <h2 className="text-lg font-medium text-white mb-2">Stored Memories</h2>
              <p className="text-sm text-gray-500">
                Browse and manage your memory collection
              </p>
            </div>
            <MemoryList />
          </div>
        )}

        {activeTab === 'metrics' && (
          <div>
            <div className="mb-6">
              <h2 className="text-lg font-medium text-white mb-2">Performance Metrics</h2>
              <p className="text-sm text-gray-500">
                Monitor system performance and latency
              </p>
            </div>
            <LatencyMetrics />
          </div>
        )}
      </main>

      {/* Footer */}
      <footer className="border-t border-surface-muted mt-auto">
        <div className="max-w-6xl mx-auto px-6 py-4">
          <div className="flex items-center justify-between text-xs text-gray-600">
            <span>Mira Memory Engine v0.1.0</span>
            <span>Powered by Deepgram • OpenAI • Claude</span>
          </div>
        </div>
      </footer>
    </div>
  )
}

export default App
