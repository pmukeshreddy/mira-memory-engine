import { useState, useRef, useCallback, useEffect } from 'react'
import { Mic, Square, Loader2 } from 'lucide-react'
import { useAudioStream } from '../hooks/useAudioStream'

interface AudioRecorderProps {
  isRecording: boolean
  onRecordingChange: (recording: boolean) => void
  onTranscript: (text: string, isFinal: boolean) => void
}

export function AudioRecorder({
  isRecording,
  onRecordingChange,
  onTranscript,
}: AudioRecorderProps) {
  const [audioLevel, setAudioLevel] = useState(0)
  const [error, setError] = useState<string | null>(null)
  
  // Refs for audio resources that need cleanup
  const analyserRef = useRef<AnalyserNode | null>(null)
  const animationRef = useRef<number | null>(null)
  const audioContextRef = useRef<AudioContext | null>(null)
  const streamRef = useRef<MediaStream | null>(null)
  const processorRef = useRef<ScriptProcessorNode | null>(null)
  const isReadyRef = useRef(false)
  const sendAudioRef = useRef<((data: string) => void) | null>(null)

  const { connect, disconnect, isConnected, isReady, sendAudio } = useAudioStream({
    onTranscript: (text, isFinal) => {
      onTranscript(text, isFinal)
    },
    onError: (err) => {
      setError(err)
      stopRecording()
    },
  })

  // Keep refs in sync with latest values
  useEffect(() => {
    isReadyRef.current = isReady
  }, [isReady])
  
  useEffect(() => {
    sendAudioRef.current = sendAudio
  }, [sendAudio])

  // Audio level visualization
  const updateAudioLevel = useCallback(() => {
    if (analyserRef.current) {
      const dataArray = new Uint8Array(analyserRef.current.frequencyBinCount)
      analyserRef.current.getByteFrequencyData(dataArray)
      
      // Calculate average level
      const average = dataArray.reduce((a, b) => a + b, 0) / dataArray.length
      setAudioLevel(average / 255)
    }
    
    if (isRecording) {
      animationRef.current = requestAnimationFrame(updateAudioLevel)
    }
  }, [isRecording])

  const startRecording = useCallback(async () => {
    try {
      setError(null)
      
      // Request microphone access
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          sampleRate: 16000,
          channelCount: 1,
          echoCancellation: true,
          noiseSuppression: true,
        },
      })
      streamRef.current = stream

      // Set up audio context
      const audioContext = new AudioContext({ sampleRate: 16000 })
      audioContextRef.current = audioContext
      
      const source = audioContext.createMediaStreamSource(stream)
      const analyser = audioContext.createAnalyser()
      analyser.fftSize = 256
      source.connect(analyser)
      analyserRef.current = analyser

      // Connect to WebSocket first
      await connect()

      // Set up audio processing for sending to server
      // Use smaller buffer (2048) to reduce latency
      const processor = audioContext.createScriptProcessor(2048, 1, 1)
      processorRef.current = processor
      
      // Get actual sample rate (browsers may not support 16kHz)
      const actualSampleRate = audioContext.sampleRate
      const targetSampleRate = 16000
      const ratio = actualSampleRate / targetSampleRate
      console.log('Audio sample rate:', actualSampleRate, '-> target:', targetSampleRate, 'ratio:', ratio)
      
      processor.onaudioprocess = (e) => {
        // Use refs to get current state (avoid stale closures)
        if (isReadyRef.current && sendAudioRef.current) {
          const inputData = e.inputBuffer.getChannelData(0)
          
          // Downsample to 16kHz
          let processedData: Float32Array
          if (ratio > 1) {
            const newLength = Math.floor(inputData.length / ratio)
            processedData = new Float32Array(newLength)
            for (let i = 0; i < newLength; i++) {
              // Simple downsampling - take every Nth sample
              processedData[i] = inputData[Math.floor(i * ratio)]
            }
          } else {
            processedData = inputData
          }
          
          // Convert float32 to int16 PCM
          const int16Data = new Int16Array(processedData.length)
          for (let i = 0; i < processedData.length; i++) {
            const s = Math.max(-1, Math.min(1, processedData[i]))
            int16Data[i] = s < 0 ? s * 0x8000 : s * 0x7FFF
          }
          
          // Convert to base64
          const base64 = btoa(
            String.fromCharCode(...new Uint8Array(int16Data.buffer))
          )
          sendAudioRef.current(base64)
        }
      }

      source.connect(processor)
      // Connect to a silent destination (required for ScriptProcessor to work)
      // Create a gain node set to 0 to prevent playback/feedback
      const silentGain = audioContext.createGain()
      silentGain.gain.value = 0
      processor.connect(silentGain)
      silentGain.connect(audioContext.destination)

      onRecordingChange(true)
      updateAudioLevel()

    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to start recording')
    }
  }, [connect, onRecordingChange, updateAudioLevel])

  const stopRecording = useCallback(() => {
    console.log('stopRecording called')
    
    // Set recording to false FIRST to update UI immediately
    onRecordingChange(false)
    setAudioLevel(0)
    isReadyRef.current = false
    
    // Then cleanup resources (errors here won't block UI update)
    try {
      disconnect()
    } catch (e) {
      console.error('Error disconnecting WebSocket:', e)
    }
    
    // Stop animation
    if (animationRef.current) {
      cancelAnimationFrame(animationRef.current)
      animationRef.current = null
    }
    
    // Disconnect and cleanup processor
    try {
      if (processorRef.current) {
        processorRef.current.disconnect()
        processorRef.current = null
      }
    } catch (e) {
      console.error('Error disconnecting processor:', e)
    }
    
    // Close audio context
    try {
      if (audioContextRef.current && audioContextRef.current.state !== 'closed') {
        audioContextRef.current.close()
        audioContextRef.current = null
      }
    } catch (e) {
      console.error('Error closing audio context:', e)
    }
    
    // Stop all media tracks
    try {
      if (streamRef.current) {
        streamRef.current.getTracks().forEach(track => track.stop())
        streamRef.current = null
      }
    } catch (e) {
      console.error('Error stopping media tracks:', e)
    }
    
    analyserRef.current = null
  }, [disconnect, onRecordingChange])

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current)
      }
      if (processorRef.current) {
        processorRef.current.disconnect()
      }
      if (audioContextRef.current && audioContextRef.current.state !== 'closed') {
        audioContextRef.current.close()
      }
      if (streamRef.current) {
        streamRef.current.getTracks().forEach(track => track.stop())
      }
    }
  }, [])

  // Audio level rings for visualization
  const rings = [0.3, 0.5, 0.7, 0.9]

  return (
    <div className="glass rounded-2xl p-8">
      <div className="flex flex-col items-center">
        {/* Audio visualization */}
        <div className="relative w-48 h-48 mb-8">
          {/* Animated rings */}
          {isRecording && rings.map((ring, i) => (
            <div
              key={i}
              className="absolute inset-0 rounded-full border-2 border-mira-500/30 pointer-events-none"
              style={{
                transform: `scale(${1 + audioLevel * ring})`,
                opacity: 1 - ring,
                transition: 'transform 0.1s ease-out',
              }}
            />
          ))}
          
          {/* Main button */}
          <button
            onClick={isRecording ? stopRecording : startRecording}
            className={`
              absolute inset-0 m-auto w-32 h-32 rounded-full 
              flex items-center justify-center
              transition-all duration-300 transform z-10
              ${isRecording
                ? 'bg-gradient-to-br from-accent-rose to-accent-amber shadow-[0_0_60px_rgba(244,63,94,0.4)] scale-110'
                : 'bg-gradient-to-br from-mira-600 to-mira-800 hover:from-mira-500 hover:to-mira-700 shadow-glow hover:scale-105'
              }
            `}
          >
            {isRecording ? (
              <Square className="w-10 h-10 text-white" fill="white" />
            ) : (
              <Mic className="w-10 h-10 text-white" />
            )}
          </button>

          {/* Pulsing ring when recording */}
          {isRecording && (
            <div className="absolute inset-0 rounded-full border-4 border-accent-rose/50 animate-ping pointer-events-none" />
          )}
        </div>

        {/* Status text */}
        <div className="text-center">
          <p className="text-lg font-medium text-white mb-1">
            {isRecording ? 'Recording...' : 'Ready to Record'}
          </p>
          <p className="text-sm text-gray-500">
            {isRecording
              ? 'Tap to stop recording'
              : 'Tap the microphone to start'}
          </p>
        </div>

        {/* Connection status */}
        {isRecording && (
          <div className="mt-4 flex items-center gap-2 text-sm">
            {isReady ? (
              <>
                <span className="w-2 h-2 rounded-full bg-accent-emerald" />
                <span className="text-gray-400">Connected & Ready</span>
              </>
            ) : isConnected ? (
              <>
                <Loader2 className="w-4 h-4 animate-spin text-mira-400" />
                <span className="text-gray-400">Initializing...</span>
              </>
            ) : (
              <>
                <Loader2 className="w-4 h-4 animate-spin text-mira-400" />
                <span className="text-gray-400">Connecting...</span>
              </>
            )}
          </div>
        )}

        {/* Error message */}
        {error && (
          <div className="mt-4 px-4 py-2 bg-accent-rose/10 border border-accent-rose/20 rounded-lg">
            <p className="text-sm text-accent-rose">{error}</p>
          </div>
        )}
      </div>
    </div>
  )
}
