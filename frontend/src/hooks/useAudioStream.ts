import { useState, useCallback, useRef } from 'react'
import { useWebSocket } from './useWebSocket'

interface UseAudioStreamOptions {
  onTranscript: (text: string, isFinal: boolean) => void
  onError?: (error: string) => void
  onIngested?: (chunkCount: number, memoryIds: string[]) => void
}

interface UseAudioStreamReturn {
  isConnected: boolean
  isReady: boolean
  connect: () => Promise<void>
  disconnect: () => void
  sendAudio: (base64Data: string) => void
}

export function useAudioStream({
  onTranscript,
  onError,
  onIngested,
}: UseAudioStreamOptions): UseAudioStreamReturn {
  const [isReady, setIsReady] = useState(false)
  const pendingConnectRef = useRef<{
    resolve: () => void
    reject: (error: Error) => void
  } | null>(null)

  const handleMessage = useCallback(
    (data: unknown) => {
      const message = data as Record<string, unknown>
      console.log('Audio WebSocket message received:', message.type, message)

      switch (message.type) {
        case 'ready':
          console.log('Ready message received, setting isReady=true')
          setIsReady(true)
          pendingConnectRef.current?.resolve()
          pendingConnectRef.current = null
          break

        case 'transcript':
          onTranscript(
            message.text as string,
            message.is_final as boolean
          )
          break

        case 'ingested':
          onIngested?.(
            message.chunks as number,
            message.memory_ids as string[]
          )
          break

        case 'error':
          onError?.(message.error as string)
          break
      }
    },
    [onTranscript, onError, onIngested]
  )

  const handleError = useCallback(
    (event: Event) => {
      console.error('Audio WebSocket error:', event)
      onError?.('WebSocket connection error')
      pendingConnectRef.current?.reject(new Error('WebSocket error'))
      pendingConnectRef.current = null
    },
    [onError]
  )

  const handleClose = useCallback(() => {
    setIsReady(false)
    // Reject any pending connection
    if (pendingConnectRef.current) {
      pendingConnectRef.current.reject(new Error('Connection closed'))
      pendingConnectRef.current = null
    }
  }, [])

  // Use environment variable for production WebSocket URL
  const getWsUrl = () => {
    const wsBase = import.meta.env.VITE_WS_URL
    if (wsBase) {
      return `${wsBase}/ws/audio`
    }
    // Fallback for development (uses Vite proxy)
    if (typeof window !== 'undefined') {
      return `${window.location.protocol === 'https:' ? 'wss:' : 'ws:'}//${window.location.host}/ws/audio`
    }
    return 'ws://localhost:8000/ws/audio'
  }
  
  const wsUrl = getWsUrl()

  const { isConnected, connect: wsConnect, disconnect, send } = useWebSocket({
    url: wsUrl,
    onMessage: handleMessage,
    onError: handleError,
    onClose: handleClose,
    reconnect: false, // Manual control for audio streaming
  })

  const connect = useCallback((): Promise<void> => {
    // Guard against multiple connection attempts
    if (pendingConnectRef.current) {
      console.log('Connection already in progress')
      return new Promise((resolve, reject) => {
        // Chain to existing pending connection
        const existing = pendingConnectRef.current!
        pendingConnectRef.current = {
          resolve: () => { existing.resolve(); resolve() },
          reject: (e) => { existing.reject(e); reject(e) },
        }
      })
    }

    return new Promise((resolve, reject) => {
      pendingConnectRef.current = { resolve, reject }
      wsConnect().catch(reject)

      // Timeout for ready message
      setTimeout(() => {
        if (pendingConnectRef.current) {
          pendingConnectRef.current.reject(new Error('Connection timeout'))
          pendingConnectRef.current = null
        }
      }, 10000)
    })
  }, [wsConnect])

  const sendAudio = useCallback(
    (base64Data: string) => {
      if (isConnected && isReady) {
        send({
          type: 'audio',
          data: base64Data,
          sample_rate: 16000,
          encoding: 'linear16',
        })
      } else {
        console.log('sendAudio skipped: isConnected=', isConnected, 'isReady=', isReady)
      }
    },
    [isConnected, isReady, send]
  )

  return {
    isConnected,
    isReady,
    connect,
    disconnect,
    sendAudio,
  }
}
