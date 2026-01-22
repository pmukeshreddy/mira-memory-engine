import { useState, useCallback, useRef, useEffect } from 'react'

type MessageHandler = (data: unknown) => void

interface UseWebSocketOptions {
  url: string
  onMessage?: MessageHandler
  onOpen?: () => void
  onClose?: () => void
  onError?: (error: Event) => void
  reconnect?: boolean
  reconnectInterval?: number
  maxReconnectAttempts?: number
}

interface UseWebSocketReturn {
  isConnected: boolean
  connect: () => Promise<void>
  disconnect: () => void
  send: (data: unknown) => void
  lastMessage: unknown | null
}

export function useWebSocket({
  url,
  onMessage,
  onOpen,
  onClose,
  onError,
  reconnect = true,
  reconnectInterval = 3000,
  maxReconnectAttempts = 5,
}: UseWebSocketOptions): UseWebSocketReturn {
  const [isConnected, setIsConnected] = useState(false)
  const [lastMessage, setLastMessage] = useState<unknown | null>(null)
  
  const wsRef = useRef<WebSocket | null>(null)
  const reconnectAttemptsRef = useRef(0)
  const reconnectTimeoutRef = useRef<NodeJS.Timeout | null>(null)
  const shouldReconnectRef = useRef(reconnect)

  const clearReconnectTimeout = useCallback(() => {
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current)
      reconnectTimeoutRef.current = null
    }
  }, [])

  const scheduleReconnect = useCallback(() => {
    if (
      shouldReconnectRef.current &&
      reconnectAttemptsRef.current < maxReconnectAttempts
    ) {
      clearReconnectTimeout()
      reconnectTimeoutRef.current = setTimeout(() => {
        reconnectAttemptsRef.current += 1
        console.log(
          `WebSocket reconnecting... Attempt ${reconnectAttemptsRef.current}/${maxReconnectAttempts}`
        )
        connect()
      }, reconnectInterval)
    }
  }, [clearReconnectTimeout, maxReconnectAttempts, reconnectInterval])

  const connect = useCallback((): Promise<void> => {
    return new Promise((resolve, reject) => {
      // Close existing connection
      if (wsRef.current) {
        wsRef.current.close()
      }

      const ws = new WebSocket(url)

      ws.onopen = () => {
        console.log('WebSocket connected')
        setIsConnected(true)
        reconnectAttemptsRef.current = 0
        onOpen?.()
        resolve()
      }

      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data)
          setLastMessage(data)
          onMessage?.(data)
        } catch {
          // Handle non-JSON messages
          setLastMessage(event.data)
          onMessage?.(event.data)
        }
      }

      ws.onclose = () => {
        console.log('WebSocket disconnected')
        setIsConnected(false)
        wsRef.current = null
        onClose?.()
        scheduleReconnect()
      }

      ws.onerror = (error) => {
        console.error('WebSocket error:', error)
        onError?.(error)
        reject(error)
      }

      wsRef.current = ws
    })
  }, [url, onMessage, onOpen, onClose, onError, scheduleReconnect])

  const disconnect = useCallback(() => {
    shouldReconnectRef.current = false
    clearReconnectTimeout()

    if (wsRef.current) {
      wsRef.current.close()
      wsRef.current = null
    }
    setIsConnected(false)
  }, [clearReconnectTimeout])

  const send = useCallback((data: unknown) => {
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      const message = typeof data === 'string' ? data : JSON.stringify(data)
      wsRef.current.send(message)
    } else {
      console.warn('WebSocket not connected, cannot send message')
    }
  }, [])

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      shouldReconnectRef.current = false
      clearReconnectTimeout()
      if (wsRef.current) {
        wsRef.current.close()
      }
    }
  }, [clearReconnectTimeout])

  return {
    isConnected,
    connect,
    disconnect,
    send,
    lastMessage,
  }
}
