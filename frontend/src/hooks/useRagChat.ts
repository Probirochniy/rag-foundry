import { useState, useRef, useEffect } from 'react'
import { fetchEventSource } from '@microsoft/fetch-event-source'
import type { Message } from '../types'

const API_BASE = import.meta.env.VITE_API_BASE_URL

export function useRagChat() {
    const [messages, setMessages] = useState<Message[]>([])
    const [input, setInput] = useState('')
    const [topK, setTopK] = useState(3)
    const [isStreamingEnabled, setIsStreamingEnabled] = useState(true)
    const [isLoading, setIsLoading] = useState(false)
    const messagesEndRef = useRef<HTMLDivElement>(null)

    useEffect(() => {
        messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
    }, [messages])

    const sendMessage = async (e: React.FormEvent) => {
        e.preventDefault()

        if (!input.trim() || isLoading) {
            return
        }

        const userText = input.trim()
        setInput('')

        const userMsgId = crypto.randomUUID()
        const assistantMsgId = crypto.randomUUID()

        setMessages((prev) => [
            ...prev,
            {
                id: userMsgId,
                role: 'user',
                content: userText,
            },
            {
                id: assistantMsgId,
                role: 'assistant',
                content: '',
                isStreaming: isStreamingEnabled,
            },
        ])

        setIsLoading(true)

        const updateAssistant = (update: Partial<Message>) => {
            setMessages((prev) =>
                prev.map((msg) =>
                    msg.id === assistantMsgId
                        ? { ...msg, ...update }
                        : msg
                )
            )
        }

        if (!isStreamingEnabled) {
            try {
                const res = await fetch(`${API_BASE}/rag/query`, {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        query: userText,
                        top_k: topK,
                    }),
                })

                if (!res.ok) {
                    throw new Error(`HTTP error ${res.status}`)
                }

                const data = await res.json()
                updateAssistant({
                    content: data.answer,
                    isStreaming: false,
                })
            } catch (err) {
                console.error('RAG sync query failed:', err)
                updateAssistant({
                    content: '❌ Server does not want to talk to you. It does be like that sometimes...',
                    isStreaming: false,
                })
            } finally {
                setIsLoading(false)
            }
            return
        }

        const ctrl = new AbortController()

        try {
            await fetchEventSource(`${API_BASE}/rag/stream`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    query: userText,
                    top_k: topK,
                }),
                signal: ctrl.signal,

                onmessage(ev) {
                    if (ev.event === 'reset') {
                        updateAssistant({
                            content: '',
                            wasReset: true,
                        })
                        return
                    }

                    if (ev.event === 'delta') {
                        const parsed = JSON.parse(ev.data)

                        setMessages((prev) =>
                            prev.map((msg) =>
                                msg.id === assistantMsgId
                                    ? {
                                        ...msg,
                                        content:
                                            msg.content +
                                            (parsed.content || ''),
                                    }
                                    : msg
                            )
                        )
                        return
                    }

                    if (ev.event === 'done') {
                        updateAssistant({
                            isStreaming: false,
                        })
                    }
                },

                onerror(err) {
                    console.error('SSE Error:', err)
                    throw err
                },
            })

            updateAssistant({
                isStreaming: false,
            })
        } catch (err) {
            console.error('RAG stream failed:', err)
            updateAssistant({
                content: '❌ Server does not want to stream. Well...',
                isStreaming: false,
            })
        } finally {
            ctrl.abort()
            setIsLoading(false)
            setMessages((prev) =>
                prev.map((msg) =>
                    msg.id === assistantMsgId
                        ? { ...msg, isStreaming: false }
                        : msg
                )
            )
        }
    }

    return {
        messages,
        input,
        setInput,
        topK,
        setTopK,
        isStreamingEnabled,
        setIsStreamingEnabled,
        isLoading,
        messagesEndRef,
        sendMessage,
    }
}
