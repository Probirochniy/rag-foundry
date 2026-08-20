import { Send, Database, AlertTriangle, RefreshCw } from 'lucide-react'
import type { Message } from '../types'

interface ChatViewProps {
  messages: Message[]
  input: string
  setInput: (val: string) => void
  isLoading: boolean
  sendMessage: (e: React.FormEvent) => void
  messagesEndRef: React.RefObject<HTMLDivElement | null>
}

export function ChatView({
  messages,
  input,
  setInput,
  isLoading,
  sendMessage,
  messagesEndRef,
}: ChatViewProps) {
  return (
    <div className="flex-1 flex flex-col h-full max-w-4xl w-full mx-auto p-4">
      <div className="flex-1 overflow-y-auto space-y-4 pr-2">
        {messages.length === 0 ? (
          <div className="h-full flex flex-col items-center justify-center text-zinc-600 font-mono text-sm">
            <Database size={48} className="mb-4 text-zinc-800" />
            <p>READY TO CHAT</p>
            <p className="text-xs text-zinc-700 mt-1">...or not :(</p>
          </div>
        ) : (
          messages.map((msg) => (
            <div
              key={msg.id}
              className={`flex flex-col ${
                msg.role === 'user' ? 'items-end' : 'items-start'
              }`}
            >
              <div
                className={`max-w-[80%] rounded-2xl px-4 py-3 text-sm leading-relaxed ${
                  msg.role === 'user'
                    ? 'bg-emerald-600 text-white rounded-br-none'
                    : 'bg-zinc-900 border border-zinc-800 text-zinc-200 rounded-bl-none shadow-lg'
                }`}
              >
                {msg.wasReset && (
                  <div className="flex items-center gap-1 text-xs text-amber-400 mb-2 font-mono bg-amber-950/40 px-2 py-1 rounded border border-amber-800/50">
                    <AlertTriangle size={12} />
                    <span>Catched hallucination! Retried.</span>
                  </div>
                )}
                <p className="whitespace-pre-wrap">{msg.content}</p>
                {msg.isStreaming && (
                  <span className="inline-block w-2 h-4 ml-1 bg-emerald-400 animate-pulse" />
                )}
              </div>
            </div>
          ))
        )}
        <div ref={messagesEndRef} />
      </div>

      <form onSubmit={sendMessage} className="mt-4 flex gap-2">
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Спроси что-нибудь у RAG..."
          disabled={isLoading}
          className="flex-1 bg-zinc-900 border border-zinc-800 rounded-xl px-4 py-3 text-sm text-zinc-100 placeholder-zinc-500 focus:outline-none focus:border-emerald-500 transition"
        />
        <button
          type="submit"
          disabled={isLoading || !input.trim()}
          className="bg-emerald-600 hover:bg-emerald-500 disabled:opacity-50 text-white px-5 py-3 rounded-xl transition flex items-center justify-center cursor-pointer"
        >
          {isLoading ? <RefreshCw size={18} className="animate-spin" /> : <Send size={18} />}
        </button>
      </form>
    </div>
  )
}
