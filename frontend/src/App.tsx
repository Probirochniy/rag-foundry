import { useState } from 'react'
import { Sidebar } from './components/Sidebar'
import { ChatView } from './components/ChatView'
import { IngestView } from './components/IngestView'
import { useRagChat } from './hooks/useRagChat'

export default function App() {
  const [activeTab, setActiveTab] = useState<'chat' | 'ingest'>('chat')
  const { messages, input, setInput, topK, setTopK, isLoading, messagesEndRef, sendMessage } = useRagChat()

  return (
    <div className="flex h-screen bg-zinc-950 text-zinc-100">
      <Sidebar
        activeTab={activeTab}
        setActiveTab={setActiveTab}
        topK={topK}
        setTopK={setTopK}
      />
      <main className="flex-1 flex flex-col h-full bg-zinc-950">
        {activeTab === 'chat' ? (
          <ChatView
            messages={messages}
            input={input}
            setInput={setInput}
            isLoading={isLoading}
            sendMessage={sendMessage}
            messagesEndRef={messagesEndRef}
          />
        ) : (
          <IngestView />
        )}
      </main>
    </div>
  )
}
