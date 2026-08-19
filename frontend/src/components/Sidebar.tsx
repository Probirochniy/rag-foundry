import { Database, MessageSquare } from 'lucide-react'

interface SidebarProps {
  activeTab: 'chat' | 'ingest'
  setActiveTab: (tab: 'chat' | 'ingest') => void
  topK: number
  setTopK: (k: number) => void
}

export function Sidebar({ activeTab, setActiveTab, topK, setTopK }: SidebarProps) {
  return (
    <aside className="w-64 border-r border-zinc-800 bg-zinc-900/50 p-4 flex flex-col justify-between">
      <div>
        <div className="flex items-center gap-2 mb-8 px-2">
          <div className="h-3 w-3 rounded-full bg-emerald-500 animate-pulse" />
          <h1 className="font-mono font-bold text-lg tracking-wider text-emerald-400">
            RAG FOUNDRY
          </h1>
        </div>

        <nav className="space-y-2">
          <button
            onClick={() => setActiveTab('chat')}
            className={`w-full flex items-center gap-3 px-3 py-2 rounded-lg text-sm font-medium transition ${
              activeTab === 'chat'
                ? 'bg-zinc-800 text-emerald-400 border border-zinc-700'
                : 'text-zinc-400 hover:bg-zinc-800/50 hover:text-zinc-200'
            }`}
          >
            <MessageSquare size={18} />
            Chat
          </button>

          <button
            onClick={() => setActiveTab('ingest')}
            className={`w-full flex items-center gap-3 px-3 py-2 rounded-lg text-sm font-medium transition ${
              activeTab === 'ingest'
                ? 'bg-zinc-800 text-emerald-400 border border-zinc-700'
                : 'text-zinc-400 hover:bg-zinc-800/50 hover:text-zinc-200'
            }`}
          >
            <Database size={18} />
            Data base(d)
          </button>
        </nav>
      </div>

      {activeTab === 'chat' && (
        <div className="bg-zinc-900 border border-zinc-800 p-3 rounded-xl">
          <label className="text-xs font-mono text-zinc-400 block mb-2">
            TOP_K: <span className="text-emerald-400 font-bold">{topK}</span>
          </label>
          <input
            type="range"
            min="1"
            max="10"
            value={topK}
            onChange={(e) => setTopK(Number(e.target.value))}
            className="w-full accent-emerald-500 cursor-pointer"
          />
        </div>
      )}
    </aside>
  )
}
