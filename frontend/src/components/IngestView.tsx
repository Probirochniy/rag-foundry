import { useState } from 'react'
import { Database, RefreshCw, CheckCircle } from 'lucide-react'

const API_BASE = import.meta.env.BACKEND_URL

export function IngestView() {
    const [sourceId, setSourceId] = useState('')
    const [docContent, setDocContent] = useState('')
    const [ingestStatus, setIngestStatus] = useState<string | null>(null)
    const [isIngesting, setIsIngesting] = useState(false)

    const handleIngest = async (e: React.FormEvent) => {
        e.preventDefault()
        if (!sourceId.trim() || !docContent.trim() || isIngesting) return

        setIsIngesting(true)
        setIngestStatus(null)

        try {
            const res = await fetch(`${API_BASE}/rag/ingest`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    source_id: sourceId.trim(),
                    content: docContent.trim(),
                }),
            })

            if (!res.ok) throw new Error('Failed')
            const data = await res.json()
            setIngestStatus(`Successfully ingested chunks: ${data.ingested_count}`)
            setDocContent('')
            setSourceId('')
        } catch {
            setIngestStatus('Oh no! Error ingesting documents!')
        } finally {
            setIsIngesting(false)
        }
    }

    return (
        <div className="max-w-2xl w-full mx-auto p-8">
            <h2 className="text-xl font-bold font-mono text-emerald-400 mb-6 flex items-center gap-2">
                <Database size={22} />
                Ingest your docs!
            </h2>

            <form onSubmit={handleIngest} className="space-y-4">
                <div>
                    <label className="block text-xs font-mono text-zinc-400 mb-2">Source name</label>
                    <input
                        type="text"
                        value={sourceId}
                        onChange={(e) => setSourceId(e.target.value)}
                        placeholder="manual_k8s.md"
                        required
                        className="w-full bg-zinc-900 border border-zinc-800 rounded-xl px-4 py-2.5 text-sm focus:outline-none focus:border-emerald-500"
                    />
                </div>

                <div>
                    <label className="block text-xs font-mono text-zinc-400 mb-2">Chunk text</label>
                    <textarea
                        value={docContent}
                        onChange={(e) => setDocContent(e.target.value)}
                        placeholder="Paste the text you want to embed here..."
                        rows={6}
                        required
                        className="w-full bg-zinc-900 border border-zinc-800 rounded-xl p-4 text-sm focus:outline-none focus:border-emerald-500 resize-none"
                    />
                </div>

                <button
                    type="submit"
                    disabled={isIngesting}
                    className="w-full bg-emerald-600 hover:bg-emerald-500 disabled:opacity-50 text-white font-medium py-3 rounded-xl transition cursor-pointer flex items-center justify-center gap-2"
                >
                    {isIngesting ? (
                        <>
                            <RefreshCw size={18} className="animate-spin" />
                            Writing writing writing...
                        </>
                    ) : (
                        'Store to vector database'
                    )}
                </button>
            </form>

            {ingestStatus && (
                <div className="mt-4 p-3 bg-zinc-900 border border-zinc-800 rounded-xl flex items-center gap-2 text-sm text-zinc-300 font-mono">
                    <CheckCircle size={16} className="text-emerald-400" />
                    {ingestStatus}
                </div>
            )}
        </div>
    )
}
