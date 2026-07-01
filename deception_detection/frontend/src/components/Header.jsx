import React, { useEffect, useState } from 'react'
import { useStore } from '../store'
import { Activity, Clock, Hash, Map } from 'lucide-react'

export default function Header() {
  const { activeSessionId, activeData } = useStore()
  
  // Local state for Context to avoid re-rendering the whole App
  const [context, setContext] = useState({ phase: 'N/A', questionId: -1, elapsedMs: 0 })

  useEffect(() => {
    // Transient subscription to globalTimeMs
    // This allows the Header to update at 60Hz without thrashing the parent DOM
    const unsub = useStore.subscribe((state, prevState) => {
      if (state.globalTimeMs === prevState.globalTimeMs) return;
      if (!state.activeData) return;
      
      const { columns, data } = state.activeData
      const tIdxStart = columns.indexOf('start_time_ms')
      const tIdxEnd = columns.indexOf('end_time_ms')
      const pIdx = columns.indexOf('context_phase')
      const qIdx = columns.indexOf('question_id')
      const eIdx = columns.indexOf('phase_elapsed_ms')
      
      if (tIdxStart === -1 || tIdxEnd === -1) return

      // Binary search could be optimized here, but for header rendering a fast loop on downsampled window sizes is usually okay.
      // Assuming sequential data
      let targetRow = data[0]
      for (let i = 0; i < data.length; i++) {
        if (state.globalTimeMs >= data[i][tIdxStart] && state.globalTimeMs <= data[i][tIdxEnd]) {
          targetRow = data[i]
          break
        }
      }

      if (targetRow) {
        setContext({
          phase: targetRow[pIdx] || 'N/A',
          questionId: targetRow[qIdx] || -1,
          elapsedMs: targetRow[eIdx] || 0
        })
      }
    })
    return () => unsub()
  }, [])

  return (
    <header className="glass-panel p-4 flex items-center justify-between shrink-0 h-16">
      <div className="flex items-center gap-3">
        <Activity className="text-indigo-400 w-6 h-6" />
        <h1 className="text-xl font-bold tracking-widest uppercase text-gray-100">SPOVNOB Control</h1>
      </div>
      
      {activeSessionId && (
        <div className="flex items-center gap-6 text-sm">
          <div className="flex items-center gap-2 bg-black/40 px-3 py-1.5 rounded-lg border border-gray-800">
            <Map className="w-4 h-4 text-emerald-400" />
            <span className="text-gray-400 font-mono">PHASE:</span>
            <span className="font-bold text-white uppercase">{context.phase}</span>
          </div>
          <div className="flex items-center gap-2 bg-black/40 px-3 py-1.5 rounded-lg border border-gray-800">
            <Hash className="w-4 h-4 text-emerald-400" />
            <span className="text-gray-400 font-mono">Q_ID:</span>
            <span className="font-bold text-white">{context.questionId}</span>
          </div>
          <div className="flex items-center gap-2 bg-black/40 px-3 py-1.5 rounded-lg border border-gray-800">
            <Clock className="w-4 h-4 text-emerald-400" />
            <span className="text-gray-400 font-mono">ELAPSED:</span>
            <span className="font-bold text-white">{(context.elapsedMs / 1000).toFixed(2)}s</span>
          </div>
        </div>
      )}
    </header>
  )
}
