import React, { useEffect, useState } from 'react'
import { useStore } from '../store'
import { FolderGit2, Search } from 'lucide-react'

export default function Sidebar() {
  const [sessions, setSessions] = useState([])
  const { activeSessionId, setActiveSessionId } = useStore()

  useEffect(() => {
    // API is now relative to the monolithic server
    fetch('/api/sessions')
      .then(res => res.json())
      .then(data => setSessions(data.sessions || []))
      .catch(err => console.error("Failed fetching sessions:", err))
  }, [])

  return (
    <aside className="w-72 glass-panel m-4 flex flex-col shrink-0 overflow-hidden">
      <div className="p-4 border-b border-gray-800/50 bg-black/20">
        <h2 className="text-sm font-semibold tracking-widest text-gray-400 uppercase flex items-center gap-2">
          <FolderGit2 className="w-4 h-4" />
          Data Registry
        </h2>
        <div className="mt-3 relative">
          <Search className="w-4 h-4 absolute left-3 top-1/2 -translate-y-1/2 text-gray-500" />
          <input 
            type="text" 
            placeholder="Search sessions..." 
            className="w-full bg-gray-900/50 border border-gray-700/50 rounded-md py-1.5 pl-9 pr-3 text-sm focus:outline-none focus:border-indigo-500 transition-colors"
          />
        </div>
      </div>
      
      <div className="flex-1 overflow-y-auto p-2 space-y-1">
        {sessions.map((s, idx) => {
          const sid = s.session_id || `SESSION_UNK_${idx}`
          const isActive = activeSessionId === sid
          return (
            <button
              key={sid}
              onClick={() => setActiveSessionId(sid)}
              className={`w-full text-left px-3 py-2.5 rounded-lg text-sm transition-all duration-200 border ${
                isActive 
                  ? 'bg-indigo-500/20 border-indigo-500/50 text-indigo-100' 
                  : 'bg-transparent border-transparent text-gray-400 hover:bg-gray-800/50 hover:text-gray-200'
              }`}
            >
              <div className="font-mono font-bold">{sid}</div>
              <div className="text-xs opacity-60 mt-1 truncate">Length: {s.video_duration_sec || '?'}s | FPS: 30</div>
            </button>
          )
        })}
        {sessions.length === 0 && (
          <div className="text-center text-gray-600 text-sm mt-10">
            No pipeline outputs found.
          </div>
        )}
      </div>
    </aside>
  )
}
