import { useEffect } from 'react'
import { useStore } from './store'
import Sidebar from './components/Sidebar'
import Header from './components/Header'
import VideoScrubber from './components/VideoScrubber'
import ChartGrid from './components/ChartGrid'
import RadarChart from './components/RadarChart'

function App() {
  const { activeSessionId, setActiveData, setIsLoading } = useStore()

  // Load Data Layer
  useEffect(() => {
    if (!activeSessionId) return
    
    setIsLoading(true)
    fetch(`/api/data/${activeSessionId}`) // API relative to monolithic server
      .then(res => {
        if (!res.ok) throw new Error("Data not found")
        return res.json()
      })
      .then(json => {
        // We now receive Pandas split format: { data: { columns: [], data: [[]] } }
        setActiveData(json.data)
        setIsLoading(false)
      })
      .catch(err => {
        console.error("Failed to load session data:", err)
        setActiveData(null)
        setIsLoading(false)
      })
  }, [activeSessionId, setActiveData, setIsLoading])

  // Context syncing loop is now moved into a transient subscriber within Header.jsx 
  // to prevent re-rendering the entire App DOM tree 60 times a second.

  return (
    <div className="flex h-screen w-full overflow-hidden bg-gradient-to-br from-gray-950 via-gray-900 to-black text-white font-sans">
      <Sidebar />
      
      <main className="flex-1 flex flex-col p-4 gap-4 overflow-hidden relative">
        <Header />
        
        <div className="flex-1 flex gap-4 h-full min-h-0">
          <div className="flex-1 flex flex-col gap-4 min-w-0">
            <div className="h-2/5 glass-panel flex flex-col overflow-hidden relative group">
              {activeSessionId ? (
                <VideoScrubber sessionId={activeSessionId} />
              ) : (
                <div className="flex items-center justify-center h-full text-gray-500">
                  Select a session from the registry.
                </div>
              )}
            </div>
            <div className="h-3/5 glass-panel p-2 flex flex-col overflow-hidden">
              <ChartGrid />
            </div>
          </div>

          <div className="w-96 glass-panel p-4 flex flex-col flex-shrink-0">
            <h2 className="text-sm font-semibold text-gray-300 tracking-wider uppercase mb-4">Instantaneous Action Units</h2>
            <div className="flex-1 relative">
              <RadarChart />
            </div>
          </div>
        </div>
      </main>
    </div>
  )
}

export default App
