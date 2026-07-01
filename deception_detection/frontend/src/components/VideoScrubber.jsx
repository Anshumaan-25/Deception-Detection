import React, { useRef, useEffect } from 'react'
import { useStore } from '../store'

export default function VideoScrubber({ sessionId }) {
  const videoRef = useRef(null)
  const isSeekingRef = useRef(false)

  useEffect(() => {
    // Transient subscriber for the video playhead
    const unsub = useStore.subscribe((state, prevState) => {
      if (!videoRef.current || isSeekingRef.current) return
      
      const newTime = state.globalTimeMs
      if (newTime !== prevState.globalTimeMs) {
        const diff = Math.abs(videoRef.current.currentTime * 1000 - newTime)
        if (diff > 100) {
          videoRef.current.currentTime = newTime / 1000
        }
      }
    })
    return () => unsub()
  }, [])

  const handleTimeUpdate = () => {
    if (!videoRef.current || isSeekingRef.current) return
    useStore.getState().setGlobalTimeMs(videoRef.current.currentTime * 1000)
  }

  const handleSeeked = () => {
    isSeekingRef.current = false
    if (videoRef.current) {
      useStore.getState().setGlobalTimeMs(videoRef.current.currentTime * 1000)
    }
  }

  const handleSeeking = () => {
    isSeekingRef.current = true
  }

  return (
    <div className="w-full h-full bg-black flex items-center justify-center group relative rounded-2xl overflow-hidden">
      <video
        ref={videoRef}
        src={`/api/video/${sessionId}`}
        controls
        controlsList="nodownload nofullscreen"
        className="max-h-full max-w-full object-contain"
        onTimeUpdate={handleTimeUpdate}
        onSeeking={handleSeeking}
        onSeeked={handleSeeked}
      />
      <div className="absolute top-0 left-0 right-0 h-16 bg-gradient-to-b from-black/60 to-transparent pointer-events-none" />
    </div>
  )
}
