import { create } from 'zustand'

export const useStore = create((set) => ({
  // Global Clock - Intentionally isolated to allow transient subscription bypassing React Renders
  globalTimeMs: 0,
  
  // Data Registry
  activeSessionId: null,
  activeData: null, // Holds the raw Pandas split dictionary { columns: [...], data: [...] }
  isLoading: false,
  
  // State Mutators
  setGlobalTimeMs: (timeMs) => set({ globalTimeMs: timeMs }),
  setActiveSessionId: (id) => set({ activeSessionId: id }),
  setActiveData: (data) => set({ activeData: data }),
  setIsLoading: (loading) => set({ isLoading: loading }),

  // Context Metadata bounds
  activeContext: {
    phase: 'N/A',
    questionId: -1,
    elapsedMs: 0
  },
  setActiveContext: (context) => set({ activeContext: context })
}))
