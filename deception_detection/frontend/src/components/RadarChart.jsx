import React, { useRef, useEffect } from 'react'
import ReactECharts from 'echarts-for-react'
import { useStore } from '../store'

export default function RadarChart() {
  const { activeData } = useStore()
  const echartsRef = useRef(null)

  // Initialize the base empty radar
  const baseOptions = {
    tooltip: {},
    radar: {
      indicator: [
        { name: 'AU01', max: 5.0 },
        { name: 'AU02', max: 5.0 },
        { name: 'AU04', max: 5.0 },
        { name: 'AU06', max: 5.0 },
        { name: 'AU12', max: 5.0 },
        { name: 'AU15', max: 5.0 }
      ],
      splitNumber: 4,
      axisName: { color: '#a1a1aa' },
      splitLine: {
        lineStyle: { color: ['rgba(255, 255, 255, 0.1)', 'rgba(255, 255, 255, 0.2)'] }
      },
      splitArea: { show: false },
      axisLine: { lineStyle: { color: 'rgba(255, 255, 255, 0.2)' } }
    },
    series: [
      {
        name: 'Instantaneous AU',
        type: 'radar',
        data: [{
          value: [0,0,0,0,0,0],
          name: 'Current Frame',
          itemStyle: { color: '#6366f1' },
          areaStyle: { color: 'rgba(99, 102, 241, 0.4)' },
          lineStyle: { color: '#818cf8', width: 2 }
        }]
      }
    ]
  }

  useEffect(() => {
    // Transient subscriber for frame-by-frame radar updating without React Renders
    const unsub = useStore.subscribe((state) => {
      if (!echartsRef.current || !state.activeData) return
      
      const echartsInstance = echartsRef.current.getEchartsInstance()
      const { columns, data } = state.activeData
      
      const tIdxStart = columns.indexOf('start_time_ms')
      const tIdxEnd = columns.indexOf('end_time_ms')
      
      const auIdxs = [
        columns.indexOf('AU01'), columns.indexOf('AU02'),
        columns.indexOf('AU04'), columns.indexOf('AU06'),
        columns.indexOf('AU12'), columns.indexOf('AU15')
      ]

      let targetRow = null
      for (let i = 0; i < data.length; i++) {
        if (state.globalTimeMs >= data[i][tIdxStart] && state.globalTimeMs <= data[i][tIdxEnd]) {
          targetRow = data[i]
          break
        }
      }

      let newValues = [0,0,0,0,0,0]
      let hasData = false

      if (targetRow) {
        newValues = auIdxs.map(idx => (idx !== -1 && targetRow[idx] !== null) ? targetRow[idx] : 0)
        hasData = newValues.some(v => v > 0)
      }

      // Update the chart series locally (NaN fallback smoothly handled via opacity drop)
      echartsInstance.setOption({
        series: [{
          data: [{
            value: newValues,
            name: 'Current Frame',
            itemStyle: { color: hasData ? '#6366f1' : '#3f3f46' },
            areaStyle: { color: hasData ? 'rgba(99, 102, 241, 0.4)' : 'rgba(63, 63, 70, 0.1)' },
            lineStyle: { color: hasData ? '#818cf8' : '#52525b', width: 2 }
          }]
        }]
      })
    })
    return () => unsub()
  }, [])

  if (!activeData || !activeData.data || activeData.data.length === 0) {
    return <div className="h-full flex items-center justify-center text-gray-500 text-sm">No Tensor Active</div>
  }

  return (
    <div className="w-full h-full">
      <ReactECharts 
        ref={echartsRef}
        option={baseOptions} 
        style={{ height: '100%', width: '100%' }} 
        opts={{ renderer: 'canvas' }}
      />
    </div>
  )
}
