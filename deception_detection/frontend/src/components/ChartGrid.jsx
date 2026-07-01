import React, { useMemo, useRef, useEffect } from 'react'
import ReactECharts from 'echarts-for-react'
import { useStore } from '../store'

export default function ChartGrid() {
  const { activeData, setGlobalTimeMs } = useStore()
  const echartsRef = useRef(null)

  // Memoize the heavy options object creation so it only runs when a new session loads.
  const options = useMemo(() => {
    if (!activeData || !activeData.data || activeData.data.length === 0) return {}

    const { columns, data } = activeData
    
    const tIdxEnd = columns.indexOf('end_time_ms')
    const lwIdx = columns.indexOf('left_wrist_velocity_mean')
    const rwIdx = columns.indexOf('right_wrist_velocity_mean')
    const fftIdx = columns.indexOf('fft_dominant_freq')
    const mfccIdx = columns.indexOf('mfcc_1')

    const times = []
    const leftWrist = []
    const rightWrist = []
    const bandPower = []
    const mfcc = []

    for (let i = 0; i < data.length; i++) {
      const row = data[i]
      times.push(row[tIdxEnd] ? row[tIdxEnd] / 1000 : i)
      leftWrist.push(row[lwIdx] || 0)
      rightWrist.push(row[rwIdx] || 0)
      bandPower.push(row[fftIdx] || 0)
      mfcc.push(row[mfccIdx] || 0)
    }

    return {
      tooltip: {
        trigger: 'axis',
        axisPointer: { type: 'cross', animation: false }
      },
      grid: [
        { left: '3%', right: '3%', height: '25%', top: '5%' },
        { left: '3%', right: '3%', height: '25%', top: '35%' },
        { left: '3%', right: '3%', height: '25%', top: '65%' }
      ],
      xAxis: [
        { gridIndex: 0, type: 'category', data: times, show: false },
        { gridIndex: 1, type: 'category', data: times, show: false },
        { gridIndex: 2, type: 'category', data: times, axisLabel: { color: '#888' } }
      ],
      yAxis: [
        { gridIndex: 0, type: 'value', splitLine: { show: false }, axisLabel: { color: '#888' } },
        { gridIndex: 1, type: 'value', splitLine: { show: false }, axisLabel: { color: '#888' } },
        { gridIndex: 2, type: 'value', splitLine: { show: false }, axisLabel: { color: '#888' } }
      ],
      dataZoom: [
        { type: 'inside', xAxisIndex: [0, 1, 2] },
        { type: 'slider', xAxisIndex: [0, 1, 2], bottom: 0, textStyle: { color: '#fff' } }
      ],
      series: [
        {
          name: 'L-Wrist Vel',
          type: 'line',
          xAxisIndex: 0,
          yAxisIndex: 0,
          data: leftWrist,
          showSymbol: false,
          sampling: 'lttb', // LTTB Signal Preservation
          lineStyle: { color: '#3b82f6', width: 1.5 }
        },
        {
          name: 'R-Wrist Vel',
          type: 'line',
          xAxisIndex: 0,
          yAxisIndex: 0,
          data: rightWrist,
          showSymbol: false,
          sampling: 'lttb',
          lineStyle: { color: '#10b981', width: 1.5 }
        },
        {
          name: 'Tremor FFT',
          type: 'line',
          xAxisIndex: 1,
          yAxisIndex: 1,
          data: bandPower,
          showSymbol: false,
          sampling: 'lttb',
          lineStyle: { color: '#f59e0b', width: 1.5 }
        },
        {
          name: 'Vocal MFCC',
          type: 'line',
          xAxisIndex: 2,
          yAxisIndex: 2,
          data: mfcc,
          showSymbol: false,
          sampling: 'lttb',
          lineStyle: { color: '#8b5cf6', width: 1.5 }
        }
      ]
    }
  }, [activeData])

  // Transient Playhead crosshair injection
  useEffect(() => {
    const unsub = useStore.subscribe((state) => {
      if (!echartsRef.current) return
      const echartsInstance = echartsRef.current.getEchartsInstance()
      if (!echartsInstance) return

      // Dispatch an action to show the tooltip/crosshair at the exact time index
      // ECharts expects the x-axis string or index. We pass the string representation of time.
      const timeStr = (state.globalTimeMs / 1000).toFixed(2).toString()
      echartsInstance.dispatchAction({
        type: 'showTip',
        seriesIndex: 0,
        name: timeStr
      })
    })
    return () => unsub()
  }, [])

  const onEvents = {
    click: (params) => {
      if (params.name) {
        const timeSec = parseFloat(params.name)
        if (!isNaN(timeSec)) {
          setGlobalTimeMs(timeSec * 1000)
        }
      }
    }
  }

  if (!activeData || !activeData.data || activeData.data.length === 0) {
    return <div className="h-full flex items-center justify-center text-gray-500">Awaiting Dense Tensor Load...</div>
  }

  return (
    <div className="w-full h-full relative">
      <ReactECharts 
        ref={echartsRef}
        option={options} 
        notMerge={true} 
        lazyUpdate={true} 
        onEvents={onEvents}
        style={{ height: '100%', width: '100%' }} 
      />
    </div>
  )
}
