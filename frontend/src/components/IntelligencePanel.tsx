import React from 'react';
import { 
  BarChart, 
  Bar, 
  XAxis, 
  YAxis, 
  Tooltip, 
  ResponsiveContainer, 
  Cell
} from 'recharts';
import { 
  Cpu, 
  Activity, 
  BarChart3
} from 'lucide-react';
import type { ChangeAnalysisResponse } from '../types/earthlens';

interface IntelligencePanelProps {
  analysis: ChangeAnalysisResponse | null;
  isLoading: boolean;
}

export const IntelligencePanel: React.FC<IntelligencePanelProps> = ({
  analysis,
  isLoading,
}) => {
  if (isLoading || !analysis) {
    return (
      <aside className="w-96 h-full flex flex-col bg-slate-950/85 backdrop-blur-xl border-l border-cyan-500/20 text-slate-200 z-20 shadow-2xl p-6 justify-center items-center">
        <div className="w-10 h-10 border-2 border-cyan-400 border-t-transparent rounded-full animate-spin mb-4" />
        <p className="font-mono text-xs text-cyan-400 uppercase tracking-widest">
          COMPUTING SATELLITE TENSOR INTELLIGENCE...
        </p>
        <span className="text-[11px] text-slate-400 font-mono mt-1">Executing Random Forest Inference</span>
      </aside>
    );
  }

  const { summary, categoryChanges, modelMetrics } = analysis;

  const chartData = categoryChanges.map((c) => ({
    name: c.name.split(' ')[0],
    fullName: c.name,
    NetChange: c.netChangeKm2,
    percent: c.percentChange,
    color: c.color,
  }));

  return (
    <aside className="w-96 h-full flex flex-col bg-slate-950/85 backdrop-blur-xl border-l border-cyan-500/20 text-slate-200 z-20 shadow-2xl overflow-y-auto">
      {/* Panel Header */}
      <div className="p-4 border-b border-cyan-500/20 flex items-center justify-between">
        <div className="flex items-center space-x-2">
          <BarChart3 className="w-4 h-4 text-cyan-400" />
          <h2 className="font-mono text-xs font-bold uppercase tracking-wider text-slate-200">
            INTELLIGENCE DASHBOARD
          </h2>
        </div>
        <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-cyan-950/60 border border-cyan-500/40 text-cyan-400">
          RANDOM FOREST
        </span>
      </div>

      <div className="p-4 space-y-5 flex-1">
        {/* Core Key Metric Cards */}
        <div className="grid grid-cols-2 gap-2.5">
          <div className="p-3 rounded-lg bg-slate-900/60 border border-slate-800">
            <span className="text-[10px] font-mono text-slate-400 uppercase block">Total Changed Surface</span>
            <div className="mt-1 flex items-baseline gap-1">
              <span className="text-xl font-bold font-mono text-cyan-400">{summary.changePercentage}%</span>
              <span className="text-[10px] text-slate-400 font-mono">({summary.totalChangedAreaKm2} km²)</span>
            </div>
            <div className="text-[10px] text-slate-400 mt-1 font-mono">Total Study Area: {summary.totalStudyAreaKm2} km²</div>
          </div>

          <div className="p-3 rounded-lg bg-slate-900/60 border border-slate-800">
            <span className="text-[10px] font-mono text-slate-400 uppercase block">Primary Dynamic Driver</span>
            <span className="mt-1 font-bold text-sm text-emerald-400 font-mono block">
              {summary.keyDriver}
            </span>
            <div className="text-[10px] text-slate-400 mt-1 font-mono flex items-center gap-1">
              <span className="w-1.5 h-1.5 rounded-full bg-emerald-400" />
              Sensor: Sentinel-2 MSI
            </div>
          </div>
        </div>

        {/* Change Breakdown Cards */}
        <div className="space-y-2">
          <div className="text-xs font-mono text-cyan-400 flex items-center gap-1.5">
            <Activity className="w-3.5 h-3.5" />
            DETECTED PHENOMENA IMPACT
          </div>

          <div className="space-y-1.5 text-xs">
            {summary.waterLossKm2 > 0 && (
              <div className="p-2.5 rounded-lg bg-blue-950/20 border border-blue-900/40 flex justify-between items-center">
                <span className="text-blue-300 font-medium">Waterbody Contraction</span>
                <span className="font-mono text-red-400 font-bold">-{summary.waterLossKm2} km²</span>
              </div>
            )}
            {summary.urbanGainKm2 > 0 && (
              <div className="p-2.5 rounded-lg bg-orange-950/20 border border-orange-900/40 flex justify-between items-center">
                <span className="text-orange-300 font-medium">Urban & Infrastructure Gain</span>
                <span className="font-mono text-orange-400 font-bold">+{summary.urbanGainKm2} km²</span>
              </div>
            )}
            {summary.vegetationLossKm2 > 0 && (
              <div className="p-2.5 rounded-lg bg-red-950/20 border border-red-900/40 flex justify-between items-center">
                <span className="text-red-300 font-medium">Canopy Loss / Clearing</span>
                <span className="font-mono text-red-400 font-bold">-{summary.vegetationLossKm2} km²</span>
              </div>
            )}
          </div>
        </div>

        {/* Categorical Land Cover Shifts Chart */}
        <div className="space-y-2 p-3 rounded-lg bg-slate-900/40 border border-slate-800">
          <div className="text-xs font-mono text-cyan-400 flex items-center justify-between">
            <span>NET LAND-COVER SHIFT (KM²)</span>
            <span className="text-[10px] text-slate-400">T0 → T1</span>
          </div>

          <div className="h-44 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={chartData} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
                <XAxis 
                  dataKey="name" 
                  stroke="#64748b" 
                  fontSize={10} 
                  tickLine={false} 
                />
                <YAxis 
                  stroke="#64748b" 
                  fontSize={10} 
                  tickLine={false} 
                />
                <Tooltip
                  content={({ active, payload }) => {
                    if (active && payload && payload.length) {
                      const data = payload[0].payload;
                      return (
                        <div className="p-2 rounded bg-slate-900 border border-slate-700 text-[11px] font-mono shadow-xl">
                          <p className="font-bold text-slate-200">{data.fullName}</p>
                          <p className={`font-semibold ${data.NetChange >= 0 ? 'text-emerald-400' : 'text-red-400'}`}>
                            Net: {data.NetChange > 0 ? `+${data.NetChange}` : data.NetChange} km² ({data.percent}%)
                          </p>
                        </div>
                      );
                    }
                    return null;
                  }}
                />
                <Bar dataKey="NetChange" radius={[4, 4, 0, 0]}>
                  {chartData.map((entry, index) => (
                    <Cell 
                      key={`cell-${index}`} 
                      fill={entry.NetChange >= 0 ? '#10b981' : '#ef4444'} 
                    />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* ML Model Performance & Verification */}
        <div className="p-3 rounded-lg bg-slate-900/60 border border-cyan-500/20 space-y-2.5">
          <div className="text-xs font-mono text-cyan-400 flex items-center justify-between">
            <span className="flex items-center gap-1.5">
              <Cpu className="w-3.5 h-3.5" />
              ML CLASSIFIER STATS
            </span>
            <span className="text-[10px] text-emerald-400 font-bold">{modelMetrics.accuracy}% ACC</span>
          </div>

          <div className="grid grid-cols-2 gap-2 text-[11px] font-mono">
            <div className="p-2 rounded bg-slate-950/70 border border-slate-800">
              <span className="text-[10px] text-slate-400 block">ESTIMATORS</span>
              <span className="font-bold text-slate-200">{modelMetrics.n_estimators} Trees</span>
            </div>
            <div className="p-2 rounded bg-slate-950/70 border border-slate-800">
              <span className="text-[10px] text-slate-400 block">F1 MACRO SCORE</span>
              <span className="font-bold text-cyan-400">{modelMetrics.f1_macro}%</span>
            </div>
          </div>

          <div className="text-[10px] font-mono text-slate-400">
            Features: {modelMetrics.features.join(', ')}
          </div>
        </div>

        {/* Land-cover Legend */}
        <div className="p-3 rounded-lg bg-slate-900/30 border border-slate-800 space-y-1.5">
          <span className="text-[10px] font-mono text-slate-400 block uppercase">Land-Cover Spectrum</span>
          <div className="grid grid-cols-2 gap-1.5 text-[11px]">
            <div className="flex items-center gap-1.5">
              <span className="w-2.5 h-2.5 rounded bg-blue-700" />
              <span className="text-slate-300">Waterbody</span>
            </div>
            <div className="flex items-center gap-1.5">
              <span className="w-2.5 h-2.5 rounded bg-emerald-700" />
              <span className="text-slate-300">Forest / Canopy</span>
            </div>
            <div className="flex items-center gap-1.5">
              <span className="w-2.5 h-2.5 rounded bg-lime-500" />
              <span className="text-slate-300">Grassland / Crop</span>
            </div>
            <div className="flex items-center gap-1.5">
              <span className="w-2.5 h-2.5 rounded bg-red-500" />
              <span className="text-slate-300">Urban Built-up</span>
            </div>
            <div className="flex items-center gap-1.5">
              <span className="w-2.5 h-2.5 rounded bg-amber-600" />
              <span className="text-slate-300">Barren / Desert</span>
            </div>
          </div>
        </div>
      </div>
    </aside>
  );
};
