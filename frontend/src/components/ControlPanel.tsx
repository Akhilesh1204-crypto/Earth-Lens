import React from 'react';
import { 
  Globe2, 
  Layers, 
  Sliders, 
  Cpu, 
  MapPin, 
  Calendar, 
  Eye, 
  EyeOff, 
  ChevronRight
} from 'lucide-react';
import type { StudyArea } from '../types/earthlens';

interface ControlPanelProps {
  studyAreas: StudyArea[];
  selectedArea: StudyArea;
  onSelectArea: (area: StudyArea) => void;
  sliderPos: number;
  onSliderChange: (val: number) => void;
  showClassOverlay: boolean;
  onToggleClassOverlay: () => void;
  showChangeOverlay: boolean;
  onToggleChangeOverlay: () => void;
  overlayOpacity: number;
  onOpacityChange: (val: number) => void;
  isRunningAnalysis: boolean;
  onRunAnalysis: () => void;
  activeViewMode: 'split' | 'before' | 'after' | 'difference';
  onViewModeChange: (mode: 'split' | 'before' | 'after' | 'difference') => void;
}

export const ControlPanel: React.FC<ControlPanelProps> = ({
  studyAreas,
  selectedArea,
  onSelectArea,
  sliderPos,
  onSliderChange,
  showClassOverlay,
  onToggleClassOverlay,
  showChangeOverlay,
  onToggleChangeOverlay,
  overlayOpacity,
  onOpacityChange,
  isRunningAnalysis,
  onRunAnalysis,
  activeViewMode,
  onViewModeChange,
}) => {
  return (
    <aside className="w-84 h-full flex flex-col bg-slate-950/85 backdrop-blur-xl border-r border-cyan-500/20 text-slate-200 z-20 shadow-2xl overflow-y-auto">
      {/* Brand Header */}
      <div className="p-4 border-b border-cyan-500/20 flex items-center justify-between">
        <div className="flex items-center space-x-2.5">
          <div className="w-9 h-9 rounded-lg bg-gradient-to-tr from-cyan-500 to-emerald-500 flex items-center justify-center shadow-lg shadow-cyan-500/20">
            <Globe2 className="w-5 h-5 text-slate-950" />
          </div>
          <div>
            <h1 className="font-bold text-base tracking-wider bg-gradient-to-r from-cyan-400 via-teal-300 to-emerald-400 bg-clip-text text-transparent">
              EARTHLENS
            </h1>
            <p className="text-[10px] uppercase font-mono tracking-widest text-cyan-400/70">
              ORBITAL ML INTELLIGENCE
            </p>
          </div>
        </div>
        <div className="flex items-center space-x-1.5 px-2 py-0.5 rounded-full bg-emerald-500/10 border border-emerald-500/30 text-emerald-400 text-[10px] font-mono">
          <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse" />
          <span>LIVE V1.0</span>
        </div>
      </div>

      <div className="p-4 space-y-5 flex-1">
        {/* Study Area Selector */}
        <div className="space-y-2">
          <div className="flex items-center justify-between text-xs font-mono text-cyan-400">
            <span className="flex items-center gap-1.5">
              <MapPin className="w-3.5 h-3.5" />
              STUDY REGION
            </span>
            <span className="text-[10px] text-slate-400">{studyAreas.length} AVAILABLE</span>
          </div>

          <div className="space-y-1.5">
            {studyAreas.map((area) => {
              const isSelected = area.id === selectedArea.id;
              return (
                <button
                  key={area.id}
                  onClick={() => onSelectArea(area)}
                  className={`w-full text-left p-2.5 rounded-lg border transition-all duration-200 flex flex-col gap-1 ${
                    isSelected
                      ? 'bg-cyan-950/40 border-cyan-400 shadow-md shadow-cyan-500/10'
                      : 'bg-slate-900/40 border-slate-800 hover:border-slate-700 hover:bg-slate-900/60'
                  }`}
                >
                  <div className="flex items-center justify-between">
                    <span className={`text-xs font-semibold ${isSelected ? 'text-cyan-300' : 'text-slate-300'}`}>
                      {area.name}
                    </span>
                    {isSelected && <ChevronRight className="w-3.5 h-3.5 text-cyan-400" />}
                  </div>
                  <p className="text-[11px] text-slate-400 line-clamp-1">{area.description}</p>
                </button>
              );
            })}
          </div>
        </div>

        {/* Temporal Dates Info */}
        <div className="p-3 rounded-lg bg-slate-900/70 border border-slate-800/80 space-y-2">
          <div className="text-[11px] font-mono text-slate-400 flex items-center gap-1.5">
            <Calendar className="w-3.5 h-3.5 text-cyan-400" />
            OBSERVATION EPOCHS
          </div>
          <div className="grid grid-cols-2 gap-2 text-xs">
            <div className="p-2 rounded bg-slate-950/60 border border-slate-800">
              <span className="text-[10px] font-mono text-slate-400 block">T0 (BEFORE)</span>
              <span className="font-mono text-cyan-300 font-semibold">{selectedArea.dates.before}</span>
            </div>
            <div className="p-2 rounded bg-slate-950/60 border border-slate-800">
              <span className="text-[10px] font-mono text-slate-400 block">T1 (AFTER)</span>
              <span className="font-mono text-emerald-400 font-semibold">{selectedArea.dates.after}</span>
            </div>
          </div>
          <div className="text-[10px] text-slate-400 font-mono pt-1 border-t border-slate-800/60">
            Source: <span className="text-slate-300">{selectedArea.attribution}</span>
          </div>
        </div>

        {/* View Mode Switcher */}
        <div className="space-y-2">
          <div className="text-xs font-mono text-cyan-400 flex items-center gap-1.5">
            <Sliders className="w-3.5 h-3.5" />
            VIEW MODE
          </div>
          <div className="grid grid-cols-2 gap-1.5 p-1 rounded-lg bg-slate-900/80 border border-slate-800">
            {[
              { id: 'split', label: 'Split Slider' },
              { id: 'difference', label: 'Change Map' },
              { id: 'before', label: 'Before (T0)' },
              { id: 'after', label: 'After (T1)' },
            ].map((mode) => (
              <button
                key={mode.id}
                onClick={() => onViewModeChange(mode.id as any)}
                className={`py-1.5 px-2 text-[11px] font-medium rounded-md transition-all ${
                  activeViewMode === mode.id
                    ? 'bg-cyan-500 text-slate-950 font-bold shadow-md shadow-cyan-500/20'
                    : 'text-slate-400 hover:text-slate-200'
                }`}
              >
                {mode.label}
              </button>
            ))}
          </div>
        </div>

        {/* Draggable Slider Control */}
        {activeViewMode === 'split' && (
          <div className="space-y-2 p-3 rounded-lg bg-slate-900/40 border border-slate-800/80">
            <div className="flex justify-between items-center text-xs font-mono">
              <span className="text-slate-400">SLIDER POSITION</span>
              <span className="text-cyan-400 font-bold">{Math.round(sliderPos)}%</span>
            </div>
            <input
              type="range"
              min={0}
              max={100}
              value={sliderPos}
              onChange={(e) => onSliderChange(Number(e.target.value))}
              className="w-full accent-cyan-400 cursor-pointer h-1.5 bg-slate-800 rounded-lg"
            />
            <div className="flex justify-between text-[10px] font-mono text-slate-400">
              <span>← Before ({selectedArea.dates.before})</span>
              <span>After ({selectedArea.dates.after}) →</span>
            </div>
          </div>
        )}

        {/* Intelligent Overlays Control */}
        <div className="space-y-3">
          <div className="text-xs font-mono text-cyan-400 flex items-center gap-1.5">
            <Layers className="w-3.5 h-3.5" />
            ML SPECTRAL OVERLAYS
          </div>

          <div className="space-y-2">
            <button
              onClick={onToggleChangeOverlay}
              className={`w-full p-2.5 rounded-lg border text-xs flex items-center justify-between transition-all ${
                showChangeOverlay
                  ? 'bg-orange-950/30 border-orange-500 text-orange-200'
                  : 'bg-slate-900/40 border-slate-800 text-slate-400 hover:border-slate-700'
              }`}
            >
              <div className="flex items-center gap-2">
                <span className="w-2.5 h-2.5 rounded-full bg-orange-500" />
                <span className="font-medium">Change Anomaly Map</span>
              </div>
              {showChangeOverlay ? <Eye className="w-3.5 h-3.5 text-orange-400" /> : <EyeOff className="w-3.5 h-3.5" />}
            </button>

            <button
              onClick={onToggleClassOverlay}
              className={`w-full p-2.5 rounded-lg border text-xs flex items-center justify-between transition-all ${
                showClassOverlay
                  ? 'bg-emerald-950/30 border-emerald-500 text-emerald-200'
                  : 'bg-slate-900/40 border-slate-800 text-slate-400 hover:border-slate-700'
              }`}
            >
              <div className="flex items-center gap-2">
                <span className="w-2.5 h-2.5 rounded-full bg-emerald-500" />
                <span className="font-medium">Land Cover Classification</span>
              </div>
              {showClassOverlay ? <Eye className="w-3.5 h-3.5 text-emerald-400" /> : <EyeOff className="w-3.5 h-3.5" />}
            </button>
          </div>

          {/* Opacity Slider */}
          {(showClassOverlay || showChangeOverlay) && (
            <div className="p-3 rounded-lg bg-slate-900/40 border border-slate-800/80 space-y-1.5">
              <div className="flex justify-between text-[11px] font-mono text-slate-400">
                <span>OVERLAY OPACITY</span>
                <span className="text-cyan-400">{Math.round(overlayOpacity * 100)}%</span>
              </div>
              <input
                type="range"
                min={0.1}
                max={1.0}
                step={0.05}
                value={overlayOpacity}
                onChange={(e) => onOpacityChange(Number(e.target.value))}
                className="w-full accent-cyan-400 cursor-pointer h-1.5 bg-slate-800 rounded-lg"
              />
            </div>
          )}
        </div>

        {/* Action Button */}
        <div className="pt-2">
          <button
            onClick={onRunAnalysis}
            disabled={isRunningAnalysis}
            className={`w-full py-3 px-4 rounded-lg font-mono text-xs uppercase tracking-wider font-bold transition-all duration-300 flex items-center justify-center gap-2 shadow-lg ${
              isRunningAnalysis
                ? 'bg-slate-800 text-slate-400 cursor-not-allowed'
                : 'bg-gradient-to-r from-cyan-500 to-emerald-500 hover:from-cyan-400 hover:to-emerald-400 text-slate-950 shadow-cyan-500/25 active:scale-[0.98]'
            }`}
          >
            <Cpu className={`w-4 h-4 ${isRunningAnalysis ? 'animate-spin' : ''}`} />
            {isRunningAnalysis ? 'INFERRING CHANGES...' : 'RUN ML CHANGE DETECTION'}
          </button>
        </div>
      </div>
    </aside>
  );
};
