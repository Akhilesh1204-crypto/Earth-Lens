import React, { useRef, useState, useEffect } from 'react';
import * as maplibregl from 'maplibre-gl';
import 'maplibre-gl/dist/maplibre-gl.css';
import { ZoomIn, ZoomOut, Compass } from 'lucide-react';
import type { StudyArea, ChangeAnalysisResponse } from '../types/earthlens';

interface SatelliteMapProps {
  studyArea: StudyArea;
  analysis: ChangeAnalysisResponse | null;
  sliderPos: number;
  onSliderChange: (val: number) => void;
  showClassOverlay: boolean;
  showChangeOverlay: boolean;
  overlayOpacity: number;
  viewMode: 'split' | 'before' | 'after' | 'difference';
}

export const SatelliteMap: React.FC<SatelliteMapProps> = ({
  studyArea,
  analysis,
  sliderPos,
  onSliderChange,
  showClassOverlay,
  showChangeOverlay,
  overlayOpacity,
  viewMode,
}) => {
  const mapBeforeContainer = useRef<HTMLDivElement>(null);
  const mapAfterContainer = useRef<HTMLDivElement>(null);
  const mapBefore = useRef<maplibregl.Map | null>(null);
  const mapAfter = useRef<maplibregl.Map | null>(null);
  const isSyncing = useRef(false);
  const [isDraggingSlider, setIsDraggingSlider] = useState(false);
  const containerRef = useRef<HTMLDivElement>(null);

  // Initialize synchronized dual high-res maps
  useEffect(() => {
    if (!mapBeforeContainer.current || !mapAfterContainer.current) return;

    const beforeTile = studyArea.tileUrls?.before || 'https://wayback.maptiles.arcgis.com/arcgis/rest/services/World_Imagery/WMTS/1.0.0/default028mm/MapServer/tile/10/{z}/{y}/{x}';
    const afterTile = studyArea.tileUrls?.after || 'https://wayback.maptiles.arcgis.com/arcgis/rest/services/World_Imagery/WMTS/1.0.0/default028mm/MapServer/tile/43291/{z}/{y}/{x}';

    // Before Map
    const mb = new maplibregl.Map({
      container: mapBeforeContainer.current,
      style: {
        version: 8,
        sources: {
          'satellite-before': {
            type: 'raster',
            tiles: [beforeTile],
            tileSize: 256,
            maxzoom: 19,
          },
        },
        layers: [
          {
            id: 'satellite-before-layer',
            type: 'raster',
            source: 'satellite-before',
          },
        ],
      },
      center: studyArea.center,
      zoom: studyArea.zoom,
      attributionControl: false,
    });

    // After Map
    const ma = new maplibregl.Map({
      container: mapAfterContainer.current,
      style: {
        version: 8,
        sources: {
          'satellite-after': {
            type: 'raster',
            tiles: [afterTile],
            tileSize: 256,
            maxzoom: 19,
          },
        },
        layers: [
          {
            id: 'satellite-after-layer',
            type: 'raster',
            source: 'satellite-after',
          },
        ],
      },
      center: studyArea.center,
      zoom: studyArea.zoom,
      attributionControl: false,
    });

    // Sync zoom, pan, and pitch across both views
    const syncMaps = (source: maplibregl.Map, target: maplibregl.Map) => {
      source.on('move', () => {
        if (isSyncing.current) return;
        isSyncing.current = true;
        target.jumpTo({
          center: source.getCenter(),
          zoom: source.getZoom(),
          bearing: source.getBearing(),
          pitch: source.getPitch(),
        });
        isSyncing.current = false;
      });
    };

    syncMaps(mb, ma);
    syncMaps(ma, mb);

    mapBefore.current = mb;
    mapAfter.current = ma;

    return () => {
      mb.remove();
      ma.remove();
      mapBefore.current = null;
      mapAfter.current = null;
    };
  }, []);

  // Update center & zoom when study area changes
  useEffect(() => {
    if (mapBefore.current && mapAfter.current) {
      mapBefore.current.flyTo({
        center: studyArea.center,
        zoom: studyArea.zoom,
        duration: 1500,
      });
      mapAfter.current.flyTo({
        center: studyArea.center,
        zoom: studyArea.zoom,
        duration: 1500,
      });
    }
  }, [studyArea]);

  // Pointer drag events for the split slider
  const handlePointerDown = (e: React.PointerEvent) => {
    setIsDraggingSlider(true);
    (e.target as HTMLElement).setPointerCapture(e.pointerId);
  };

  const handlePointerMove = (e: React.PointerEvent) => {
    if (!isDraggingSlider || !containerRef.current) return;
    const rect = containerRef.current.getBoundingClientRect();
    const x = Math.max(0, Math.min(e.clientX - rect.left, rect.width));
    const percent = (x / rect.width) * 100;
    onSliderChange(percent);
  };

  const handlePointerUp = (e: React.PointerEvent) => {
    if (isDraggingSlider) {
      setIsDraggingSlider(false);
      try {
        (e.target as HTMLElement).releasePointerCapture(e.pointerId);
      } catch {}
    }
  };

  return (
    <div
      ref={containerRef}
      onPointerMove={handlePointerMove}
      onPointerUp={handlePointerUp}
      className="relative flex-1 h-full w-full bg-slate-950 overflow-hidden select-none"
    >
      {/* Background: After (Recent) Full Interactive Map */}
      <div
        ref={mapAfterContainer}
        className="absolute inset-0 w-full h-full"
        style={{
          display: viewMode === 'before' ? 'none' : 'block',
        }}
      />

      {/* Foreground: Before (Historical) Full Interactive Map clipped by slider */}
      <div
        ref={mapBeforeContainer}
        className="absolute inset-0 w-full h-full"
        style={{
          clipPath:
            viewMode === 'split'
              ? `polygon(0 0, ${sliderPos}% 0, ${sliderPos}% 100%, 0 100%)`
              : viewMode === 'before'
              ? 'polygon(0 0, 100% 0, 100% 100%, 0 100%)'
              : 'none',
          display: viewMode === 'after' ? 'none' : 'block',
        }}
      />

      {/* Real-time ML Overlays */}
      {analysis && (showClassOverlay || showChangeOverlay || viewMode === 'difference') && (
        <div className="absolute inset-0 pointer-events-none flex items-center justify-center p-8">
          <div className="relative w-full max-w-2xl aspect-square rounded-2xl overflow-hidden border border-cyan-500/40 shadow-2xl bg-slate-950/20 backdrop-blur-[2px]">
            {showClassOverlay && (
              <img
                src={sliderPos > 50 ? analysis.images.classificationAfter : analysis.images.classificationBefore}
                alt="ML Classification"
                style={{ opacity: overlayOpacity }}
                className="absolute inset-0 w-full h-full object-cover transition-opacity duration-200"
              />
            )}
            {(showChangeOverlay || viewMode === 'difference') && (
              <img
                src={analysis.images.changeOverlay}
                alt="ML Change Anomaly"
                style={{ opacity: viewMode === 'difference' ? 0.95 : overlayOpacity }}
                className="absolute inset-0 w-full h-full object-cover mix-blend-screen transition-opacity duration-200"
              />
            )}
            <div className="absolute bottom-3 left-3 px-2 py-1 rounded bg-slate-950/80 border border-cyan-500/30 font-mono text-[10px] text-cyan-300">
              ML TENSOR OVERLAY ACTIVATED
            </div>
          </div>
        </div>
      )}

      {/* Draggable Divider Bar */}
      {viewMode === 'split' && (
        <div
          onPointerDown={handlePointerDown}
          style={{ left: `${sliderPos}%` }}
          className="absolute top-0 bottom-0 w-1 -ml-0.5 bg-gradient-to-b from-cyan-400 via-emerald-400 to-cyan-400 cursor-ew-resize z-30 shadow-[0_0_15px_rgba(0,240,255,1)] flex items-center justify-center hover:w-2 transition-[width]"
        >
          <div className="w-9 h-9 rounded-full bg-slate-950 border-2 border-cyan-400 flex items-center justify-center shadow-xl shadow-cyan-500/60 text-cyan-300 font-mono text-xs hover:scale-110 transition-transform">
            ⬄
          </div>
        </div>
      )}

      {/* Floating Epoch Badges */}
      <div className="absolute top-4 left-4 z-20 px-3 py-1.5 rounded-lg bg-slate-950/85 border border-cyan-500/30 text-cyan-400 font-mono text-xs backdrop-blur-md shadow-lg flex items-center gap-2">
        <span className="w-2 h-2 rounded-full bg-cyan-400" />
        <span>HISTORICAL EPOCH: <strong>{studyArea.dates.before}</strong></span>
      </div>

      <div className="absolute top-4 right-16 z-20 px-3 py-1.5 rounded-lg bg-slate-950/85 border border-emerald-500/30 text-emerald-400 font-mono text-xs backdrop-blur-md shadow-lg flex items-center gap-2">
        <span className="w-2 h-2 rounded-full bg-emerald-400" />
        <span>RECENT EPOCH: <strong>{studyArea.dates.after}</strong></span>
      </div>

      {/* HUD Controls */}
      <div className="absolute top-4 right-4 z-20 flex flex-col gap-1.5 rounded-lg bg-slate-950/85 backdrop-blur-md border border-cyan-500/30 p-1 shadow-lg">
        <button
          onClick={() => {
            mapBefore.current?.zoomIn();
            mapAfter.current?.zoomIn();
          }}
          className="p-2 text-slate-300 hover:text-cyan-400 hover:bg-slate-900 rounded transition-colors"
          title="Zoom In"
        >
          <ZoomIn className="w-4 h-4" />
        </button>
        <button
          onClick={() => {
            mapBefore.current?.zoomOut();
            mapAfter.current?.zoomOut();
          }}
          className="p-2 text-slate-300 hover:text-cyan-400 hover:bg-slate-900 rounded transition-colors"
          title="Zoom Out"
        >
          <ZoomOut className="w-4 h-4" />
        </button>
        <button
          onClick={() => {
            mapBefore.current?.resetNorthPitch();
            mapAfter.current?.resetNorthPitch();
          }}
          className="p-2 text-slate-300 hover:text-cyan-400 hover:bg-slate-900 rounded transition-colors"
          title="Reset Orientation"
        >
          <Compass className="w-4 h-4" />
        </button>
      </div>

      {/* HUD Info Watermark */}
      <div className="absolute bottom-4 left-4 z-20 px-3.5 py-1.5 rounded-lg bg-slate-950/85 border border-slate-800 text-[11px] font-mono text-slate-300 backdrop-blur-md flex items-center gap-3 shadow-lg">
        <div className="flex items-center gap-1.5">
          <span className="w-2 h-2 rounded-full bg-cyan-400 animate-pulse" />
          <span className="text-cyan-400 font-semibold">{studyArea.name}</span>
        </div>
        <span className="text-slate-600">|</span>
        <span>IMAGERY: SUB-METER HIGH-RES (MAXAR / ESRI)</span>
        <span className="text-slate-600">|</span>
        <span>COORDINATES: {studyArea.center[1].toFixed(4)}°N, {studyArea.center[0].toFixed(4)}°E</span>
      </div>
    </div>
  );
};
