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
  const mapContainer = useRef<HTMLDivElement>(null);
  const mapRef = useRef<maplibregl.Map | null>(null);
  const [isDraggingSlider, setIsDraggingSlider] = useState(false);
  const containerRef = useRef<HTMLDivElement>(null);

  // Initialize MapLibre GL instance
  useEffect(() => {
    if (!mapContainer.current || mapRef.current) return;

    const map = new maplibregl.Map({
      container: mapContainer.current,
      style: {
        version: 8,
        sources: {
          'esri-satellite': {
            type: 'raster',
            tiles: [
              'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
            ],
            tileSize: 256,
            attribution: 'Esri, Maxar, Earthstar Geographics, USDA, USGS, AeroGRID, IGN, and the GIS User Community',
          },
        },
        layers: [
          {
            id: 'esri-satellite-layer',
            type: 'raster',
            source: 'esri-satellite',
            minzoom: 0,
            maxzoom: 19,
          },
        ],
      },
      center: studyArea.center,
      zoom: studyArea.zoom,
      attributionControl: false,
    });

    map.addControl(new maplibregl.AttributionControl({ compact: true }), 'bottom-left');
    mapRef.current = map;

    return () => {
      map.remove();
      mapRef.current = null;
    };
  }, []);

  // Update map center when study area changes
  useEffect(() => {
    if (mapRef.current) {
      mapRef.current.flyTo({
        center: studyArea.center,
        zoom: studyArea.zoom,
        essential: true,
        duration: 2000,
      });
    }
  }, [studyArea]);

  // Handle slider drag events
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
      {/* Basemap Canvas */}
      <div ref={mapContainer} className="absolute inset-0 w-full h-full" />

      {/* Satellite Imagery & Analysis Canvas Overlay */}
      {analysis && (
        <div className="absolute inset-0 pointer-events-none flex items-center justify-center p-4">
          <div className="relative w-full max-w-4xl aspect-[4/3] rounded-2xl overflow-hidden border border-cyan-500/30 shadow-2xl shadow-cyan-950/80 bg-slate-950/40 backdrop-blur-sm pointer-events-auto">
            {/* View Mode: AFTER / Right Side */}
            <div className="absolute inset-0 w-full h-full">
              <img
                src={analysis.images.satelliteAfter}
                alt="Satellite After"
                className="w-full h-full object-cover"
              />
              {/* Optional Land Cover Classification After */}
              {showClassOverlay && (
                <img
                  src={analysis.images.classificationAfter}
                  alt="Classification After"
                  style={{ opacity: overlayOpacity }}
                  className="absolute inset-0 w-full h-full object-cover transition-opacity duration-200"
                />
              )}
            </div>

            {/* Split Slider: BEFORE / Left Side (Clipped via clip-path) */}
            {viewMode === 'split' && (
              <div
                className="absolute inset-0 w-full h-full overflow-hidden"
                style={{
                  clipPath: `polygon(0 0, ${sliderPos}% 0, ${sliderPos}% 100%, 0 100%)`,
                }}
              >
                <img
                  src={analysis.images.satelliteBefore}
                  alt="Satellite Before"
                  className="w-full h-full object-cover"
                />
                {showClassOverlay && (
                  <img
                    src={analysis.images.classificationBefore}
                    alt="Classification Before"
                    style={{ opacity: overlayOpacity }}
                    className="absolute inset-0 w-full h-full object-cover transition-opacity duration-200"
                  />
                )}
              </div>
            )}

            {/* Full Before View Mode */}
            {viewMode === 'before' && (
              <div className="absolute inset-0 w-full h-full">
                <img
                  src={analysis.images.satelliteBefore}
                  alt="Satellite Before"
                  className="w-full h-full object-cover"
                />
                {showClassOverlay && (
                  <img
                    src={analysis.images.classificationBefore}
                    alt="Classification Before"
                    style={{ opacity: overlayOpacity }}
                    className="absolute inset-0 w-full h-full object-cover transition-opacity duration-200"
                  />
                )}
              </div>
            )}

            {/* Change Anomaly Overlay (shows pixel transitions) */}
            {(showChangeOverlay || viewMode === 'difference') && (
              <img
                src={analysis.images.changeOverlay}
                alt="Change Detection Anomaly Overlay"
                style={{ opacity: viewMode === 'difference' ? 0.95 : overlayOpacity }}
                className="absolute inset-0 w-full h-full object-cover mix-blend-screen pointer-events-none transition-opacity duration-200"
              />
            )}

            {/* Draggable Divider Bar */}
            {viewMode === 'split' && (
              <div
                onPointerDown={handlePointerDown}
                style={{ left: `${sliderPos}%` }}
                className="absolute top-0 bottom-0 w-1 -ml-0.5 bg-gradient-to-b from-cyan-400 via-emerald-400 to-cyan-400 cursor-ew-resize z-30 shadow-[0_0_12px_rgba(0,240,255,0.8)] flex items-center justify-center hover:w-1.5 transition-[width]"
              >
                {/* Thumb Pill */}
                <div className="w-8 h-8 rounded-full bg-slate-950 border-2 border-cyan-400 flex items-center justify-center shadow-lg shadow-cyan-500/50 text-cyan-300 font-mono text-[10px]">
                  ⬄
                </div>
              </div>
            )}

            {/* In-viewport Temporal Epoch Badges */}
            <div className="absolute top-3 left-3 z-20 px-2.5 py-1 rounded bg-slate-950/80 border border-cyan-500/30 text-cyan-400 font-mono text-xs backdrop-blur-md">
              <span className="text-slate-400">EPOCH T0:</span> {studyArea.dates.before}
            </div>

            <div className="absolute top-3 right-3 z-20 px-2.5 py-1 rounded bg-slate-950/80 border border-emerald-500/30 text-emerald-400 font-mono text-xs backdrop-blur-md">
              <span className="text-slate-400">EPOCH T1:</span> {studyArea.dates.after}
            </div>

            {/* Change rate watermark badge */}
            <div className="absolute bottom-3 right-3 z-20 px-3 py-1.5 rounded-lg bg-slate-950/85 border border-cyan-500/30 backdrop-blur-md flex items-center gap-2">
              <span className="w-2 h-2 rounded-full bg-red-500 animate-ping" />
              <span className="text-[11px] font-mono text-slate-300">
                CHANGED SURFACE: <strong className="text-cyan-400">{analysis.summary.changePercentage}%</strong> ({analysis.summary.totalChangedAreaKm2} km²)
              </span>
            </div>
          </div>
        </div>
      )}

      {/* Map Control HUD Overlay */}
      <div className="absolute top-4 right-4 z-10 flex flex-col gap-2">
        <div className="flex flex-col rounded-lg bg-slate-950/80 backdrop-blur-md border border-cyan-500/20 p-1 shadow-lg">
          <button
            onClick={() => mapRef.current?.zoomIn()}
            className="p-2 text-slate-300 hover:text-cyan-400 hover:bg-slate-900 rounded transition-colors"
            title="Zoom In"
          >
            <ZoomIn className="w-4 h-4" />
          </button>
          <button
            onClick={() => mapRef.current?.zoomOut()}
            className="p-2 text-slate-300 hover:text-cyan-400 hover:bg-slate-900 rounded transition-colors"
            title="Zoom Out"
          >
            <ZoomOut className="w-4 h-4" />
          </button>
          <button
            onClick={() => mapRef.current?.resetNorthPitch()}
            className="p-2 text-slate-300 hover:text-cyan-400 hover:bg-slate-900 rounded transition-colors"
            title="Reset North Orientation"
          >
            <Compass className="w-4 h-4" />
          </button>
        </div>
      </div>

      {/* Floating HUD status in bottom corner */}
      <div className="absolute bottom-4 left-4 z-10 px-3 py-1.5 rounded bg-slate-950/80 border border-slate-800 text-[10px] font-mono text-slate-400 backdrop-blur-md flex items-center gap-2">
        <span className="w-1.5 h-1.5 rounded-full bg-cyan-400" />
        <span>COORDINATES: {studyArea.center[1].toFixed(4)}°N, {studyArea.center[0].toFixed(4)}°E</span>
        <span className="text-slate-400">|</span>
        <span>RESOLUTION: 10M/PX (SENTINEL-2 MSI)</span>
      </div>
    </div>
  );
};
