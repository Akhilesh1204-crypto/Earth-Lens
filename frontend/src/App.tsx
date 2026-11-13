import { useState, useEffect } from 'react';
import { ControlPanel } from './components/ControlPanel';
import { SatelliteMap } from './components/SatelliteMap';
import { IntelligencePanel } from './components/IntelligencePanel';
import { LandingHero } from './components/LandingHero';
import type { StudyArea, ChangeAnalysisResponse } from './types/earthlens';

const API_BASE = 'http://127.0.0.1:8000';

export function App() {
  const [showLanding, setShowLanding] = useState(true);
  const [studyAreas, setStudyAreas] = useState<StudyArea[]>([]);
  const [selectedArea, setSelectedArea] = useState<StudyArea | null>(null);
  const [analysis, setAnalysis] = useState<ChangeAnalysisResponse | null>(null);
  const [isLoading, setIsLoading] = useState(false);

  // Explorer states
  const [sliderPos, setSliderPos] = useState(50);
  const [showClassOverlay, setShowClassOverlay] = useState(false);
  const [showChangeOverlay, setShowChangeOverlay] = useState(false);
  const [overlayOpacity, setOverlayOpacity] = useState(0.7);
  const [activeViewMode, setActiveViewMode] = useState<'split' | 'before' | 'after' | 'difference'>('split');

  // Load study areas on mount
  useEffect(() => {
    fetch(`${API_BASE}/api/study-areas`)
      .then((res) => res.json())
      .then((data: StudyArea[]) => {
        setStudyAreas(data);
        if (data.length > 0) {
          setSelectedArea(data[0]);
          loadAnalysis(data[0].id);
        }
      })
      .catch((err) => {
        console.error('Failed to load study areas from backend:', err);
      });
  }, []);

  const loadAnalysis = async (areaId: string) => {
    setIsLoading(true);
    try {
      const res = await fetch(`${API_BASE}/api/analysis/${areaId}`);
      if (!res.ok) throw new Error('API failed');
      const data: ChangeAnalysisResponse = await res.json();
      setAnalysis(data);
    } catch (err) {
      console.error('Error fetching change analysis:', err);
    } finally {
      setIsLoading(false);
    }
  };

  const handleSelectArea = (area: StudyArea) => {
    setSelectedArea(area);
    loadAnalysis(area.id);
  };

  return (
    <div className="relative w-screen h-screen bg-slate-950 text-slate-100 flex overflow-hidden font-sans">
      {/* Cinematic Landing Screen */}
      {showLanding && <LandingHero onExplore={() => setShowLanding(false)} />}

      {/* Main 3-Section Dashboard Explorer */}
      {selectedArea && (
        <>
          {/* Left Section: Controls & Epoch Selectors */}
          <ControlPanel
            studyAreas={studyAreas}
            selectedArea={selectedArea}
            onSelectArea={handleSelectArea}
            sliderPos={sliderPos}
            onSliderChange={setSliderPos}
            showClassOverlay={showClassOverlay}
            onToggleClassOverlay={() => setShowClassOverlay(!showClassOverlay)}
            showChangeOverlay={showChangeOverlay}
            onToggleChangeOverlay={() => setShowChangeOverlay(!showChangeOverlay)}
            overlayOpacity={overlayOpacity}
            onOpacityChange={setOverlayOpacity}
            isRunningAnalysis={isLoading}
            onRunAnalysis={() => loadAnalysis(selectedArea.id)}
            activeViewMode={activeViewMode}
            onViewModeChange={setActiveViewMode}
          />

          {/* Center Section: Dominant MapLibre Satellite View & Draggable Slider */}
          <SatelliteMap
            studyArea={selectedArea}
            analysis={analysis}
            sliderPos={sliderPos}
            onSliderChange={setSliderPos}
            showClassOverlay={showClassOverlay}
            showChangeOverlay={showChangeOverlay}
            overlayOpacity={overlayOpacity}
            viewMode={activeViewMode}
          />

          {/* Right Section: Quantitative ML Change Intelligence Dashboard */}
          <IntelligencePanel
            analysis={analysis}
            isLoading={isLoading}
          />
        </>
      )}
    </div>
  );
}

export default App;
