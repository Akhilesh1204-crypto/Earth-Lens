import React from 'react';
import { motion } from 'framer-motion';
import { Globe2, ArrowRight, Cpu, Layers } from 'lucide-react';

interface LandingHeroProps {
  onExplore: () => void;
}

export const LandingHero: React.FC<LandingHeroProps> = ({ onExplore }) => {
  return (
    <div className="absolute inset-0 z-50 flex items-center justify-center bg-slate-950/90 backdrop-blur-2xl overflow-hidden">
      {/* Animated Cybernetic Grid Background */}
      <div className="absolute inset-0 bg-[radial-gradient(#00f0ff15_1px,transparent_1px)] [background-size:32px_32px] pointer-events-none" />

      {/* Atmospheric Glow Orbs */}
      <div className="absolute -top-32 -left-32 w-96 h-96 bg-cyan-500/20 rounded-full blur-3xl pointer-events-none" />
      <div className="absolute -bottom-32 -right-32 w-96 h-96 bg-emerald-500/20 rounded-full blur-3xl pointer-events-none" />

      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.8 }}
        className="relative max-w-3xl mx-auto px-6 text-center space-y-8 z-10"
      >
        {/* Status Badge */}
        <div className="inline-flex items-center gap-2 px-3 py-1.5 rounded-full bg-cyan-950/60 border border-cyan-500/40 text-cyan-300 text-xs font-mono">
          <span className="w-2 h-2 rounded-full bg-cyan-400 animate-ping" />
          <span>AUTONOMOUS ORBITAL ML CHANGE DETECTION</span>
        </div>

        {/* Main Title */}
        <div className="space-y-3">
          <h1 className="text-6xl sm:text-7xl font-extrabold tracking-tight bg-gradient-to-r from-cyan-400 via-teal-300 to-emerald-400 bg-clip-text text-transparent">
            EARTHLENS
          </h1>
          <p className="text-xl sm:text-2xl font-light text-slate-300">
            See how our planet changes.
          </p>
        </div>

        {/* Description */}
        <p className="text-sm sm:text-base text-slate-400 max-w-xl mx-auto leading-relaxed">
          Explore multi-temporal Sentinel-2 satellite observations with real-time Random Forest land-cover classification, pixel-level change detection, and dynamic geospatial metrics.
        </p>

        {/* Key Features Badges */}
        <div className="grid grid-cols-3 gap-3 max-w-lg mx-auto text-xs font-mono text-slate-300">
          <div className="p-3 rounded-xl bg-slate-900/60 border border-slate-800 flex flex-col items-center gap-1.5">
            <Globe2 className="w-5 h-5 text-cyan-400" />
            <span>Sentinel-2 MSI</span>
          </div>
          <div className="p-3 rounded-xl bg-slate-900/60 border border-slate-800 flex flex-col items-center gap-1.5">
            <Cpu className="w-5 h-5 text-emerald-400" />
            <span>Random Forest ML</span>
          </div>
          <div className="p-3 rounded-xl bg-slate-900/60 border border-slate-800 flex flex-col items-center gap-1.5">
            <Layers className="w-5 h-5 text-blue-400" />
            <span>Temporal Slider</span>
          </div>
        </div>

        {/* Call to action button */}
        <div className="pt-4">
          <button
            onClick={onExplore}
            className="group relative inline-flex items-center gap-3 px-8 py-4 rounded-xl bg-gradient-to-r from-cyan-500 to-emerald-500 text-slate-950 font-bold font-mono tracking-wider text-sm shadow-xl shadow-cyan-500/25 hover:from-cyan-400 hover:to-emerald-400 transition-all duration-300 active:scale-95 cursor-pointer"
          >
            <span>LAUNCH PLANETARY EXPLORER</span>
            <ArrowRight className="w-4 h-4 group-hover:translate-x-1 transition-transform" />
          </button>
        </div>
      </motion.div>
    </div>
  );
};
