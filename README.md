# EarthLens — Planetary AI Satellite Change Detection

> **"See how our planet changes."**

EarthLens is an interactive geospatial machine learning web platform that empowers researchers, analysts, and explorers to visualize, measure, and analyze physical surface transformations across the globe over multi-temporal satellite epochs.

---

## 🛰️ Architecture & Capabilities

1. **Frontend Experience**
   - **Interactive Map:** Built with MapLibre GL JS and high-resolution Esri/Sentinel-2 basemaps.
   - **Before-and-After Split Slider:** Draggable dual-observation swipe reveal with multi-touch / mouse support.
   - **Futuristic Intelligence HUD:** Dark-mode glassmorphic cybernetic interface styled with Tailwind CSS v4 & Lucide icons.
   - **Real-Time Analytics:** Recharts visualizations detailing net categorical land-cover area shifts (km²), water retreat, canopy loss, and urban expansion.

2. **Backend & Geospatial ML Engine**
   - **Framework:** FastAPI with asynchronous ASGI execution (`uvicorn`).
   - **Multispectral Bands:** Processing calibrated Blue, Green, Red, Near-Infrared (NIR), and Shortwave-Infrared (SWIR) bands.
   - **Spectral Indices:** Calculation of standard biophysical indices:
     - **NDVI** (Normalized Difference Vegetation Index)
     - **NDWI** (Normalized Difference Water Index)
     - **NDBI** (Normalized Difference Built-up Index)
   - **Machine Learning Classification:** Trained lightweight **Random Forest Classifier** (`scikit-learn`, `joblib`) providing continuous pixel-level inference across 5 land-cover spectrums:
     - *Water bodies*
     - *Dense forest / Tree canopy*
     - *Agriculture / Grassland*
     - *Built-up / Urban infrastructure*
     - *Barren land / Desert soil*
   - **Change Anomaly Engine:** Pixel-level categorical difference mapping identifying critical degradation, waterbody desiccation, and urban sprawl.

---

## 🌍 Study Areas Included

- **Aral Sea, Central Asia:** Catastrophic water desiccation and desertification (2018 vs. 2023).
- **Dubai Coastal Zone, UAE:** Massive artificial island land reclamation and coastal urban sprawl (2016 vs. 2024).
- **Rondônia, Brazil:** Amazonian fishbone deforestation and canopy loss (2017 vs. 2023).

---

## 🚀 Running EarthLens Locally

### 1. Backend (Python + FastAPI + Scikit-Learn)
```powershell
cd A:\project2\backend
.\venv\Scripts\activate
python -m uvicorn main:app --host 127.0.0.1 --port 8000
```
Backend API will be running on: `http://127.0.0.1:8000`

### 2. Frontend (React 19 + Vite + TypeScript + Tailwind)
```powershell
cd A:\project2\frontend
npm run dev
```
Frontend Web Application will be running on: `http://127.0.0.1:5173`

---

## 📊 ML Model Status
- **Classifier:** Random Forest (80 Estimators, Max Depth 12)
- **Features Used:** `[Blue, Green, Red, NIR, SWIR, NDVI, NDWI, NDBI]`
- **Inference Speed:** < 200ms per 256x256 multispectral tile
- **Model Persistence:** Serialized to `backend/models/random_forest_landcover.joblib`
