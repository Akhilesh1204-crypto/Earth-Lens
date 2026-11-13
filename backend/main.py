"""
EarthLens FastAPI Backend Application
Exposes REST APIs for study areas, Sentinel-2 comparative observation data,
ML inference metrics, and multi-temporal change detection analytics.
"""

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import geo_engine

app = FastAPI(
    title="EarthLens Geospatial & ML Intelligence API",
    description="Sentinel-2 and ML-driven planetary change detection service",
    version="1.0.0"
)

# Enable CORS for frontend Vite application
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def health_check():
    return {
        "status": "online",
        "service": "EarthLens Geospatial ML Engine",
        "version": "1.0.0",
        "capabilities": [
            "Sentinel-2 Multispectral Processing",
            "Spectral Indices (NDVI, NDWI, NDBI)",
            "Random Forest Land-Cover Classification",
            "Pixel-based Temporal Change Detection",
            "Continuous Area Statistics"
        ]
    }

@app.get("/api/study-areas")
def get_study_areas():
    """Lists available historical satellite study areas with geographic bounds and dates."""
    return geo_engine.get_study_areas_list()

@app.get("/api/analysis/{area_id}")
def get_change_analysis(area_id: str, retrain: bool = Query(False)):
    """
    Computes and returns full change intelligence:
    - Before & After satellite imagery
    - Random Forest land-cover classification
    - Color-coded change anomaly layer
    - Net land-cover shifts and categorical charts
    - Machine learning evaluation metrics
    """
    if area_id not in geo_engine.STUDY_AREAS:
        raise HTTPException(status_code=404, detail=f"Study area '{area_id}' not found.")
    
    if retrain:
        geo_engine.train_or_load_rf_classifier(force_retrain=True)
        
    analysis = geo_engine.compute_change_analysis(area_id)
    return analysis

@app.get("/api/model/metrics")
def get_model_metrics():
    """Returns classification report, confusion matrix, and feature importances."""
    _, metrics = geo_engine.train_or_load_rf_classifier()
    return metrics

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
