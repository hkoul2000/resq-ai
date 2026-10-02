"""ResQ-AI FastAPI service.

Endpoints:
  POST /predict - Flood prediction with uncertainty
  POST /impact - Impact assessment
  POST /allocate - Resource allocation optimization  
  GET /health - Health check
  GET /config - Current configuration
"""

import logging
from typing import Dict, Any, List, Optional
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="ResQ-AI API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class PredictRequest(BaseModel):
    uq_method: str = "ensemble"
    use_demo: bool = True

class PredictResponse(BaseModel):
    probability_map: List[List[float]]
    uncertainty_map: List[List[float]]
    confidence_interval: List[float]
    geojson_extent: Dict[str, Any]

class ImpactRequest(BaseModel):
    prediction_id: str
    population_data: Dict[str, Any]
    building_data: Dict[str, Any]

class ImpactResponse(BaseModel):
    impact_distribution: Dict[str, Any]
    affected_population: Dict[str, Any]

class AllocateRequest(BaseModel):
    impact_scenarios: List[Dict[str, Any]]
    depots: List[Dict[str, Any]]
    method: str = "cvar"
    alpha: float = 0.95
    num_scenarios: int = 100

class AllocateResponse(BaseModel):
    allocation_plan: List[Dict[str, Any]]
    unmet_demand: Dict[str, Any]

@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    try:
        return PredictResponse(
            probability_map=[[0.1, 0.9], [0.2, 0.8]],
            uncertainty_map=[[0.05, 0.1], [0.02, 0.05]],
            confidence_interval=[0.1, 0.9],
            geojson_extent={"type": "FeatureCollection", "features": []}
        )
    except Exception as e:
        logger.error(f"Error in predict: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/impact", response_model=ImpactResponse)
async def impact(request: ImpactRequest):
    try:
        return ImpactResponse(
            impact_distribution={"mean": 100, "var": 10},
            affected_population={"95_ci": [80, 120]}
        )
    except Exception as e:
        logger.error(f"Error in impact: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/allocate", response_model=AllocateResponse)
async def allocate(request: AllocateRequest):
    try:
        return AllocateResponse(
            allocation_plan=[],
            unmet_demand={"total": 0}
        )
    except Exception as e:
        logger.error(f"Error in allocate: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.get("/config")
async def config():
    return {"uq_methods": ["ensemble", "mc_dropout", "evidential", "conformal"]}
