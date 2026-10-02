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

import numpy as np
import torch
from src.models.resqnet import ResQNet
from src.optimization.allocation import AllocationProblem, GreedyAllocation

_CACHED_MODEL: Optional[ResQNet] = None

def get_model() -> ResQNet:
    global _CACHED_MODEL
    if _CACHED_MODEL is None:
        _CACHED_MODEL = ResQNet(pretrained=False)
        _CACHED_MODEL.eval()
    return _CACHED_MODEL

@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    try:
        model = get_model()
        # Synthetic demo chip matching ResQNet channels (2 SAR, 13 Optical, 6 Geo, 30 Rain)
        sar = torch.randn(1, 2, 64, 64)
        optical = torch.randn(1, 13, 64, 64)
        geo = torch.randn(1, 6, 64, 64)
        rain = torch.randn(1, 30, 1)

        with torch.no_grad():
            if request.uq_method == "mc_dropout":
                model.train()
                preds = [torch.sigmoid(model(sar, optical, geo, rain)["logits"]).squeeze().cpu().numpy() for _ in range(3)]
                model.eval()
                prob = np.mean(preds, axis=0)
                unc = np.var(preds, axis=0)
            else:
                model.eval()
                out = model(sar, optical, geo, rain)["logits"]
                prob = torch.sigmoid(out).squeeze().cpu().numpy()
                unc = prob * (1.0 - prob)

        step = max(1, prob.shape[0] // 8)
        p_sub = prob[::step, ::step].round(4).tolist()
        u_sub = unc[::step, ::step].round(4).tolist()
        ci = [float(round(np.percentile(prob, 5), 4)), float(round(np.percentile(prob, 95), 4))]

        return PredictResponse(
            probability_map=p_sub,
            uncertainty_map=u_sub,
            confidence_interval=ci,
            geojson_extent={"type": "FeatureCollection", "features": []}
        )
    except Exception as e:
        logger.error(f"Error in predict: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/impact", response_model=ImpactResponse)
async def impact(request: ImpactRequest):
    try:
        pop_count = float(request.population_data.get("count", 1000.0))
        bldg_count = float(request.building_data.get("count", 200.0))

        rng = np.random.default_rng(42)
        flood_rates = rng.beta(2, 5, size=200)
        pop_affected = flood_rates * pop_count
        bldg_affected = flood_rates * bldg_count

        return ImpactResponse(
            impact_distribution={
                "mean": float(round(np.mean(pop_affected), 2)),
                "var": float(round(np.var(pop_affected), 2)),
                "p05": float(round(np.percentile(pop_affected, 5), 2)),
                "p50": float(round(np.percentile(pop_affected, 50), 2)),
                "p95": float(round(np.percentile(pop_affected, 95), 2)),
                "buildings_mean": float(round(np.mean(bldg_affected), 2))
            },
            affected_population={
                "mean": float(round(np.mean(pop_affected), 2)),
                "95_ci": [float(round(np.percentile(pop_affected, 2.5), 2)), float(round(np.percentile(pop_affected, 97.5), 2))]
            }
        )
    except Exception as e:
        logger.error(f"Error in impact: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/allocate", response_model=AllocateResponse)
async def allocate(request: AllocateRequest):
    try:
        scenarios_input = request.impact_scenarios or [{"zone_1": 50.0}]
        depots_input = request.depots or [{"id": "depot_1", "capacity": 100.0}]

        zones = list(scenarios_input[0].keys())
        depot_ids = [d["id"] for d in depots_input]
        capacities = {d["id"]: float(d.get("capacity", 100.0)) for d in depots_input}

        scenarios = {f"s_{i}": {z: float(sc.get(z, 0.0)) for z in zones} for i, sc in enumerate(scenarios_input)}
        probs = {s: 1.0 / len(scenarios) for s in scenarios}
        travel_times = {(z, d): 10.0 for z in zones for d in depot_ids}

        problem = AllocationProblem(
            zones=zones,
            depots=depot_ids,
            capacities=capacities,
            travel_times=travel_times,
            scenarios=scenarios,
            scenario_probs=probs
        )

        solver = GreedyAllocation()
        solution = solver.solve(problem)

        plan = [{"from": j, "to": i, "amount": float(round(amt, 2))} for (i, j), amt in solution.x_ij.items() if amt > 0]
        unmet = {z: float(round(v, 2)) for z, v in solution.unmet_demand.items()}
        unmet["total"] = float(round(sum(unmet.values()), 2))

        return AllocateResponse(
            allocation_plan=plan,
            unmet_demand=unmet
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
