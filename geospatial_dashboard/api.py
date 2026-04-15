import requests as req
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import os
import json
import numpy as np
from gini import calculate_gini, get_accessibility_scores

# Load environment variables
load_dotenv()
OCMAP_API_KEY = os.getenv("OCMAP_API_KEY")

app = FastAPI(
    title="EVolvAI API",
    description="REST API for EV Infrastructure Dashboard (NYC) — IEEE IES Hackathon",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load mock data once at startup
with open("mock_data.json") as f:
    RAW_DATA = json.load(f)

NODES = RAW_DATA["nodes"]

# Scenario configs — multipliers for charger demand
SCENARIOS = {
    "baseline": {
        "label": "Baseline",
        "demand_multiplier": 1.0,
        "temp_drop": 0,
        "overload_threshold": 0.80
    },
    "extreme_winter_storm": {
        "label": "Extreme winter storm + 2.5x fleet",
        "demand_multiplier": 2.5,
        "temp_drop": 15,
        "overload_threshold": 0.45
    },
    "summer_peak": {
        "label": "High summer temperatures + 1.5x fleet",
        "demand_multiplier": 1.5,
        "temp_drop": -10,
        "overload_threshold": 0.65
    },
    "full_electrification": {
        "label": "Normal weather + 3.0x full fleet electrification",
        "demand_multiplier": 3.0,
        "temp_drop": 0,
        "overload_threshold": 0.35
    },
    "extreme_winter_v2": {
        "label": "Winter storm + full electrification + weekend",
        "demand_multiplier": 2.5,
        "temp_drop": 15,
        "overload_threshold": 0.45
    },
    "rush_hour_gridlock": {
        "label": "Peak rush hour with 2x fleet electrification",
        "demand_multiplier": 2.0,
        "temp_drop": 0,
        "overload_threshold": 0.55
    }
}


def apply_scenario(nodes: list, scenario_key: str) -> list:
    """Apply scenario multipliers or actual model output to node data."""
    config = SCENARIOS[scenario_key]
    
    # Try to load real model output from ../output/
    # The generative core output is typically [24, 32] --> [SEQ_LEN, NUM_NODES]
    model_output = None
    # Check both potential naming conventions seen in output/
    paths_to_check = [
        os.path.join("..", "output", f"{scenario_key}.npy"),
        os.path.join("..", "output", f"scenario_{scenario_key}.npy")
    ]
    
    for path in paths_to_check:
        if os.path.exists(path):
            try:
                data = np.load(path)
                # If shape is [24, 32], take the peak demand per node
                if data.ndim == 2:
                    model_output = np.max(data, axis=0)
                    break
            except Exception as e:
                print(f"Error loading model output from {path}: {e}")

    result = []
    for node in nodes:
        adjusted = dict(node)
        
        # Determine demand
        if model_output is not None and node["node_id"] > 1:
            # Map node_id (2..33) to model_output index (0..31)
            node_idx = min(node["node_id"] - 2, len(model_output) - 1)
            demand = float(model_output[node_idx])
            adjusted["effective_demand_kw"] = round(demand, 2)
        else:
            # Fallback or substation baseline
            base = node["charger_count"] * 7.2
            if node["node_id"] == 1:
                # Substation node usually has high base demand but we use it as a reference
                adjusted["effective_demand_kw"] = round(base * config["demand_multiplier"], 2)
            else:
                adjusted["effective_demand_kw"] = round(base * config["demand_multiplier"], 2)

        # Update overload status based on the new demand vs threshold
        # In a real system this would be based on xfmr capacity, here we use gini as a proxy for stress
        adjusted["transformer_overload"] = node["gini_score"] > config["overload_threshold"]
        
        result.append(adjusted)
    return result


# ─── Endpoints ───────────────────────────────────────────────

@app.get("/")
def root():
    return {"message": "EVolvAI API is running", "docs": "/docs"}


@app.get("/api/nodes")
def get_all_nodes():
    """Return all 33 IEEE bus nodes with baseline data."""
    return {
        "scenario": "baseline",
        "node_count": len(NODES),
        "nodes": NODES
    }


@app.get("/api/nodes/{scenario}")
def get_nodes_by_scenario(scenario: str):
    """
    Return nodes adjusted for a specific scenario.
    Options: baseline | extreme_winter_storm | summer_peak | full_electrification | extreme_winter_v2 | rush_hour_gridlock
    """
    if scenario not in SCENARIOS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown scenario '{scenario}'. Choose from: {list(SCENARIOS.keys())}"
        )
    
    adjusted_nodes = apply_scenario(NODES, scenario)
    
    return {
        "scenario": scenario,
        "scenario_label": SCENARIOS[scenario]["label"],
        "demand_multiplier": SCENARIOS[scenario]["demand_multiplier"],
        "node_count": len(adjusted_nodes),
        "nodes": adjusted_nodes
    }


@app.get("/api/gini")
def get_gini_score():
    """Return the Gini accessibility index for baseline charger distribution."""
    scores = get_accessibility_scores(NODES)
    gini = calculate_gini(scores)
    
    overloaded = [n for n in NODES if n["transformer_overload"]]
    
    return {
        "gini_index": gini,
        "interpretation": "0 = perfectly equal access, 1 = fully unequal",
        "total_nodes": len(NODES),
        "overloaded_nodes": len(overloaded),
        "total_chargers": sum(n["charger_count"] for n in NODES),
        "nodes_with_zero_chargers": sum(1 for n in NODES if n["charger_count"] == 0)
    }


@app.get("/api/gini/{scenario}")
def get_gini_by_scenario(scenario: str):
    """Return Gini score for a specific scenario."""
    if scenario not in SCENARIOS:
        raise HTTPException(status_code=400, detail=f"Unknown scenario: {scenario}")
    
    adjusted_nodes = apply_scenario(NODES, scenario)
    scores = get_accessibility_scores(adjusted_nodes)
    gini = calculate_gini(scores)
    overloaded = [n for n in adjusted_nodes if n["transformer_overload"]]
    
    return {
        "scenario": scenario,
        "gini_index": gini,
        "overloaded_nodes": len(overloaded),
        "demand_multiplier": SCENARIOS[scenario]["demand_multiplier"]
    }


@app.get("/api/scenarios")
def get_scenarios():
    """Return list of available scenarios."""
    return {"scenarios": list(SCENARIOS.keys()), "details": SCENARIOS}

@app.get("/api/optimal-layout")
def get_optimal_layout():
    """Return the final optimal charger layout from the Risk Engine GA."""
    # Check both potential output directory locations
    paths = [
        os.path.join("..", "output", "final_optimal_layout.json"),
        os.path.join("..", "output", "output", "final_optimal_layout.json")
    ]
    for path in paths:
        if os.path.exists(path):
            with open(path) as f:
                return json.load(f)
    return {"error": "Optimal layout not found. Make sure to run risk_engine/optimizer_ga.py first."}

@app.get("/api/real_chargers")
def get_real_chargers():
    try:
        response = req.get(
            "https://api.openchargemap.io/v3/poi/",
            params={
                "key": OCMAP_API_KEY,
                "latitude": 40.7580,
                "longitude": -73.9855,
                "distance": 20,
                "distanceunit": "km",
                "maxresults": 100,
                "compact": True,
                "verbose": False,
                "countrycode": "US",
            },
            headers={"User-Agent": "EVolvAI-Dashboard/1.0"},
            timeout=10
        )
        if response.status_code != 200:
            return {"error": f"API returned status {response.status_code}", "chargers": []}
        
        data = response.json()
        chargers = []
        for item in data:
            if item.get("AddressInfo"):
                chargers.append({
                    "id": item.get("ID"),
                    "name": item["AddressInfo"].get("Title", "Unknown"),
                    "lat": item["AddressInfo"].get("Latitude"),
                    "lng": item["AddressInfo"].get("Longitude"),
                    "address": item["AddressInfo"].get("AddressLine1", ""),
                    "num_points": item.get("NumberOfPoints", 1),
                    "status": item.get("StatusType", {}).get("Title", "Unknown") if item.get("StatusType") else "Unknown",
                    "operator": item.get("OperatorInfo", {}).get("Title", "Unknown") if item.get("OperatorInfo") else "Unknown",
                })
        return {"source": "OpenChargeMap", "count": len(chargers), "chargers": chargers}
    except Exception as e:
        print(f"Error fetching chargers: {e}")
        return {"error": str(e), "chargers": []}