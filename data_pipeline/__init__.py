"""
physics_penalty_engine
======================
IEEE 33-bus distribution feeder constraint checker for EV charging placement.

Part of the EVolvAI Spatio-Temporal EV Infrastructure Optimizer project.

Public API
----------
    physics_penalty_engine(ev_scenario, ...)  → PenaltyResult
    evaluate_charger_placement(bus_ids, power_kw, ...)  → float | (float, PenaltyResult)
    evaluate_from_demand_tensor(demand, hour, ...)  → float | (float, PenaltyResult)
    bus_sweep(power_kw, ...)  → dict[int, float]
    hourly_sweep(demand, ...)  → dict[int, float]
    get_network_data()  → dict
    run_all_scenarios(...)  → dict

Quick start
-----------
    from physics_penalty_engine import physics_penalty_engine, evaluate_charger_placement

    # Single charger at Bus 18
    result = physics_penalty_engine([{"bus_id": 18, "p_kw": 150}])
    print(result.penalty_score, result.feasible)

    # Optimiser-friendly scalar call
    score = evaluate_charger_placement([18, 25, 33], 150)

    # From EVolvAI demand tensor
    import numpy as np
    demand = np.load("output/extreme_winter_storm.npy")
    from physics_penalty_engine import evaluate_from_demand_tensor
    score = evaluate_from_demand_tensor(demand, hour=18)

Solver
------
    Simplified DistFlow Forward-Backward Sweep (Baran & Wu, IEEE Trans. Power Del., 1989)
    No external power-system toolboxes required (pure NumPy).
"""

from .ieee33bus_data import get_network_data, BASE_MVA, BASE_KV, N_BUS, N_BRANCH, XFMR_KVA

from .physics_penalty_engine import (
    physics_penalty_engine,
    PenaltyResult,
)

from .evaluate_charger_placement import (
    evaluate_charger_placement,
    evaluate_from_demand_tensor,
    bus_sweep,
    hourly_sweep,
)

from .run_penalty_scenarios import run_all_scenarios

__all__ = [
    # Core engine
    "physics_penalty_engine",
    "PenaltyResult",
    # Wrappers
    "evaluate_charger_placement",
    "evaluate_from_demand_tensor",
    "bus_sweep",
    "hourly_sweep",
    # Network data
    "get_network_data",
    "BASE_MVA",
    "BASE_KV",
    "N_BUS",
    "N_BRANCH",
    "XFMR_KVA",
    # Scenario runner
    "run_all_scenarios",
]

__version__ = "1.0.0"
__author__  = "EVolvAI Team"
