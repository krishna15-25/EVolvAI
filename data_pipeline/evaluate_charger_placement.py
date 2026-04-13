"""
evaluate_charger_placement.py
==============================
Convenience wrapper around physics_penalty_engine() designed for
integration with optimisation algorithms (GA, PSO, greedy search, etc.)
and with the EVolvAI generative demand tensors.

Translated from: evaluate_charger_placement.m (MATLAB)
Project:         EVolvAI — Physics Penalty Engine module

Usage
-----
    # Simple scalar call (optimiser loop)
    from physics_penalty_engine.evaluate_charger_placement import evaluate_charger_placement

    score = evaluate_charger_placement(bus_ids=[18, 25], power_kw=150)
    score, result = evaluate_charger_placement(bus_ids=[18, 25], power_kw=[150, 100],
                                               return_result=True)

    # From EVolvAI demand tensor
    import numpy as np
    demand = np.load("output/extreme_winter_storm.npy")  # [24, 50] kW
    score = evaluate_from_demand_tensor(demand, hour=18, bus_offset=2)
"""

from __future__ import annotations

from typing import Union

import numpy as np

from .physics_penalty_engine import PenaltyResult, physics_penalty_engine
from .ieee33bus_data import V_MIN_DEFAULT, V_MAX_DEFAULT, I_LIM_DEFAULT, XFMR_KVA


# ---------------------------------------------------------------------------
# Primary API — mirrors MATLAB evaluate_charger_placement()
# ---------------------------------------------------------------------------
def evaluate_charger_placement(
    bus_ids:       Union[int, list[int], np.ndarray],
    power_kw:      Union[float, list[float], np.ndarray],
    q_kvar:        Union[float, list[float], None] = None,
    v_min:         float = V_MIN_DEFAULT,
    v_max:         float = V_MAX_DEFAULT,
    i_lim_pu:      float = I_LIM_DEFAULT,
    xfmr_kva:      float = XFMR_KVA,
    w_voltage:     float = 1000.0,
    w_thermal:     float = 500.0,
    w_xfmr:        float = 800.0,
    verbose:       bool  = False,
    return_result: bool  = False,
) -> Union[float, tuple[float, PenaltyResult]]:
    """
    Evaluate the penalty score for a proposed EV charger placement.

    Designed as a drop-in objective function for optimisation algorithms.
    By default returns only the scalar penalty score for minimal overhead
    in tight optimisation loops.

    Parameters
    ----------
    bus_ids : int | list[int] | ndarray
        Bus number(s) where chargers are placed. Each must be in 2–33.
        Scalar input places one charger.

    power_kw : float | list[float] | ndarray
        Charger active power demand(s) [kW].
        If scalar, the same power is applied to all buses in bus_ids.
        If array, must have the same length as bus_ids.

    q_kvar : float | list[float] | None
        Reactive power demand(s) [kVAr].
        If None (default), computed at pf = 0.90 lagging per charger.

    v_min, v_max : float
        Voltage bounds [pu]. Default 0.95 – 1.05 pu.

    i_lim_pu : float
        Branch thermal limit [pu]. Default 1.00.

    xfmr_kva : float
        Transformer rating [kVA]. Default 5000.

    w_voltage, w_thermal, w_xfmr : float
        Penalty weights.

    verbose : bool
        If True, prints full analysis report. Default False (quiet for
        use inside optimisation loops).

    return_result : bool
        If False (default), returns only the scalar penalty score.
        If True, returns (score, PenaltyResult).

    Returns
    -------
    float
        Penalty score (lower = better; 0.0 = fully feasible).
    or
    tuple[float, PenaltyResult]
        If return_result=True.

    Examples
    --------
    # Greedy best-bus search for a single 150 kW charger
    scores = {}
    for bus in range(2, 34):
        scores[bus] = evaluate_charger_placement(bus, 150)
    best_bus = min(scores, key=scores.get)
    print(f"Best bus: {best_bus}  (penalty = {scores[best_bus]:.2f})")

    # Multi-charger evaluation
    score = evaluate_charger_placement([7, 14, 26], [100, 150, 75])
    """
    # ---- Normalise inputs to lists ----------------------------------------
    if isinstance(bus_ids, (int, np.integer)):
        bus_ids = [int(bus_ids)]
    else:
        bus_ids = [int(b) for b in bus_ids]

    if isinstance(power_kw, (int, float, np.floating)):
        power_kw = [float(power_kw)] * len(bus_ids)
    else:
        power_kw = [float(p) for p in power_kw]

    if len(bus_ids) != len(power_kw):
        raise ValueError(
            f"bus_ids (len={len(bus_ids)}) and power_kw (len={len(power_kw)}) "
            "must have the same length."
        )

    if q_kvar is not None:
        if isinstance(q_kvar, (int, float, np.floating)):
            q_kvar_list: list[float | None] = [float(q_kvar)] * len(bus_ids)
        else:
            q_kvar_list = [float(q) for q in q_kvar]
    else:
        q_kvar_list = [None] * len(bus_ids)

    # ---- Build scenario struct list ----------------------------------------
    scenario = [
        {"bus_id": b, "p_kw": p, "q_kvar": q}
        for b, p, q in zip(bus_ids, power_kw, q_kvar_list)
    ]

    # ---- Run engine --------------------------------------------------------
    result = physics_penalty_engine(
        scenario,
        v_min=v_min, v_max=v_max, i_lim_pu=i_lim_pu, xfmr_kva=xfmr_kva,
        w_voltage=w_voltage, w_thermal=w_thermal, w_xfmr=w_xfmr,
        verbose=verbose,
    )

    if return_result:
        return result.penalty_score, result
    return result.penalty_score


# ---------------------------------------------------------------------------
# EVolvAI integration — evaluate directly from demand tensor
# ---------------------------------------------------------------------------
def evaluate_from_demand_tensor(
    demand:     np.ndarray,
    hour:       int,
    bus_offset: int = 2,
    **kwargs,
) -> Union[float, tuple[float, PenaltyResult]]:
    """
    Evaluate penalty directly from an EVolvAI demand tensor.

    The generative module outputs tensors of shape [24, N_NODES] in kW.
    This function maps tensor columns to IEEE 33-bus load buses and runs
    the physics penalty engine for a specified hour.

    Parameters
    ----------
    demand : ndarray  shape [24, N_NODES]
        Demand tensor from EVolvAI generative module (kW per node per hour).
        Loaded from e.g. ``np.load("output/extreme_winter_storm.npy")``.

    hour : int
        Hour index (0 = midnight, 18 = 6 PM peak, etc.).

    bus_offset : int
        First bus number to map to tensor column 0. Default 2 (since Bus 1
        is the slack bus and cannot have a charger).

        Mapping:  Bus (bus_offset + i) ← demand[hour, i]
        With default offset=2: Bus 2 ← col 0, Bus 3 ← col 1, ...,
        Bus 33 ← col 31 (first 32 columns used; remainder ignored).

    **kwargs
        Passed through to evaluate_charger_placement() — use to override
        constraint limits, penalty weights, verbose, return_result, etc.

    Returns
    -------
    float or tuple[float, PenaltyResult]
        Penalty score (or score + full result if return_result=True).

    Example
    -------
        import numpy as np
        demand = np.load("output/extreme_winter_storm.npy")  # [24, 50]
        score  = evaluate_from_demand_tensor(demand, hour=18)
        print(f"Peak hour penalty: {score:.2f}")

        # Full result with verbose report
        score, res = evaluate_from_demand_tensor(
            demand, hour=18, verbose=True, return_result=True
        )
    """
    if demand.ndim != 2 or demand.shape[0] != 24:
        raise ValueError(
            f"demand must have shape [24, N_NODES], got {demand.shape}."
        )
    if not (0 <= hour <= 23):
        raise ValueError(f"hour must be 0–23, got {hour}.")

    n_nodes     = demand.shape[1]
    max_buses   = 33 - bus_offset + 1          # how many load buses fit
    n_cols      = min(n_nodes, max_buses)       # columns we can map

    bus_ids  = list(range(bus_offset, bus_offset + n_cols))
    power_kw = demand[hour, :n_cols].tolist()

    return evaluate_charger_placement(bus_ids, power_kw, **kwargs)


# ---------------------------------------------------------------------------
# Batch evaluation — sweep across all buses or all hours
# ---------------------------------------------------------------------------
def bus_sweep(
    power_kw: float = 150.0,
    **kwargs,
) -> dict[int, float]:
    """
    Evaluate penalty for a single charger at every load bus (2–33).

    Parameters
    ----------
    power_kw : float
        Charger power [kW]. Default 150 kW.
    **kwargs
        Passed to evaluate_charger_placement().

    Returns
    -------
    dict[int, float]
        {bus_id: penalty_score} for buses 2–33.

    Example
    -------
        scores = bus_sweep(150)
        best  = min(scores, key=scores.get)
        worst = max(scores, key=scores.get)
        print(f"Best bus: {best}, Worst bus: {worst}")
    """
    return {
        bus: evaluate_charger_placement(bus, power_kw, **kwargs)
        for bus in range(2, 34)
    }


def hourly_sweep(
    demand:     np.ndarray,
    bus_offset: int = 2,
    **kwargs,
) -> dict[int, float]:
    """
    Evaluate penalty for every hour of a 24-hour demand tensor.

    Parameters
    ----------
    demand : ndarray shape [24, N_NODES]
        EVolvAI demand tensor (kW).
    bus_offset : int
        First bus number (default 2).
    **kwargs
        Passed to evaluate_from_demand_tensor().

    Returns
    -------
    dict[int, float]
        {hour: penalty_score} for hours 0–23.

    Example
    -------
        import numpy as np
        demand = np.load("output/extreme_winter_storm.npy")
        hourly = hourly_sweep(demand)
        peak_hour = max(hourly, key=hourly.get)
        print(f"Most constrained hour: {peak_hour}:00  (penalty={hourly[peak_hour]:.2f})")
    """
    return {
        hour: evaluate_from_demand_tensor(demand, hour, bus_offset, **kwargs)
        for hour in range(24)
    }
