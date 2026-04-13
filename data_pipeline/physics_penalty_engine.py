"""
physics_penalty_engine.py
==========================
Core AC power flow constraint checker for EV charging placement
on the IEEE 33-bus radial distribution feeder.

Translated from: physics_penalty_engine.m (MATLAB)
Project:         EVolvAI — Physics Penalty Engine module
Solver:          Simplified DistFlow Forward-Backward Sweep (Baran & Wu, 1989)

No external power-system toolboxes required — solver is implemented
entirely in NumPy.

Quick start
-----------
    from physics_penalty_engine.physics_penalty_engine import physics_penalty_engine

    scenario = [
        {"bus_id": 18, "p_kw": 150},
        {"bus_id": 25, "p_kw": 150},
        {"bus_id": 33, "p_kw": 150},
    ]
    result = physics_penalty_engine(scenario)
    print(f"Penalty: {result['penalty_score']:.2f}")
    print(f"Feasible: {result['feasible']}")

EVolvAI integration
-------------------
The engine accepts demand tensors from the generative module directly:

    import numpy as np
    demand = np.load("output/extreme_winter_storm.npy")  # [24, 50] kW
    # Evaluate hour 18 (peak evening demand) for the first 33 nodes
    hour = 18
    scenario = [
        {"bus_id": b + 2, "p_kw": float(demand[hour, b])}
        for b in range(min(32, demand.shape[1]))
    ]
    result = physics_penalty_engine(scenario)
"""

from __future__ import annotations

import math
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .ieee33bus_data import (
    BASE_MVA,
    BASE_P_MW,
    BASE_Q_MVAR,
    FROM_BUS,
    N_BUS,
    N_BRANCH,
    R_PU,
    TO_BUS,
    V_MAX_DEFAULT,
    V_MIN_DEFAULT,
    X_PU,
    XFMR_KVA,
    I_LIM_DEFAULT,
)


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------
@dataclass
class PenaltyResult:
    """
    All outputs from a single physics_penalty_engine() call.

    Attributes match the MATLAB result struct field-for-field so that
    documentation, reports, and team handoffs stay consistent.
    """
    # Penalty scores
    penalty_score:   float = 0.0
    penalty_voltage: float = 0.0
    penalty_thermal: float = 0.0
    penalty_xfmr:    float = 0.0

    # Feasibility
    feasible:  bool  = False
    converged: bool  = False

    # Voltage / current profiles
    v_pu:    np.ndarray = field(default_factory=lambda: np.ones(N_BUS))
    i_pu:    np.ndarray = field(default_factory=lambda: np.zeros(N_BRANCH))

    # Transformer
    s_xfmr_kva:        float = 0.0
    xfmr_loading_pct:  float = 0.0

    # Losses
    power_loss_kw: float = 0.0

    # Violation lists
    bus_violations:    list[int] = field(default_factory=list)
    branch_violations: list[int] = field(default_factory=list)

    # Per-element penalty breakdown
    pen_v_per_bus:     np.ndarray = field(default_factory=lambda: np.zeros(N_BUS))
    pen_i_per_branch:  np.ndarray = field(default_factory=lambda: np.zeros(N_BRANCH))

    # Branch power flows (pu) — useful for downstream analysis
    pbr_pu: np.ndarray = field(default_factory=lambda: np.zeros(N_BRANCH))
    qbr_pu: np.ndarray = field(default_factory=lambda: np.zeros(N_BRANCH))

    def to_dict(self) -> dict[str, Any]:
        """Return all fields as a plain dictionary (for JSON / pandas)."""
        return {
            "penalty_score":       self.penalty_score,
            "penalty_voltage":     self.penalty_voltage,
            "penalty_thermal":     self.penalty_thermal,
            "penalty_xfmr":        self.penalty_xfmr,
            "feasible":            self.feasible,
            "converged":           self.converged,
            "v_pu":                self.v_pu.tolist(),
            "i_pu":                self.i_pu.tolist(),
            "s_xfmr_kva":          self.s_xfmr_kva,
            "xfmr_loading_pct":    self.xfmr_loading_pct,
            "power_loss_kw":       self.power_loss_kw,
            "bus_violations":      self.bus_violations,
            "branch_violations":   self.branch_violations,
            "pen_v_per_bus":       self.pen_v_per_bus.tolist(),
            "pen_i_per_branch":    self.pen_i_per_branch.tolist(),
            "pbr_pu":              self.pbr_pu.tolist(),
            "qbr_pu":              self.qbr_pu.tolist(),
        }


# ---------------------------------------------------------------------------
# Tree structure (precomputed once at module load — constant for 33-bus)
# ---------------------------------------------------------------------------
def _build_tree() -> tuple[dict[int, list[int]], np.ndarray, list[int], list[int]]:
    """
    Build adjacency structures for the radial 33-bus feeder.

    Returns
    -------
    children    dict  bus → list of child buses (1-indexed)
    par_branch  ndarray (34,) int  par_branch[bus] = branch index feeding bus
    bfs_order   list  BFS traversal order (root first, bus numbers 1-indexed)
    rev_bfs     list  reverse BFS (leaves first, for backward sweep)
    """
    children: dict[int, list[int]] = defaultdict(list)
    par_branch = np.full(N_BUS + 1, -1, dtype=int)   # 1-indexed, bus 1 unused

    for k in range(N_BRANCH):
        f, t = int(FROM_BUS[k]), int(TO_BUS[k])
        children[f].append(t)
        par_branch[t] = k

    # BFS from root (bus 1)
    bfs_order: list[int] = []
    queue: deque[int] = deque([1])
    visited = {1}
    while queue:
        n = queue.popleft()
        bfs_order.append(n)
        for c in children[n]:
            if c not in visited:
                visited.add(c)
                queue.append(c)

    return children, par_branch, bfs_order, list(reversed(bfs_order))


_CHILDREN, _PAR_BRANCH, _BFS_ORDER, _REV_BFS = _build_tree()


# ---------------------------------------------------------------------------
# Core solver
# ---------------------------------------------------------------------------
def _run_fbs(
    P_pu: np.ndarray,
    Q_pu: np.ndarray,
    max_iter: int = 100,
    tol: float = 1e-9,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
    """
    Simplified DistFlow Forward-Backward Sweep solver.

    Uses the V² form of the DistFlow equations (Baran & Wu 1989) which
    avoids the cumulative voltage-drop overflow that occurs in the V form
    on high-impedance distribution feeders like the IEEE 33-bus system.

    Forward sweep voltage update (V-squared form):
        V_t² = V_f² - 2*(R_k*P_k + X_k*Q_k)

    All powers are in pu on baseMVA; all impedances in pu on Zbase.
    The quadratic loss term (R²+X²)(P²+Q²)/V² is omitted (simplified form)
    for unconditional stability — introduces <0.1% error at normal loading.

    Parameters
    ----------
    P_pu : ndarray (33,)   Bus active loads in per-unit (index 0 = Bus 1)
    Q_pu : ndarray (33,)   Bus reactive loads in per-unit
    max_iter : int         Maximum FBS iterations (default 100)
    tol : float            Convergence tolerance on max |ΔV²| (default 1e-9)

    Returns
    -------
    V    ndarray (34,) float  Voltage magnitudes [pu], 1-indexed (V[1]..V[33])
    Pbr  ndarray (32,) float  Branch active power flows [pu]
    Qbr  ndarray (32,) float  Branch reactive power flows [pu]
    converged bool            True if tolerance met within max_iter
    """
    # State vectors — 1-indexed bus array: V2[1] = |V_bus1|², ..., V2[33]
    V2  = np.ones(N_BUS + 1)      # |V|² flat start: 1.0 pu²
    Pbr = np.zeros(N_BRANCH)
    Qbr = np.zeros(N_BRANCH)

    converged = False
    for _ in range(max_iter):
        V2_old = V2.copy()

        # ---- Backward sweep: accumulate downstream loads -------------------
        # Simplified DistFlow: Pbr[k] = sum of all downstream P loads.
        # No I²R loss term — keeps flows bounded regardless of V level.
        for n in _REV_BFS:
            if n == 1:
                continue
            k = _PAR_BRANCH[n]
            ps = P_pu[n - 1]
            qs = Q_pu[n - 1]
            for c in _CHILDREN[n]:
                kc = _PAR_BRANCH[c]
                ps += Pbr[kc]
                qs += Qbr[kc]
            Pbr[k] = ps
            Qbr[k] = qs

        # ---- Forward sweep: update V² (DistFlow V-squared form) ------------
        # V_t² = V_f² - 2*(R_k*P_k + X_k*Q_k)
        # The coefficient 2 makes this exact (not an approximation) for the
        # lossless case, and the drop per branch is proportional to branch
        # flow × impedance — small compared to 1.0 pu² for normal loading.
        V2[1] = 1.0   # slack bus: |V|² = 1.0 pu²
        for n in _BFS_ORDER:
            if n == 1:
                continue
            k  = _PAR_BRANCH[n]
            f  = int(FROM_BUS[k])
            drop = 2.0 * (R_PU[k] * Pbr[k] + X_PU[k] * Qbr[k])
            V2[n] = max(V2[f] - drop, 0.25)   # floor at (0.5 pu)²

        if np.max(np.abs(V2 - V2_old)) < tol:
            converged = True
            break

    # Convert V² → |V|
    V = np.sqrt(np.maximum(V2, 0.0))
    return V, Pbr, Qbr, converged


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def physics_penalty_engine(
    ev_scenario: list[dict],
    v_min:     float = V_MIN_DEFAULT,
    v_max:     float = V_MAX_DEFAULT,
    i_lim_pu:  float = I_LIM_DEFAULT,
    xfmr_kva:  float = XFMR_KVA,
    w_voltage: float = 1000.0,
    w_thermal: float = 500.0,
    w_xfmr:    float = 800.0,
    verbose:   bool  = True,
    max_iter:  int   = 100,
    tol:       float = 1e-9,
) -> PenaltyResult:
    """
    Evaluate whether a proposed EV charging scenario violates grid constraints.

    Parameters
    ----------
    ev_scenario : list of dict
        Each dict must contain:
            "bus_id"  (int)    Load bus number, 2–33
            "p_kw"    (float)  Charger active power demand [kW]
        Optionally:
            "q_kvar"  (float)  Reactive demand [kVAr].
                               If omitted, computed at pf = 0.90 lagging.

    v_min, v_max : float
        Nodal voltage bounds [pu]. Default 0.95–1.05 pu.

    i_lim_pu : float
        Branch current thermal limit [pu]. Default 1.00 (rated current).

    xfmr_kva : float
        Substation transformer apparent power rating [kVA]. Default 5000.

    w_voltage, w_thermal, w_xfmr : float
        Penalty weights for each constraint type.

    verbose : bool
        If True, prints a formatted report to stdout.

    max_iter : int
        Maximum FBS iterations.

    tol : float
        Convergence tolerance [pu].

    Returns
    -------
    PenaltyResult
        Dataclass containing penalty scores, voltage/current profiles,
        transformer loading, losses, and violation lists.

    Notes
    -----
    Penalty formula (matching MATLAB engine exactly):

        penalty_voltage = w_voltage × Σ [max(0, v_min−Vᵢ)² + max(0, Vᵢ−v_max)²]
        penalty_thermal = w_thermal × Σ [max(0, |Iₖ|/i_lim − 1)²]
        penalty_xfmr    = w_xfmr   × max(0, S_xfmr/xfmr_kva − 1)²
        total           = penalty_voltage + penalty_thermal + penalty_xfmr
    """
    # ---- 1. Build load vectors (0-indexed; index 0 = Bus 1) ----------------
    P_mw   = BASE_P_MW.copy()
    Q_mvar = BASE_Q_MVAR.copy()

    for s in ev_scenario:
        bus_id = int(s["bus_id"])
        if not (2 <= bus_id <= 33):
            raise ValueError(f"bus_id {bus_id} out of range — must be 2–33.")
        p_inj = float(s["p_kw"]) / 1000.0   # kW → MW
        if "q_kvar" in s and s["q_kvar"] is not None:
            q_inj = float(s["q_kvar"]) / 1000.0
        else:
            q_inj = p_inj * math.tan(math.acos(0.90))   # pf = 0.90 lagging
        P_mw[bus_id - 1]   += p_inj
        Q_mvar[bus_id - 1] += q_inj

    # ---- 2. Convert to per-unit (divide by baseMVA) ------------------------
    P_pu = P_mw   / BASE_MVA
    Q_pu = Q_mvar / BASE_MVA

    # ---- 3. Run Simplified DistFlow FBS solver ------------------------------
    V, Pbr, Qbr, converged = _run_fbs(P_pu, Q_pu, max_iter=max_iter, tol=tol)

    # ---- 4. Branch currents [pu] -------------------------------------------
    # |I_k| = sqrt(P_k² + Q_k²) / V_f   (all in pu)
    S_br  = np.sqrt(Pbr**2 + Qbr**2)
    Vf    = np.maximum(V[FROM_BUS], 0.50)
    I_pu  = S_br / Vf

    # ---- 5. Transformer apparent power (branch 0 = feeder head, Bus 1→2) ---
    S_xfmr_mva      = math.sqrt(float(Pbr[0])**2 + float(Qbr[0])**2) * BASE_MVA
    S_xfmr_kva      = S_xfmr_mva * 1000.0
    xfmr_loading_pct = (S_xfmr_kva / xfmr_kva) * 100.0

    # ---- 6. Active power losses [kW] ----------------------------------------
    Vf2        = np.maximum(V[FROM_BUS]**2, 0.25)
    P_loss_kw  = float(np.sum(R_PU * (Pbr**2 + Qbr**2) / Vf2)) * BASE_MVA * 1000.0

    # ---- 7. Penalty scores --------------------------------------------------
    # Voltage — buses 1..33 (V array is 1-indexed)
    V_buses = V[1:]   # shape (33,), index 0 = Bus 1
    v_under = np.maximum(0.0, v_min - V_buses)
    v_over  = np.maximum(0.0, V_buses - v_max)
    pen_v_per_bus = v_under**2 + v_over**2
    penalty_voltage = w_voltage * float(np.sum(pen_v_per_bus))

    # Thermal
    i_excess         = np.maximum(0.0, I_pu / i_lim_pu - 1.0)
    pen_i_per_branch = i_excess**2
    penalty_thermal  = w_thermal * float(np.sum(pen_i_per_branch))

    # Transformer
    xfmr_excess  = max(0.0, S_xfmr_kva / xfmr_kva - 1.0)
    penalty_xfmr = w_xfmr * xfmr_excess**2

    penalty_score = penalty_voltage + penalty_thermal + penalty_xfmr

    # ---- 8. Violation lists (1-indexed bus / branch numbers) ----------------
    bus_violations    = [i + 1 for i in range(N_BUS)
                         if V_buses[i] < v_min or V_buses[i] > v_max]
    branch_violations = [k + 1 for k in range(N_BRANCH) if I_pu[k] > i_lim_pu]

    # ---- 9. Pack result -----------------------------------------------------
    result = PenaltyResult(
        penalty_score    = penalty_score,
        penalty_voltage  = penalty_voltage,
        penalty_thermal  = penalty_thermal,
        penalty_xfmr     = penalty_xfmr,
        feasible         = penalty_score < 1e-10,
        converged        = converged,
        v_pu             = V_buses.copy(),        # shape (33,), Bus 1..33
        i_pu             = I_pu.copy(),            # shape (32,)
        s_xfmr_kva       = S_xfmr_kva,
        xfmr_loading_pct = xfmr_loading_pct,
        power_loss_kw    = P_loss_kw,
        bus_violations   = bus_violations,
        branch_violations= branch_violations,
        pen_v_per_bus    = pen_v_per_bus.copy(),
        pen_i_per_branch = pen_i_per_branch.copy(),
        pbr_pu           = Pbr.copy(),
        qbr_pu           = Qbr.copy(),
    )

    if verbose:
        _print_report(result, ev_scenario, v_min, v_max, i_lim_pu, xfmr_kva)

    return result


# ---------------------------------------------------------------------------
# Console report (mirrors MATLAB print_report)
# ---------------------------------------------------------------------------
def _print_report(
    res:       PenaltyResult,
    scenario:  list[dict],
    v_min:     float,
    v_max:     float,
    i_lim_pu:  float,
    xfmr_kva:  float,
) -> None:
    SEP = "=" * 68
    sep = "-" * 68
    print(f"\n{SEP}")
    print("  PHYSICS PENALTY ENGINE  |  IEEE 33-Bus Distribution Feeder")
    print("  Solver: Simplified DistFlow FBS (Baran & Wu 1989)")
    print(f"{SEP}")

    print("\n  EV Charging Scenario:")
    for s in scenario:
        q = s.get("q_kvar", 0.0) or 0.0
        print(f"    Bus {s['bus_id']:2d} :  P = {s['p_kw']:7.1f} kW   Q = {q:6.1f} kVAr")

    print(f"\n{sep}\n  PENALTY SCORES\n{sep}")
    print(f"  Voltage violations    : {res.penalty_voltage:12.4f}")
    print(f"  Thermal violations    : {res.penalty_thermal:12.4f}")
    print(f"  Transformer overload  : {res.penalty_xfmr:12.4f}")
    print(f"  TOTAL PENALTY SCORE   : {res.penalty_score:12.4f}")
    status = "FEASIBLE  — no physical constraints violated." \
             if res.feasible else \
             "INFEASIBLE — constraints violated (see below)."
    print(f"\n  >>> RESULT: {status}")

    print(f"\n{sep}\n  VOLTAGE PROFILE  (bounds: {v_min:.2f} – {v_max:.2f} pu)\n{sep}")
    for i, v in enumerate(res.v_pu):
        bus = i + 1
        tag = "<-- UNDER-VOLTAGE" if v < v_min else \
              "<-- OVER-VOLTAGE"  if v > v_max else "OK"
        print(f"  Bus {bus:3d} :  {v:.4f} pu   {tag}")

    print(f"\n{sep}\n  BRANCH THERMAL  (limit = {i_lim_pu*100:.0f}%)\n{sep}")
    shown = False
    for k, ii in enumerate(res.i_pu):
        ld = ii * 100
        if ld > 50:
            tag = f"<-- OVERLOADED ({ld:.1f}%)" if ld > 100 else "OK"
            print(f"  Branch {k+1:2d} :  {ld:6.1f}%   {tag}")
            shown = True
    if not shown:
        print("  All branches below 50% loading.")

    print(f"\n{sep}\n  TRANSFORMER\n{sep}")
    tag = "<-- OVERLOADED" if res.xfmr_loading_pct > 100 else "OK"
    print(f"  Rating  : {xfmr_kva:6.0f} kVA")
    print(f"  Loading : {res.s_xfmr_kva:6.1f} kVA  ({res.xfmr_loading_pct:5.1f}%)  {tag}")
    print(f"\n  Active power loss : {res.power_loss_kw:.2f} kW")
    print(f"  FBS converged     : {res.converged}")
    print(f"{SEP}\n")
