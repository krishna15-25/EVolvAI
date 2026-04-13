"""
ieee33bus_data.py
=================
IEEE 33-bus radial distribution feeder — network data.

Translated from: ieee33bus_data.m (MATLAB)
Project:         EVolvAI — Physics Penalty Engine module
Reference:       Baran & Wu, IEEE Trans. Power Del., 1989

Returns the complete network description as NumPy arrays so the rest
of the Python pipeline (physics_penalty_engine, optimisers, etc.) can
import it without any file I/O.

System base
-----------
    baseMVA  = 10 MVA
    baseKV   = 12.66 kV
    Zbase    = baseKV² / baseMVA = 16.024 Ω

All impedances (R, X) are already in per-unit on this base.
All loads (P, Q) are in MW / MVAr — divide by baseMVA to get pu.
"""

import numpy as np


# ---------------------------------------------------------------------------
# System constants
# ---------------------------------------------------------------------------
BASE_MVA: float = 10.0      # MVA base
BASE_KV:  float = 12.66     # kV base (distribution voltage)
N_BUS:    int   = 33        # total buses (Bus 1 = slack/substation)
N_BRANCH: int   = 32        # total branches (cables)

# Substation transformer rating
XFMR_KVA: float = 5_000.0  # kVA  (5 MVA)
XFMR_KV_HV: float = 33.0   # HV side [kV]
XFMR_KV_LV: float = 12.66  # LV side [kV]

# Default constraint bounds
V_MIN_DEFAULT: float = 0.95   # pu
V_MAX_DEFAULT: float = 1.05   # pu
I_LIM_DEFAULT: float = 1.00   # pu (100 % rated current)


# ---------------------------------------------------------------------------
# Branch data — shape (32, 4)
# Columns: [from_bus, to_bus, R_pu, X_pu]   (buses are 1-indexed integers)
# ---------------------------------------------------------------------------
BRANCH_DATA: np.ndarray = np.array([
    [ 1,  2,  0.0575, 0.0293],
    [ 2,  3,  0.3075, 0.1563],
    [ 3,  4,  0.2279, 0.1161],
    [ 4,  5,  0.2379, 0.1209],
    [ 5,  6,  0.5110, 0.4411],
    [ 6,  7,  0.1167, 0.3861],
    [ 7,  8,  1.0640, 0.7707],
    [ 8,  9,  1.0640, 0.7707],
    [ 9, 10,  0.9977, 0.4400],
    [10, 11,  0.1967, 0.0651],
    [11, 12,  0.3744, 0.1238],
    [12, 13,  1.4680, 1.1549],
    [13, 14,  0.5416, 0.7129],
    [14, 15,  0.5910, 0.5260],
    [15, 16,  0.7463, 0.5450],
    [16, 17,  1.2890, 1.7210],
    [17, 18,  0.7320, 0.5739],
    [ 2, 19,  0.1640, 0.1565],
    [19, 20,  1.5042, 1.3554],
    [20, 21,  0.4095, 0.4784],
    [21, 22,  0.7089, 0.9373],
    [ 3, 23,  0.4512, 0.3083],
    [23, 24,  0.8980, 0.7091],
    [24, 25,  0.8960, 0.7011],
    [ 6, 26,  0.2030, 0.1034],
    [26, 27,  0.2842, 0.1447],
    [27, 28,  1.0590, 0.9337],
    [28, 29,  0.8042, 0.7006],
    [29, 30,  0.5075, 0.2585],
    [30, 31,  0.9744, 0.9630],
    [31, 32,  0.3105, 0.3619],
    [32, 33,  0.3410, 0.5302],
], dtype=np.float64)

# Convenience column slices (0-indexed)
FROM_BUS: np.ndarray = BRANCH_DATA[:, 0].astype(int)   # shape (32,)
TO_BUS:   np.ndarray = BRANCH_DATA[:, 1].astype(int)   # shape (32,)
R_PU:     np.ndarray = BRANCH_DATA[:, 2]               # resistance [pu]
X_PU:     np.ndarray = BRANCH_DATA[:, 3]               # reactance  [pu]


# ---------------------------------------------------------------------------
# Base bus loads — shape (33,)
# Index i corresponds to Bus (i+1), so index 0 = Bus 1 (slack, zero load).
# Units: MW for P, MVAr for Q.
# ---------------------------------------------------------------------------
BASE_P_MW: np.ndarray = np.array([
    0.000,   # Bus  1  (slack — substation, no load)
    0.100,   # Bus  2
    0.090,   # Bus  3
    0.120,   # Bus  4
    0.060,   # Bus  5
    0.060,   # Bus  6
    0.200,   # Bus  7
    0.200,   # Bus  8
    0.060,   # Bus  9
    0.060,   # Bus 10
    0.045,   # Bus 11
    0.060,   # Bus 12
    0.060,   # Bus 13
    0.120,   # Bus 14
    0.060,   # Bus 15
    0.060,   # Bus 16
    0.060,   # Bus 17
    0.090,   # Bus 18
    0.090,   # Bus 19
    0.090,   # Bus 20
    0.090,   # Bus 21
    0.090,   # Bus 22
    0.090,   # Bus 23
    0.420,   # Bus 24
    0.420,   # Bus 25
    0.060,   # Bus 26
    0.060,   # Bus 27
    0.060,   # Bus 28
    0.120,   # Bus 29
    0.200,   # Bus 30
    0.150,   # Bus 31
    0.210,   # Bus 32
    0.060,   # Bus 33
], dtype=np.float64)

BASE_Q_MVAR: np.ndarray = np.array([
    0.000,   # Bus  1
    0.060,   # Bus  2
    0.040,   # Bus  3
    0.080,   # Bus  4
    0.030,   # Bus  5
    0.035,   # Bus  6
    0.1025,  # Bus  7
    0.100,   # Bus  8
    0.020,   # Bus  9
    0.020,   # Bus 10
    0.030,   # Bus 11
    0.035,   # Bus 12
    0.035,   # Bus 13
    0.080,   # Bus 14
    0.010,   # Bus 15
    0.020,   # Bus 16
    0.020,   # Bus 17
    0.040,   # Bus 18
    0.040,   # Bus 19
    0.040,   # Bus 20
    0.040,   # Bus 21
    0.040,   # Bus 22
    0.050,   # Bus 23
    0.200,   # Bus 24
    0.200,   # Bus 25
    0.025,   # Bus 26
    0.025,   # Bus 27
    0.020,   # Bus 28
    0.070,   # Bus 29
    0.600,   # Bus 30
    0.070,   # Bus 31
    0.100,   # Bus 32
    0.040,   # Bus 33
], dtype=np.float64)


# ---------------------------------------------------------------------------
# Bus metadata (for reporting / topology)
# ---------------------------------------------------------------------------
BUS_TYPE: dict = {
    1: "slack",   # Bus 1 — substation / voltage reference
    **{b: "PQ" for b in range(2, 34)}
}

# Human-readable lateral description
LATERALS: dict = {
    "main":     list(range(1, 19)),   # Bus 1 → 18
    "lateral_1": [2, 19, 20, 21, 22],
    "lateral_2": [3, 23, 24, 25],
    "lateral_3": [6, 26, 27, 28, 29, 30, 31, 32, 33],
}

# End-of-feeder buses (electrically weakest — most sensitive to EV loading)
WEAK_BUSES: list = [18, 22, 25, 33]


# ---------------------------------------------------------------------------
# Accessor function (mirrors MATLAB function call style)
# ---------------------------------------------------------------------------
def get_network_data() -> dict:
    """
    Return a dictionary containing all IEEE 33-bus network data.

    Returns
    -------
    dict with keys:
        base_mva        float   System MVA base
        base_kv         float   System kV base
        n_bus           int     Number of buses
        n_branch        int     Number of branches
        xfmr_kva        float   Transformer rating [kVA]
        branch_data     ndarray (32, 4) [from, to, R_pu, X_pu]
        from_bus        ndarray (32,)  int
        to_bus          ndarray (32,)  int
        R               ndarray (32,)  branch resistance [pu]
        X               ndarray (32,)  branch reactance  [pu]
        base_P_mw       ndarray (33,)  base active loads [MW]
        base_Q_mvar     ndarray (33,)  base reactive loads [MVAr]
        v_min           float   default lower voltage bound [pu]
        v_max           float   default upper voltage bound [pu]
        i_lim_pu        float   default thermal limit [pu]
        weak_buses      list    end-of-feeder buses
        laterals        dict    feeder topology description
    """
    return {
        "base_mva":    BASE_MVA,
        "base_kv":     BASE_KV,
        "n_bus":       N_BUS,
        "n_branch":    N_BRANCH,
        "xfmr_kva":   XFMR_KVA,
        "branch_data": BRANCH_DATA.copy(),
        "from_bus":    FROM_BUS.copy(),
        "to_bus":      TO_BUS.copy(),
        "R":           R_PU.copy(),
        "X":           X_PU.copy(),
        "base_P_mw":   BASE_P_MW.copy(),
        "base_Q_mvar": BASE_Q_MVAR.copy(),
        "v_min":       V_MIN_DEFAULT,
        "v_max":       V_MAX_DEFAULT,
        "i_lim_pu":    I_LIM_DEFAULT,
        "weak_buses":  WEAK_BUSES.copy(),
        "laterals":    LATERALS.copy(),
    }


if __name__ == "__main__":
    nd = get_network_data()
    print(f"IEEE 33-Bus Feeder — network data loaded")
    print(f"  Buses    : {nd['n_bus']}")
    print(f"  Branches : {nd['n_branch']}")
    print(f"  Base MVA : {nd['base_mva']} MVA  |  Base kV : {nd['base_kv']} kV")
    print(f"  Total base load: P = {nd['base_P_mw'].sum():.3f} MW, "
          f"Q = {nd['base_Q_mvar'].sum():.3f} MVAr")
    print(f"  Transformer rating: {nd['xfmr_kva']:.0f} kVA")
    print(f"  Voltage bounds: [{nd['v_min']}, {nd['v_max']}] pu")
    print(f"  Weak buses (end-of-feeder): {nd['weak_buses']}")
