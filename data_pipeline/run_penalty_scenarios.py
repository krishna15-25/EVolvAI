"""
run_penalty_scenarios.py
========================
Demonstrates the Physics Penalty Engine across three scenarios and
generates result plots. Entry point for standalone testing of the
physics module within the EVolvAI project.

Translated from: run_penalty_scenarios.m (MATLAB)
Project:         EVolvAI — Physics Penalty Engine module

Usage
-----
    # All scenarios + plots
    python -m physics_penalty_engine.run_penalty_scenarios

    # Programmatic use
    from physics_penalty_engine.run_penalty_scenarios import run_all_scenarios
    results = run_all_scenarios(verbose=True, save_plots=True)

Scenarios
---------
    A  Single 150 kW charger swept across every load bus (bus 2–33)
    B  Multi-charger cluster: 3×150 kW at buses 18, 25, 33 (stress test)
    C  Custom user scenario  ← edit CUSTOM_SCENARIO below
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np

from .physics_penalty_engine import physics_penalty_engine, PenaltyResult
from .evaluate_charger_placement import evaluate_charger_placement, bus_sweep

# ---------------------------------------------------------------------------
# Scenario C definition — edit this to test custom configurations
# ---------------------------------------------------------------------------
CUSTOM_SCENARIO: list[dict] = [
    {"bus_id":  7, "p_kw": 100},
    {"bus_id": 12, "p_kw": 200},
    {"bus_id": 30, "p_kw":  75},
]


# ---------------------------------------------------------------------------
# Scenario runners
# ---------------------------------------------------------------------------
def run_scenario_a(p_kw: float = 150.0, verbose: bool = False) -> dict[str, Any]:
    """
    Scenario A: single charger bus sweep.

    Places one charger of `p_kw` kW at each load bus (2–33) in turn,
    records penalty score, minimum voltage, transformer loading, and losses.

    Returns
    -------
    dict with arrays indexed by bus number (2–33):
        scores, v_min, xfmr_pct, loss_kw, feasible
    """
    print(f"\n>>> Scenario A: single-charger bus sweep ({p_kw:.0f} kW)...")

    buses    = list(range(2, 34))
    scores   = np.zeros(34)
    v_min_v  = np.ones(34)
    xfmr_pct = np.zeros(34)
    loss_kw  = np.zeros(34)
    feasible = np.zeros(34, dtype=bool)

    for bus in buses:
        score, res = evaluate_charger_placement(
            bus, p_kw, verbose=False, return_result=True
        )
        scores[bus]   = res.penalty_score
        v_min_v[bus]  = float(np.min(res.v_pu))
        xfmr_pct[bus] = res.xfmr_loading_pct
        loss_kw[bus]  = res.power_loss_kw
        feasible[bus] = res.feasible

    valid_scores = scores[2:]
    best_bus  = int(np.argmin(valid_scores)) + 2
    worst_bus = int(np.argmax(valid_scores)) + 2
    n_feasible = int(np.sum(feasible[2:]))

    print(f"\n  Results (150 kW single-charger sweep):")
    print(f"  Best  placement : Bus {best_bus:2d}  "
          f"(penalty = {scores[best_bus]:.4f},  V_min = {v_min_v[best_bus]:.4f} pu)")
    print(f"  Worst placement : Bus {worst_bus:2d}  "
          f"(penalty = {scores[worst_bus]:.4f},  V_min = {v_min_v[worst_bus]:.4f} pu)")
    print(f"  Feasible buses  : {n_feasible} / 32")

    return {
        "buses":    buses,
        "scores":   scores,
        "v_min":    v_min_v,
        "xfmr_pct": xfmr_pct,
        "loss_kw":  loss_kw,
        "feasible": feasible,
        "best_bus": best_bus,
        "worst_bus": worst_bus,
    }


def run_scenario_b(verbose: bool = True) -> PenaltyResult:
    """
    Scenario B: 3×150 kW cluster at weak-end buses 18, 25, 33 (stress test).
    """
    print("\n>>> Scenario B: 3×150 kW cluster at buses 18, 25, 33...")
    scenario = [
        {"bus_id": 18, "p_kw": 150},
        {"bus_id": 25, "p_kw": 150},
        {"bus_id": 33, "p_kw": 150},
    ]
    return physics_penalty_engine(
        scenario,
        w_voltage=1000, w_thermal=500, w_xfmr=800,
        verbose=verbose,
    )


def run_scenario_c(
    scenario: list[dict] | None = None,
    verbose: bool = True,
) -> PenaltyResult:
    """
    Scenario C: custom user-defined scenario.
    Edit CUSTOM_SCENARIO at the top of this file, or pass `scenario` directly.
    """
    sc = scenario if scenario is not None else CUSTOM_SCENARIO
    buses_str = ", ".join(f"Bus {s['bus_id']} ({s['p_kw']} kW)" for s in sc)
    print(f"\n>>> Scenario C: custom scenario — {buses_str}...")
    return physics_penalty_engine(sc, verbose=verbose)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def generate_plots(
    res_a: dict[str, Any],
    res_b: PenaltyResult,
    res_c: PenaltyResult,
    save_path: Path | None = None,
) -> None:
    """
    Generate the 2×3 subplot figure matching the MATLAB run_penalty_scenarios output.

    Parameters
    ----------
    res_a : dict      Output of run_scenario_a()
    res_b, res_c : PenaltyResult
    save_path : Path  If given, saves figure to this path instead of displaying.
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError:
        print("  [plots] matplotlib not installed — skipping plots.")
        print("  Install with: pip install matplotlib")
        return

    bus_ids  = np.array(res_a["buses"])          # 2..33
    scores   = res_a["scores"][2:]
    v_min_v  = res_a["v_min"][2:]
    xfmr_pct = res_a["xfmr_pct"][2:]
    feasible = res_a["feasible"][2:]

    BLUE  = (0.18, 0.45, 0.75)
    RED   = (0.75, 0.12, 0.12)
    AMBER = (0.93, 0.69, 0.13)
    GREEN = (0.45, 0.72, 0.30)

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle("Physics Penalty Engine — IEEE 33-Bus Distribution Feeder",
                 fontsize=14, fontweight="bold")

    # ---- (1) Penalty score by bus ------------------------------------------
    ax = axes[0, 0]
    colors = [BLUE if f else RED for f in feasible]
    ax.bar(bus_ids, scores, color=colors, edgecolor="none")
    ax.set_xlabel("Bus No.")
    ax.set_ylabel("Penalty Score")
    ax.set_title("Penalty Score vs. Charger Location (150 kW)")
    ax.set_xlim(1, 34)
    ax.grid(True, alpha=0.3)
    ax.legend(handles=[
        mpatches.Patch(color=BLUE, label="Feasible"),
        mpatches.Patch(color=RED,  label="Violated"),
    ], loc="upper left")

    # ---- (2) Minimum voltage by placement ----------------------------------
    ax = axes[0, 1]
    ax.bar(bus_ids, v_min_v, color=(0.20, 0.63, 0.90), edgecolor="none")
    ax.axhline(0.95, color="red", linestyle="--", linewidth=1.5, label="V_min=0.95 pu")
    ax.set_xlabel("Bus No.")
    ax.set_ylabel("Min Voltage [pu]")
    ax.set_title("Worst Nodal Voltage vs. Charger Location")
    ylo = max(0.88, float(v_min_v.min()) - 0.02)
    ax.set_ylim(ylo, 1.02)
    ax.set_xlim(1, 34)
    ax.grid(True, alpha=0.3)
    ax.legend()

    # ---- (3) Transformer loading by placement ------------------------------
    ax = axes[0, 2]
    ax.bar(bus_ids, xfmr_pct, color=GREEN, edgecolor="none")
    ax.axhline(100, color="red", linestyle="--", linewidth=1.5, label="100% Rated")
    ax.set_xlabel("Bus No.")
    ax.set_ylabel("Transformer Loading [%]")
    ax.set_title("Transformer Loading vs. Charger Location")
    ax.set_xlim(1, 34)
    ax.grid(True, alpha=0.3)
    ax.legend()

    # ---- (4) Scenario B: voltage profile -----------------------------------
    ax = axes[1, 0]
    buses_all = np.arange(1, 34)
    ax.fill_between(buses_all, res_b.v_pu, alpha=0.6,
                    color=(0.70, 0.85, 0.98), label="Profile")
    ax.plot(buses_all, res_b.v_pu, "-o", color=(0.13, 0.40, 0.75),
            markersize=4, linewidth=1.4, label="V [pu]")
    ax.axhline(0.95, color="red", linestyle="--", linewidth=1.5)
    ax.axhline(1.05, color="red", linestyle="--", linewidth=1.5,
               label="V_min/V_max")
    ax.set_xlabel("Bus No.")
    ax.set_ylabel("Voltage [pu]")
    ax.set_title("Scenario B: Voltage Profile (3×150 kW at 18,25,33)")
    ax.set_ylim(0.88, 1.07)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower left")

    # ---- (5) Scenario B: branch loading ------------------------------------
    ax = axes[1, 1]
    loading = res_b.i_pu * 100
    bar_colors = [RED if l > 100 else AMBER for l in loading]
    ax.bar(range(1, len(loading) + 1), loading, color=bar_colors, edgecolor="none")
    ax.axhline(100, color="red", linestyle="--", linewidth=1.5, label="Thermal Limit")
    ax.set_xlabel("Branch No.")
    ax.set_ylabel("Loading [%]")
    ax.set_title("Scenario B: Branch Loading")
    ax.grid(True, alpha=0.3)
    ax.legend()

    # ---- (6) Penalty breakdown: B vs C -------------------------------------
    ax = axes[1, 2]
    categories = ["Voltage", "Thermal", "Transformer"]
    x = np.arange(len(categories))
    w = 0.35
    ax.bar(x - w/2, [res_b.penalty_voltage, res_b.penalty_thermal, res_b.penalty_xfmr],
           w, color=(0.18, 0.45, 0.75), label="Sc. B (18,25,33)")
    ax.bar(x + w/2, [res_c.penalty_voltage, res_c.penalty_thermal, res_c.penalty_xfmr],
           w, color=(0.85, 0.40, 0.10), label="Sc. C (custom)")
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.set_xlabel("Constraint Type")
    ax.set_ylabel("Penalty Score")
    ax.set_title("Penalty Breakdown: Scenario B vs C")
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"\n  Plots saved to: {save_path}")
    else:
        plt.show()


# ---------------------------------------------------------------------------
# Top-level runner
# ---------------------------------------------------------------------------
def run_all_scenarios(
    verbose:    bool = True,
    plot:       bool = True,
    save_plots: bool = False,
    output_dir: Path = Path("output"),
) -> dict[str, Any]:
    """
    Run all three scenarios and optionally generate/save plots.

    Parameters
    ----------
    verbose    : bool   Print detailed reports for Scenarios B and C.
    plot       : bool   Generate matplotlib plots.
    save_plots : bool   If True, saves plots to output_dir instead of displaying.
    output_dir : Path   Directory for saved plots.

    Returns
    -------
    dict with keys "scenario_a", "scenario_b", "scenario_c".
    """
    SEP = "=" * 56
    print(SEP)
    print("  Physics Penalty Engine — EVolvAI Scenario Runner")
    print("  Solver: Simplified DistFlow FBS (Baran & Wu 1989)")
    print(SEP)

    res_a = run_scenario_a(verbose=False)
    res_b = run_scenario_b(verbose=verbose)
    res_c = run_scenario_c(verbose=verbose)

    if plot:
        save_path = None
        if save_plots:
            output_dir.mkdir(parents=True, exist_ok=True)
            save_path = output_dir / "penalty_engine_results.png"
        generate_plots(res_a, res_b, res_c, save_path=save_path)

    print("\nAll scenarios complete.\n")
    return {"scenario_a": res_a, "scenario_b": res_b, "scenario_c": res_c}


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="EVolvAI Physics Penalty Engine — scenario runner"
    )
    parser.add_argument("--no-plot",   action="store_true", help="Skip plots")
    parser.add_argument("--save-plot", action="store_true", help="Save plots to output/")
    parser.add_argument("--quiet",     action="store_true", help="Suppress verbose reports")
    args = parser.parse_args()

    run_all_scenarios(
        verbose=not args.quiet,
        plot=not args.no_plot,
        save_plots=args.save_plot,
    )
