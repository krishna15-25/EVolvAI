#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════╗
║  EVolvAI — ChargerOptimizerGA Test Suite                             ║
║  Validates GA operators, CVaR math, fitness, and end-to-end run.     ║
╚══════════════════════════════════════════════════════════════════════╝
"""

import numpy as np
import pytest

from risk_engine.optimizer_ga import (
    EVOptimizerConfig,
    ChargerOptimizerGA,
    calculate_cvar,
    evaluate_fitness,
    get_grid_penalty_dummy,
)

# ─────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────
N_NODES = 10
N_SCENARIOS = 50
SEED = 123


@pytest.fixture
def config() -> EVOptimizerConfig:
    return EVOptimizerConfig(
        pop_size=20,
        max_ports_per_node=15,
        max_generations=30,
        convergence_window=10,
        seed=SEED,
    )


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(SEED)


@pytest.fixture
def demand_scenarios(rng) -> np.ndarray:
    return rng.lognormal(mean=2.5, sigma=0.5, size=(N_SCENARIOS, N_NODES))


@pytest.fixture
def ga(config) -> ChargerOptimizerGA:
    return ChargerOptimizerGA(config=config, n_nodes=N_NODES)


# ─────────────────────────────────────────────────────────────────────
# §1  CVaR unit tests
# ─────────────────────────────────────────────────────────────────────
class TestCVaR:
    def test_uniform_costs_returns_var(self):
        """When all costs are identical, CVaR == VaR == that value."""
        costs = np.full(100, 42.0)
        assert calculate_cvar(costs, alpha=0.99) == pytest.approx(42.0)

    def test_known_distribution(self):
        """CVaR of sorted costs [1..100] at alpha=0.95 → mean of top 5."""
        costs = np.arange(1, 101, dtype=np.float64)
        # VaR at 0.95 = 95.25 (np.quantile interpolation)
        # tail: values > 95.25 → {96, 97, 98, 99, 100}
        cvar = calculate_cvar(costs, alpha=0.95)
        assert cvar == pytest.approx(np.mean([96, 97, 98, 99, 100]))

    def test_cvar_geq_var(self):
        """CVaR should always be ≥ VaR for any distribution."""
        rng = np.random.default_rng(0)
        costs = rng.exponential(scale=100.0, size=1000)
        alpha = 0.99
        var = float(np.quantile(costs, alpha))
        cvar = calculate_cvar(costs, alpha=alpha)
        assert cvar >= var

    def test_single_element(self):
        """Edge case: single cost."""
        assert calculate_cvar(np.array([7.0]), alpha=0.99) == pytest.approx(7.0)


# ─────────────────────────────────────────────────────────────────────
# §2  Grid penalty tests
# ─────────────────────────────────────────────────────────────────────
class TestGridPenalty:
    def test_non_negative(self, rng):
        chrom = np.array([3, 5, 2, 0, 1, 4, 2, 3, 1, 0], dtype=np.int32)
        penalty = get_grid_penalty_dummy(chrom, 5.0, rng)
        assert penalty >= 0.0

    def test_higher_ports_higher_penalty(self, rng):
        """More total ports should generally yield a higher penalty."""
        low = np.ones(N_NODES, dtype=np.int32)
        high = np.full(N_NODES, 15, dtype=np.int32)
        # average over multiple draws to wash out noise
        low_penalties = [
            get_grid_penalty_dummy(low, 5.0, np.random.default_rng(i))
            for i in range(50)
        ]
        high_penalties = [
            get_grid_penalty_dummy(high, 5.0, np.random.default_rng(i))
            for i in range(50)
        ]
        assert np.mean(high_penalties) > np.mean(low_penalties)

    def test_zero_ports_minimal_penalty(self, rng):
        chrom = np.zeros(N_NODES, dtype=np.int32)
        penalty = get_grid_penalty_dummy(chrom, 5.0, rng)
        # should be very small (just noise)
        assert penalty < 1.0


# ─────────────────────────────────────────────────────────────────────
# §3  Fitness function tests
# ─────────────────────────────────────────────────────────────────────
class TestFitness:
    def test_returns_finite_float(self, config, demand_scenarios, rng):
        chrom = rng.integers(0, 16, size=N_NODES, dtype=np.int32)
        f = evaluate_fitness(chrom, demand_scenarios, config, rng)
        assert np.isfinite(f)
        assert isinstance(f, float)

    def test_zero_ports_high_wait_cost(self, demand_scenarios):
        """Zero ports everywhere → huge wait-time & unserved penalty.
        Use low CapEx so service-quality penalties dominate."""
        # low capex so that CapEx doesn't mask the wait/unserved penalties
        low_capex_cfg = EVOptimizerConfig(
            pop_size=20,
            max_ports_per_node=15,
            capex_per_port=100.0,  # $100 instead of $45,000
            seed=SEED,
        )
        zero_chrom = np.zeros(N_NODES, dtype=np.int32)
        moderate_chrom = np.full(N_NODES, 5, dtype=np.int32)
        # average over seeds to wash out grid noise
        f_zeros, f_mods = [], []
        for s in range(20):
            r = np.random.default_rng(s)
            f_zeros.append(
                evaluate_fitness(zero_chrom, demand_scenarios, low_capex_cfg, r)
            )
            r2 = np.random.default_rng(s + 1000)
            f_mods.append(
                evaluate_fitness(moderate_chrom, demand_scenarios, low_capex_cfg, r2)
            )
        # zero-port fitness should be worse on average
        assert np.mean(f_zeros) > np.mean(f_mods)

    def test_more_ports_higher_capex(self, config, rng):
        """With no demand, more ports = higher fitness (pure CapEx)."""
        # zero demand → wait times and opex vanish
        zero_demand = np.zeros((N_SCENARIOS, N_NODES))
        low = np.ones(N_NODES, dtype=np.int32)
        high = np.full(N_NODES, 15, dtype=np.int32)
        f_low = evaluate_fitness(low, zero_demand, config, rng)
        f_high = evaluate_fitness(high, zero_demand, config, rng)
        assert f_high > f_low


# ─────────────────────────────────────────────────────────────────────
# §4  GA operator tests
# ─────────────────────────────────────────────────────────────────────
class TestGAOperators:
    def test_population_shape(self, ga):
        pop = ga.initialize_population()
        assert pop.shape == (ga.config.pop_size, N_NODES)

    def test_population_bounds(self, ga):
        pop = ga.initialize_population()
        assert np.all(pop >= 0)
        assert np.all(pop <= ga.config.max_ports_per_node)

    def test_tournament_returns_valid_chromosome(self, ga, demand_scenarios):
        ga.initialize_population()
        ga._evaluate_population(demand_scenarios)
        parent = ga.tournament_selection()
        assert parent.shape == (N_NODES,)
        assert np.all(parent >= 0)
        assert np.all(parent <= ga.config.max_ports_per_node)

    def test_crossover_preserves_bounds(self, ga):
        ga.initialize_population()
        p_a = ga.population[0].copy()
        p_b = ga.population[1].copy()
        c_a, c_b = ga.two_point_crossover(p_a, p_b)
        assert c_a.shape == (N_NODES,)
        assert np.all(c_a >= 0) and np.all(c_a <= ga.config.max_ports_per_node)
        assert np.all(c_b >= 0) and np.all(c_b <= ga.config.max_ports_per_node)

    def test_mutation_preserves_bounds(self, ga):
        ga.initialize_population()
        chrom = ga.population[0].copy()
        # force high mutation to ensure some genes flip
        ga.mutation_rate = 0.5
        mutated = ga.mutate_integer_reset(chrom)
        assert np.all(mutated >= 0)
        assert np.all(mutated <= ga.config.max_ports_per_node)


# ─────────────────────────────────────────────────────────────────────
# §5  End-to-end GA run
# ─────────────────────────────────────────────────────────────────────
class TestGARun:
    def test_run_returns_valid_result(self, ga, demand_scenarios):
        result = ga.run(demand_scenarios, verbose=False)
        assert "best_chromosome" in result
        assert "best_fitness" in result
        assert "history" in result
        assert result["best_chromosome"].shape == (N_NODES,)
        assert np.isfinite(result["best_fitness"])
        assert len(result["history"]) > 0

    def test_fitness_improves_overall(self, ga, demand_scenarios):
        """Best fitness at the end should be ≤ initial best.
        Stochastic grid noise can cause small per-gen jitter, so we
        check overall trend instead of strict monotonicity."""
        result = ga.run(demand_scenarios, verbose=False)
        bests = [h["best_fitness"] for h in result["history"]]
        # final best should be better than or equal to first-gen best
        assert bests[-1] <= bests[0]

    def test_deterministic_with_seed(self, config, demand_scenarios):
        """Same seed → identical results."""
        ga1 = ChargerOptimizerGA(config=config, n_nodes=N_NODES)
        r1 = ga1.run(demand_scenarios, verbose=False)

        ga2 = ChargerOptimizerGA(config=config, n_nodes=N_NODES)
        r2 = ga2.run(demand_scenarios, verbose=False)

        np.testing.assert_array_equal(
            r1["best_chromosome"], r2["best_chromosome"]
        )
        assert r1["best_fitness"] == pytest.approx(r2["best_fitness"])


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
