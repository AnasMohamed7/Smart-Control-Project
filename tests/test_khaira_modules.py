"""Unit tests for Khaira's three components:

  1. PIDController        (smart_control/controllers/pid.py)
  2. GeneticAlgorithm     (smart_control/optimizers/genetic_algorithm.py)
  3. GPController         (smart_control/controllers/gp_controller.py)

Run with either:
    python -m pytest tests/test_khaira_modules.py -v
    python tests/test_khaira_modules.py

Dependencies: numpy, control  (same as the project)
"""
import contextlib
import math
import pathlib
import sys

import numpy as np

# Allow running from the repo root or from the tests/ subdirectory
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from smart_control.controllers.pid import PIDController
from smart_control.controllers.gp_controller import GPController
from smart_control.optimizers.genetic_algorithm import GeneticAlgorithm
from smart_control.gp_tree import DEFAULT_GRAMMAR, GPGrammar, encode_tree, decode_vector
from smart_control.types import OptimizationResult


# ---------------------------------------------------------------------------
# Tiny helper so we can run as a plain script without pytest installed
# ---------------------------------------------------------------------------
@contextlib.contextmanager
def raises(exc):
    """Assert that the body raises exc (mimics pytest.raises)."""
    try:
        yield
    except exc:
        return
    raise AssertionError(f"Expected {exc.__name__} to be raised")


# ===========================================================================
# 1. PIDController tests
# ===========================================================================

def test_pid_inherits_base_controller():
    from smart_control.controllers.base import BaseController
    assert issubclass(PIDController, BaseController)


def test_pid_default_construction():
    pid = PIDController()
    assert pid.Kp == 1.0
    assert pid.Ki == 0.0
    assert pid.Kd == 0.0


def test_pid_set_parameters_basic():
    pid = PIDController()
    pid.set_parameters([2.0, 0.5, 0.1])
    assert pid.Kp == 2.0
    assert pid.Ki == 0.5
    assert pid.Kd == 0.1


def test_pid_set_parameters_wrong_length():
    pid = PIDController()
    with raises(ValueError):
        pid.set_parameters([1.0, 2.0])          # too few
    with raises(ValueError):
        pid.set_parameters([1.0, 2.0, 3.0, 4.0])  # too many


def test_pid_set_parameters_non_finite():
    pid = PIDController()
    with raises(ValueError):
        pid.set_parameters([float("nan"), 0.0, 0.0])
    with raises(ValueError):
        pid.set_parameters([1.0, float("inf"), 0.0])


def test_pid_parameter_bounds_shape():
    bounds = PIDController().parameter_bounds()
    assert len(bounds) == 3
    for lo, hi in bounds:
        assert hi > lo >= 0.0  # gains must be non-negative and bounded


def test_pid_pure_proportional():
    """P-only: u = Kp * e, no I or D contribution."""
    pid = PIDController()
    pid.set_parameters([3.0, 0.0, 0.0])
    pid.reset()
    u = pid.compute(2.0, 0.01)
    assert abs(u - 6.0) < 1e-10


def test_pid_integral_accumulation():
    """I-only: after n steps, integral = sum(e*dt)."""
    pid = PIDController()
    pid.set_parameters([0.0, 1.0, 0.0])
    pid.reset()
    errors = [1.0, 0.5, 0.25]
    dt     = 0.1
    expected_integral = 0.0
    for e in errors:
        expected_integral += e * dt
        u = pid.compute(e, dt)
    assert abs(u - expected_integral) < 1e-10


def test_pid_derivative_first_step_is_zero():
    """On the very first call after reset(), de/dt must be 0."""
    pid = PIDController()
    pid.set_parameters([0.0, 0.0, 10.0])   # D-only
    pid.reset()
    u = pid.compute(5.0, 0.01)
    # D term on first step = Kd * 0 = 0
    assert u == 0.0


def test_pid_derivative_finite_difference():
    """D term = Kd * (e[k] - e[k-1]) / dt after first step."""
    pid = PIDController()
    pid.set_parameters([0.0, 0.0, 2.0])   # D-only, Kd=2
    pid.reset()
    pid.compute(1.0, 0.1)      # first step, D=0
    u = pid.compute(3.0, 0.1)  # second step, D = 2*(3-1)/0.1 = 40
    assert abs(u - 40.0) < 1e-10


def test_pid_anti_windup_clamp():
    """Integral is clamped at integral_limit."""
    limit = 5.0
    pid = PIDController({"integral_limit": limit})
    pid.set_parameters([0.0, 1.0, 0.0])  # I-only, Ki=1
    pid.reset()
    # Feed large error for many steps; integral must saturate at limit
    for _ in range(1000):
        pid.compute(10.0, 0.1)
    assert pid._integral <= limit + 1e-10


def test_pid_output_saturation():
    """Output must not exceed u_limit regardless of gains."""
    pid = PIDController({"u_limit": 10.0})
    pid.set_parameters([1000.0, 0.0, 0.0])  # huge Kp
    pid.reset()
    u = pid.compute(5.0, 0.01)
    assert abs(u) <= 10.0 + 1e-10


def test_pid_reset_clears_state_not_gains():
    """reset() must zero integral/prev_error but keep Kp/Ki/Kd unchanged."""
    pid = PIDController()
    pid.set_parameters([2.0, 1.0, 0.5])
    # Build up some state
    for _ in range(10):
        pid.compute(1.0, 0.01)
    # Reset
    pid.reset()
    assert pid._integral   == 0.0
    assert pid._prev_error == 0.0
    # Gains must survive
    assert pid.Kp == 2.0
    assert pid.Ki == 1.0
    assert pid.Kd == 0.5


def test_pid_non_finite_error_returns_zero():
    """compute() with NaN error should return 0 without raising."""
    pid = PIDController()
    pid.set_parameters([2.0, 1.0, 0.5])
    pid.reset()
    u = pid.compute(float("nan"), 0.01)
    assert u == 0.0


def test_pid_invalid_dt_raises():
    pid = PIDController()
    pid.set_parameters([1.0, 0.0, 0.0])
    pid.reset()
    with raises(ValueError):
        pid.compute(1.0, 0.0)      # dt == 0 is invalid
    with raises(ValueError):
        pid.compute(1.0, -0.01)    # negative dt


def test_pid_derivative_filter():
    """With tau_f > 0, the derivative is low-pass filtered."""
    pid_raw = PIDController({"tau_f": 0.0})
    pid_flt = PIDController({"tau_f": 0.1})
    for pid in (pid_raw, pid_flt):
        pid.set_parameters([0.0, 0.0, 1.0])
        pid.reset()
    errors = [0.0, 10.0, 0.0, 10.0, 0.0]  # noisy spike
    dt = 0.01
    raw_outputs = []
    flt_outputs = []
    for e in errors:
        raw_outputs.append(pid_raw.compute(e, dt))
        flt_outputs.append(pid_flt.compute(e, dt))
    # Filtered output should have smaller max magnitude (less noisy)
    assert max(abs(v) for v in flt_outputs) < max(abs(v) for v in raw_outputs)


def test_pid_full_pid_step():
    """Full PID with known values: verify exact arithmetic."""
    pid = PIDController()
    pid.set_parameters([1.0, 2.0, 0.5])
    pid.reset()
    # Step 1: e=1.0, dt=0.1 -> D=0, I=0.1, P=1  -> u = 1 + 0.2 + 0 = 1.2
    u1 = pid.compute(1.0, 0.1)
    assert abs(u1 - 1.2) < 1e-10
    # Step 2: e=0.5, dt=0.1 -> D=(0.5-1.0)/0.1=-5, I=0.15, P=0.5
    # u = 0.5*1 + 0.15*2 + (-5)*0.5 = 0.5 + 0.3 - 2.5 = -1.7
    u2 = pid.compute(0.5, 0.1)
    assert abs(u2 - (-1.7)) < 1e-10


def test_pid_with_simulation_loop():
    """Integration test: PID + second-order plant closes the loop."""
    from smart_control.plants.models import get_second_order_plant
    from smart_control.plants.simulation import simulate_closed_loop

    plant = get_second_order_plant()
    pid   = PIDController()
    pid.set_parameters([2.0, 1.0, 0.5])

    res = simulate_closed_loop(plant, pid, {"time_step": 0.01, "time_end": 20.0})
    assert np.all(np.isfinite(res.output)), "simulation became unstable"
    # With these gains the second-order plant should settle close to 1.0
    ss_error = abs(res.output[-1] - 1.0)
    assert ss_error < 0.05, f"steady-state error too large: {ss_error:.4f}"


# ===========================================================================
# 2. GeneticAlgorithm tests
# ===========================================================================

def test_ga_inherits_base_optimizer():
    from smart_control.optimizers.base import BaseOptimizer
    assert issubclass(GeneticAlgorithm, BaseOptimizer)


def test_ga_returns_optimization_result():
    result = GeneticAlgorithm().optimize(
        lambda x: float(np.sum(np.square(x))),
        [(-5, 5)] * 3,
        {"population_size": 10, "max_iterations": 5},
    )
    assert isinstance(result, OptimizationResult)


def test_ga_minimises_sphere_function():
    """GA should find a near-zero solution for f(x) = sum(x^2)."""
    result = GeneticAlgorithm().optimize(
        lambda x: float(np.sum(np.square(x))),
        [(-5, 5)] * 5,
        {"population_size": 30, "max_iterations": 200, "random_seed": 42},
    )
    assert result.best_fitness < 0.5, f"GA failed on sphere: {result.best_fitness:.4f}"


def test_ga_convergence_history_non_increasing():
    """best-so-far must never increase across generations."""
    result = GeneticAlgorithm().optimize(
        lambda x: float(np.sum(np.square(x))),
        [(-3, 3)] * 4,
        {"population_size": 20, "max_iterations": 50},
    )
    history = result.convergence_history
    assert len(history) > 0
    for a, b in zip(history, history[1:]):
        assert b <= a + 1e-12, "convergence history increased!"


def test_ga_history_length_matches_iterations():
    """History should have one entry per generation actually run."""
    max_iter = 15
    result = GeneticAlgorithm().optimize(
        lambda x: float(np.sum(np.square(x))),
        [(-1, 1)] * 2,
        {"population_size": 8, "max_iterations": max_iter},
    )
    assert len(result.convergence_history) == max_iter


def test_ga_reproducible():
    """Same seed → same result; different seed → (usually) different result."""
    obj    = lambda x: float(np.sum(np.square(x)))
    bounds = [(-5, 5)] * 3
    cfg    = {"population_size": 20, "max_iterations": 30, "random_seed": 7}
    r1 = GeneticAlgorithm().optimize(obj, bounds, cfg)
    r2 = GeneticAlgorithm().optimize(obj, bounds, cfg)
    assert r1.best_solution == r2.best_solution
    assert r1.convergence_history == r2.convergence_history


def test_ga_no_global_rng_side_effects():
    """GA must use its own local RNG (not np.random global state)."""
    rng_state = np.random.get_state()
    GeneticAlgorithm().optimize(
        lambda x: float(np.sum(np.square(x))),
        [(-5, 5)] * 3,
        {"population_size": 10, "max_iterations": 10, "random_seed": 42},
    )
    new_state = np.random.get_state()
    # Compare the actual random arrays inside the state tuples
    old_arr = rng_state[1]
    new_arr = new_state[1]
    assert np.array_equal(old_arr, new_arr), "GA polluted the global NumPy RNG"


def test_ga_solution_respects_bounds():
    """Every gene of the best solution must lie within its bound."""
    bounds = [(-2.0, 3.0), (1.0, 5.0), (-10.0, -1.0)]
    result = GeneticAlgorithm().optimize(
        lambda x: float(np.sum(np.square(x))),
        bounds,
        {"population_size": 20, "max_iterations": 30},
    )
    for v, (lo, hi) in zip(result.best_solution, bounds):
        assert lo - 1e-10 <= v <= hi + 1e-10, f"{v} outside [{lo}, {hi}]"


def test_ga_metadata_keys():
    result = GeneticAlgorithm().optimize(
        lambda x: 0.0, [(-1, 1)] * 2,
        {"population_size": 4, "max_iterations": 2},
    )
    meta = result.metadata
    for key in ("algorithm", "iterations_run", "n_evaluations", "random_seed"):
        assert key in meta, f"metadata missing key: {key}"
    assert meta["algorithm"] == "GA"


def test_ga_handles_nan_objective():
    """NaN objective values must not break the search."""
    def flaky(x):
        v = float(np.sum(np.square(x)))
        return float("nan") if v > 10 else v

    result = GeneticAlgorithm().optimize(
        flaky, [(-5, 5)] * 2,
        {"population_size": 10, "max_iterations": 20},
    )
    assert math.isfinite(result.best_fitness)


def test_ga_on_rosenbrock():
    """Rosenbrock function (banana): f* = 0 at (1,1)."""
    def rosenbrock(x):
        return float(100.0 * (x[1] - x[0] ** 2) ** 2 + (1.0 - x[0]) ** 2)

    result = GeneticAlgorithm().optimize(
        rosenbrock,
        [(-2, 2), (-2, 2)],
        {"population_size": 50, "max_iterations": 300, "random_seed": 42},
    )
    # Rosenbrock is hard; require only a reasonable improvement
    assert result.best_fitness < 10.0, f"Rosenbrock too large: {result.best_fitness:.4f}"


def test_ga_tunes_pid_on_second_order_plant():
    """Integration: GA optimising PID on the second-order plant."""
    from smart_control.plants.models import get_second_order_plant
    from smart_control.evaluation.objective import build_objective
    from smart_control.plants.simulation import simulate_closed_loop

    plant = get_second_order_plant()
    pid   = PIDController()
    obj   = build_objective(pid, plant)
    result = GeneticAlgorithm().optimize(
        obj, pid.parameter_bounds(),
        {"population_size": 20, "max_iterations": 50, "random_seed": 42},
    )
    pid.set_parameters(result.best_solution)
    sim = simulate_closed_loop(plant, pid)
    assert np.all(np.isfinite(sim.output)), "optimised PID produced unstable response"
    assert abs(sim.output[-1] - 1.0) < 0.1


def test_ga_tunes_fuzzy_controller():
    """GA must work with FuzzyController, not just PID."""
    from smart_control.plants.models import get_second_order_plant
    from smart_control.controllers.fuzzy import FuzzyController
    from smart_control.evaluation.objective import build_objective

    plant = get_second_order_plant()
    ctrl  = FuzzyController()
    obj   = build_objective(ctrl, plant)
    result = GeneticAlgorithm().optimize(
        obj, ctrl.parameter_bounds(),
        {"population_size": 15, "max_iterations": 30, "random_seed": 1},
    )
    assert isinstance(result, OptimizationResult)
    assert math.isfinite(result.best_fitness)


def test_ga_tunes_nn_controller():
    """GA must work with NeuralNetworkController (21-dimensional search)."""
    from smart_control.plants.models import get_second_order_plant
    from smart_control.controllers.neural_network import NeuralNetworkController
    from smart_control.evaluation.objective import build_objective

    plant = get_second_order_plant()
    ctrl  = NeuralNetworkController()
    obj   = build_objective(ctrl, plant)
    result = GeneticAlgorithm().optimize(
        obj, ctrl.parameter_bounds(),
        {"population_size": 10, "max_iterations": 20, "random_seed": 5},
    )
    assert isinstance(result, OptimizationResult)
    assert math.isfinite(result.best_fitness)


def test_ga_early_stopping_on_tolerance():
    """When tolerance is set and population converges, GA stops early."""
    call_count = [0]

    def counting_fn(x):
        call_count[0] += 1
        return 0.0  # all candidates have identical fitness -> converges immediately

    result = GeneticAlgorithm().optimize(
        counting_fn, [(-1, 1)] * 2,
        {"population_size": 6, "max_iterations": 100, "tolerance": 1e-3},
    )
    # Should stop well before 100 generations
    assert result.metadata["iterations_run"] < 100


# ===========================================================================
# 3. GPController tests
# ===========================================================================

def test_gp_inherits_base_controller():
    from smart_control.controllers.base import BaseController
    assert issubclass(GPController, BaseController)


def test_gp_default_construction_no_tree():
    ctrl = GPController()
    assert ctrl.tree is None


def test_gp_set_tree_validates():
    ctrl = GPController()
    # Valid tree
    ctrl.set_tree(("add", "e", "ie"))
    assert ctrl.tree == ("add", "e", "ie")
    # Invalid tree should raise
    with raises(ValueError):
        ctrl.set_tree(("sin", "e"))    # "sin" not in function set
    with raises(ValueError):
        ctrl.set_tree("speed")         # unknown variable


def test_gp_compute_requires_tree():
    ctrl = GPController()
    with raises(RuntimeError):
        ctrl.compute(1.0, 0.01)        # no tree set yet


def test_gp_compute_proportional_tree():
    """Tree 'e' should produce u = e (after safe_output clipping)."""
    ctrl = GPController()
    ctrl.set_tree("e")
    ctrl.reset()
    u = ctrl.compute(3.0, 0.01)
    assert abs(u - 3.0) < 1e-10


def test_gp_compute_pid_law_tree():
    """Standard PID-law tree should give the expected numeric result."""
    # ("add", ("add", Kp*e, Ki*ie), Kd*de)
    # Kp=2, Ki=0.5, Kd=0.1
    tree = (
        "add",
        ("add", ("mul", 2.0, "e"), ("mul", 0.5, "ie")),
        ("mul", 0.1, "de"),
    )
    ctrl = GPController()
    ctrl.set_tree(tree)
    ctrl.reset()

    dt = 0.1
    # Step 1: e=1, de=0 (first step), ie=0.1
    u1 = ctrl.compute(1.0, dt)
    # ie = 1*0.1 = 0.1, de = 0
    # u  = (2*1 + 0.5*0.1) + 0.1*0 = 2.05
    assert abs(u1 - 2.05) < 1e-10

    # Step 2: e=0.5, de=(0.5-1)/0.1=-5, ie=0.1+0.5*0.1=0.15
    u2 = ctrl.compute(0.5, dt)
    # u = (2*0.5 + 0.5*0.15) + 0.1*(-5) = 1.0 + 0.075 - 0.5 = 0.575
    assert abs(u2 - 0.575) < 1e-10


def test_gp_reset_clears_integral_not_tree():
    ctrl = GPController()
    ctrl.set_tree(("mul", 1.0, "ie"))
    ctrl.reset()
    ctrl.compute(2.0, 0.1)   # ie accumulates
    assert ctrl._ie != 0.0
    ctrl.reset()
    assert ctrl._ie      == 0.0
    assert ctrl._prev_e  == 0.0
    assert ctrl.tree is not None   # tree must survive reset


def test_gp_integral_clamp():
    """Error integral must stay within ±IE_LIMIT."""
    from smart_control.controllers.gp_controller import _IE_LIMIT
    ctrl = GPController()
    ctrl.set_tree("ie")
    ctrl.reset()
    for _ in range(10000):
        ctrl.compute(100.0, 0.1)
    assert ctrl._ie <= _IE_LIMIT + 1e-10


def test_gp_output_clamp():
    """safe_output must clip to ±U_LIMIT."""
    from smart_control.controllers.gp_controller import _U_LIMIT
    ctrl = GPController()
    ctrl.set_tree(("mul", 1000.0, "e"))   # huge gain
    ctrl.reset()
    u = ctrl.compute(5.0, 0.01)
    assert abs(u) <= _U_LIMIT + 1e-10


def test_gp_non_finite_error_handled():
    """NaN error should not raise; controller treats it as 0."""
    ctrl = GPController()
    ctrl.set_tree("e")
    ctrl.reset()
    u = ctrl.compute(float("nan"), 0.01)
    assert math.isfinite(u)


def test_gp_invalid_dt_raises():
    ctrl = GPController()
    ctrl.set_tree("e")
    ctrl.reset()
    ctrl.compute(1.0, 0.1)      # initialise prev
    with raises(ValueError):
        ctrl.compute(1.0, 0.0)  # zero dt


def test_gp_set_parameters_numeric_vector():
    """set_parameters() with a valid numeric vector should decode and compile."""
    grammar = DEFAULT_GRAMMAR
    # Encode a known tree and decode it back via set_parameters
    known_tree = ("add", ("mul", 2.0, "e"), ("mul", 0.5, "ie"))
    vector     = encode_tree(known_tree, grammar)

    ctrl = GPController(grammar=grammar)
    ctrl.set_parameters(vector)
    assert ctrl.tree == known_tree


def test_gp_parameter_bounds_matches_grammar():
    ctrl   = GPController()
    bounds = ctrl.parameter_bounds()
    expected = DEFAULT_GRAMMAR.parameter_bounds()
    assert len(bounds) == len(expected)
    for (lo, hi), (elo, ehi) in zip(bounds, expected):
        assert lo == elo and hi == ehi


def test_gp_set_parameters_wrong_length():
    ctrl = GPController()
    with raises(ValueError):
        ctrl.set_parameters([0.0] * 5)   # wrong length


def test_gp_mode_b_via_ga_optimizer():
    """Full integration: GA optimises GPController via numeric vector encoding."""
    from smart_control.plants.models import get_second_order_plant
    from smart_control.evaluation.objective import build_objective

    plant  = get_second_order_plant()
    ctrl   = GPController()
    obj    = build_objective(ctrl, plant)
    bounds = ctrl.parameter_bounds()

    result = GeneticAlgorithm().optimize(
        obj, bounds,
        {"population_size": 10, "max_iterations": 20, "random_seed": 7},
    )
    assert isinstance(result, OptimizationResult)
    assert math.isfinite(result.best_fitness)


def test_gp_tree_property():
    ctrl = GPController()
    assert ctrl.tree is None
    ctrl.set_tree("e")
    assert ctrl.tree == "e"


def test_gp_custom_grammar():
    """GPController respects a custom grammar with different max_depth."""
    grammar = GPGrammar(max_depth=2, variables=("e", "de", "ie"))
    ctrl    = GPController(grammar=grammar)
    bounds  = ctrl.parameter_bounds()
    # max_depth=2 → n_slots = 2^3-1 = 7, n_params = 14
    assert len(bounds) == 14


# ===========================================================================
# Script entry point (no pytest required)
# ===========================================================================

if __name__ == "__main__":
    tests = [
        (name, fn)
        for name, fn in sorted(globals().items())
        if name.startswith("test_") and callable(fn)
    ]
    passed = 0
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"  ok  {name}")
            passed += 1
        except Exception as exc:
            print(f"FAIL  {name}")
            print(f"      {exc}")
            failed += 1

    print(f"\n{passed + failed} tests run: {passed} passed, {failed} failed.")
    if failed:
        sys.exit(1)
