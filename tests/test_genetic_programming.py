"""Tests for gp_tree.py, optimizers/base.py and optimizers/genetic_programming.py.

Run with either:
    python -m pytest tests/test_genetic_programming.py -q
    python tests/test_genetic_programming.py
Standard library only (no numpy / control needed).
"""
import contextlib
import math
import pathlib
import random
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from smart_control import gp_tree as gt
from smart_control.gp_tree import DEFAULT_GRAMMAR, GPGrammar
from smart_control.optimizers import genetic_programming as gp
from smart_control.optimizers.base import BaseOptimizer, CountedObjective, PENALTY_FITNESS
from smart_control.optimizers.genetic_programming import GeneticProgramming
from smart_control.types import OptimizationResult

G = DEFAULT_GRAMMAR
BOUNDS = G.parameter_bounds()


@contextlib.contextmanager
def raises(exc):
    try:
        yield
    except exc:
        return
    raise AssertionError(f"expected {exc.__name__}")


def _random_trees(grammar, n=200, seed=1):
    rng = random.Random(seed)
    return [gp.random_tree(rng, grammar, rng.randint(0, grammar.max_depth), rng.choice(["full", "grow"]))
            for _ in range(n)]


# ---------------------------------------------------------------------------
# gp_tree: primitives, helpers, evaluation
# ---------------------------------------------------------------------------
def test_protected_division_and_safe_output():
    assert gt.protected_div(1.0, 0.0) == 1.0
    assert gt.protected_div(1.0, 1e-9) == 1.0
    assert gt.protected_div(1.0, float("nan")) == 1.0
    assert gt.protected_div(6.0, 3.0) == 2.0
    assert gt.safe_output(float("nan"), 5.0) == 0.0
    assert gt.safe_output(float("inf"), 5.0) == 5.0
    assert gt.safe_output(-1e300, 5.0) == -5.0
    assert gt.safe_output(2.0, 5.0) == 2.0


def test_depth_size_and_string():
    tree = ("add", ("mul", 2.0, "e"), ("tanh", "ie"))
    assert gt.tree_depth(tree) == 2 and gt.tree_size(tree) == 6
    assert gt.tree_depth("e") == 0 and gt.tree_size(1.5) == 1
    assert gt.tree_to_string(tree) == "((2 * e) + tanh(ie))"


def test_validate_tree_rejects_bad_trees():
    gt.validate_tree(("add", "e", 1.0))
    with raises(ValueError):
        gt.validate_tree(("sin", "e"))                      # unknown function
    with raises(ValueError):
        gt.validate_tree(("add", "e"))                      # wrong arity
    with raises(ValueError):
        gt.validate_tree("speed")                           # unknown variable
    with raises(ValueError):
        gt.validate_tree(float("nan"))                      # non-finite constant
    with raises(TypeError):
        gt.validate_tree(["add", "e", "e"])                 # lists are not trees
    deep = ("tanh", ("tanh", ("tanh", ("tanh", "e"))))
    with raises(ValueError):
        gt.validate_tree(deep)                              # deeper than max_depth=3
    gt.validate_tree(deep, G, max_depth=4)


def test_uses_variable():
    assert gt.uses_variable(("add", ("mul", 2.0, "e"), "ie"), "e")
    assert gt.uses_variable("ie", "ie")
    assert not gt.uses_variable(("tanh", ("tanh", 1.26)), "e")
    assert not gt.uses_variable(("mul", "ie", 2.0), "e")


def test_grammar_validation():
    with raises(ValueError):
        GPGrammar(variables=("__import__",))                # no private / dunder names
    with raises(ValueError):
        GPGrammar(variables=("e", "e"))
    with raises(ValueError):
        GPGrammar(functions=("pow",))
    with raises(ValueError):
        GPGrammar(const_range=(1.0, 1.0))
    with raises(ValueError):
        GPGrammar(max_depth=0)


def test_compile_matches_interpreter():
    rng = random.Random(7)
    for tree in _random_trees(G, 300):
        fn = gt.compile_tree(tree, G)
        for _ in range(5):
            e, de, ie = (rng.choice([rng.uniform(-5, 5), 0.0, 1e-9, 1e150]) for _ in range(3))
            a = fn(e, de, ie)
            b = gt.evaluate_tree(tree, {"e": e, "de": de, "ie": ie})
            assert a == b or (a != a and b != b), (tree, a, b)


def test_compiled_tree_never_raises_and_has_no_builtins():
    tree = ("div", ("mul", "e", "e"), ("sub", "e", "e"))    # x*x / 0 -> protected
    fn = gt.compile_tree(tree, G)
    assert fn(1e200, 0.0, 0.0) == 1.0
    assert gt.safe_output(fn(float("inf"), 0.0, 0.0), 10.0) in (1.0, 0.0, 10.0)
    with raises(ValueError):
        gt.compile_tree("__import__('os')", G)              # unknown "variable" is rejected


def test_pid_law_fits_default_grammar():
    pid = ("add", ("add", ("mul", 2.0, "e"), ("mul", 0.5, "ie")), ("mul", 0.1, "de"))
    gt.validate_tree(pid, G)
    assert gt.tree_depth(pid) == G.max_depth == 3
    fn = gt.compile_tree(pid, G)
    assert abs(fn(1.0, 2.0, 4.0) - (2.0 + 2.0 + 0.2)) < 1e-12


# ---------------------------------------------------------------------------
# gp_tree: vector encoding
# ---------------------------------------------------------------------------
def test_vector_length_and_bounds():
    assert G.n_slots == 15 and G.n_params == 30 == len(BOUNDS)
    assert BOUNDS[0] == (0.0, 9.0)          # interior: 5 functions + 3 vars + const
    assert BOUNDS[1] == G.const_range
    assert BOUNDS[-2] == (0.0, 4.0)         # leaf slot: 3 vars + const
    assert gt.depth_for_n_params(30) == 3 and gt.depth_for_n_params(62) == 4
    assert gt.depth_for_n_params(31) is None


def test_encode_decode_roundtrip_all_depths():
    for depth in (1, 2, 3, 4):
        grammar = GPGrammar(max_depth=depth)
        for tree in _random_trees(grammar, 150, seed=depth):
            vector = gt.encode_tree(tree, grammar)
            assert len(vector) == grammar.n_params
            assert all(lo <= v <= hi for v, (lo, hi) in zip(vector, grammar.parameter_bounds()))
            assert gt.decode_vector(vector, grammar) == tree


def test_decode_is_total_and_depth_safe():
    rng = random.Random(3)
    for _ in range(300):
        vector = [rng.uniform(lo - 3, hi + 3) for lo, hi in BOUNDS]
        tree = gt.decode_vector(vector, G)
        gt.validate_tree(tree, G)
    for bad in (float("nan"), float("inf"), -float("inf")):
        gt.validate_tree(gt.decode_vector([bad] * G.n_params, G), G)
    with raises(ValueError):
        gt.decode_vector([0.0] * 29, G)


def test_decode_is_robust_to_small_symbol_noise():
    """Symbol genes sit at index+0.5, so +-0.4 noise (what DE/PSO/GA do) keeps the tree."""
    rng = random.Random(11)
    for tree in _random_trees(G, 100, seed=5):
        vector = gt.encode_tree(tree, G)
        noisy = [v + (rng.uniform(-0.4, 0.4) if i % 2 == 0 else 0.0) for i, v in enumerate(vector)]
        assert gt.decode_vector(noisy, G) == tree


def test_constants_are_clipped_to_range_on_encode():
    vector = gt.encode_tree(("mul", 99.0, "e"), G)
    assert max(vector[1::2]) <= G.const_range[1]
    assert gt.decode_vector(vector, G) == ("mul", G.const_range[1], "e")


# ---------------------------------------------------------------------------
# Variation operators
# ---------------------------------------------------------------------------
def test_operators_keep_trees_valid_and_within_depth():
    rng = random.Random(21)
    for depth in (2, 3, 4):
        grammar = GPGrammar(max_depth=depth)
        trees = _random_trees(grammar, 60, seed=depth)
        for a in trees:
            b = trees[rng.randrange(len(trees))]
            for child in (
                gp.subtree_crossover(rng, a, b, depth),
                gp.subtree_mutation(rng, a, grammar, depth),
                gp.point_mutation(rng, a, grammar),
                gp.constant_mutation(rng, a, grammar, 0.1),
            ):
                gt.validate_tree(child, grammar)
                assert gt.tree_depth(child) <= depth


def test_crossover_and_mutation_respect_max_size():
    rng = random.Random(4)
    trees = _random_trees(G, 80, seed=9)
    for a in trees:
        b = trees[rng.randrange(len(trees))]
        small = min(gt.tree_size(a), 5)
        child = gp.subtree_crossover(rng, a, b, 3, max_size=max(gt.tree_size(a), 5))
        assert gt.tree_size(child) <= max(gt.tree_size(a), 5)
        mut = gp.subtree_mutation(rng, a, G, 3, max_size=gt.tree_size(a))
        assert gt.tree_size(mut) <= gt.tree_size(a) or mut == a
        assert small >= 1


def test_constant_mutation_stays_in_range_and_changes_something():
    rng = random.Random(8)
    tree = ("add", ("mul", 9.9, "e"), -9.9)
    for _ in range(50):
        child = gp.constant_mutation(rng, tree, G, 0.3)
        for _p, node in gp.enumerate_points(child):
            if gt._is_const(node):
                assert G.const_range[0] <= node <= G.const_range[1]
    assert any(gp.constant_mutation(rng, tree, G, 0.3) != tree for _ in range(10))


def test_random_tree_methods():
    rng = random.Random(2)
    for _ in range(50):
        full = gp.random_tree(rng, G, 3, "full")
        assert gt.tree_depth(full) == 3
        assert gt.tree_depth(gp.random_tree(rng, G, 3, "grow")) <= 3
    with raises(ValueError):
        gp.random_tree(rng, G, 3, "nope")


# ---------------------------------------------------------------------------
# GP optimizer: contract and behaviour
# ---------------------------------------------------------------------------
def _target(e, de, ie):
    return 3.0 * e + 2.0 * ie


_SAMPLES = [tuple(random.Random(i).uniform(-2, 2) for _ in range(3)) for i in range(30)]


def regression_objective(vector):
    fn = gt.compile_tree(gt.decode_vector(vector, G), G)
    return sum((fn(*s) - _target(*s)) ** 2 for s in _SAMPLES) / len(_SAMPLES)


FAST = {"population_size": 24, "max_iterations": 15}


def test_result_contract():
    result = GeneticProgramming().optimize(regression_objective, BOUNDS, FAST)
    assert isinstance(result, OptimizationResult)
    assert len(result.best_solution) == len(BOUNDS)
    assert all(isinstance(v, float) for v in result.best_solution)
    assert len(result.convergence_history) == FAST["max_iterations"]
    hist = result.convergence_history
    assert all(b <= a for a, b in zip(hist, hist[1:])), "best-so-far must be non-increasing"
    assert hist[-1] == result.best_fitness <= result.metadata["initial_best_fitness"]
    assert abs(regression_objective(result.best_solution) - result.best_fitness) < 1e-12
    meta = result.metadata
    assert meta["algorithm"] == "GP" and meta["iterations_run"] == FAST["max_iterations"]
    assert meta["best_tree"] == gt.decode_vector(result.best_solution, G)
    assert 1 <= meta["n_evaluations"] <= FAST["population_size"] * (FAST["max_iterations"] + 1)
    assert isinstance(meta["best_expression"], str) and meta["stopped_by"] == "max_iterations"


def test_every_candidate_is_inside_bounds_and_depth_limit():
    seen = []

    def spy(vector):
        seen.append(list(vector))
        return regression_objective(vector)

    GeneticProgramming().optimize(spy, BOUNDS, FAST)
    assert seen
    for vector in seen:
        assert len(vector) == len(BOUNDS)
        assert all(lo <= v <= hi for v, (lo, hi) in zip(vector, BOUNDS))
        assert gt.tree_depth(gt.decode_vector(vector, G)) <= G.max_depth


def test_reproducible_and_no_global_rng_side_effects():
    state = random.getstate()
    r1 = GeneticProgramming().optimize(regression_objective, BOUNDS, {**FAST, "random_seed": 5})
    assert random.getstate() == state, "optimizer must use its own RNG"
    r2 = GeneticProgramming().optimize(regression_objective, BOUNDS, {**FAST, "random_seed": 5})
    r3 = GeneticProgramming().optimize(regression_objective, BOUNDS, {**FAST, "random_seed": 6})
    assert r1.best_solution == r2.best_solution and r1.convergence_history == r2.convergence_history
    assert r1.convergence_history != r3.convergence_history


def test_gp_recovers_a_known_expression():
    """3*e + 2*ie is representable at depth 2; at least one of a few seeds must find it."""
    best = min(
        GeneticProgramming().optimize(
            regression_objective, BOUNDS, {"population_size": 60, "max_iterations": 60, "random_seed": s}
        ).best_fitness
        for s in range(4)
    )
    assert best < 1e-3, best


def test_gp_beats_its_own_initial_population():
    result = GeneticProgramming().optimize(
        regression_objective, BOUNDS, {"population_size": 40, "max_iterations": 40})
    assert result.best_fitness < 0.5 * result.metadata["initial_best_fitness"]


def test_budget_and_early_stop_settings():
    calls = CountedObjective(regression_objective)
    result = GeneticProgramming().optimize(calls, BOUNDS, {**FAST, "max_evaluations": 50})
    assert calls.n_calls <= 50 and result.metadata["n_evaluations"] == calls.n_calls
    assert result.metadata["stopped_by"] == "max_evaluations"
    stopped = GeneticProgramming().optimize(
        regression_objective, BOUNDS, {**FAST, "max_iterations": 200, "target_fitness": 1e9})
    assert stopped.metadata["stopped_by"] == "target_fitness" and stopped.convergence_history == []


def test_cache_saves_evaluations_without_changing_result():
    on = GeneticProgramming().optimize(regression_objective, BOUNDS, {**FAST, "use_cache": True})
    off = GeneticProgramming().optimize(regression_objective, BOUNDS, {**FAST, "use_cache": False})
    assert on.best_fitness == off.best_fitness and on.best_solution == off.best_solution
    assert on.metadata["n_evaluations"] <= off.metadata["n_evaluations"]


def test_max_size_is_respected():
    result = GeneticProgramming().optimize(regression_objective, BOUNDS, {**FAST, "max_size": 7})
    assert gt.tree_size(result.metadata["best_tree"]) <= 7


def test_invalid_objective_values_do_not_break_the_search():
    def flaky(vector):
        value = regression_objective(vector)
        return float("nan") if value > 50 else (float("inf") if value > 40 else value)

    result = GeneticProgramming().optimize(flaky, BOUNDS, FAST)
    assert math.isfinite(result.best_fitness) and result.best_fitness < PENALTY_FITNESS


def test_depth_four_grammar_via_bounds():
    grammar = GPGrammar(max_depth=4, const_range=(-5.0, 5.0))
    seen = []

    def spy(vector):
        seen.append(len(vector))
        return sum(abs(v) for v in vector[1::2])

    result = GeneticProgramming().optimize(spy, grammar.parameter_bounds(), FAST)
    assert set(seen) == {62} and result.metadata["max_depth"] == 4
    assert result.metadata["const_range"] == [-5.0, 5.0]


def test_bounds_are_validated():
    objective = regression_objective
    with raises(ValueError):
        GeneticProgramming().optimize(objective, [(-1.0, 1.0)] * 21, FAST)       # e.g. NN bounds
    with raises(ValueError):
        GeneticProgramming().optimize(objective, [(-5.0, 5.0)] * 30, FAST)       # right length, wrong grammar
    with raises(ValueError):
        GeneticProgramming().optimize(objective, [], FAST)
    with raises(ValueError):
        GeneticProgramming().optimize(objective, [(2.0, 1.0)] * 30, FAST)


def test_settings_are_validated():
    for bad in ({"population_size": 1}, {"max_iterations": 0}, {"elite_count": 30},
                {"crossover_rate": 1.5}, {"max_evaluations": 0}, {"tournament_size": 0}):
        with raises(ValueError):
            GeneticProgramming().optimize(regression_objective, BOUNDS, {**FAST, **bad})


def test_settings_precedence_defaults_constructor_call():
    opt = GeneticProgramming({"population_size": 10, "max_iterations": 5})
    assert opt.resolve_settings()["population_size"] == 10
    assert opt.resolve_settings({"population_size": 12})["population_size"] == 12
    assert opt.resolve_settings()["tournament_size"] == 5           # untouched default
    result = opt.optimize(regression_objective, BOUNDS)
    assert len(result.convergence_history) == 5 and result.metadata["population_size"] == 10


# ---------------------------------------------------------------------------
# base.py helpers
# ---------------------------------------------------------------------------
def test_base_optimizer_helpers():
    with raises(NotImplementedError):
        BaseOptimizer().optimize(lambda x: 0.0, [(0, 1)])
    assert BaseOptimizer.validate_bounds([(0, 1), (2, 3)]) == ([0.0, 2.0], [1.0, 3.0])
    for bad in (None, [], [(1,)], [(0, float("inf"))], [(3, 1)]):
        with raises(ValueError):
            BaseOptimizer.validate_bounds(bad)
    assert BaseOptimizer.safe_fitness(float("nan")) == PENALTY_FITNESS
    assert BaseOptimizer.safe_fitness("oops") == PENALTY_FITNESS
    assert BaseOptimizer.safe_fitness(2) == 2.0
    counted = CountedObjective(lambda x: float("inf"))
    assert counted([0.0]) == PENALTY_FITNESS and counted.n_calls == 1


def test_subclass_that_skips_super_init_still_works():
    """PSO (and possibly others) never call super().__init__; settings must still resolve."""
    class Legacy(BaseOptimizer):
        def __init__(self):  # noqa: D107 - intentionally does not call super()
            pass

        def default_settings(self):
            return {"population_size": 7}

    assert Legacy().resolve_settings({"max_iterations": 3}) == {"population_size": 7, "max_iterations": 3}


def test_existing_pso_still_subclasses_base():
    try:
        from smart_control.optimizers.particle_swarm import ParticleSwarmOptimization
    except ImportError:  # numpy missing
        return
    assert issubclass(ParticleSwarmOptimization, BaseOptimizer)
    result = ParticleSwarmOptimization({"population_size": 8, "max_iterations": 5}).optimize(
        lambda x: sum(v * v for v in x), [(-1, 1)] * 3)
    assert isinstance(result, OptimizationResult)


if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    for name, fn in tests:
        fn()
        print(f"ok  {name}")
    print(f"\n{len(tests)} tests passed")
