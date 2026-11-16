"""Genetic Programming (GP) optimizer -- implemented from scratch.

The GP optimizer evolves *expression trees* (programs).  A separate component,
``controllers/gp_controller.py``, turns one tree into a control signal.  Both
share the representation defined in :mod:`smart_control.gp_tree`.

How it plugs into the common optimizer interface
------------------------------------------------
``optimize(objective_function, bounds, settings)`` is called exactly like every
other optimizer.  The objective receives a flat parameter vector, so each tree
is passed through ``gp_tree.encode_tree`` (heap layout, see gp_tree.py) before
evaluation, and the controller's ``set_parameters`` decodes it again.
``bounds`` must be the GP controller's ``parameter_bounds()``
(= ``GPGrammar.parameter_bounds()``); the tree depth limit and the constant
range are read from them, so the controller stays the single source of truth.

Algorithm (generational, elitist)
---------------------------------
1. Initialization : ramped half-and-half (depth 1..max_depth, alternating
                    "full" / "grow"), duplicates avoided where possible.
2. Fitness        : ``objective_function(encode(tree))`` (minimized); NaN/inf
                    is replaced by a large finite penalty.  Evaluations of
                    identical trees are cached (``use_cache``).
3. Selection      : tournament (``tournament_size``) on
                    ``fitness + parsimony_coeff * size``.
4. Crossover      : subtree crossover; crossover points follow Koza's 90 % /
                    10 % internal/leaf rule and the donor subtree is chosen so
                    the child never exceeds ``max_depth`` (or ``max_size``).
5. Mutation       : subtree (30 %), point (30 %) or constant-perturbation
                    (40 %) mutation, each size/depth safe.
6. Elitism        : the best ``elite_count`` trees survive unchanged.
7. Tracking       : best-so-far tree/fitness; ``convergence_history`` has one
                    entry per generation (same convention as PSO: the initial
                    population is generation 0 and is not an entry).

Settings (all optional)
-----------------------
shared      : population_size=30, max_iterations=100, random_seed=42
GP specific : tournament_size=5, crossover_rate=0.8, mutation_rate=0.3,
              elite_count=1, internal_point_prob=0.9, const_sigma_ratio=0.1,
              max_size=None, parsimony_coeff=0.0, max_evaluations=None,
              target_fitness=None, use_cache=True, grammar=None
Defaults can also be set in ``config.GP_SETTINGS`` if that dict exists.

The result's ``best_solution`` is the encoded best tree; the tree itself and
its readable form are in ``metadata["best_tree"]`` / ``["best_expression"]``.
"""
from __future__ import annotations

import math
import random
from dataclasses import replace
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

from smart_control import gp_tree as gt
from smart_control.gp_tree import ARITY, CONST_TOKEN, GPGrammar, Tree
from smart_control.optimizers.base import BaseOptimizer, CountedObjective
from smart_control.types import OptimizationResult

_DEFAULT_GP_SETTINGS = {
    "population_size": 30,
    "max_iterations": 100,
    "random_seed": 42,
    "tournament_size": 5,        # tuned on the closed-loop objective (3 converged prematurely)
    "crossover_rate": 0.8,       # P(child made by crossover); else reproduction
    "mutation_rate": 0.3,        # P(child additionally mutated)
    "elite_count": 1,
    "internal_point_prob": 0.9,  # Koza: pick a function node as crossover/mutation point
    "const_sigma_ratio": 0.1,    # gaussian sigma = ratio * (const_hi - const_lo)
    "max_size": None,            # optional cap on node count (depth is capped by bounds)
    "parsimony_coeff": 0.0,      # selection-only size penalty (objective scale dependent)
    "max_evaluations": None,     # hard cap on objective calls
    "target_fitness": None,      # stop early once best <= target
    "use_cache": True,
    "grammar": None,             # GPGrammar override (functions / variables); depth from bounds
}

Path = Tuple[int, ...]


# ---------------------------------------------------------------------------
# Tree generation and variation operators (module level -> easy to unit test)
# ---------------------------------------------------------------------------
def _terminal_from_token(rng: random.Random, grammar: GPGrammar, token: str) -> Tree:
    if token == CONST_TOKEN:
        lo, hi = grammar.const_range
        return rng.uniform(lo, hi)
    return token


def random_terminal(rng: random.Random, grammar: GPGrammar) -> Tree:
    return _terminal_from_token(rng, grammar, grammar.terminals[rng.randrange(len(grammar.terminals))])


def random_tree(rng: random.Random, grammar: GPGrammar, max_depth: int, method: str = "grow") -> Tree:
    """Random tree of depth <= max_depth. method: 'full' | 'grow'."""
    if method not in ("full", "grow"):
        raise ValueError("method must be 'full' or 'grow'")
    if max_depth <= 0:
        return random_terminal(rng, grammar)
    if method == "full":
        name = grammar.functions[rng.randrange(len(grammar.functions))]
    else:
        symbol = grammar.interior_symbols[rng.randrange(len(grammar.interior_symbols))]
        if symbol not in ARITY:
            return _terminal_from_token(rng, grammar, symbol)
        name = symbol
    return (name,) + tuple(random_tree(rng, grammar, max_depth - 1, method) for _ in range(ARITY[name]))


def enumerate_points(tree: Tree, path: Path = ()) -> Iterator[Tuple[Path, Tree]]:
    """Yield (path, subtree) for every node, pre-order. A path is a tuple of tuple-indexes."""
    yield path, tree
    if isinstance(tree, tuple):
        for k in range(1, len(tree)):
            yield from enumerate_points(tree[k], path + (k,))


def replace_subtree(tree: Tree, path: Path, new: Tree) -> Tree:
    if not path:
        return new
    children = list(tree)
    children[path[0]] = replace_subtree(tree[path[0]], path[1:], new)
    return tuple(children)


def _pick_point(rng: random.Random, points: Sequence[Tuple[Path, Tree]], internal_prob: float):
    """Koza point selection: function node with prob internal_prob, else a leaf."""
    internal = [p for p in points if isinstance(p[1], tuple)]
    leaves = [p for p in points if not isinstance(p[1], tuple)]
    if internal and leaves:
        pool = internal if rng.random() < internal_prob else leaves
    else:
        pool = internal or leaves
    return pool[rng.randrange(len(pool))]


def subtree_crossover(rng: random.Random, parent_a: Tree, parent_b: Tree, max_depth: int,
                      max_size: Optional[int] = None, internal_prob: float = 0.9,
                      tries: int = 10) -> Tree:
    """Replace a subtree of parent_a by a subtree of parent_b; child respects the limits."""
    points_a = list(enumerate_points(parent_a))
    points_b = [(p, s, gt.tree_depth(s)) for p, s in enumerate_points(parent_b)]
    for _ in range(tries):
        path_a, _sub_a = _pick_point(rng, points_a, internal_prob)
        room = max_depth - len(path_a)  # how deep a donor subtree may be
        donors = [(p, s) for p, s, h in points_b if h <= room]
        if not donors:
            continue
        _path_b, donor = _pick_point(rng, donors, internal_prob)
        child = replace_subtree(parent_a, path_a, donor)
        if max_size is None or gt.tree_size(child) <= max_size:
            return child
    return parent_a


def subtree_mutation(rng: random.Random, tree: Tree, grammar: GPGrammar, max_depth: int,
                     max_size: Optional[int] = None, internal_prob: float = 0.9) -> Tree:
    """Replace a random subtree with a freshly grown one (depth-safe)."""
    path, _ = _pick_point(rng, list(enumerate_points(tree)), internal_prob)
    new = random_tree(rng, grammar, max_depth - len(path), "grow")
    child = replace_subtree(tree, path, new)
    return child if (max_size is None or gt.tree_size(child) <= max_size) else tree


def point_mutation(rng: random.Random, tree: Tree, grammar: GPGrammar) -> Tree:
    """Swap one node for another of the same kind (function->same arity, terminal->terminal)."""
    points = list(enumerate_points(tree))
    for _ in range(8):
        path, node = points[rng.randrange(len(points))]
        if isinstance(node, tuple):
            options = [f for f in grammar.functions if ARITY[f] == ARITY[node[0]] and f != node[0]]
            if options:
                return replace_subtree(tree, path, (options[rng.randrange(len(options))],) + node[1:])
        else:
            return replace_subtree(tree, path, random_terminal(rng, grammar))
    return tree


def constant_mutation(rng: random.Random, tree: Tree, grammar: GPGrammar, sigma_ratio: float) -> Tree:
    """Gaussian-perturb a random subset (at least one) of the constants, clipped to range."""
    consts = [(p, n) for p, n in enumerate_points(tree) if gt._is_const(n)]
    if not consts:
        return point_mutation(rng, tree, grammar)
    lo, hi = grammar.const_range
    sigma = sigma_ratio * (hi - lo)
    chosen = [c for c in consts if rng.random() < 0.5] or [consts[rng.randrange(len(consts))]]
    for path, value in chosen:
        tree = replace_subtree(tree, path, max(lo, min(hi, value + rng.gauss(0.0, sigma))))
    return tree


# ---------------------------------------------------------------------------
# Optimizer
# ---------------------------------------------------------------------------
class GeneticProgramming(BaseOptimizer):
    """Tree-based Genetic Programming minimizer (see module docstring)."""

    name = "GP"

    def __init__(self, settings: Optional[dict] = None) -> None:
        super().__init__(settings)

    # -- settings --------------------------------------------------------
    def default_settings(self) -> dict:
        merged = dict(_DEFAULT_GP_SETTINGS)
        try:
            import smart_control.config as cfg  # noqa: PLC0415
            merged.update(getattr(cfg, "GP_SETTINGS", {}) or {})
        except Exception:  # pragma: no cover - config is optional here
            pass
        return merged

    @staticmethod
    def _grammar_from_bounds(options: dict, lower: List[float], upper: List[float]) -> GPGrammar:
        base = options.get("grammar") or gt.DEFAULT_GRAMMAR
        depth = gt.depth_for_n_params(len(lower))
        if depth is None:
            raise ValueError(
                f"GeneticProgramming evolves expression trees: bounds must be "
                f"GPGrammar.parameter_bounds() (length 2*(2**(D+1)-1), e.g. 30 for depth 3), "
                f"got {len(lower)} bounds")
        grammar = replace(base, max_depth=depth, const_range=(lower[1], upper[1]))
        for i, (lo, hi) in enumerate(grammar.parameter_bounds()):
            if abs(lo - lower[i]) > 1e-9 or abs(hi - upper[i]) > 1e-9:
                raise ValueError(
                    f"bounds[{i}]=({lower[i]}, {upper[i]}) do not match the GP grammar "
                    f"(expected ({lo}, {hi})); build bounds with GPGrammar.parameter_bounds() "
                    f"using the same functions/variables")
        return grammar

    @staticmethod
    def _check_settings(o: dict) -> None:
        if int(o["population_size"]) < 2:
            raise ValueError("population_size must be >= 2")
        if int(o["max_iterations"]) < 1:
            raise ValueError("max_iterations must be >= 1")
        if int(o["tournament_size"]) < 1:
            raise ValueError("tournament_size must be >= 1")
        if not 0 <= int(o["elite_count"]) < int(o["population_size"]):
            raise ValueError("elite_count must be in [0, population_size)")
        for key in ("crossover_rate", "mutation_rate", "internal_point_prob"):
            if not 0.0 <= float(o[key]) <= 1.0:
                raise ValueError(f"{key} must be in [0, 1]")
        if o["max_evaluations"] is not None and int(o["max_evaluations"]) < 1:
            raise ValueError("max_evaluations must be >= 1")

    # -- main loop -------------------------------------------------------
    def optimize(self, objective_function, bounds, settings=None) -> OptimizationResult:
        """Evolve trees; return best encoded tree, its fitness and convergence history."""
        o = self.resolve_settings(settings)
        self._check_settings(o)
        lower, upper = self.validate_bounds(bounds)
        grammar = self._grammar_from_bounds(o, lower, upper)

        pop_size = int(o["population_size"])
        n_generations = int(o["max_iterations"])
        elite_count = int(o["elite_count"])
        tournament = int(o["tournament_size"])
        p_cross, p_mut = float(o["crossover_rate"]), float(o["mutation_rate"])
        internal_prob = float(o["internal_point_prob"])
        sigma_ratio = float(o["const_sigma_ratio"])
        max_size = None if o["max_size"] is None else int(o["max_size"])
        parsimony = float(o["parsimony_coeff"])
        budget = None if o["max_evaluations"] is None else int(o["max_evaluations"])
        target = o["target_fitness"]
        max_depth = grammar.max_depth
        if max_size is not None and max_size < 1:
            raise ValueError("max_size must be >= 1")

        rng = random.Random(o["random_seed"])  # local RNG: no global side effects
        counted = CountedObjective(objective_function)
        cache: Optional[Dict[tuple, float]] = {} if o["use_cache"] else None

        best = {"tree": None, "fitness": math.inf}

        def evaluate(tree: Tree) -> Optional[float]:
            """Fitness of a tree, or None if the evaluation budget is exhausted."""
            vector = gt.encode_tree(tree, grammar)
            key = tuple(vector)
            if cache is not None and key in cache:
                return cache[key]
            if budget is not None and counted.n_calls >= budget:
                return None
            fitness = counted(vector)
            if cache is not None:
                cache[key] = fitness
            if fitness < best["fitness"]:
                best["tree"], best["fitness"] = tree, fitness
            return fitness

        def evaluate_all(trees: Sequence[Tree]) -> Tuple[List[float], bool]:
            values, exhausted = [], False
            for tree in trees:
                value = None if exhausted else evaluate(tree)
                if value is None:
                    exhausted, value = True, math.inf
                values.append(value)
            return values, exhausted

        def adjusted(tree: Tree, fitness: float) -> float:
            return fitness + parsimony * gt.tree_size(tree) if parsimony else fitness

        # ---- generation 0: ramped half-and-half -------------------------
        population: List[Tree] = []
        seen = set()
        for i in range(pop_size):
            depth = 1 + i % max_depth
            method = "full" if (i // max_depth) % 2 == 0 else "grow"
            tree = random_tree(rng, grammar, depth, method)
            for _ in range(10):  # try to avoid duplicate individuals
                if tree not in seen and (max_size is None or gt.tree_size(tree) <= max_size):
                    break
                tree = random_tree(rng, grammar, depth, method)
            seen.add(tree)
            population.append(tree)

        fitness, exhausted = evaluate_all(population)
        if best["tree"] is None:
            raise ValueError("no objective evaluation was possible (check max_evaluations)")
        initial_best = best["fitness"]

        history: List[float] = []
        stopped_by = "max_iterations"
        if target is not None and best["fitness"] <= float(target):
            stopped_by = "target_fitness"
        elif exhausted:
            stopped_by = "max_evaluations"

        # ---- generations 1..N -------------------------------------------
        generation = 0
        while stopped_by == "max_iterations" and generation < n_generations:
            generation += 1

            def select() -> Tree:
                picks = [rng.randrange(pop_size) for _ in range(tournament)]
                winner = min(picks, key=lambda j: adjusted(population[j], fitness[j]))
                return population[winner]

            order = sorted(range(pop_size), key=lambda j: adjusted(population[j], fitness[j]))
            next_pop = [population[j] for j in order[:elite_count]]
            next_fit = [fitness[j] for j in order[:elite_count]]

            def mutate(child: Tree) -> Tree:
                r = rng.random()
                if r < 0.3:
                    return subtree_mutation(rng, child, grammar, max_depth, max_size, internal_prob)
                if r < 0.6:
                    return point_mutation(rng, child, grammar)
                return constant_mutation(rng, child, grammar, sigma_ratio)

            taken = set(next_pop)  # trees already in the next generation
            children: List[Tree] = []
            while len(next_pop) + len(children) < pop_size:
                if rng.random() < p_cross:
                    child = subtree_crossover(rng, select(), select(), max_depth, max_size, internal_prob)
                else:
                    child = select()
                if rng.random() < p_mut:
                    child = mutate(child)
                # Diversity: a clone of an existing individual is mutated until it is new.
                for _ in range(5):
                    if child not in taken:
                        break
                    child = mutate(child)
                taken.add(child)
                children.append(child)

            child_fit, exhausted = evaluate_all(children)
            population, fitness = next_pop + children, next_fit + child_fit
            history.append(best["fitness"])

            if target is not None and best["fitness"] <= float(target):
                stopped_by = "target_fitness"
            elif exhausted:
                stopped_by = "max_evaluations"

        best_tree = best["tree"]
        return self.make_result(
            gt.encode_tree(best_tree, grammar), best["fitness"], history,
            algorithm="GP",
            best_tree=best_tree,
            best_expression=gt.tree_to_string(best_tree),
            best_depth=gt.tree_depth(best_tree),
            best_size=gt.tree_size(best_tree),
            iterations_run=len(history),
            n_evaluations=counted.n_calls,
            population_size=pop_size,
            max_depth=max_depth,
            initial_best_fitness=initial_best,
            stopped_by=stopped_by,
            random_seed=o["random_seed"],
            functions=list(grammar.functions),
            variables=list(grammar.variables),
            const_range=list(grammar.const_range),
        )


# ---------------------------------------------------------------------------
# Standalone smoke test: recover 3*e + 2*ie + 0.5*de from samples
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    grammar = gt.DEFAULT_GRAMMAR
    sample_rng = random.Random(0)
    samples = [(sample_rng.uniform(-2, 2), sample_rng.uniform(-2, 2), sample_rng.uniform(-2, 2))
               for _ in range(40)]

    def objective(vector):
        fn = gt.compile_tree(gt.decode_vector(vector, grammar), grammar)
        return sum((fn(e, de, ie) - (3 * e + 2 * ie + 0.5 * de)) ** 2 for e, de, ie in samples) / len(samples)

    result = GeneticProgramming().optimize(objective, grammar.parameter_bounds(),
                                           {"population_size": 60, "max_iterations": 80})
    print("Best expression :", result.metadata["best_expression"])
    print("Best fitness    :", f"{result.best_fitness:.3e}")
    print("Evaluations     :", result.metadata["n_evaluations"])
    print("History length  :", len(result.convergence_history))
