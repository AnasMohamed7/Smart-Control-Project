"""Shared Genetic-Programming tree representation (optimizer <-> controller).

This module is the single agreed contract between

* ``optimizers/genetic_programming.py``  (the GP *optimizer*, evolves trees), and
* ``controllers/gp_controller.py``       (the GP *controller*, evaluates a tree).

It also lets GA / DE / PSO / SA optimize the GP controller, because every tree
of depth <= ``max_depth`` has a lossless fixed-length numeric encoding
(see "Vector encoding" below).  Standard library only.

Tree format (plain nested tuples / strings / floats)
----------------------------------------------------
    terminal variable : "e", "de", "ie"            (a str)
    terminal constant : 2.5                         (a float)
    function node     : ("add", left, right)        (tuple: name, *children)
                        ("tanh", child)

Example -- a PID law ``Kp*e + Ki*ie + Kd*de``::

    ("add", ("add", ("mul", 2.0, "e"), ("mul", 0.5, "ie")), ("mul", 0.1, "de"))

Function set (``ARITY``): add, sub, mul, div (protected), tanh.
Variables (default): e (error), de (error derivative), ie (error integral).
The controller decides how de / ie are computed; the tree only sees numbers.

Protection against invalid operations
-------------------------------------
* ``div(a, b)`` returns 1.0 when ``|b| <= 1e-6`` (never raises).
* ``tanh`` is bounded; no exp / pow / sqrt / log, so no OverflowError or
  domain errors.  Float overflow can still give inf/nan, so the controller
  must pass its output through :func:`safe_output`.

Vector encoding (heap layout)
-----------------------------
A tree of depth <= D is stored in a complete binary tree of ``2**(D+1)-1``
slots (slot ``i`` has children ``2i+1`` and ``2i+2``) with two genes per slot::

    vector = [sym_0, const_0, sym_1, const_1, ...]      # length 2 * n_slots

* ``sym_i``  : symbol index + 0.5, decoded with ``floor`` (so small
  perturbations from a real-valued optimizer do not change the symbol).
  Interior slots (depth < D) index ``functions + variables + ["const"]``;
  leaf slots (depth == D) index ``variables + ["const"]`` only.
* ``const_i``: value used when slot i is the "const" terminal.
* Slots below a terminal (and the unused 2nd child of unary nodes) are
  ignored when decoding (introns).

Every vector inside ``GPGrammar.parameter_bounds()`` decodes to a valid tree,
and ``decode_vector(encode_tree(t)) == t``.  With ``max_depth=3`` (default,
30 genes) a full PID law fits exactly.
"""
from __future__ import annotations

import keyword
import math
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

Tree = Union[float, str, tuple]

# Function name -> arity.  Order here is the order used by the vector encoding.
ARITY: Dict[str, int] = {"add": 2, "sub": 2, "mul": 2, "div": 2, "tanh": 1}
CONST_TOKEN = "const"
DIV_EPS = 1e-6


# ---------------------------------------------------------------------------
# Protected primitives
# ---------------------------------------------------------------------------
def protected_div(a: float, b: float) -> float:
    """a / b, or 1.0 when the denominator is (almost) zero or NaN."""
    return a / b if abs(b) > DIV_EPS else 1.0


def safe_output(value: float, limit: float) -> float:
    """Make a tree output usable as a control signal: NaN -> 0, clip to +-limit."""
    value = float(value)
    if value != value:
        return 0.0
    return max(-limit, min(limit, value))


# ---------------------------------------------------------------------------
# Grammar
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class GPGrammar:
    """Function/terminal sets plus the depth limit and constant range."""

    functions: Tuple[str, ...] = ("add", "sub", "mul", "div", "tanh")
    variables: Tuple[str, ...] = ("e", "de", "ie")
    const_range: Tuple[float, float] = (-10.0, 10.0)
    max_depth: int = 3

    def __post_init__(self):
        if not self.functions:
            raise ValueError("grammar needs at least one function")
        for f in self.functions:
            if f not in ARITY:
                raise ValueError(f"unknown function {f!r}; allowed: {sorted(ARITY)}")
        if not self.variables:
            raise ValueError("grammar needs at least one variable")
        for v in self.variables:
            if (not isinstance(v, str) or not v.isidentifier() or keyword.iskeyword(v)
                    or v.startswith("_") or v == CONST_TOKEN or v in ARITY):
                raise ValueError(f"invalid variable name {v!r}")
        if len(set(self.variables)) != len(self.variables):
            raise ValueError("duplicate variable names")
        lo, hi = self.const_range
        if not (math.isfinite(lo) and math.isfinite(hi) and lo < hi):
            raise ValueError("const_range must be finite with lo < hi")
        if int(self.max_depth) != self.max_depth or self.max_depth < 1:
            raise ValueError("max_depth must be an integer >= 1")

    # -- symbol tables -------------------------------------------------
    @property
    def terminals(self) -> Tuple[str, ...]:
        return tuple(self.variables) + (CONST_TOKEN,)

    @property
    def interior_symbols(self) -> Tuple[str, ...]:
        return tuple(self.functions) + self.terminals

    # -- vector layout -------------------------------------------------
    @property
    def n_slots(self) -> int:
        return 2 ** (self.max_depth + 1) - 1

    @property
    def n_params(self) -> int:
        return 2 * self.n_slots

    def parameter_bounds(self) -> List[Tuple[float, float]]:
        """Bounds for the flat vector -- use this in ``GPController.parameter_bounds``."""
        lo, hi = self.const_range
        bounds: List[Tuple[float, float]] = []
        for slot in range(self.n_slots):
            n_sym = (len(self.interior_symbols) if slot_depth(slot) < self.max_depth
                     else len(self.terminals))
            bounds.append((0.0, float(n_sym)))
            bounds.append((float(lo), float(hi)))
        return bounds


DEFAULT_GRAMMAR = GPGrammar()


def slot_depth(slot: int) -> int:
    return (slot + 1).bit_length() - 1


def depth_for_n_params(n_params: int) -> Optional[int]:
    """Inverse of ``GPGrammar.n_params``: the max_depth whose vector has this length."""
    for depth in range(1, 16):
        if 2 * (2 ** (depth + 1) - 1) == n_params:
            return depth
    return None


# ---------------------------------------------------------------------------
# Basic tree helpers
# ---------------------------------------------------------------------------
def _is_const(node) -> bool:
    return isinstance(node, (int, float)) and not isinstance(node, bool)


def tree_depth(tree: Tree) -> int:
    """Number of edges on the longest root-to-leaf path (a terminal has depth 0)."""
    if isinstance(tree, tuple):
        return 1 + max(tree_depth(c) for c in tree[1:])
    return 0


def tree_size(tree: Tree) -> int:
    """Total number of nodes."""
    if isinstance(tree, tuple):
        return 1 + sum(tree_size(c) for c in tree[1:])
    return 1


def validate_tree(tree: Tree, grammar: GPGrammar = DEFAULT_GRAMMAR,
                  max_depth: Optional[int] = None) -> Tree:
    """Raise ValueError/TypeError if ``tree`` is not valid for ``grammar``; else return it."""
    limit = grammar.max_depth if max_depth is None else max_depth

    def check(node, depth):
        if depth > limit:
            raise ValueError(f"tree deeper than max_depth={limit}")
        if isinstance(node, tuple):
            if not node or node[0] not in grammar.functions:
                raise ValueError(f"unknown/disallowed function in node {node!r}")
            if len(node) - 1 != ARITY[node[0]]:
                raise ValueError(f"function {node[0]!r} needs {ARITY[node[0]]} children")
            for child in node[1:]:
                check(child, depth + 1)
        elif isinstance(node, str):
            if node not in grammar.variables:
                raise ValueError(f"unknown variable {node!r}; allowed: {grammar.variables}")
        elif _is_const(node):
            if not math.isfinite(node):
                raise ValueError("constants must be finite")
        else:
            raise TypeError(f"invalid tree node {node!r}")

    check(tree, 0)
    return tree


def uses_variable(tree: Tree, name: str) -> bool:
    """True if the variable ``name`` appears anywhere in the tree.

    Useful as a feedback-validity check: a tree without ``"e"`` ignores the
    tracking error, i.e. it is an open-loop constant/ramp, not a controller.
    """
    if isinstance(tree, tuple):
        return any(uses_variable(child, name) for child in tree[1:])
    return tree == name


def tree_to_string(tree: Tree, precision: int = 4) -> str:
    """Human-readable infix form, e.g. ``((2 * e) + (0.5 * ie))``."""
    if isinstance(tree, tuple):
        name, args = tree[0], [tree_to_string(c, precision) for c in tree[1:]]
        symbol = {"add": "+", "sub": "-", "mul": "*", "div": "/"}.get(name)
        if symbol is not None:
            return f"({args[0]} {symbol} {args[1]})"
        return f"{name}({', '.join(args)})"
    if isinstance(tree, str):
        return tree
    return f"{float(tree):.{precision}g}"


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def evaluate_tree(tree: Tree, variables: Dict[str, float]) -> float:
    """Reference (slow) interpreter. Use :func:`compile_tree` inside simulations."""
    if isinstance(tree, tuple):
        name = tree[0]
        args = [evaluate_tree(c, variables) for c in tree[1:]]
        if name == "add":
            return args[0] + args[1]
        if name == "sub":
            return args[0] - args[1]
        if name == "mul":
            return args[0] * args[1]
        if name == "div":
            return protected_div(args[0], args[1])
        if name == "tanh":
            return math.tanh(args[0])
        raise ValueError(f"unknown function {name!r}")
    if isinstance(tree, str):
        return float(variables[tree])
    return float(tree)


def compile_tree(tree: Tree, grammar: GPGrammar = DEFAULT_GRAMMAR) -> Callable[..., float]:
    """Compile a *validated* tree into ``f(*variable_values)`` (fast).

    Arguments are positional in ``grammar.variables`` order, e.g. ``f(e, de, ie)``.
    The source is generated only from validated tokens (known function names,
    grammar variables, finite floats) and run with empty ``__builtins__``, so
    nothing outside the grammar can be executed.
    """
    validate_tree(tree, grammar)

    def src(node) -> str:
        if isinstance(node, tuple):
            name, args = node[0], [src(c) for c in node[1:]]
            if name == "add":
                return f"({args[0]} + {args[1]})"
            if name == "sub":
                return f"({args[0]} - {args[1]})"
            if name == "mul":
                return f"({args[0]} * {args[1]})"
            if name == "div":
                return f"_div({args[0]}, {args[1]})"
            return f"_tanh({args[0]})"
        if isinstance(node, str):
            return node
        return repr(float(node))

    expression = src(tree)
    namespace = {"_div": protected_div, "_tanh": math.tanh, "__builtins__": {}}
    fn = eval(f"lambda {', '.join(grammar.variables)}: {expression}", namespace)  # noqa: S307
    fn.expression = expression
    return fn


# ---------------------------------------------------------------------------
# Vector encoding / decoding
# ---------------------------------------------------------------------------
def _to_index(gene: float, n: int) -> int:
    gene = float(gene)
    if gene != gene or gene <= 0.0:   # NaN and anything below the first interval
        return 0
    if gene >= n:                     # includes +inf
        return n - 1
    return int(gene)                  # floor, since gene > 0


def _clean_const(value: float, lo: float, hi: float) -> float:
    value = float(value)
    if value != value:
        value = 0.0
    return max(lo, min(hi, value))


def encode_tree(tree: Tree, grammar: GPGrammar = DEFAULT_GRAMMAR) -> List[float]:
    """Tree -> flat vector of length ``grammar.n_params`` (constants clipped to range)."""
    validate_tree(tree, grammar)
    lo, hi = grammar.const_range
    n_func = len(grammar.functions)
    filler_const = _clean_const(0.0, lo, hi)
    genes: List[float] = []
    for slot in range(grammar.n_slots):  # filler: first variable, valid for any slot level
        genes += [(0.5 if slot_depth(slot) == grammar.max_depth else n_func + 0.5), filler_const]

    def put(node, slot):
        leaf_level = slot_depth(slot) == grammar.max_depth
        offset = 0 if leaf_level else n_func
        if isinstance(node, tuple):
            genes[2 * slot] = grammar.functions.index(node[0]) + 0.5
            for k, child in enumerate(node[1:]):
                put(child, 2 * slot + 1 + k)
        elif isinstance(node, str):
            genes[2 * slot] = offset + grammar.variables.index(node) + 0.5
        else:
            genes[2 * slot] = offset + len(grammar.variables) + 0.5
            genes[2 * slot + 1] = _clean_const(node, lo, hi)

    put(tree, 0)
    return genes


def decode_vector(vector: Sequence[float], grammar: GPGrammar = DEFAULT_GRAMMAR) -> Tree:
    """Flat vector -> tree. Total function: any finite/NaN/out-of-range vector decodes."""
    genes = [float(g) for g in vector]
    if len(genes) != grammar.n_params:
        raise ValueError(f"expected {grammar.n_params} genes, got {len(genes)}")
    lo, hi = grammar.const_range

    def build(slot):
        leaf_level = slot_depth(slot) == grammar.max_depth
        symbols = grammar.terminals if leaf_level else grammar.interior_symbols
        symbol = symbols[_to_index(genes[2 * slot], len(symbols))]
        if symbol in ARITY:
            return (symbol,) + tuple(build(2 * slot + 1 + k) for k in range(ARITY[symbol]))
        if symbol == CONST_TOKEN:
            return _clean_const(genes[2 * slot + 1], lo, hi)
        return symbol

    return build(0)
