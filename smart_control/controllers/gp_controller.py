"""Genetic Programming Controller – Khaira's implementation.

This is the *controller-side* wrapper for an evolved GP expression/tree.
The GP *optimizer* (Fayad's genetic_programming.py) evolves trees; this
class evaluates one fixed tree to produce a control signal u(t).

Both components share smart_control/gp_tree.py as the single agreed
contract between them (tree format, grammar, vector encoding).

Design decisions (from docs/design_decisions.md §7, Fayad/team agreement)
--------------------------------------------------------------------------
Tree format
  Nested tuples / strings / floats – defined in gp_tree.py.
  Example PID law:
    ("add", ("add", ("mul", Kp, "e"), ("mul", Ki, "ie")), ("mul", Kd, "de"))

Variables fed to the tree each step
  e   – tracking error     e = r − y
  de  – error derivative   (e[k] − e[k-1]) / dt,  0 on first step
  ie  – error integral     Σ e·dt,  clamped to ±IE_LIMIT

Controller-side protection
  NaN/inf output is mapped to 0 and clipped to ±U_LIMIT by safe_output()
  (gp_tree.safe_output is the agreed function; we use the same U_LIMIT = 50
  that the simulator uses so the two clamps are consistent).

set_parameters() / parameter_bounds()
  The GP controller supports *two* usage modes:

  Mode A – direct tree assignment (used by the GP optimizer):
    controller.set_tree(tree)      # stores a validated tree directly
    controller.set_parameters() raises an error in this mode.

  Mode B – numeric vector (used by GA/DE/PSO/SA to optimise the GP tree):
    controller.set_parameters(vector)  # decodes via GPGrammar → tree
    controller.parameter_bounds()      # returns GPGrammar.parameter_bounds()

  Calling set_parameters() with a numeric vector automatically switches
  to Mode B, overwriting any tree previously set by set_tree().
  Calling set_tree() switches back to Mode A.
"""
from __future__ import annotations

import math
from typing import Optional, Sequence

from smart_control.controllers.base import BaseController
from smart_control.gp_tree import (
    DEFAULT_GRAMMAR,
    GPGrammar,
    Tree,
    compile_tree,
    decode_vector,
    depth_for_n_params,
    safe_output,
    validate_tree,
)

# Control output limit – must match U_LIMIT in plants/simulation.py
_U_LIMIT     = 50.0
# Integral clamp – prevent integral windup
_IE_LIMIT    = 50.0


class GPController(BaseController):
    """Evaluates a GP expression tree to produce a scalar control signal.

    Parameters
    ----------
    expression : Tree, optional
        A validated GP tree.  If provided the controller starts in Mode A
        (direct tree assignment).  Set to None to start in Mode B (numeric
        vector, requires set_parameters() before the first compute() call).
    grammar : GPGrammar, optional
        Grammar used for tree validation and vector encoding/decoding.
        Defaults to DEFAULT_GRAMMAR (max_depth=3, functions=add/sub/mul/div/tanh,
        variables=e/de/ie).
    """

    def __init__(self, expression: Optional[Tree] = None,
                 grammar: Optional[GPGrammar] = None):
        self._grammar = grammar if grammar is not None else DEFAULT_GRAMMAR
        self._tree: Optional[Tree] = None
        self._fn = None          # compiled callable f(e, de, ie) -> float

        if expression is not None:
            self.set_tree(expression)

        # Dynamic state
        self.reset()

    # ------------------------------------------------------------------
    # Tree assignment (Mode A – direct tree, used by GP optimizer)
    # ------------------------------------------------------------------

    def set_tree(self, tree: Tree) -> None:
        """Store and compile a GP tree directly.

        Parameters
        ----------
        tree : Tree
            A valid GP tree (nested tuple / str / float).  It is validated
            against the grammar before being stored.

        Raises
        ------
        ValueError
            If the tree violates the grammar.
        """
        validate_tree(tree, self._grammar)
        self._tree = tree
        # compile_tree returns a fast callable: f(e, de, ie) -> float
        self._fn   = compile_tree(tree, self._grammar)

    # ------------------------------------------------------------------
    # BaseController interface
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Reset dynamic state only (integral and previous error).

        Per §7.3 of design_decisions.md: reset() must clear dynamic state
        only, never the tree/parameters.  The compiled function and tree
        survive a reset so the simulator can call reset() between runs
        without losing the optimised expression.
        """
        self._ie      = 0.0   # error integral (clamped to ±IE_LIMIT)
        self._prev_e  = 0.0   # previous error (for de/dt)
        self._has_prev = False # True after first compute()

    def set_parameters(self, parameters: Sequence[float]) -> None:
        """Decode a numeric vector into a GP tree (Mode B).

        Called by GA / DE / PSO / SA.  The vector length must match
        the grammar (e.g. 30 genes for max_depth=3).  The tree decoded
        from the vector is immediately compiled into a fast callable.

        Parameters
        ----------
        parameters : sequence of float
            Flat numeric vector, length == grammar.n_params.
            Any vector within grammar.parameter_bounds() decodes to a valid
            tree; out-of-range values are handled gracefully by decode_vector.

        Raises
        ------
        ValueError
            If the vector length does not match the grammar.
        """
        # Allow decode_vector to validate the length and report a clear error
        tree = decode_vector(parameters, self._grammar)
        # Store and compile (validate_tree is called inside compile_tree)
        self._tree = tree
        self._fn   = compile_tree(tree, self._grammar)

    def compute(self, error: float, dt: float, **signals) -> float:
        """Evaluate the GP tree and return the control signal.

        The tree receives three inputs at each step:
          e  – tracking error (provided directly)
          de – error rate  (computed internally from previous error)
          ie – error integral (accumulated internally, clamped)

        Parameters
        ----------
        error : float
            Tracking error  e = r − y  at the current time step.
        dt : float
            Time step in seconds.  Must be finite and > 0.
        **signals : dict
            Optional pre-computed signals; recognised keys:
              ``derivative`` – overrides the internal de/dt calculation.
              ``integral``   – overrides the internal integral accumulation.

        Returns
        -------
        float
            Control signal u, passed through safe_output (NaN→0, clipped to
            ±U_LIMIT).

        Raises
        ------
        RuntimeError
            If compute() is called before a tree has been assigned.
        ValueError
            If dt is not a finite positive number.
        """
        if self._fn is None:
            raise RuntimeError(
                "GPController has no expression yet. "
                "Call set_tree() or set_parameters() before compute()."
            )

        error = float(error)
        dt    = float(dt)

        if not math.isfinite(error):
            # Treat non-finite error as zero to keep the simulation running
            error = 0.0

        if not math.isfinite(dt) or dt <= 0.0:
            raise ValueError("dt must be a finite positive number")

        # --- Error derivative de ---
        if "derivative" in signals:
            de = float(signals["derivative"])
        elif not self._has_prev:
            # First step: no previous sample, so de/dt = 0
            de = 0.0
        else:
            de = (error - self._prev_e) / dt

        if not math.isfinite(de):
            de = 0.0

        # --- Error integral ie (with clamp) ---
        if "integral" in signals:
            self._ie = float(signals["integral"])
        else:
            self._ie += error * dt
            # Clamp to prevent integral windup
            self._ie = max(-_IE_LIMIT, min(_IE_LIMIT, self._ie))

        # Store state for next step
        self._prev_e   = error
        self._has_prev = True

        # --- Evaluate tree ---
        # self._fn is f(e, de, ie) -> float (variables in grammar order)
        raw_output = self._fn(error, de, self._ie)

        # --- Safe output: NaN → 0, clip to ±U_LIMIT ---
        return safe_output(raw_output, _U_LIMIT)

    def parameter_bounds(self):
        """Return numeric bounds for the GP vector encoding (Mode B).

        Returns the same bounds as GPGrammar.parameter_bounds() so that
        any real-valued optimizer (GA, DE, PSO, SA) can search the space.
        Every vector within these bounds decodes to a valid tree of depth
        <= grammar.max_depth.

        Returns
        -------
        list of (lower, upper) pairs, length == grammar.n_params
        """
        return self._grammar.parameter_bounds()

    # ------------------------------------------------------------------
    # Convenience / introspection
    # ------------------------------------------------------------------

    @property
    def tree(self) -> Optional[Tree]:
        """The current GP tree, or None if not yet set."""
        return self._tree

    @property
    def grammar(self) -> GPGrammar:
        """The grammar used by this controller."""
        return self._grammar

    def __repr__(self) -> str:
        from smart_control.gp_tree import tree_to_string
        if self._tree is None:
            expr = "<no tree>"
        else:
            expr = tree_to_string(self._tree)
        return f"GPController(grammar={self._grammar!r}, expr={expr!r})"
