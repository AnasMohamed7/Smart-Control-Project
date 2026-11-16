"""
Plant Transfer Function Models
==============================

This module defines three continuous-time plant models used for controller
design and optimisation experiments in the Smart Control project.

Plant Summary
-------------

second_order
    G(s) = 1 / (s² + 1.2s + 1)

    Classic underdamped second-order system.
    - Natural frequency  : ωn = 1 rad/s
    - Damping ratio      : ζ  = 0.6
    - Poles              : s = -0.6 ± j0.8   (stable, complex conjugate)
    - DC gain            : G(0) = 1

third_order
    G(s) = 10 / (s³ + 6s² + 11s + 6)

    Three distinct real poles; factors as (s+1)(s+2)(s+3).
    - Poles              : s = -1, s = -2, s = -3  (all stable, real)
    - DC gain            : G(0) = 10/6 ≈ 1.667

fourth_order
    G(s) = 1 / (s⁴ + 2.6s³ + 3.44s² + 2.4s + 0.64)

    Two second-order sections in series:
        (s² + 0.8s + 0.64)(s² + 1.8s + 1)
    - Poles              : s = -0.4 ± j0.72  (ζ ≈ 0.49, ωn = 0.8 rad/s)
                           s = -0.9 ± j0.436 (ζ ≈ 0.90, ωn = 1.0 rad/s)
    - DC gain            : G(0) = 1/0.64 = 1.5625

All poles have strictly negative real parts → all plants are BIBO stable.
"""

import control


# ---------------------------------------------------------------------------
# Individual plant constructors
# ---------------------------------------------------------------------------

def get_second_order_plant() -> control.TransferFunction:
    """Return the second-order plant transfer function.

    G(s) = 1 / (s² + 1.2s + 1)

    Mathematical properties
    -----------------------
    - Damping ratio ζ = 0.6  (underdamped, 0 < ζ < 1)
    - Natural frequency ωn = 1 rad/s
    - Poles at s = -0.6 ± j0.8
    - Steady-state gain = 1

    Returns
    -------
    control.TransferFunction
        Continuous-time transfer function object.
    """
    num = [1]
    den = [1, 1.2, 1]
    return control.TransferFunction(num, den)


def get_third_order_plant() -> control.TransferFunction:
    """Return the third-order plant transfer function.

    G(s) = 10 / (s³ + 6s² + 11s + 6)
         = 10 / [(s+1)(s+2)(s+3)]

    Mathematical properties
    -----------------------
    - Three distinct, real, stable poles: s = -1, -2, -3
    - Steady-state gain = 10/6 ≈ 1.667
    - No oscillatory behaviour (all poles are real)

    Returns
    -------
    control.TransferFunction
        Continuous-time transfer function object.
    """
    num = [10]
    den = [1, 6, 11, 6]
    return control.TransferFunction(num, den)


def get_fourth_order_plant() -> control.TransferFunction:
    """Return the fourth-order plant transfer function.

    G(s) = 1 / (s⁴ + 2.6s³ + 3.44s² + 2.4s + 0.64)
         = 1 / [(s² + 0.8s + 0.64)(s² + 1.8s + 1)]

    Mathematical properties
    -----------------------
    - First pair  : s = -0.4 ± j0.72   (ζ ≈ 0.49, ωn = 0.8 rad/s)
    - Second pair : s = -0.9 ± j0.436  (ζ = 0.90, ωn = 1.0 rad/s)
    - Steady-state gain = 1/0.64 ≈ 1.5625
    - Mildly oscillatory due to the first complex pair

    Returns
    -------
    control.TransferFunction
        Continuous-time transfer function object.
    """
    num = [1]
    den = [1, 2.6, 3.44, 2.4, 0.64]
    return control.TransferFunction(num, den)


# ---------------------------------------------------------------------------
# Convenience aggregator
# ---------------------------------------------------------------------------

def get_plant_models() -> dict:
    """Return all plant models as a dictionary.

    Returns
    -------
    dict
        Keys   : "second_order", "third_order", "fourth_order"
        Values : corresponding ``control.TransferFunction`` objects

    Example
    -------
    >>> from smart_control.plants.models import get_plant_models
    >>> plants = get_plant_models()
    >>> plants["second_order"]
    TransferFunction(array([1]), array([1. , 1.2, 1. ]))
    """
    return {
        "second_order": get_second_order_plant(),
        "third_order":  get_third_order_plant(),
        "fourth_order": get_fourth_order_plant(),
    }
