"""Closed-loop simulation with unity feedback.

The plant is converted to discrete state space (ZOH) once, then the loop is
stepped with the controller: e = r - y, u = controller(e), x[k+1] = Ad x + Bd u.
"""
import math

import control
import numpy as np

from smart_control import config
from smart_control.types import SimulationResult

U_LIMIT = 50.0
Y_LIMIT = 1e3

_cache = {}


def _discretize(plant, dt):
    key = (id(plant), dt)
    if key in _cache and _cache[key][0] is plant:
        return _cache[key][1]

    sys_ss = control.ss(plant)
    sysd = control.c2d(sys_ss, dt, method="zoh")
    Ad = np.asarray(sysd.A, dtype=float)
    Bd = np.asarray(sysd.B, dtype=float).ravel()
    Cd = np.asarray(sysd.C, dtype=float).ravel()
    Dd = float(np.asarray(sysd.D).ravel()[0])

    _cache[key] = (plant, (Ad, Bd, Cd, Dd))
    return Ad, Bd, Cd, Dd


def simulate_closed_loop(plant, controller, simulation_config=None):
    """Simulate a unit step response of controller + plant.

    simulation_config keys (all optional): time_step, time_end,
    reference_value, u_limit. Defaults come from config.SIMULATION.
    Returns a SimulationResult. If the response blows up, the output is
    filled with NaN from that point on so the metrics treat it as unstable.
    """
    cfg = dict(config.SIMULATION)
    if simulation_config:
        cfg.update(simulation_config)

    dt = float(cfg["time_step"])
    t_end = float(cfg["time_end"])
    t_start = float(cfg.get("time_start", 0.0))
    ref = float(cfg["reference_value"])
    u_limit = float(cfg.get("u_limit", U_LIMIT))

    n_steps = int(round((t_end - t_start) / dt))
    time = t_start + np.arange(n_steps) * dt

    Ad, Bd, Cd, Dd = _discretize(plant, dt)

    controller.reset()
    x = np.zeros(Ad.shape[0])
    y_arr = np.empty(n_steps)
    u_arr = np.empty(n_steps)

    for k in range(n_steps):
        y = float(Cd @ x)
        if not math.isfinite(y) or abs(y) > Y_LIMIT:
            y_arr[k:] = np.nan
            u_arr[k:] = np.nan
            break

        e = ref - y
        u = float(controller.compute(e, dt))
        if not math.isfinite(u):
            y_arr[k:] = np.nan
            u_arr[k:] = np.nan
            break
        u = max(-u_limit, min(u_limit, u))

        y_arr[k] = y
        u_arr[k] = u
        x = Ad @ x + Bd * u

    return SimulationResult(time=time, reference=ref, output=y_arr, control_signal=u_arr)
