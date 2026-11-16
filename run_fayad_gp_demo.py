"""GP optimizer demo: evolve a GP-controller expression for a project plant.

Vertical-slice check using only numpy, matplotlib and ``control``:

    plant (plants.models) + GP controller (tree from gp_tree) + objective
    -> GeneticProgramming.optimize   and   ParticleSwarmOptimization.optimize
       (same objective, same flat parameter vector -> proves the encoding lets
        the other optimizers drive the GP controller too)

The mini simulator / metrics / controller below are *stand-ins* until the real
plants/simulation.py, evaluation/* and controllers/gp_controller.py land.
Run:  python run_fayad_gp_demo.py [second_order|third_order|fourth_order]
"""
import sys
import time

sys.path.insert(0, ".")

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import control

import smart_control.config as cfg
from smart_control import gp_tree as gt
from smart_control.controllers.base import BaseController
from smart_control.optimizers.genetic_programming import GeneticProgramming
from smart_control.optimizers.particle_swarm import ParticleSwarmOptimization
from smart_control.plants.models import get_plant_models

PLANT_NAME = sys.argv[1] if len(sys.argv) > 1 else "second_order"
POP, ITERS = 30, 40
U_LIMIT, IE_LIMIT = 50.0, 50.0
REQUIRE_FEEDBACK = True  # reject trees that never look at the error e
GRAMMAR = gt.DEFAULT_GRAMMAR


class DemoGPController(BaseController):
    """Reference for Khaira: how a controller consumes gt trees / vectors."""

    def __init__(self, grammar=GRAMMAR):
        self.grammar = grammar
        self.set_expression(("mul", 1.0, "e"))
        self.reset()

    def set_expression(self, tree):
        self.tree = gt.validate_tree(tree, self.grammar)
        self._fn = gt.compile_tree(tree, self.grammar)

    def reset(self):  # clears dynamic state only -- never the parameters
        self._prev_e, self._ie = None, 0.0

    def set_parameters(self, parameters):
        self.set_expression(gt.decode_vector(parameters, self.grammar))

    def parameter_bounds(self):
        return self.grammar.parameter_bounds()

    def compute(self, error, dt, **signals):
        de = 0.0 if self._prev_e is None else (error - self._prev_e) / dt
        self._ie = max(-IE_LIMIT, min(IE_LIMIT, self._ie + error * dt))
        self._prev_e = error
        return gt.safe_output(self._fn(error, de, self._ie), U_LIMIT)


def discretize(plant, dt):
    sysd = control.sample_system(control.tf2ss(plant), dt, method="zoh")
    return (np.asarray(sysd.A), np.asarray(sysd.B).ravel(), np.asarray(sysd.C).ravel())


def simulate(Ad, Bd, Cd, controller, n_steps, dt, ref):
    controller.reset()
    x = np.zeros(Ad.shape[0])
    y_arr, u_arr = np.empty(n_steps), np.empty(n_steps)
    for k in range(n_steps):
        y = float(Cd @ x)
        if not np.isfinite(y) or abs(y) > 1e3:
            return None, None
        u = controller.compute(ref - y, dt)
        x = Ad @ x + Bd * u
        y_arr[k], u_arr[k] = y, u
    return y_arr, u_arr


def metrics(t, y, ref=cfg.REFERENCE, tol=cfg.SETTLING_TOLERANCE):
    dt = t[1] - t[0]
    out = np.where(np.abs(y - ref) > tol * ref)[0]
    i10, i90 = np.where(y >= 0.1 * ref)[0], np.where(y >= 0.9 * ref)[0]
    return {
        "iae": float(np.sum(np.abs(ref - y)) * dt),
        "overshoot": float(max(0.0, (y.max() - ref) / ref * 100)),
        "settling": float(t[out[-1]]) if len(out) else 0.0,
        "rise": float(t[i90[0]] - t[i10[0]]) if len(i10) and len(i90) else float("nan"),
        "sse": float(abs(ref - y[-max(1, len(y) // 20):].mean())),
    }


def main():
    dt, t_end, ref = cfg.DT, cfg.T_END, cfg.REFERENCE
    n_steps = int(t_end / dt)
    t = np.linspace(0, t_end, n_steps)
    Ad, Bd, Cd = discretize(get_plant_models()[PLANT_NAME], dt)
    ctrl, w = DemoGPController(), cfg.OBJECTIVE_WEIGHTS

    def objective(vector):
        ctrl.set_parameters(vector)
        if REQUIRE_FEEDBACK and not gt.uses_variable(ctrl.tree, "e"):
            return 1e3  # open-loop constant: not a controller (see design_decisions.md)
        y, _ = simulate(Ad, Bd, Cd, ctrl, n_steps, dt, ref)
        if y is None:
            return 1e4
        m = metrics(t, y)
        return w["iae"] * m["iae"] + w["overshoot"] * m["overshoot"] + w["settling_time"] * m["settling"]

    bounds = ctrl.parameter_bounds()
    t0 = time.time()
    pid_like = ("add", ("add", ("mul", 4.0, "e"), ("mul", 1.0, "ie")), ("mul", 2.0, "de"))
    objective(gt.encode_tree(pid_like, GRAMMAR))
    print(f"cost per candidate: {(time.time() - t0) * 1000:.1f} ms | plant={PLANT_NAME} "
          f"| dims={len(bounds)} | pop={POP} iters={ITERS}")

    runs = {}
    for name, opt in (("GP", GeneticProgramming()), ("PSO (same GP-controller encoding)", ParticleSwarmOptimization())):
        t0 = time.time()
        res = opt.optimize(objective, bounds, {"population_size": POP, "max_iterations": ITERS})
        runs[name] = (res, time.time() - t0)
        print(f"{name:36s} best={res.best_fitness:9.4f}  time={runs[name][1]:5.1f}s")

    gp_res = runs["GP"][0]
    print("\nGP best expression :", gp_res.metadata["best_expression"])
    ctrl.set_parameters(gp_res.best_solution)
    y, u = simulate(Ad, Bd, Cd, ctrl, n_steps, dt, ref)
    print("GP metrics         :", {k: round(v, 4) for k, v in metrics(t, y).items()})
    print("evaluations (GP)   :", gp_res.metadata["n_evaluations"], "of", POP * (ITERS + 1))

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(t, y, label="GP controller"); ax[0].axhline(ref, color="k", ls="--", label="reference")
    ax[0].set(title=f"Step response - {PLANT_NAME}", xlabel="Time (s)", ylabel="Output"); ax[0].legend(); ax[0].grid(alpha=.3)
    for name, (res, _) in runs.items():
        ax[1].semilogy(res.convergence_history, label=name)
    ax[1].set(title="Convergence", xlabel="Iteration", ylabel="Best fitness"); ax[1].legend(); ax[1].grid(alpha=.3, which="both")
    fig.tight_layout(); fig.savefig(f"gp_demo_{PLANT_NAME}.png", dpi=120)
    print(f"saved gp_demo_{PLANT_NAME}.png")


if __name__ == "__main__":
    main()
