import time

import control
import numpy as np

from smart_control.plants.models import get_plant_models
from smart_control.plants.simulation import simulate_closed_loop
from smart_control.optimizers.differential_evolution import DifferentialEvolution
from smart_control.evaluation.objective import build_objective
from smart_control.evaluation.metrics import calculate_all_metrics


class SimplePID:
    def __init__(self, kp=1.0, ki=0.0, kd=0.0):
        self.kp, self.ki, self.kd = kp, ki, kd
        self.reset()

    def reset(self):
        self.integral = 0.0
        self.prev_error = 0.0

    def set_parameters(self, p):
        self.kp, self.ki, self.kd = [float(v) for v in p]

    def parameter_bounds(self):
        return [(0.0, 20.0), (0.0, 20.0), (0.0, 10.0)]

    def compute(self, error, dt, **signals):
        self.integral += error * dt
        d = (error - self.prev_error) / dt
        self.prev_error = error
        return self.kp * error + self.ki * self.integral + self.kd * d


def test_plants_are_stable():
    for name, g in get_plant_models().items():
        assert np.all(np.real(control.poles(g)) < 0), name


def test_p_control_matches_control_library():
    cfg = {"time_step": 0.001, "time_end": 10.0}
    for name, g in get_plant_models().items():
        res = simulate_closed_loop(g, SimplePID(kp=2.0), cfg)
        closed = control.feedback(2.0 * g, 1)
        _, y_ref = control.step_response(closed, res.time)
        assert np.max(np.abs(res.output - y_ref)) < 0.01, name


def test_simulation_shapes():
    g = get_plant_models()["second_order"]
    res = simulate_closed_loop(g, SimplePID(kp=1.0))
    assert len(res.time) == len(res.output) == len(res.control_signal)
    assert res.output[0] == 0.0


def test_unstable_loop_gives_nan():
    g = get_plant_models()["fourth_order"]
    res = simulate_closed_loop(g, SimplePID(kp=500.0, ki=0, kd=0), {"u_limit": 1e9})
    assert np.isnan(res.output[-1])


def test_de_sphere():
    r = DifferentialEvolution().optimize(lambda x: float(np.sum(np.square(x))), [(-5, 5)] * 4,
                                         {"max_iterations": 150})
    assert r.best_fitness < 1e-6
    assert len(r.convergence_history) == 150
    h = r.convergence_history
    assert all(a >= b for a, b in zip(h, h[1:]))


def test_de_is_reproducible_and_in_bounds():
    f = lambda x: float(np.sum(np.square(np.asarray(x) - 1.0)))
    b = [(0, 3), (0, 3)]
    r1 = DifferentialEvolution().optimize(f, b, {"max_iterations": 30})
    r2 = DifferentialEvolution().optimize(f, b, {"max_iterations": 30})
    assert r1.best_solution == r2.best_solution
    assert all(lo <= v <= hi for v, (lo, hi) in zip(r1.best_solution, b))


def test_de_handles_nan_objective():
    def f(x):
        return float("nan") if x[0] > 2.5 else float(x[0] ** 2)
    r = DifferentialEvolution().optimize(f, [(0, 5)], {"max_iterations": 30})
    assert np.isfinite(r.best_fitness)


def test_de_tunes_pid_on_each_plant():
    for name, g in get_plant_models().items():
        pid = SimplePID()
        obj = build_objective(pid, g)
        t0 = time.time()
        r = DifferentialEvolution().optimize(obj, pid.parameter_bounds(),
                                             {"population_size": 20, "max_iterations": 30})
        pid.set_parameters(r.best_solution)
        res = simulate_closed_loop(g, pid)
        m = calculate_all_metrics(res)
        print(name, [round(v, 3) for v in r.best_solution], m.to_dict(), round(time.time() - t0, 1), "s")
        assert r.best_fitness < 1e4
        assert m.steady_state_error < 0.05
