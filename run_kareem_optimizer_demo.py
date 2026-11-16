"""
PSO + Neural Network Controller Demo
=====================================
Choose which plant to run by commenting/uncommenting below.
Includes:
- Divergence penalty (fast handling of unstable parameters)
- Live progress reporting (no terminal freezing)
- Automatic plot saving without blocking GUI hang
"""

import sys
import time
sys.path.insert(0, '.')

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-blocking headless backend: saves PNG without freezing terminal
import matplotlib.pyplot as plt
from scipy import signal as sig

import smart_control.config as cfg
from smart_control.plants.models import get_plant_models
from smart_control.controllers.neural_network import NeuralNetworkController
from smart_control.types import OptimizationResult

# =============================================================================
#  SELECT YOUR PLANT  (uncomment ONE, comment the others)
# =============================================================================

PLANT_NAME = "second_order"
#PLANT_NAME = "third_order"
#PLANT_NAME = "fourth_order"

# =============================================================================
#  TUNING SETTINGS
# =============================================================================
POPULATION_SIZE = 25    # Swarm particles
MAX_ITERATIONS  = 50    # Optimization iterations (50 gives fast & high quality convergence)
# =============================================================================


def get_discrete_ss(plant_tf, dt):
    """Convert transfer function to discrete-time state-space matrices."""
    A, B, C, D = sig.tf2ss(plant_tf.num[0][0], plant_tf.den[0][0])
    Ad, Bd, Cd, Dd, _ = sig.cont2discrete((A, B, C, D), dt, method='zoh')
    return Ad, Bd.flatten(), Cd.flatten(), float(Dd.flat[0])


def unpack_nn(params):
    """Split flat 21-parameter vector into W1(5x2), b1(5), W2(1x5), b2(1)."""
    p  = np.asarray(params, dtype=float)
    W1 = p[0:10].reshape(5, 2)
    b1 = p[10:15]
    W2 = p[15:20].reshape(1, 5)
    b2 = p[20:21]
    return W1, b1, W2, b2


def simulate_fast(Ad, Bd, Cd, n_steps, W1, b1, W2, b2, ref=1.0, u_clip=50.0):
    """
    Fast closed-loop simulation with early-exit on divergence.
    """
    n_states = Ad.shape[0]
    x        = np.zeros(n_states)
    prev_err = 0.0

    y_arr = np.empty(n_steps)
    u_arr = np.empty(n_steps)

    for k in range(n_steps):
        y = float(Cd @ x)

        # Early exit if system goes unstable to prevent NaN/Inf and save CPU time
        if abs(y) > 50.0 or np.isnan(y):
            # Fill remaining with divergence indicator
            y_arr[k:] = 100.0
            u_arr[k:] = 0.0
            return y_arr, u_arr, True  # True = diverged

        e = ref - y

        # NN forward pass
        inp = np.array([e, prev_err])
        h   = np.tanh(W1 @ inp + b1)
        u   = float((W2 @ h + b2)[0])
        u   = max(-u_clip, min(u_clip, u))

        x        = Ad @ x + Bd * u
        prev_err = e
        y_arr[k] = y
        u_arr[k] = u

    return y_arr, u_arr, False


def build_objective(Ad, Bd, Cd, n_steps, t_arr, ref=cfg.REFERENCE):
    tol = cfg.SETTLING_TOLERANCE * ref
    w   = cfg.OBJECTIVE_WEIGHTS

    def objective(params):
        W1, b1, W2, b2 = unpack_nn(params)
        y, u, diverged = simulate_fast(Ad, Bd, Cd, n_steps, W1, b1, W2, b2, ref)

        # Penalize unstable responses strongly
        if diverged or np.isnan(y[-1]):
            return 1e4

        iae       = float(np.sum(np.abs(ref - y)) * cfg.DT)
        overshoot = float(max(0.0, (np.max(y) - ref) / ref * 100.0))

        out_idx   = np.where(np.abs(y - ref) > tol)[0]
        settling  = float(t_arr[out_idx[-1]]) if len(out_idx) > 0 else 0.0

        return float(w['iae'] * iae + w['overshoot'] * overshoot + w['settling_time'] * settling)

    return objective


def run_pso_with_progress(objective_function, bounds, pop_size=POPULATION_SIZE, max_iter=MAX_ITERATIONS):
    """
    PSO with live command-line progress updates so terminal never looks stuck.
    """
    np.random.seed(cfg.RANDOM_SEED)
    n_particles = pop_size
    n_dims      = len(bounds)

    lower = np.array([b[0] for b in bounds], dtype=float)
    upper = np.array([b[1] for b in bounds], dtype=float)
    v_max = 0.2 * (upper - lower)

    positions  = np.random.uniform(lower, upper, (n_particles, n_dims))
    velocities = np.random.uniform(-v_max, v_max, (n_particles, n_dims))

    fitness = np.array([objective_function(positions[i]) for i in range(n_particles)])
    pbest_positions = positions.copy()
    pbest_fitness   = fitness.copy()

    best_idx       = int(np.argmin(pbest_fitness))
    gbest_position = pbest_positions[best_idx].copy()
    gbest_fitness  = float(pbest_fitness[best_idx])

    convergence = [gbest_fitness]
    w_max = 0.7
    c1, c2 = 1.5, 1.5

    for iteration in range(1, max_iter + 1):
        w = w_max - (w_max - 0.4) * (iteration / max_iter)
        r1 = np.random.uniform(0.0, 1.0, (n_particles, n_dims))
        r2 = np.random.uniform(0.0, 1.0, (n_particles, n_dims))

        velocities = w * velocities + c1 * r1 * (pbest_positions - positions) + c2 * r2 * (gbest_position - positions)
        velocities = np.clip(velocities, -v_max, v_max)
        positions  = np.clip(positions + velocities, lower, upper)

        fitness = np.array([objective_function(positions[i]) for i in range(n_particles)])

        improved = fitness < pbest_fitness
        pbest_positions[improved] = positions[improved]
        pbest_fitness[improved]   = fitness[improved]

        best_idx = int(np.argmin(pbest_fitness))
        if pbest_fitness[best_idx] < gbest_fitness:
            gbest_fitness  = float(pbest_fitness[best_idx])
            gbest_position = pbest_positions[best_idx].copy()

        convergence.append(gbest_fitness)

        # Progress reporting every 10 iterations or on last iteration
        if iteration % 10 == 0 or iteration == max_iter:
            print(f"  [Progress] Iteration {iteration:3d}/{max_iter} | Best Cost: {gbest_fitness:.4f}")

    return OptimizationResult(
        best_solution=gbest_position.tolist(),
        best_fitness=float(gbest_fitness),
        convergence_history=convergence,
        metadata={"iterations": max_iter, "particles": n_particles}
    )


def compute_metrics(t, y, ref=cfg.REFERENCE, tol=cfg.SETTLING_TOLERANCE):
    dt = t[1] - t[0]
    iae = np.sum(np.abs(ref - y)) * dt
    overshoot = max(0.0, (np.max(y) - ref) / ref * 100.0)

    idx_10 = np.where(y >= 0.10 * ref)[0]
    idx_90 = np.where(y >= 0.90 * ref)[0]
    rise_time = (t[idx_90[0]] - t[idx_10[0]]) if (len(idx_10) and len(idx_90)) else float('nan')

    out_idx = np.where(np.abs(y - ref) > tol * ref)[0]
    settling_time = t[out_idx[-1]] if len(out_idx) > 0 else 0.0

    ss_error = abs(ref - np.mean(y[-max(1, int(0.05 * len(y))):]))

    return dict(iae=iae, overshoot=overshoot,
                rise_time=rise_time, settling_time=settling_time,
                ss_error=ss_error)


def main():
    plants = get_plant_models()
    if PLANT_NAME not in plants:
        print(f"Unknown plant: {PLANT_NAME}")
        return

    plant = plants[PLANT_NAME]
    dt    = cfg.DT
    t_end = cfg.T_END
    ref   = cfg.REFERENCE

    n_steps = int(t_end / dt)
    t_arr   = np.linspace(0, t_end, n_steps)

    Ad, Bd, Cd, _ = get_discrete_ss(plant, dt)

    print("=" * 60)
    print(f"  PLANT TARGET : {PLANT_NAME.upper()}")
    print(f"  Swarm Size   : {POPULATION_SIZE} particles")
    print(f"  Iterations   : {MAX_ITERATIONS}")
    print(f"  Sim Duration : {t_end}s (dt = {dt}s, {n_steps} steps)")
    print("=" * 60)
    print("Starting PSO Optimization...\n")

    t_start = time.time()
    nn      = NeuralNetworkController()
    bounds  = nn.parameter_bounds()
    obj     = build_objective(Ad, Bd, Cd, n_steps, t_arr, ref)

    result  = run_pso_with_progress(obj, bounds, pop_size=POPULATION_SIZE, max_iter=MAX_ITERATIONS)
    elapsed = time.time() - t_start

    print(f"\nOptimization finished in {elapsed:.1f} seconds!")

    # Final detailed simulation with best parameters
    W1, b1, W2, b2 = unpack_nn(result.best_solution)
    y, u, _ = simulate_fast(Ad, Bd, Cd, n_steps, W1, b1, W2, b2, ref)
    m = compute_metrics(t_arr, y)

    print("-" * 60)
    print(f"  Best Fitness       : {result.best_fitness:.4f}")
    print(f"  IAE (Tracking Err) : {m['iae']:.4f}")
    print(f"  Overshoot          : {m['overshoot']:.2f} %")
    print(f"  Rise Time          : {m['rise_time']:.3f} s")
    print(f"  Settling Time      : {m['settling_time']:.3f} s")
    print(f"  Steady-State Error : {m['ss_error']:.5f}")
    print("-" * 60)

    # Generate and save plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle(f"PSO + NN Controller  |  Plant: {PLANT_NAME}", fontsize=13, fontweight='bold')

    # 1. Step Response
    ax = axes[0]
    ax.plot(t_arr, y, color='steelblue', linewidth=2, label='Output y(t)')
    ax.axhline(ref, color='k', linewidth=1.2, linestyle='--', label='Reference (1.0)')
    ax.axhline(ref * (1 + cfg.SETTLING_TOLERANCE), color='gray', linewidth=0.8, linestyle=':', label='±2% band')
    ax.axhline(ref * (1 - cfg.SETTLING_TOLERANCE), color='gray', linewidth=0.8, linestyle=':')
    ax.set_title("Closed-Loop Step Response")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Output y(t)")
    ax.set_xlim(0, t_end)
    ax.legend(fontsize=8, loc='lower right')
    ax.grid(True, alpha=0.3)
    txt = (f"IAE = {m['iae']:.3f}\n"
           f"OS  = {m['overshoot']:.2f}%\n"
           f"Tr  = {m['rise_time']:.3f}s\n"
           f"Ts  = {m['settling_time']:.3f}s\n"
           f"Ess = {m['ss_error']:.4f}")
    ax.text(0.98, 0.55, txt, transform=ax.transAxes, fontsize=8,
            va='center', ha='right', bbox=dict(boxstyle='round,pad=0.3', fc='lightyellow', alpha=0.9))

    # 2. Control Action
    ax = axes[1]
    ax.plot(t_arr, u, color='darkorange', linewidth=1.5)
    ax.set_title("Control Action u(t)")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("u(t)")
    ax.set_xlim(0, t_end)
    ax.grid(True, alpha=0.3)

    # 3. Convergence
    ax = axes[2]
    ax.semilogy(result.convergence_history, color='green', linewidth=1.8)
    ax.set_title("PSO Convergence Curve")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Best Fitness (log)")
    ax.grid(True, alpha=0.3, which='both')
    ax.text(0.98, 0.95, f"Final: {result.best_fitness:.4f}",
            transform=ax.transAxes, fontsize=9, va='top', ha='right', color='green')

    plt.tight_layout()
    fname = f"result_{PLANT_NAME}.png"
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close(fig)

    print(f"\n[DONE] Plot successfully saved to: {fname}")
    print("=" * 60)


if __name__ == "__main__":
    main()
