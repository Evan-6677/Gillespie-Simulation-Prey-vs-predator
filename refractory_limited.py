# implements terms for mature vs immature bdellovibrio
# also considers non-zero bdelloplast death

# === Necessary libraries ===
import os
import torch
import numpy as np
from scipy.interpolate import interp1d
from matplotlib import pyplot as plt
import pandas as pd
import json

import sys

if len(sys.argv) >= 2:
    seed = int(sys.argv[1])
else:
    seed = 157

print('seed',seed)
np.random.seed(seed)
torch.manual_seed(seed)

# === Device ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# === Stoichiometry Matrix for Nutrient-aware Chemostat Model ===
# Species order: E, Bv_mature, Bp, N, Bv_immature
S_nutrient = torch.tensor([
    [+1,  -1,   0,  -1,   0,   0,  -1,   0,   0,    0,   0,   0],  # E
    [ 0,  -1,   0,   0,  -1,   0,   0,  -1,   0,    0,   0,  +1],  # Bv_mature
    [ 0,  +1,  -1,   0,   0,  -1,   0,   0,  -1,    0,   0,   0],  # Bp
    [-1,   0,   0,   0,   0,   0,   0,   0,   0,   -1,   0,   0],  # N
    [ 0,   0,   0,   0,   0,   0,   0,   0,   0,    0,  -1,  -1],  # Bv_immature
], dtype=torch.float, device=device)

# === Rate Function for Nutrient-aware Model ===
def get_rates_with_nutrients(X, params):
    E, Bv_m, Bp, N, Bv_imm = X[:, 0], X[:, 1], X[:, 2], X[:, 3], X[:, 4]
    mu, rho, h, d1, d2, omega, N_in, m = [params[:, i] for i in range(8)]
    n_offspring = 5  # or sample per event if stochastic

    return torch.stack([
        mu * E * N,         # R1: E growth
        rho * E * Bv_m,     # R2: predation
        h * Bp,             # R3: hatching
        d1 * E,             # R4: E death
        d2 * Bv_m,          # R5: Bv_m death
        omega * Bp,         # R6: Bp outflow
        omega * E,          # R7: E outflow
        omega * Bv_m,       # R8: Bv_m outflow
        omega * Bp,         # R9: Bp outflow
        omega * N,          # R10: N outflow
        omega * Bv_imm,     # R11: Bv_imm outflow
        m * Bv_imm          # R12: maturation
    ], dim=1).to(device)

# === Square wave inflow function ===

def square_wave_inflow(t, amplitude=50.0, period=30.0, duty_cycle=0.25):
    """
    Returns square-wave style nutrient inflow.
    t can be scalar or tensor. Units should match simulation time.
    """
    on_time = period * duty_cycle
    return torch.where((t % period) < on_time, amplitude, 0.0)

# === Define sweep parameters ===
def generate_param_sweep_5species(num_conditions):
    mu_range     = (1.0, 3.0)
    rho_range    = (0.1, 2.0)
    h_range      = (0.1, 0.6)
    d1_range     = (0.01, 0.07)
    d2_range     = (0.01, 0.07)
    omega_range  = (0.05, 0.2)
    N_in_range   = (10.0, 200.0)
    m_range      = (0.2, 2.0)
    ranges = [mu_range, rho_range, h_range, d1_range, d2_range, omega_range, N_in_range, m_range]
    sweep = [torch.rand(num_conditions) * (hi - lo) + lo for (lo, hi) in ranges]
    params = torch.stack(sweep, dim=1).to(device)
    return params

# === Updated Gillespie step with deterministic square wave inflow pulses ===
def gillespie_step(X, params, S, t, t_max):
    move = t < t_max
    if not move.any():
        return t, X, torch.zeros_like(t)

    # Square wave inflow parameters
    inflow_period = 30.0
    inflow_duty_cycle = 0.25
    inflow_amplitude = 50.0
    inflow_on_time = inflow_period * inflow_duty_cycle

    # Calculate time until next inflow pulse for each trajectory
    time_since_period_start = t % inflow_period
    # Determine time to next rising edge (start of inflow pulse)
    time_to_next_inflow = inflow_period - time_since_period_start
    # Calculate stochastic reaction propensities
    rates = get_rates_with_nutrients(X, params)
    dt_prop = -torch.log(torch.rand_like(rates)) / rates.clamp(min=1e-8)
    dt_min, rxn = dt_prop.min(dim=1)

    # Determine which event happens first for each simulation: inflow pulse or stochastic reaction
    inflow_first = time_to_next_inflow < dt_min
    dt = torch.where(inflow_first, time_to_next_inflow, dt_min)
    t_new = t + dt
    X_new = X.clone()

    # for i in range(X.shape[0]):
    #     if not move[i]:
    #         continue
    #     if inflow_first[i]:
    #         # Apply deterministic inflow pulse
    #         X_new[i, 3] += inflow_amplitude 
    #     else:
    #         # Apply stochastic reaction update
    #         dS = S[:, rxn[i].item()].clone()
    #         # Modify for hatching reaction offspring count
    #         if rxn[i].item() == 2:
    #             n_offspring = np.random.poisson(3.5)
    #             dS[4] = n_offspring
    #         X_new[i] += dS
    X_new[torch.logical_and(move,inflow_first), 3] += inflow_amplitude

    dS = S[:,rxn].clone()
    dS[4,(rxn==2)] = torch.poisson(3.5*torch.ones_like(rxn)[rxn==2])
    dS = dS.T
    dS[torch.logical_or(~move,inflow_first)] *= 0
    X_new += dS

    X_new = torch.clamp(X_new, min=0)
    return t_new, X_new, dt

# === Interpolation function to calculate mean trajectory ===
def interpolate_replicates(T_all, X_all, time_grid):
    T_np = T_all.cpu().numpy()
    X_np = X_all.cpu().numpy()
    N_reps = T_all.shape[1]
    G = len(time_grid)
    X_interp = np.zeros((N_reps, G, 5))
    for i in range(N_reps):
        t_i = T_np[:, i]
        for j in range(5):
            x_i = X_np[:, i, j]
            f = interp1d(t_i, x_i, kind='previous', bounds_error=False,
                         fill_value=(x_i[0], x_i[-1]))
            X_interp[i, :, j] = f(time_grid)
    return X_interp

# === Plotting function ===
def plot_outflow_results(results, time_grid, outdir="plots", log_scale=False):
    os.makedirs(outdir, exist_ok=True)
    labels = ['E (Prey)', 'Bv (Mature)', 'Bp (Bdelloplast)', 'Nutrient', 'Bv (Immature)']
    param_labels = ["$\\mu$", "$\\rho$", "$h$", "$d_1$", "$d_2$", "$\\omega$", "$N_{in}$", "$m$"]
    species_keys = ["E", "Bv_mature", "Bp", "N", "Bv_immature"]

    for label, (T_all, X_all, theta) in results.items():
        X_interp = interpolate_replicates(T_all, X_all, time_grid)
        mean = X_interp.mean(axis=0)
        std = X_interp.std(axis=0)

        fig, axs = plt.subplots(1, 5, figsize=(30, 4), sharex=True)
        for j in range(5):
            for r in range(X_interp.shape[0]):
                axs[j].plot(time_grid, X_interp[r, :, j], color='gray', alpha=0.3)
            axs[j].plot(time_grid, mean[:, j], color='black', linewidth=2)
            axs[j].fill_between(time_grid, mean[:, j] - std[:, j], mean[:, j] + std[:, j],
                                color='blue', alpha=0.2)
            axs[j].set_title(labels[j])
            axs[j].set_xlabel("Time (hr)")
            axs[j].set_ylabel("Population" if j != 3 else "Nutrient")
            axs[j].grid(True)
            if log_scale:
                axs[j].set_yscale("log")

        param_str = "\n".join(f"{k} = {v:.2f}" for k, v in zip(param_labels, theta))
        axs[-1].text(
            0.98, 0.98, param_str,
            transform=axs[-1].transAxes,
            verticalalignment='top',
            horizontalalignment='right',
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="gray", alpha=0.9)
        )

        plt.suptitle(f"Condition: {label}", fontsize=14)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(os.path.join(outdir, f"{label}_replicates.png"))
        plt.close()

        pd.DataFrame(mean, columns=species_keys).to_csv(
            os.path.join(outdir, f"{label}_mean.csv"), index=False)
        with open(os.path.join(outdir, f"{label}_params.json"), "w") as f:
            json.dump({k: float(v) for k, v in zip(param_labels, theta)}, f, indent=2)

# === Driver function ===
def run_nutrient_param_sweep(num_conditions=100, N=50, t_max=200.0, time_grid=None, max_steps=100000):
    if time_grid is None:
        time_grid = np.linspace(0, t_max, 200)
    param_matrix = generate_param_sweep_5species(num_conditions)
    t0 = torch.zeros(N, device=device)
    X0 = torch.zeros(N, 5, device=device)
    X0[:, 0] = 100  # E
    X0[:, 1] = 15   # Bv_mature
    X0[:, 2] = 0    # Bp
    X0[:, 3] = 100  # N
    X0[:, 4] = 0    # Bv_immature

    results = {}
    for i in range(num_conditions):
        label = f"cond_{i:03d}"
        params = param_matrix[i].unsqueeze(0).repeat(N, 1)
        t = t0.clone()
        X = X0.clone()
        history_t, history_X = [t.clone()], [X.clone()]
        for step in range(max_steps):
            t, X, dt = gillespie_step(X, params, S_nutrient, t, t_max)
            history_t.append(t.clone())
            history_X.append(X.clone())
            if (t >= t_max).all():
                break
        results[label] = (torch.stack(history_t), torch.stack(history_X), param_matrix[i].tolist())
    outdir = "nutrient_param_sweep_"+ str(seed)
    plot_outflow_results(results, time_grid, outdir=outdir, log_scale=False)
    return results

# === Now, actually running the simulation ===
if __name__ == "__main__":
    num_conditions = 100
    N = 50
    t_max = 200.0
    max_steps = 100000
    time_grid = np.linspace(0, t_max, 200)

    results = run_nutrient_param_sweep(
        num_conditions=num_conditions,
        N=N,
        t_max=t_max,
        time_grid=time_grid,
        max_steps=max_steps
    )
