import numpy as np
from numba import njit
from matplotlib import pyplot as plt
import pandas as pd

# Simulation parameters
N_r = 15          # Number of stochastic runs
T = 10 * 24       # Total simulation time (in hours)
case = 'Case1'

# Random choice function for stochastic reaction selection
categorical = lambda p: np.random.choice(p.size, p=p)

@njit
def normalize(x):
    return x / x.sum()

@njit
def get_rates(x, value):
    ingestion_E, mu_E, k, d, ingestion_B, nu, k_E, xi, m = value
    E, B, P = x
    rates = np.array([
        ingestion_E,                             # Resource ingestion (constant influx)
        mu_E * E,                                # Linear growth (normal reproduction)
        ((mu_E - d) / k) * (E ** 2),            # Quadratic term (crowding/death)
        d * E,                                   # Death of E. coli
        ingestion_B,                             # Bb ingestion
        nu * (E / (E + k_E)) * B,               # Infection/growth of Bb depending on E
        xi * P,                                  # Phage infecting Bb
        m * P,                                   # Phage decay
        m * B                                    # Bb death
    ])
    return rates

# Reaction stoichiometry matrix: how populations change per reaction
g = 4  # Phage burst size
S = np.array([
    [1, 0, 0],    # Reaction 0: +1 E. coli (resource ingestion)
    [1, 0, 0],    # Reaction 1: +1 E. coli (reproduction)
    [-1, 0, 0],   # Reaction 2: -1 E. coli (crowding death)
    [-1, 0, 0],   # Reaction 3: -1 E. coli (death)
    [0, 1, 0],    # Reaction 4: +1 Bb
    [-1, -1, 1],  # Reaction 5: E. coli & Bb decrease, phage produced (lysis)
    [0, g, -1],   # Reaction 6: Bb increase, phage decrease (infection)
    [0, 0, -1],   # Reaction 7: -1 phage (decay)
    [0, -1, 0]    # Reaction 8: -1 Bb (death)
])

def run_forward(x0, T, value):
    t = 0
    xs = [x0 * 1.0]
    x = x0 * 1.0
    ts = [0.0]

    while True:
        rates = get_rates(x, value)
        if rates.sum() <= 0:
            return np.array(ts), np.stack(xs)

        jump_t = np.random.exponential(1 / rates.sum())
        t += jump_t

        if t > T:
            return np.array(ts), np.stack(xs)

        reaction = categorical(normalize(rates))
        x += S[reaction]

        xs.append(x.copy())
        ts.append(t)

# Parameter values (example)
ground = np.array([
    50 / 1000,   # ingestion_E
    0.1,         # mu_E
    6000,        # k
    0.01,        # d
    0.1 * 50 / 1000,  # ingestion_B
    6,           # nu
    100000,      # k_E
    1 / 4,       # xi
    0.1          # m
])

# Initial population: random E. coli count, Bb and Phage start at 0
x0 = np.array([int(np.random.normal(300, 10)), 0, 0])

# Run simulations
trajs = [run_forward(x0, T, ground) for _ in range(N_r)]

# Plotting E. coli trajectories (with labels)
for i, (ts, xs) in enumerate(trajs):
    plt.plot(ts / 24, xs[:, 0], label=f'Run {i+1}')
plt.xlabel('Time (days)')
plt.ylabel('E. coli within the gut')
plt.title('E. coli Dynamics')
plt.legend()
plt.show()

# Plotting Bb trajectories (with labels)
for i, (ts, xs) in enumerate(trajs):
    plt.plot(ts / 24, xs[:, 1], label=f'Run {i+1}')
plt.xlabel('Time (days)')
plt.ylabel('Bb within the gut')
plt.title('Bb Dynamics')
plt.legend()
plt.show()

# Plotting Phage trajectories (with labels)
for i, (ts, xs) in enumerate(trajs):
    plt.plot(ts / 24, xs[:, 2], label=f'Run {i+1}')
plt.xlabel('Time (days)')
plt.ylabel('Phage within the gut')
plt.title('Phage Dynamics')
plt.legend()
plt.show()
