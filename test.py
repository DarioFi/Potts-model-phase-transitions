import copy

from utilities import energy_fun, update_energy_multiple, Flipper, metropolis_rule, ising_ferro
import numpy as np

import matplotlib.pyplot as plt

# %%
n_iters = 5 * 10 ** 5
q = 5
side = 20
N = side ** 2
energies = {}
magnetizations = {}
acceptance = {}
burnin = 5000
multiple = 1


# %%
# ayo = []
# %%
def metropolis(T):
    flipper = Flipper(N, q, n_iters + burnin, multiple)
    energies[T] = []
    magnetizations[T] = []
    ens = energies[T]
    mags = magnetizations[T]
    acceptance[T] = 0
    # J = -np.ones((N, N))
    J = ising_ferro(N, side)
    sigma = np.random.randint(0, q, N)
    energy = energy_fun(J, sigma)
    # run for some iterations to reach stationary distribution
    for _ in range(burnin):
        indices, vals = flipper.propose_multiple(J, sigma, multiple)
        delta_e = update_energy_multiple(J, sigma, indices, vals)
        if metropolis_rule(delta_e, T):
            for ind, val in zip(indices, vals):
                sigma[ind] = val
            energy += delta_e

    mags.append([int(sum(sigma == x)) for x in range(q)])
    ens.append(energy)

    for it in range(n_iters):
        indices, vals = flipper.propose_multiple(J, sigma, multiple)
        delta_e = update_energy_multiple(J, sigma, indices, vals)

        nm = copy.copy(mags[-1])

        # ayo.append(np.exp(- delta_e / T))

        if metropolis_rule(delta_e, T):
            acceptance[T] += 1

            for ind, val in zip(indices, vals):
                nm[sigma[ind]] -= 1
                nm[val] += 1
                sigma[ind] = val

            energy += delta_e

        mags.append(nm)
        ens.append(energy)
    acceptance[T] /= n_iters


# %%
# temps = np.arange(.0001, .01, .0001)
tmatt = 2e-4
temps = [tmatt]
for t in temps:
    print(f"Computing {t=}")
    metropolis(t)
