import copy
import time

import scipy

from utilities import energy_fun, update_energy, propose_flip, metropolis_rule, ising_ferro, Flipper
import numpy as np

import matplotlib.pyplot as plt

n_iters = 10 ** 6
q = 5
side = 30
N = side ** 2
energies = {}
magnetizations = {}
acceptance = {}
burnin = 3000


def metropolis(T):
    flipper = Flipper(N, q, n_iters+burnin)
    energies[T] = []
    magnetizations[T] = []
    ens = energies[T]
    mags = magnetizations[T]
    acceptance[T] = 0
    # J = -np.ones((N, N))
    J = ising_ferro(N, side)

    J = np.sparse

    sigma = np.random.randint(0, q, N)
    energy = energy_fun(J, sigma)
    # run for some iterations to reach stationary distribution
    for _ in range(burnin):
        index, val = flipper.propose(J, sigma)
        delta_e = update_energy(J, sigma, index, val)
        if metropolis_rule(delta_e, T):
            sigma[index] = val
            energy += delta_e

    mags.append([int(sum(sigma == x)) for x in range(q)])
    ens.append(energy)

    for it in range(n_iters):
        index, val = flipper.propose(J, sigma)
        delta_e = update_energy(J, sigma, index, val)

        nm = copy.copy(mags[-1])

        if metropolis_rule(delta_e, T):
            acceptance[T] += 1
            nm[sigma[index]] -= 1
            nm[val] += 1

            sigma[index] = val
            energy += delta_e

        mags.append(nm)
        ens.append(energy)
    acceptance[T] /= n_iters


tmatt = 1e-4
temps = [tmatt]
for t in temps:
    print(f"Computing {t}")
    metropolis(t)
print(time.process_time())
