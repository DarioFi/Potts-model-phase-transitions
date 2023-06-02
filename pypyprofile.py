import math
import random
import time

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns



L = 20  # side length
N = L ** 2  # number of sites
q = 4  # number of states per site
J = 1  # ferromagnetic interaction strength


def energy_nn(sigma, J):
    en = 0
    for i in range(L):
        for j in range(L):
            if sigma[i][j] == sigma[i][(j + 1) % L]: en -= J
            if sigma[i][j] == sigma[(i + 1) % L][j]: en -= J
    return en


# controlla ordine i, j corretto
neighs_cache = [[((i, (j + 1) % L), (i, (j - 1) % L), ((i + 1) % L, j), ((i - 1) % L, j)) for j in range(L)] for i in
                range(L)]


def delta_energy_nn(sigma, J, i, j, new_q):
    # i, j = ind
    neighs = neighs_cache[i][j]

    delta_en = 0
    for x, y in neighs:
        if sigma[i][j] == sigma[x][y]:
            delta_en += J
        if new_q == sigma[x][y]:
            delta_en -= J
    return delta_en


# PRECOMP = 10 ** 6
# precomputed_indexes = [random.randint(0, N - 1) for x in range(PRECOMP)]
# precomp_index = 0


def propose_flip(sigma, J):
    # global precomp_index, precomputed_indexes

    # if precomp_index >= PRECOMP:
    #     precomputed_indexes = [random.randint(0, N - 1) for x in range(PRECOMP)]
    #     precomp_index = 0
    #
    # index = precomputed_indexes[precomp_index]
    # precomp_index += 1

    index = random.randint(0, N - 1)
    index1, index2 = index // L, index % L

    x = random.randint(0, q - 1)
    while x == sigma[index1][index2]:
        x = random.randint(0, q - 1)
    return index1, index2, x


# metropolis acceptance with symmetric proposal. returns a Boolean
def metropolis(delta_en, t):
    if delta_en < 0:
        return True
    if random.uniform(0, 1) < math.exp(- delta_en / t):
        return True
    return False


# metropolis simulation. Accept parameters specifying system and simulation config. Store results in global dicts.

avgs = []


def MCMC(L, q, t, nstep, burnin, J=1):
    random.seed(42)

    # random initial configuration
    # sigma = np.random.randint(0, q, (L, L))

    sigma = [[random.randint(0, q) for _1 in range(L)] for _2 in range(L)]
    en = energy_nn(sigma, J)

    # run a few steps to reach stationarity
    for istep in range(burnin):
        # propose random flip
        ind1, ind2, q_new = propose_flip(sigma, J)
        # compute energy difference
        delta_en = delta_energy_nn(sigma, J, ind1, ind2, q_new)
        # metropolis update rule
        if metropolis(delta_en, t):
            # update state
            q_old = sigma[ind1][ind2]
            sigma[ind1][ind2] = q_new
            # update energy
            en += delta_en

    # prepare to store metrics
    # mag_history[t] = np.zeros((q, nstep + 1))
    mag_history[t] = [[0 for x in range(q)] for y in range(nstep + 1)]
    # mag_history[t][:, 0] = np.bincount(sigma.reshape(-1), minlength=q)
    prob_history[t] = []
    en_history[t] = []
    n_accepted[t] = 0

    # main loop
    for istep in range(nstep):
        # propose random flip
        ind1, ind2, q_new = propose_flip(sigma, J)
        # compute energy difference
        delta_en = delta_energy_nn(sigma, J, ind1, ind2, q_new)
        # update probability history
        if delta_en > 0:
            prob_history[t].append(math.exp(- delta_en / t))
        else:
            prob_history[t].append(1)
        # prepare magnetization update
        # mag_history[t][:, istep + 1] = mag_history[t][:, istep]
        mag_history[t][istep + 1] = mag_history[t][istep].copy()
        # metropolis update rule
        if metropolis(delta_en, t):
            # update state
            q_old = sigma[ind1][ind2]
            sigma[ind1][ind2] = q_new
            n_accepted[t] += 1
            # update energy
            en += delta_en
            # update magnetization history
            mag_history[t][istep + 1][q_old] -= 1
            mag_history[t][istep + 1][q_new] += 1
        # update energy history
        en_history[t].append(en)

    # print(en_history)
    # en_history[t] = np.array(en_history[t])
    prob_history[t] = np.array(prob_history[t])
    avgs.append(sum(en_history[t]) / len(en_history[t]))

    del en_history[t]  # memory reason, keep only average
    # maybe extract features also from prob_history and delete that too
    del prob_history[t]
    del mag_history[t]  # not used at the moment so freeing memory


en_history = {}
mag_history = {}
prob_history = {}
n_accepted = {}

burnin = 7 * 10 ** 5
nstep = 5 * 10 ** 6

# temps for q = 5
# temps = [0.1, 0.3, 0.7, 0.75, 0.78, 0.8, 0.82, 0.83, 0.84, 0.85, 0.86, 0.87, 0.9, 0.95, 1.0, 1.1, 1.2, 1.5]

temps = [.5, .5, .5, .5, .5]

tempo = time.time()
for t in temps:
    print(f'temperature: {t}')
    MCMC(L, q, t, nstep, burnin, J)  # seed 42
    print(f"Passed: {time.time() - tempo}")
    tempo = time.time()

print(f"Total time:  {time.process_time()}")
exit()
# avgs = [sum(en_history[t]) / len(en_history[t]) for t in temps]


plt.plot(temps, avgs)
plt.xlabel('Temperature')
plt.ylabel('average Energy')
plt.title('Energy vs temperature')
plt.show()

plt.plot(temps[2:-4], avgs[2:-4])
plt.xlabel('Temperature')
plt.ylabel('average Energy')
plt.title('Energy vs temperature - zoom on the phase transition')
plt.show()

spec_heats = [(avgs[i + 1] - avgs[i]) / (temps[i + 1] - temps[i]) for i in range(0, len(temps) - 1)]
avg_temps = [(temps[i + 1] + temps[i]) / 2 for i in range(0, len(temps) - 1)]

plt.plot(avg_temps, spec_heats)
plt.ylabel('Specific Heat')
plt.xlabel('temperature')
plt.title('Specific Heat vs temperature')
plt.show()

print(list(zip(temps, avgs)))
print(list(zip(temps, spec_heats)))
