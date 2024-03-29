# metropolis simulation. Accept parameters specifying system and simulation config. Store results in global dicts.
import json
import math
import platform
import random
import time

random.seed(2212)


def energy_nn(sigma, J, L):
    en = 0
    for i in range(L):
        for j in range(L):
            if sigma[i][j] == sigma[i][(j + 1) % L]: en -= J
            if sigma[i][j] == sigma[(i + 1) % L][j]: en -= J
    return en


def delta_energy_nn(sigma, J, i, j, new_q, L):
    # i, j = ind
    neighs = [(i, (j + 1) % L), (i, (j - 1) % L), ((i + 1) % L, j), ((i - 1) % L, j)]
    delta_en = 0
    for x, y in neighs:
        if sigma[i][j] == sigma[x][y]:
            delta_en += J
        if new_q == sigma[x][y]:
            delta_en -= J
    return delta_en


def propose_flip(sigma, J, L, N, q):
    index = random.randint(0, N - 1)
    index1, index2 = index // L, index % L

    x = random.randint(0, q - 1)
    while x == sigma[index1][index2]:
        x = random.randint(0, q - 1)
    return index1, index2, x


def metropolis(delta_en, t):
    if delta_en < 0:
        return True
    if random.uniform(0, 1) < math.exp(- delta_en / t):
        return True
    return False


def bincount(x, q, double=True):
    # flatten double list
    if double:
        flattened = []
        for sublist in x:
            for item in sublist:
                flattened.append(item)
        x = flattened
    # count appearences of integers between 0 and q-1 (error raised if q is wrong)
    counts = [0] * q
    for i in x:
        counts[i] += 1
    return counts


def MCMC(L, q, t, nstep, burnin, J=1):
    # random initial configuration
    N = L ** 2
    sigma = [[random.randint(0, q - 1) for _1 in range(L)] for _2 in range(L)]

    # initial energy
    en = energy_nn(sigma, J, L)

    # run a few steps to reach stationarity
    for istep in range(burnin):
        # propose random flip
        ind1, ind2, q_new = propose_flip(sigma, J, L, N, q)
        # compute energy difference
        delta_en = delta_energy_nn(sigma, J, ind1, ind2, q_new, L)
        # metropolis update rule
        if metropolis(delta_en, t):
            # update state
            q_old = sigma[ind1][ind2]
            sigma[ind1][ind2] = q_new
            # update energy
            en += delta_en

    # mag_state = list(np.bincount(np.array(sigma).reshape(-1), minlength=q))
    mag_state = bincount(sigma, q)
    mag_avg = 0

    avg_t = 0
    # main loop
    for istep in range(nstep):
        # propose random flip
        ind1, ind2, q_new = propose_flip(sigma, J, L, N, q)
        # compute energy difference
        delta_en = delta_energy_nn(sigma, J, ind1, ind2, q_new, L)

        # metropolis update rule
        if metropolis(delta_en, t):
            # update state
            q_old = sigma[ind1][ind2]
            sigma[ind1][ind2] = q_new

            # update energy
            en += delta_en
            # update magnetization history
            mag_state[q_old] -= 1
            mag_state[q_new] += 1

            # print(max(mag_state))
            # print(mag_state)
            # time.sleep(0.1)

        mag_avg += max(mag_state)

        avg_t += en

    return avg_t / nstep, mag_avg / nstep / N


def critical_Temperature(q, J=1):
    return J / math.log(1 + math.sqrt(q))


n2 = 5
n1 = 10
# dt = 0.005
dt = 0.02


def arange(start, stop, step):
    result = []
    eps = 0.001
    current = start
    while current < stop - eps:
        result.append(current)
        current += step
    return result


def get_temps(q, J=1, n1=n1, n2=n2, dt=dt, zero=0.4, infinity=3.0):
    crit = critical_Temperature(q, J)
    crit = round(crit, 4)

    low = crit - n1 * dt
    high = crit + n1 * dt

    core = arange(low, high, dt)
    out1 = arange(zero, low, (low - zero) / n2)
    out2 = arange(high, infinity, (infinity - high) / (n2 + 1))

    core = list(core)
    out1 = list(out1)[:-1]
    out2 = list(out2)

    for arr in [core, out1, out2]:
        for i, x in enumerate(arr):
            arr[i] = round(x, 4)

    return out1, core, out2


def simulate(L, q):
    avg_en = []
    avg_mag = []

    temps_triple = get_temps(q)

    ordered_temps = []

    steps = (10 ** 8, 2 * 10 ** 8, 5 * 10 ** 7)
    burnin = (10 ** 8, 2 * 10 ** 8, 5 * 10 ** 7)

    for temps, step, burn in zip(temps_triple, steps, burnin):
        for t in temps:
            tempo = time.time()
            ordered_temps.append(t)
            if abs(t - critical_Temperature(q)) < 0.05:
                exstp = 5 * 10 ** 8
                print(f"Starting simulation for {q=} {L=} {t=} with step={exstp} and burnin={exstp}")
                en, mag = MCMC(L, q, t, exstp, exstp)
            else:
                print(f"Starting simulation for {q=} {L=} {t=} with step={step} and burnin={burn}")
                en, mag = MCMC(L, q, t, step, burn)
            avg_en.append(en)
            avg_mag.append(mag)
            print(f"Elapsed time: {time.time() - tempo}")

    return ordered_temps, avg_en, avg_mag


def simulate_save(inp):
    q, L = inp
    tempo = time.time()
    print(f"Simulating {q=} {L=}")
    temps, avg_en, avg_mag = simulate(L, q)
    print(f"Elapsed time: {time.time() - tempo}")
    spec_heat = [(avg_en[i + 1] - avg_en[i]) / (temps[i + 1] - temps[i]) for i in range(0, len(temps) - 1)]
    spec_heat_temps = [(temps[i + 1] + temps[i]) / 2 for i in range(0, len(temps) - 1)]

    for i, x in enumerate(spec_heat_temps):
        spec_heat_temps[i] = round(x, 3)

    with open(f"simulation_{q=}_{L=}.json", "w") as file:
        json.dump({"temps": temps, "avg_en": avg_en, "avg_mag": avg_mag,
                   "spec_heat": spec_heat, "spec_heat_temps": spec_heat_temps}, file)

    # if "pypy" in platform.python_implementation().lower():
    #     import matplotlib.pyplot as plt
    #
    #     fig = plt.figure(figsize=(10, 10))
    #
    #     plt.plot(spec_heat_temps, spec_heat)
    #     plt.xlabel('Temperature')
    #     plt.ylabel('Specific Heat')
    #     plt.title('Specific Heat vs Temperature')
    #     plt.savefig(f"specific_heat_{q=}_{L=}.png")
    #
    #     plt.close(fig)
    #
    #     fig = plt.figure(figsize=(10, 10))
    #     plt.plot(temps, avg_en)
    #     plt.xlabel('Temperature')
    #     plt.ylabel('average Energy')
    #     plt.title('Energy vs Temperature')
    #     plt.savefig(f"energy_{q=}_{L=}.png")
    #     plt.close(fig)
    #
    #     fig = plt.figure(figsize=(10, 10))
    #     plt.plot(temps, avg_mag)
    #     plt.xlabel('Temperature')
    #     plt.ylabel('Max magnetization')
    #     plt.title('Max magnetization vs Temperature')
    #     plt.savefig(f"max_mag_{q=}_{L=}.png")
    #     plt.close(fig)
    #
    #     fig = plt.figure(figsize=(10, 10))
    #     plt.plot(temps[n2:-n2], avg_en[n2:-n2])
    #     plt.xlabel('Temperature')
    #     plt.ylabel('average Energy')
    #     plt.title('Energy vs Temperature zoomed in')
    #     plt.savefig(f"energy_zoom_{q=}_{L=}.png")
    #     plt.close(fig)
    #
    #     fig = plt.figure(figsize=(10, 10))
    #     plt.plot(temps[n2:-n2], avg_mag[n2:-n2])
    #     plt.xlabel('Temperature')
    #     plt.ylabel('Max magnetization')
    #     plt.title('Max magnetization vs Temperature zoomed in')
    #     plt.savefig(f"max_mag_zoom_{q=}_{L=}.png")
    #     plt.close(fig)
    #
    #     fig = plt.figure(figsize=(10, 10))
    #     plt.plot(spec_heat_temps[n2:-n2], spec_heat[n2:-n2])
    #     plt.xlabel('Temperature')
    #     plt.ylabel('Specific Heat')
    #     plt.title('Specific Heat vs Temperature zoomed in')
    #     plt.savefig(f"specific_heat_zoom_{q=}_{L=}.png")
    #     plt.close(fig)


import multiprocessing as mp

# print(critical_Temperature(5))
# print(get_temps(8))
#
# print("------------------")
# print("------------------")
# print("------------------")
#
# print(critical_Temperature(2))
# print(get_temps(2))

qs_Ls = [
    (3, 50),
]

if __name__ == '__main__':
    with mp.Pool(1) as p:
        print(p.map(simulate_save, qs_Ls))
    # print(get_temps(8))
