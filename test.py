import time
from collections import defaultdict

import numpy as np

# size of the lattice
side = 50
N = side ** 2
q = 4


def _u_slow_test(J, sigma):
    """function used to test the faster one"""
    en = 0
    for i in range(N):
        for j in range(i + 1, N):
            if sigma[i] == sigma[j]:
                en += J[i, j]
    return en / 2


def U(J, sigma):
    """internal energy, only ran once at the beginning"""
    en = 0
    for i in range(N):
        delta = sigma[i] == sigma
        en += np.dot(J[i], delta)
    return en / 2 - J.trace()


# temperature
T = 3.
n_steps = 100000000
stride = 100000

bins = 300
min_en = -N * N - 1
max_en = N * N + 1


def propose_flip(sigma, J):
    """propose random flip uniformly and checks that it is actually a flip"""
    index = np.random.randint(0, N)
    x = np.random.randint(0, q)
    while x == sigma[index]:
        x = np.random.randint(0, q)
    return index, x


def energy_flip(sigma, J, i, new_q):
    """compute what the internal energy would be after flipping one spin without modifying the state"""
    # new energy - old energy
    delta_old = sigma == sigma[i]
    delta_new = sigma == new_q
    delta_new[i] = True

    en_old = np.dot(J[i], delta_old)
    en_new = (np.dot(J[i], delta_new))
    return + en_new - en_old


# initialize J to be a random symmetric matrix with weights in [-1, +1]
# J = np.random.uniform(-1, 1, (N, N))
# for i in range(N):
#     for j in range(i, N):
#         J[i, j] = J[j, i]
#
J = -np.ones((N, N))

for i in range(N):
    for j in range(N):
        if abs(i % side - j % side) < 2:
            J[i, j] = 0
        if abs(i // side - j // side) < 2:
            J[i, j] = 0

# initialize the state at a uniform random
sigma = np.random.randint(0, q, N)
energy = U(J, sigma)

g: defaultdict[int] = defaultdict(lambda: 1)
# sf = 2
sf = 1.00001
M = 1000

f = sf
energies = []
magnetization = []

# region attempt to do non-uniform bins
# def binner(x):
# global memo
# if int(x) in memo:
#     return memo[int(x)]
# res = floor(nbin * tanh(alfabin * x))
# memo[int(x)] = res
# return res


# def debinner(k):
#     return atanh(k / nbin) / alfabin

precision_bins = side


def binner(x) -> int:
    return int(x / precision_bins)


def debinner(x):
    return x * precision_bins


ps = []
for step in range(n_steps):
    ind, new_q = propose_flip(sigma, J)
    diff = energy_flip(sigma, J, ind, new_q)

    # if energy != U(J, sigma):
    #     print("orrore")
    energy_new = energy + diff

    p = min(1, g[binner(energy)] / g[binner(energy_new)])

    # if energy == -39600 and step > 100:
    #     debug = 0
    # if energy != energy_new:
    #     print(energy, energy_new)
    #     print(g[binner(energy)], g[binner(energy_new)])

    ps.append(p)
    if np.random.uniform(0, 1) < p:
        # here it is possible to add more histogram for other state functions y
        sigma[ind] = new_q
        energy = energy_new
        g[binner(energy)] *= f
    if step % M == 0:
        f **= .5

    if step % stride == 0:
        print(f"Step {step}/{n_steps} - {round(time.process_time(), 2)}s")
        if time.process_time() > 60:
            break

    energies.append(energy)
    magnetization.append(sigma.mean())

import matplotlib.pyplot as plt

x, y = [], []
for i, v in g.items():
    # print(i)
    x.append((debinner(i), v))
    y.append(v)

x.sort(key=lambda x: x[0])
plt.plot([z[0] for z in x], [z[1] for z in x])
plt.title(f"Entropy / Energy {sf=} {step=}")
plt.grid()
plt.show()

# fig, ax1 = plt.subplots()
# ax2 = ax1.twinx()

# ax1.plot(energies, color="blue", label="energy")
# ax2.plot(magnetization, color="red", label="magnetization")
# ax1.legend()
# ax2.legend()
# plt.show()

# plt.hist(ps)
# plt.show()
