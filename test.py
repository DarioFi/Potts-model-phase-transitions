import numpy as np

# parameters
N = 20 ** 2
q = 2


# J = np.ones((N,N))
# sigma = np.ones(N)
# print(sigma)
def _u_slow_test(J, sigma):
    en = 0
    for i in range(N):
        for j in range(i + 1, N):
            if sigma[i] == sigma[j]:
                en += J[i, j]
    return en / 2


def U(J, sigma):
    en = 0
    for i in range(N):
        delta = sigma[i] == sigma
        en += np.dot(J[i], delta)
    return en / 2 - J.trace()


# U = _u_slow_test

T = 3.
n_steps = 15000
stride = 1e3

bins = 500000
min_en = -N * N - 1
max_en = N * N + 1


def propose_flip(sigma, J):
    index = np.random.randint(0, N)
    x = np.random.randint(0, q)
    while x == sigma[index]:
        x = np.random.randint(0, q)
    return index, x
    # while (x := np.random.randint(0, N), np.random.randint(0, q)) == (x[0], sigma[x[0]]):
    #     pass
    # return x


def energy_flip(sigma, J, i, new_q):
    # new energy - old energy
    delta_old = sigma == sigma[i]
    delta_new = sigma == new_q
    delta_new[i] = True

    en_old = np.dot(J[i], delta_old)
    en_new = (np.dot(J[i], delta_new))
    return + en_new - en_old


# J = np.random.uniform(-1, 1, (N, N))
# for i in range(N):
#     for j in range(i, N):
#         J[i, j] = J[j, i]
#
J = -np.ones((N, N))

sigma = 2 * np.random.randint(0, q, N) - 1
energy = U(J, sigma)

g = np.ones(bins + 1)
f = 1.1
M = 1000

energies = []
magnetization = []

en_to_bin = lambda x: int((x - min_en) / (max_en - min_en) * bins)

for step in range(n_steps):
    ind, new_q = propose_flip(sigma, J)

    diff = energy_flip(sigma, J, ind, new_q)
    energy_new = energy + diff

    p = min(1, g[en_to_bin(energy)] / g[en_to_bin(energy_new)])
    if np.random.uniform(0, 1) < p:
        # print(f"accepted {energy}")
        # here it is possible to add more histogram for other state functions y
        sigma[ind] = new_q
        energy = energy_new
        g[en_to_bin(energy)] *= f
    if step % M == 0:
        print(f"Step {step}/{n_steps}")
        f **= .5

    if energy != U(J, sigma):
        x = 0

    energies.append(energy)
    magnetization.append(sigma.mean())

import matplotlib.pyplot as plt

# plt.hist(g, bins=100)
# plt.show()
fig, ax1 = plt.subplots()
ax2 = ax1.twinx()

ax1.plot(energies, color="blue")
ax2.plot(magnetization, color="red")
plt.show()
