import numpy as np


# def energy_nn(J, sigma):
#     raise NotImplemented
#
#
# def update_energy_nn(J, sigma, index, value):
#     raise NotImplemented


def update_energy(J, sigma, index, value):
    # interaction masks of site i before and after flip
    delta_old = (sigma == sigma[index])
    delta_new = (sigma == value)
    delta_new[index], delta_old[index] = False, False

    # energy due to interactions that stop happening after flip
    en_old = np.dot(J[index], delta_old)
    # energy due to interactions that begin happening after flip
    en_new = (np.dot(J[index], delta_new))

    # return change in energy due to flip
    return (en_new - en_old)


def update_energy_multiple(J, sigma, indexes, values):
    individuals = 0
    os = [sigma[i] for i in indexes]
    for ind, val in zip(indexes, values):
        individuals += update_energy(J, sigma, ind, val)
        sigma[ind] = val

    for ind, x in zip(indexes, os):
        sigma[ind] = x

    return individuals


def energy_fun(J, sigma):
    # initialize energy to 0
    en = 0
    N = len(J)
    # loop over sites
    for i in range(N):
        # interaction mask of site i: True at position j iff site i and site j interact.
        delta = (sigma[i] == sigma)
        delta[i] = False
        # energy due to interaction of i with all other sites --> double counting non-diagonal entries
        en += np.dot(J[i], delta)

    # fix double counting before returning
    return 0.5 * en


class Flipper:
    def __init__(self, N, q, n_iters, multiple=0):
        self.ci = 0
        self.rnindex = np.random.randint(0, N, n_iters + 1)
        self.qs = np.random.randint(0, q, int((q + 3) / q * n_iters) * (multiple * 2 + 1) ** 2)
        self.cq = 0
        self.multiple = multiple
        self.nn = [(x + int(N ** 0.5) * y) for x in range(multiple + 1) for y in
                   range(multiple + 1)]

    def reset(self):
        self.ci = 0
        self.cq = 0

    def propose(self, J, sigma):
        index = self.rnindex[self.ci]
        self.ci += 1
        x = self.qs[self.cq]
        self.cq += 1
        while x == sigma[index]:
            x = self.qs[self.cq]
            self.cq += 1
        return index, x

    def pick_ind(self):
        self.ci += 1
        return self.rnindex[self.ci - 1]

    def pick_q(self):
        self.cq += 1
        return self.qs[self.cq - 1]

    def propose_multiple(self, J, sigma):
        N = len(sigma)

        i = self.pick_ind()

        inds = [(x + i) % N for x in self.nn]

        qs = []
        for w in inds:
            q = self.pick_q()
            # while q == sigma[w]:
            #     q = self.pick_q()
            qs.append(q)

        return inds, qs


def propose_flip(J, sigma, q_max):
    # sample a random site and state
    index = np.random.randint(0, len(sigma))
    x = np.random.randint(0, q_max)
    # ensure new state is different from the old one
    while x == sigma[index]:
        x = np.random.randint(0, q_max)
    # return
    return index, x


def metropolis_rule(delta_e, T):
    if delta_e < 0:
        return True
    if np.random.uniform(0, 1) < np.exp(- delta_e / T):
        return True
    return False


def ising_ferro(N, L):
    J = np.zeros((N, N))
    assert np.sqrt(N) == int(np.sqrt(N))  # N must be a perfect square
    for i in range(N):
        for j in [(i // L) * L + (i + 1) % L, (i // L) * L + (i - 1) % L, ((i // L + 1) % L) * L + i % L,
                  ((i // L - 1) % L) * L + i % L]:
            J[i, j] = -1
    return J


if __name__ == '__main__':

    np.random.seed(1234)
    L = 10
    q = 5
    N = L ** 2
    for test in range(1):
        J = ising_ferro(N, L)
        sigma = np.random.randint(0, q, N)

        se = energy_fun(J, sigma)
        il = []
        iv = []

        f = Flipper(N, q, n_iters=1000, multiple=1)

        for _ in range(10):
            x = f.propose_multiple(J, sigma)
            print(x)

        ue = update_energy_multiple(J, sigma, il, iv)

        for i, v in zip(il, iv):
            sigma[i] = v
        fe = energy_fun(J, sigma)

        error = fe - se - ue
        if abs(error) > 1e-3:
            print(il)
            print("dramma")
            print("start:  ", se)
            print("end:   ", fe)
            print("update:   ", ue)
            print(f"actual: {fe - se}")
            print(f"error: {error}")
