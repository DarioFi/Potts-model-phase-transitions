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
    return (en_new - en_old) / len(sigma)


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
    return 0.5 * en / N


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
