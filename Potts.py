#%%
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#%%
# global parameters

L = 20             # side length
N = L ** 2        # number of sites
q = 5             # number of states per site
#%%
# energy function of the potts model with long-range interactions.
def energy(sigma, J):
    # initialize energy to 0
    en = 0
    # loop over sites
    for i in range(N):
        # interaction mask of site i: True at position j iff site i and site j interact.
        delta = (sigma[i] == sigma)
        delta[i] = False
        # energy due to interaction of i with all other sites --> double counting non-diagonal entries
        en += np.dot(J[i], delta)

    # fix double counting before returning
    return 0.5 * en / N

# # slow but safe implementation of the energy
# def energy_test(sigma, J):
#     # initialize energy to 0
#     en = 0
#     # loop over pairs of sites
#     for i in range(N):
#         for j in range(i + 1, N):
#             # add interaction of sites i and j
#             if sigma[i] == sigma[j]:
#                 en += J[i, j]
#     return en / N
#%%
# propose a random flip in a random site
def propose_flip(sigma, J):
    # sample a random site and state
    index = np.random.randint(0, N)
    x = np.random.randint(0, q)
    # ensure new state is different than the old one
    while x == sigma[index]:
        x = np.random.randint(0, q)
    # return
    return index, x
#%%
# compute change in energy after flipping site i to state new_q
def delta_energy(sigma, J, i, new_q):

    # interaction masks of site i before and after flip
    delta_old = (sigma == sigma[i])
    delta_new = (sigma == new_q)
    delta_new[i], delta_old[i] = False, False

    # energy due to interactions that stop happening after flip
    en_old = np.dot(J[i], delta_old)
    # energy due to interactions that begin happening after flip
    en_new = (np.dot(J[i], delta_new))

    # return change in energy due to flip
    return (en_new - en_old) / N
#%%
# metropolis acceptance with symmetric proposal. returns a Boolean
def metropolis(delta_en, t):
    if delta_en < 0:
        return True
    if np.random.uniform(0, 1) < np.exp( - delta_en / t ):
        return True
    return False
#%%
# # random symmetric interaction matrix.
# J = np.random.randn(N, N)
# for i in range(N):
#     for j in range(i+1, N):
#         J[i, j] = J[j, i]

# # random nearest-neighbour interaction matrix
J = np.zeros((N, N))
assert np.sqrt(N) == int(np.sqrt(N))    # N must be a perfect square
for i in range(N):
    for j in [(i//L)*L + (i+1)%L, (i//L)*L + (i-1)%L, ((i//L+1)%L)*L + i%L, ((i//L-1)%L)*L + i%L]:
        J[i, j] = np.random.randn()

# # complete ferromagnetic interaction
# J = - np.ones((N, N))

# ferromagnetic nearest-neighbour interaction matrix
k = - 1   # interaction strength
J = np.zeros((N, N))
assert np.sqrt(N) == int(np.sqrt(N))    # N must be a perfect square
for i in range(N):
    for j in [(i//L)*L + (i+1)%L, (i//L)*L + (i-1)%L, ((i//L+1)%L)*L + i%L, ((i//L-1)%L)*L + i%L]:
        J[i, j] = k
#%%
# Monte Carlo simulation

# random initial configuration
sigma = np.random.randint(0, q, N)

t = 0.0001
nstep = 300000
stride = 1000

en = energy(sigma, J)

en_history = []
mag_history = np.zeros((q, nstep+1))
prob_history = []
n_accepted = 0
mag_history[:, 0] = np.bincount(sigma, minlength=q)

for istep in range(nstep):

    # propose random flip
    ind, q_new = propose_flip(sigma, J)
    # compute energy difference
    delta_en = delta_energy(sigma, J, ind, q_new)

    # update probability history
    prob_history.append(min(1, np.exp(- delta_en / t)))

    # prepare magnetization update
    mag_history[:, istep+1] = mag_history[:, istep]

    # metropolis update rule
    if metropolis(delta_en, t):
        # update state
        q_old = sigma[ind]
        sigma[ind] = q_new
        n_accepted += 1

        # update energy
        en += delta_en
        # update magnetization history
        mag_history[q_old, istep+1] -= 1
        mag_history[q_new, istep+1] += 1

    # update energy history
    en_history.append(en)

    if istep % stride == 0:
        print(istep, en)

en_history = np.array(en_history)
prob_history = np.array(prob_history)
#%%
burnin = 0
en_history_full = en_history
mag_history_full = mag_history
en_history = en_history[burnin:]
mag_history = mag_history[:, burnin:]
#%%
plt.plot(en_history)
plt.xlabel("step")
plt.ylabel("E")
print(f'Average energy: {en_history.mean()}')
plt.show()
#%%
print(f"Fraction of accepted proposals: {n_accepted / nstep}")
plt.hist(prob_history, bins=100)
plt.show()
#%%
# convolve acceptance probabilities with a kernel to improve visualization
window_size = 3000
kernel = np.ones(window_size) / window_size
smoothed_probs = np.convolve(prob_history, kernel, mode='valid')

# plot
plt.plot(smoothed_probs)
plt.yscale('log')
plt.show()
#%%
# print magnetization evolution in time#
for i in range(q):
    plt.plot(mag_history[i] / N, label=i)
plt.show()
#%%
sns.heatmap(sigma.reshape(L, L), cbar=False, xticklabels=False, yticklabels=False)
plt.show()