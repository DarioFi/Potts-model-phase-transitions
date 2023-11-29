# Potts model and phase transitions

In this project we used Monte Carlo methods to study numerically a classical model in the field of statistical physics and disordered systems: the Potts model. By sampling from the Boltzmann distribution we found evidence, in the thermodynamic limit, of a continuous phase transition when $q \le 4$ and a discontinuous one for $q > 4$.

This is joint work with Dario Filatrella. A detailed report of our experiments and findings, with code and figures, can be found in `report.ipynb`.

![energy-and_specific_heat](https://github.com/DarioFi/Potts-model-phase-transitions/assets/42577667/62dc7795-7a8a-49e3-97a6-13270d935443)

## Potts model

The Potts model consists of a system of interacting spins that can be in either of $q$ states. Spins are located on a two-dimensional square lattice of length $L$, and neighbouring spins interact according to the Hamiltonian:
```math
$$ H(s_1, ..., s_N) = - J \sum_{(ij)} \delta (s_i, s_j) $$
```
where the summation is over all pairs of neighbouring spins on the lattice (with periodic boundary conditions). $J$ is a positive constant and it represents the strength of the ferromagnetic interaction.

## Numerical Simulations

To estimate thermodynamic averages, we used the Metropolis-Hastings algorithm. Since Monte Carlo simulations are computationally very expensive, we had to resort to a few simple optimizations to speed up the simulations, which are detailed in the report. Furthermore, we ran our code with Pypy, a JIT-compiled alternative to CPython. The following is an example trajectory sampled with Metropolis, $q = 4$, below the critical temperature. Each colour corresponds to a different state, and we report magnetizations on the y axis.

![dynamics](https://github.com/DarioFi/Potts-model-phase-transitions/assets/42577667/1ba625a6-364e-4903-9c9d-774c5d4de9a1)
