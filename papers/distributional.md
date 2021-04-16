# Distributional RL

Distributional as in probability, not distributed computing.

> (1) "A Distributional Perspective on Reinforcement Learning"

A good intro paper.
- Represents a value(s,a) distribution with Pr(Vi|s,a) for i=1,..N, where Vi = Vmin + (Vmax-Vmin)*(i-1)/(n-1).
- Proves that Wasserstein(p=1) + Bellman operator is a contraction.
- However, the Wasserstein Bellman operator does not work with SGD. Hence they used KL-divergence in experiments.
- This was SOTA.

> (2) Distributional Reinforcement Learning with Quantile Regression

An improvement of (1). The model is called QR-DQN.
- Instead of representing Pr(value=Vi|s,a), model quantiles theta_1, ..., theta_n at tau_1=0, 1/n, 2/n, .. 1. However, it still does not work with SGD without modification.
- Using a dual problem of quantile regression (see [Wikipedia](https://en.wikipedia.org/wiki/Quantile_regression)),
  it proposes the loss function compatible with SGD. With this, we can use Wasserstein (not KL-divergence) in experiments.
- This beats (1) by a largin margin.
- The concept of tilted absolute value function seems useful not only in distributional problem but in general quantil regression.
  According to authors, it's already popular in Economics.

> (3) Distributed Distributional Deterministic Policy Gradients (D4PG)

The full name has 4 Ds, *Distributed Distributional Deep Deterministic Policy Gradient*.

A continuous version of (1), (2). Or, an improvement of DDPG.

> (4) Implicit Quantile Networks for Distributional Reinforcement Learning

