

## Algos

See: https://spinningup.openai.com/en/latest/algorithms/sac.html

- Rainbow
    start. DQN = Q(s,a) + e-greedy + experience-replay(1 million) + RMSProp + target-network(copy)/online-network
    - Double Q-learning: q value with target-network is evaluated at the action from online-network
    - Prioritized replay: samples with prob relative to last absolute TD error
    - Dueling Network: Q(s,a) = V(s) + Advantage(s,a)
    - Multi-step learning: Q(s_t, a) is updated from Q(s_[t+n], a)
    - Distributional RL: Use a distribution P(s,a) instead of Q(s,a), and update with D_KL.
    - Noisy Network: Add noise terms, anneal over time.

- VPG: on-policy, PolicyFunction + ValueFunction(s,a)
- TRPO: on-policy, additional policy update step-size constraint terms 
- PPO: same idea as TRPO, better loss term
- DDPG: off-policy, only works with continuous actions, learns two fu nctions Q(s,a)+u(s). Policy is learnt by gradient-ascent on Q(s,mu(a)), 
- TD3 (Twin Delayed DDPG): Add 3 tricks to DDPG
- SAC: off-policy, only works with continuous actions (but can be altered for discrete case). Learns Q(s,a) and pi(s). Q(s,a) has additional entropy(policy) term.

## Impls
- sac-basic.py: replay buffer, w/o target networks
- sac.py: sac-basic + target-networks
