import wandb
import itertools
import gym
import copy
import tinder
import torch
import torch.nn as nn
import torch.nn.functional as F
from types import SimpleNamespace
import numpy as np
from torch.distributions.normal import Normal


class Policy(nn.Module):
    def __init__(self, state_dim, action_dim, action_low, action_high):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
        )

        self.mean_layer = nn.Linear(256, action_dim)
        self.log_std_layer = nn.Linear(256, action_dim)

        self.action_low = torch.FloatTensor(action_low).cuda()
        self.action_high = torch.FloatTensor(action_high).cuda()

    # with_logprob: if True, return (action [N], log Pr[action]).
    def forward(self, x, with_logprob=False):
        x = self.layers(x)
        mean = self.mean_layer(x)
        std = self.log_std_layer(x).clamp(-20, 2).exp()
        pi_distribution = Normal(mean, std)
        pi_action = pi_distribution.rsample()

        if with_logprob:
            # Compute logprob from Gaussian, and then apply correction for Tanh squashing.
            # NOTE: The correction formula is a little bit magic. To get an understanding
            # of where it comes from, check out the original SAC paper (arXiv 1801.01290)
            # and look in appendix C. This is a more numerically-stable equivalent to Eq 21.
            # Try deriving it yourself as a (very difficult) exercise. :)
            logp = pi_distribution.log_prob(pi_action).sum(axis=-1)
            logp -= (2 * (np.log(2) - pi_action - F.softplus(-2 * pi_action))).sum(
                axis=1
            )
        else:
            logp = None

        x = torch.tanh(pi_action)  # [N, action_dim]

        # scale (-1, 1) to [action.low, action_high]
        action = (x + 1) * (self.action_high - self.action_low) / 2 + self.action_low

        if with_logprob:
            return (action, logp)
        else:
            return action


class Q(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, x, actions):
        # x : [N, state_dim]
        # action: [N, action_dim]
        x = torch.cat([x, actions], dim=1)
        return self.layers(x).squeeze(1)


class ReplayBuffer(object):
    def __init__(self, num_records, state_dim, action_dim):
        # (state, action, reward, next_state)
        #  dim     1
        self.states = np.zeros((num_records, state_dim), dtype=np.float32)
        self.actions = np.zeros((num_records, action_dim), dtype=np.float32)
        self.rewards = np.zeros(num_records, dtype=np.float32)
        self.next_states = np.zeros((num_records, state_dim), dtype=np.float32)
        self.is_dones = np.zeros(num_records, dtype=np.float32)
        self.i = 0
        self.size = 0
        self.capacity = num_records

    def add(
        self, state, action, reward, next_state, is_done
    ):  # store ndarray, not tensor
        assert isinstance(state, np.ndarray)
        assert isinstance(next_state, np.ndarray)
        assert isinstance(action, np.ndarray)
        assert np.isscalar(reward)
        assert np.isscalar(is_done)
        i = self.i
        self.states[i] = state
        self.actions[i] = action
        self.rewards[i] = reward
        self.next_states[i] = next_state
        self.is_dones[i] = is_done

        self.i = (self.i + 1) % self.capacity
        self.size = min(self.capacity, self.size + 1)

    def sample(self, batch_size):
        if self.size < batch_size:
            return None

        indexes = np.random.randint(0, self.size, size=batch_size)
        return SimpleNamespace(
            states=torch.FloatTensor(self.states[indexes]).cuda(),
            actions=torch.FloatTensor(self.actions[indexes]).cuda(),
            rewards=torch.FloatTensor(self.rewards[indexes]).cuda(),
            next_states=torch.FloatTensor(self.next_states[indexes]).cuda(),
            is_dones=torch.FloatTensor(self.is_dones[indexes]).cuda(),
        )


def train(config):
    wandb.init(project="lunar-lander", name="target")

    # EPISODE = 1_000
    STEP = 3_000_000
    EXPERIENCE_REPLAY = 1_000_000
    BATCH_SIZE = 32
    ENTROPY_TERM_COEFFICIENT = 0.2
    # ENTROPY_TERM_COEFFICIENT = 0.002
    GAMMA = 0.99
    POLYAK = 0.995
    LR = 0.001
    START_STEP = 10000

    env = gym.make(config.env)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    if not isinstance(env.action_space, gym.spaces.Box):
        raise RuntimeError("action space is not continuous")

    q1 = Q(state_dim, action_dim).cuda()
    q2 = Q(state_dim, action_dim).cuda()
    q1_target = copy.deepcopy(q1)
    q2_target = copy.deepcopy(q2)

    q_params = itertools.chain(q1.parameters(), q2.parameters())
    q_opt = torch.optim.Adam(q_params, LR)
    del q_params

    policy = Policy(
        state_dim, action_dim, env.action_space.low, env.action_space.high
    ).cuda()
    policy_opt = torch.optim.Adam(policy.parameters(), LR)

    wandb.watch([q1, q2, policy], log="all", log_freq=10000)

    replay_buffer = ReplayBuffer(EXPERIENCE_REPLAY, state_dim, action_dim)
    episode = 0
    step = 0

    while True:
        episode += 1
        state = env.reset()

        episode_reward = 0
        is_done = False

        while (not is_done) and step < STEP:
            step += 1
            # get action from policy net
            state_tensor = (
                torch.from_numpy(state).type(torch.FloatTensor).unsqueeze(0).cuda()
            )
            with torch.no_grad():
                action = policy(state_tensor)[0].cpu().numpy()

            # take action
            next_state, reward, is_done, _info = env.step(action)
            bonus = 0
            reward += bonus

            # record
            replay_buffer.add(
                state=state,
                action=action,
                reward=reward,
                next_state=next_state,
                is_done=is_done,
            )
            if config.render:
                env.render()
            episode_reward += reward

            # clean up
            state = next_state
            del action, state_tensor, next_state, _info

            # train from replay
            if step < START_STEP:
                continue

            batch = replay_buffer.sample(BATCH_SIZE)

            # update Q
            # Q(s,a) = Q(next_state,a')

            with torch.no_grad():
                next_action, logp = policy(batch.next_states, with_logprob=True)

            target = batch.rewards + GAMMA * (1 - batch.is_dones) * (
                torch.min(
                    q1_target(batch.next_states, next_action),
                    q2_target(batch.next_states, next_action),
                )
                - ENTROPY_TERM_COEFFICIENT * logp
            )
            del next_action, logp

            # Ex_a'[Q(s',a')] = Sum_a'(Q(s',a')*Pr[a'] - alpha * log(Pr[a'|s']))
            # Since a' is continuous,
            # We sample a single value of a' from policy and use it

            q_loss = F.mse_loss(q1(batch.states, batch.actions), target) + F.mse_loss(
                q2(batch.states, batch.actions), target
            )
            q_opt.zero_grad()
            q_loss.backward()
            q_opt.step()
            del target

            # update Policy, ascend on Q+H
            action, logp = policy(
                batch.states, with_logprob=True
            )  # at this time, differentiable action
            policy_profit = (
                torch.min(q1(batch.states, action), q2(batch.states, action))
                - ENTROPY_TERM_COEFFICIENT * logp
            )
            policy_loss = -policy_profit.mean()
            policy_opt.zero_grad()
            policy_loss.backward()
            policy_opt.step()

            # Update target networks by polyak averaging.
            with torch.no_grad():
                for p, p_targ in zip(q1.parameters(), q1_target.parameters()):
                    p_targ.data.mul_(POLYAK)
                    p_targ.data.add_((1 - POLYAK) * p.data)
                for p, p_targ in zip(q2.parameters(), q2_target.parameters()):
                    p_targ.data.mul_(POLYAK)
                    p_targ.data.add_((1 - POLYAK) * p.data)

            wandb.log(dict(reward=reward, q_loss=q_loss.item()), step=step)

            if step % 10000 == 0:
                torch.save(dict(policy=policy.state_dict()), "target.pt")
        print(episode, step, episode_reward)
        wandb.log(dict(episode_reward=episode_reward), step=step)


def test(config):
    env = gym.make(config.env)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    if not isinstance(env.action_space, gym.spaces.Box):
        raise RuntimeError("action space is not continuous")

    policy = Policy(
        state_dim, action_dim, env.action_space.low, env.action_space.high
    ).cuda()
    policy.load_state_dict(torch.load("target.pt")["policy"])
    policy.eval()

    while True:
        state = env.reset()
        total_reward = 0
        while True:
            state_tensor = (
                torch.from_numpy(state).type(torch.FloatTensor).unsqueeze(0).cuda()
            )
            with torch.no_grad():
                action = policy(state_tensor)[0].cpu().numpy()
            state, reward, is_done, _info = env.step(action)
            env.render()
            total_reward += reward

            if is_done:
                break
        print(total_reward)


def main():
    tinder.bootstrap()

    config = SimpleNamespace(
        name="",
        env="LunarLanderContinuous-v2",
        mode="train",
        render=False,
        # per=True,
        # noisy=True,
        # dueling=True,
        # clip=True,
        # episodic_life=True,
        # episodic_score=False,
    )

    tinder.config.override(config)

    if config.mode not in ["train", "test"]:
        raise RuntimeError("unknown mode")
    elif config.mode == "train":
        train(config)
    elif config.mode == "test":
        test(config)


if __name__ == "__main__":
    main()
