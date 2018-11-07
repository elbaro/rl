import time
import math
import random
from collections import deque
from types import SimpleNamespace
import tinder
import torch
from torch import nn
from torch.nn import functional as F
import gym
from tensorboardX import SummaryWriter
import tqdm
from .replay import ReplayBuffer, PrioritizedReplayBuffer

device = 'cuda'


class NoisyLinear(nn.Module):
    # Borrowed from kaixhin/rainbow

    def __init__(self, in_dim, out_dim, std_init=0.4):
        super(NoisyLinear, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.std_init = std_init
        self.weight_mean = nn.Parameter(torch.Tensor(out_dim, in_dim))
        self.weight_var = nn.Parameter(torch.Tensor(out_dim, in_dim))
        self.bias_mean = nn.Parameter(torch.Tensor(out_dim))
        self.bias_var = nn.Parameter(torch.Tensor(out_dim))
        self.reset_parameters()

    def reset_parameters(self):
        mu_range = 1 / math.sqrt(self.weight_mean.size(1))
        self.weight_mean.data.uniform_(-mu_range, mu_range)
        self.weight_var.data.fill_(self.std_init / math.sqrt(self.weight_var.size(1)))
        self.bias_mean.data.uniform_(-mu_range, mu_range)
        self.bias_var.data.fill_(self.std_init / math.sqrt(self.bias_var.size(0)))

    def _scale_noise(self, size):
        x = torch.randn(size, device=device)
        x = x.sign().mul(x.abs().sqrt())
        return x

    def reset_noise(self):
        epsilon_in = self._scale_noise(self.in_dim)
        epsilon_out = self._scale_noise(self.out_dim)
        self.weight_epsilon = epsilon_out.ger(epsilon_in)
        self.bias_epsilon = self._scale_noise(self.out_dim)

    def forward(self, input):
        self.reset_noise()

        if self.training:
            return F.linear(input, self.weight_mean + self.weight_var.mul(self.weight_epsilon), self.bias_mean + self.bias_var.mul(self.bias_epsilon))
        else:
            return F.linear(input, self.weight_mean, self.bias_mean)


class AtariNetwork(nn.Module):
    """The original Atari DQN architecture

    84×84×4 input, 16 8x8 filters stride=4, 32 4x4 stride=2, fc to 256
    """

    def __init__(self, action_count, dueling=True, noisy=True):
        super().__init__()
        self.dueling = dueling

        if noisy:
            Linear = NoisyLinear
        else:
            Linear = nn.Linear

        if dueling:
            self.common = nn.Sequential(
                tinder.layers.AssertSize(None, 4, 84, 84),
                nn.Conv2d(4, 16, kernel_size=8, stride=4),  # 20,20
                nn.ReLU(inplace=True),
                nn.Conv2d(16, 32, kernel_size=4, stride=2),  # 9,9
                nn.ReLU(inplace=True),
                tinder.layers.Flatten(),
            )

            self.state_value = nn.Sequential(
                Linear(2592, 256),
                nn.ReLU(inplace=True),
                Linear(256, 1),
            )

            # FC
            self.advantage = nn.Sequential(
                Linear(2592, 256),
                nn.ReLU(inplace=True),
                Linear(256, action_count),
            )
        else:
            self.seq = nn.Sequential(
                tinder.layers.AssertSize(None, 4, 84, 84),
                nn.Conv2d(4, 16, kernel_size=8, stride=4),  # 20,20
                nn.ReLU(inplace=True),
                nn.Conv2d(16, 32, kernel_size=4, stride=2),  # 9,9
                nn.ReLU(inplace=True),
                tinder.layers.Flatten(),
                Linear(2592, 256),
                nn.ReLU(inplace=True),
                Linear(256, action_count),
            )

    def forward(self, x):
        # x: 0~1

        if self.dueling:
            x = self.common(x)
            state_value = self.state_value(x)  # [B,1]
            advantage = self.advantage(x)  # [B,action_count]
            q = state_value + advantage - advantage.mean(dim=1, keepdim=True)  # [B,action_count]
            return q
        else:
            x = self.seq(x)
            return x

    def get_relative_advantage(self, x):
        # x: 0~1, not 0~255

        if self.dueling:
            x = self.common(x)
            adv = self.advantage(x)
            return adv
        else:
            return self.seq(x)

# def byte_to_gpu_float(x):
#     return x.to(device, dtype=torch.float32, non_blocking=True) / 255.0


class Agent(object):
    """A stateless agent.

    - Agent receives states (not observations).
    - Agent decides the action on a state (not observation).
    - Agent returns a loss on a batch of (s0a0r0s1a1done).

    - Agent doesn't know how to train (adam/sgd/etc)
    - Agent doesn't have a memory (replay buffer, stack of observations)
    - Agent doesn't skip a frame, stack observations, or repeat an action

    Arguments:
        nn {[type]} -- [description]

    Returns:
        [type] -- [description]
    """

    def __init__(self, action_count, network, decay=0.99):
        super().__init__()
        self.action_count = action_count
        self.net = network
        self.decay = decay

    def get_network(self):
        return self.net

    def get_actions_for_states(self, states) -> torch.Tensor:
        """
        cpu input->cpu output
        gpu input->gpu output

        2D: states = [N, stack*ch, H, W]
        1D: states = [N, stack*ch, dim]
        """
        with torch.no_grad():
            if self.net.dueling:
                return self.net.get_relative_advantage(states).argmax(dim=1)
            else:
                return self.net(states).argmax(dim=1)

    def losses_on_batch(self, batch, return_q, target_net=None):
        """batch of (s0a0r0s1a1done)

        An external trainer is responsible for opt.zero_grad() and opt.step()

        Arguments:
            batch {dict}
        """
        assert isinstance(batch, dict)
        batch = SimpleNamespace(**batch)

        # a1 = argmax Q(s1, ?)
        # Q(s0,a0) ~= r0+decay*Q'(s1,a1)*(done?0:1)
        with torch.no_grad():
            if target_net is None:
                next_q, _ = self.net(batch.s1.to(device, dtype=torch.float32, non_blocking=True) / 255.0).max(dim=1)
            else:

                s1 = batch.s1.to(device, dtype=torch.float32, non_blocking=True) / 255.0
                a1 = self.net.get_relative_advantage(s1).argmax(dim=1)
                next_q = target_net(s1)
                next_q = torch.gather(next_q, dim=1, index=a1.unsqueeze(1)).squeeze(1)
            Q1 = (
                batch.r0.to(device, dtype=torch.float32, non_blocking=True) +
                self.decay*next_q * (1-batch.done.float().to(device, non_blocking=True))
            )

        Q0 = torch.gather(self.net(batch.s0.to(device, dtype=torch.float32, non_blocking=True) / 255.0), dim=1,
                          index=batch.a0.unsqueeze(1).to(device, non_blocking=True)).squeeze(1)
        losses = F.smooth_l1_loss(Q0, Q1.detach(), reduction='none')

        if return_q:
            with torch.no_grad():
                q = Q0.mean()
            return losses, q.item()
        return losses

    def get_next_action(self, state, exploration_rate):
        if random.random() < exploration_rate:
            action = random.randint(0, self.action_count-1)
        else:
            action = self.get_actions_for_states(
                state.unsqueeze(0).to(device, dtype=torch.float32, non_blocking=True) / 255.0
            ).item()

        return action


class AgentTrainer(object):
    """
        Atari:

        buffer_size=10000,
        exploration_fraction=0.1,
        exploration_final_eps=0.01,
        gamma=0.99,
        prioritized_replay=True,
        prioritized_replay_alpha=0.6,
    """

    def __init__(self, env, name=None, use_prioritized=True, replay_size=1_000_000, lr=2e-4, dueling=True, noisy=True):
        self.env = env
        self.use_prioritized = use_prioritized

        if use_prioritized:
            self.memory = PrioritizedReplayBuffer(replay_size)
        else:
            self.memory = ReplayBuffer(replay_size)

        net = AtariNetwork(env.action_space.n, dueling=dueling, noisy=noisy).to(device, non_blocking=True)
        self.agent = Agent(action_count=env.action_space.n, network=net)
        self.target_net = AtariNetwork(self.agent.action_count, dueling=dueling,
                                       noisy=noisy).to(device, non_blocking=True)
        self.opt = torch.optim.RMSprop(self.agent.get_network().parameters(), lr=lr)

        if name:
            name = env.spec.id + '-'+name
        else:
            name = env.spec.id

        self.tb = SummaryWriter(log_dir=f'/data/rl/{name}/logs')
        self.saver = tinder.saver.Saver(f'/data/rl/', name)
        self.model = SimpleNamespace(
            net=self.agent.net,
            target_net=self.target_net,
            opt=self.opt,
            frame_i=0,
            episode_i=0,
        )

        if self.saver.load_latest(self.model):
            print('restored model')
        else:
            print('new model from scratch')

    def play(self, sleep=0):
        model = self.model
        is_playing = True
        episode_reward = 0
        s0 = self.env.reset()  # list [o1,o2,o3,o4]
        frame_i = 0

        while is_playing:
            frame_i += 1
            action = self.agent.get_next_action(s0, exploration_rate=0)
            # print(action, self.env.unwrapped.get_action_meanings()[action])

            s1, reward, is_done, _info = self.env.step(action)
            self.env.render()
            episode_reward += reward

            if is_done:
                is_playing = False

            s0 = s1

            if sleep > 0:
                time.sleep(sleep)

        print('episode reward:', episode_reward)
        print('frame #: ', frame_i)

    def train(self,
              frame_total=10*1_000_000,
              frame_exploration_count=1*1_000_000,
              render_episode_period=10,
              batch_size=32,
              frame_target_update_period=1_000,  # 1k (pong in 30m) or 10k
              episode_save_period=100,
              frame_train_period=4,
              frame_learning_start=10_000,
              exploration_min=0.1,):
        model = self.model

        frame_i = model.frame_i
        episode_i = model.episode_i

        bar_episode = tqdm.tqdm(initial=episode_i, leave=True, ncols=70)
        bar_frame = tqdm.tqdm(initial=frame_i, total=frame_total, leave=True, ncols=70)

        while frame_i < frame_total:  # new episode
            episode_i += 1
            episode_reward = 0
            bar_episode.update()

            is_playing = True
            s0 = self.env.reset()

            while is_playing and frame_i < frame_total:
                frame_i += 1
                bar_frame.update()

                if frame_i < frame_exploration_count:
                    exploration_rate = 1 - frame_i/frame_exploration_count*(1-exploration_min)
                else:
                    exploration_rate = exploration_min

                action = self.agent.get_next_action(s0, exploration_rate=exploration_rate)

                s1, reward, is_done, _ = self.env.step(action)
                episode_reward += reward

                if (render_episode_period is not None) and (episode_i % render_episode_period == 0):
                    self.env.render()

                if is_done:
                    is_playing = False

                self.memory.push({
                    's0': s0,  # lazy
                    'a0': action,
                    'r0': reward,
                    's1': s1,  # lazy
                    'done': is_done,
                })

                s0 = s1

                # train
                if frame_i % frame_train_period == 0 and frame_i >= frame_learning_start:
                    if self.use_prioritized:
                        ret = self.memory.sample(batch_size*frame_train_period)
                        if ret is None:
                            continue
                        batch, rma, weights = ret
                    else:
                        batch = self.memory.sample(batch_size*frame_train_period)
                        if batch is None:
                            continue

                    self.opt.zero_grad()
                    losses, q = self.agent.losses_on_batch(batch, return_q=True, target_net=self.target_net)
                    if self.use_prioritized:
                        self.memory.update_priority(rma, losses.detach())
                        losses *= weights.detach().to(device)
                    loss = losses.mean()
                    loss.backward()
                    self.opt.step()

                    self.tb.add_scalar('Q', q, frame_i)
                    self.tb.add_scalar('TD error', loss.item(), frame_i)

                    if frame_i % frame_target_update_period == 0:
                        tinder.rl.copy_params(src=self.agent.get_network(), dst=self.target_net)

            self.tb.add_scalar('episode.reward', episode_reward, frame_i)

            if episode_i % episode_save_period == 0:
                model.episode_i = episode_i
                model.frame_i = frame_i
                self.saver.save(model, epoch=episode_i, score=episode_reward)

        self.saver.save(model, epoch=episode_i, score=episode_reward)

        bar_frame.close()
        bar_episode.close()
