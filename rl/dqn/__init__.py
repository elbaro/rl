import random
from collections import deque
from types import SimpleNamespace
import tinder
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data.dataloader import default_collate
import numpy as np
import gym
from tensorboardX import SummaryWriter
import tqdm

device = 'cuda'


class Network1D(nn.Module):
    def __init__(self, input_ch, action_count):
        super().__init__()
        self.seq = nn.Sequential(
            tinder.layers.Flatten(),
            nn.Linear(input_ch, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, action_count),
        )

    def forward(self, x):
        x = self.seq(x)
        return x


class AtariNetwork(nn.Module):
    """The original Atari DQN architecture

    env = 210 x 160 -> grayscale -> resize to 110 x 84 -> center crop to 84 x 84
    84×84×4 input, 16 8x8 filters stride=4, 32 4x4 stride=2, fc to 256

    Arguments:
        nn {[type]} -- [description]

    Returns:
        [type] -- [description]
    """

    def __init__(self, action_count):
        super().__init__()
        self.seq = nn.Sequential(
            tinder.layers.AssertSize(None, 4, 84, 84),
            nn.Conv2d(4, 16, kernel_size=8, stride=4),  # 20,20
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=4, stride=2),  # 9,9
            nn.ReLU(inplace=True),
            tinder.layers.Flatten(),
            nn.Linear(2592, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, action_count),
        )

    def forward(self, x):
        x = self.seq(x)
        return x


class ReplayBuffer(object):
    """
    strategy: circular write
    """

    def __init__(self, size, use_float32=True):
        super().__init__()
        self.capacity = size
        self.buf = []
        self.next_row = 0
        self.use_float32 = use_float32

    def push(self, record):
        if len(self.buf) < self.capacity:
            self.buf.append(record)
        else:
            self.buf[self.next_row] = record
            self.next_row = (self.next_row+1) % self.capacity

    def sample(self, batch_size):
        if len(self.buf) < batch_size:
            return None
            # raise RuntimeError('Replay Buffer is not buffered enough to sample')
        rows = np.random.randint(0, len(self.buf), batch_size)
        batch = [self.buf[row] for row in rows]
        batch = default_collate(batch)

        # Python float is 64bit giving us DoubleTensor
        if self.use_float32:
            for key in batch:
                if isinstance(batch[key], torch.Tensor) and batch[key].dtype == torch.float64:
                    batch[key] = batch[key].float()  # Double to Float
        return batch


class IndexTree(object):
    def __init__(self, n, init_value):
        super().__init__()
        self.n = n
        N = 1
        while N < n:
            N <<= 1
        N -= 1
        self.offset = N

        self.tree = np.full(n, init_value)


class SumTree(IndexTree):
    def __init__(self, n):
        super().__init__(n, init_value=0)

    def update(self, index, new_value):
        delta = new_value - self.tree[index+self.offset]
        self.add(index, delta)

    def add(self, index, delta):
        index += self.offset
        while index:
            self.tree[index] += delta

    def query_sum(self, left, right):
        left += self.offset
        right += self.offset
        s = 0
        while left <= right:
            if left & 1:
                s += self.tree[left]
            if right & 1 ^ 1:
                s += self.tree[right]
            left = (left+1) >> 1
            right = (right-1) >> 1
        return s


class MinTree(IndexTree):
    def __init__(self, n):
        # n+1 to prevent oob
        super().__init__(n+1, init_value=float('-inf'))

    def update(self, index, new_value):
        """update a point

        Arguments:
            index {int} -- 1<=index<=n
            new_value {any}
        """

        index += self.offset
        self.tree[index] = new_value
        index >>= 1
        while index:
            m = min(self.tree[index*2], self.tree[index*2+1])
            self.tree[index] = m
            index >>= 1

    def query_min(self, left, right):
        left += self.offset
        right += self.offset
        m = float('+inf')
        while left <= right:
            if self.tree[left] < m:
                m = self.tree[left]
            if self.tree[right] < m:
                m = self.tree[right]
            left = (left+1) >> 1
            right = (right-1) >> 1
        return m


class PrioritizedReplayBuffer(ReplayBuffer):
    """
    strategy: sample with TD error
    """

    def __init__(self, size, alpha=0.6):  # 0.6 for atari
        raise NotImplementedError

        # super().__init__()
        # self.capacity = size
        # self.buf = []
        # self.next_row = 0
        # self.alpha = alpha

        # self.sum_tree = SumTree(size)
        # self.min_tree = MinTree(size)

    def push(self, record):
        if len(self.buf) < self.capacity:
            row = len(self.buf)
            self.buf.append(record)
        else:
            row = self.next_row
            self.buf[self.next_row] = record
            self.next_row = (self.next_row+1) % self.capacity

        # update row
        pass

    def sample(self, batch_size):
        # if len(self.buf) < batch_size:
        #     raise RuntimeError('Replay Buffer is not buffered enough to sample')
        # rows = np.random.randint(0, len(self.buf), batch_size)
        pass

        batch = [self.buf[row] for row in rows]
        return default_collate(batch)

    def update_priority(self, index, priority):
        assert priority > 0
        assert 0 <= index < len(self.buf)


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

    def __init__(self, action_count, network, decay=0.9):
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
            return self.net(states).argmax(dim=1)

    def get_actions_and_qs_for_states(self, states) -> torch.Tensor:
        """
        cpu input->cpu output
        gpu input->gpu output

        2D: states = [N, stack*ch, H, W]
        1D: states = [N, stack*ch, dim]
        """

        with torch.no_grad():
            q, a = self.net(states).max(dim=1)
            return a, q

    def get_qs_for_states(self, states):
        with torch.no_grad():
            q, _ = self.net(states).max(dim=1)
            return q

    def loss_on_batch(self, batch, second_net=None):
        """batch of (s0a0r0s1a1done)

        An external trainer is responsible for opt.zero_grad() and opt.step()

        Arguments:
            batch {torch.Tensor}
        """
        assert isinstance(batch, dict)
        batch = SimpleNamespace(**batch)
        if second_net is None:
            # Q(s0,a0) ~= r0+decay*maxQ(s1,?)*(done?0:1)
            with torch.no_grad():
                next_q, _ = self.net(batch.s1.to(device)).max(dim=1)
                Q1 = batch.r0.to(device) + self.decay*next_q*(1-batch.done.float().to(device))

            Q0 = torch.gather(self.net(batch.s0.to(device)), dim=1, index=batch.a0.unsqueeze(1).to(device)).squeeze(1)
            loss = F.mse_loss(Q0, Q1.detach())
        else:
            # Q(s0,a0) ~= r0+decay*Q(s1,?)*(done?0:1)
            raise NotImplementedError
        return loss


class AtariPlayer(Agent):
    """
        Player is aware of observations.
        Player stacks observations to create states.
        Player may repeat an action.
    """

    def __init__(self, env, net=None, obs_stack=4):
        assert obs_stack >= 1

        if net is None:
            net = AtariNetwork(env.action_space.n).to(device)

        # # 1D : obs_shape=[128]
        # if len(obs_shape) == 1:
        #     self.net = Network1D(obs_shape[0]*obs_stack, action_count)
        # # ? 2D -> atari?
        # else:
        #     self.net = AtariNetwork(action_count)
        # 1D?
        super().__init__(
            action_count=env.action_space.n,
            network=net,
        )
        # obs_shape=env.observation_space.shape,
        self.obs_stack_count = obs_stack
        self.obs_stack = None

    def reset_episode(self):
        self.obs_stack = None

    def push_obs(self, obs) -> torch.Tensor:
        obs = torch.from_numpy(obs).float()/255.0  # [210,160,3]
        obs = F.interpolate(obs.mean(dim=2).view(1, 1, 210, 160), size=(110, 84), mode='nearest')[0][0]  # [110,84]
        # center crop - 110 = 13+84+13
        obs = obs[13:13+84, :]  # [84,84]

        if self.obs_stack is None:
            self.obs_stack = deque([obs]*self.obs_stack_count)
        else:
            self.obs_stack.popleft()
            self.obs_stack.append(obs)

        self.state = torch.stack(list(self.obs_stack), dim=0)  # [4,84,84]
        return self.state

    def get_next_action(self, exploration_rate):
        if random.random() < exploration_rate:
            action = random.randint(0, self.action_count-1)
        else:
            action = self.get_actions_for_states(
                self.state.unsqueeze(0).to(device)
            )[0].item()

        return action

    def get_next_action_with_q(self, exploration_rate):
        state = self.state.unsqueeze(0).to(device)
        if random.random() < exploration_rate:
            action = random.randint(0, self.action_count-1)
            q = self.get_qs_for_states(state).item()
        else:
            action, q = self.get_actions_and_qs_for_states(state)
            action, q = action.item(), q.item()

        return action, q


class AgentTrainer(object):
    """
        lr=1e-4,
        buffer_size=10000,
        exploration_fraction=0.1,
        exploration_final_eps=0.01,
        train_freq=4,
        learning_starts=10000,
        target_network_update_freq=1000,
        gamma=0.99,
        prioritized_replay=True,
        prioritized_replay_alpha=0.6,
        checkpoint_freq=10000,
        checkpoint_path=None,
        dueling=True

        clip_rewards=?
        scaled float frame?
        episodic life?
        fire reset?
    """

    def __init__(self, env, use_replay=1, replay_size=None):
        # TODO: action_repeat=3 for space invader
        assert 0 <= use_replay <= 2  # 1 for basic, 2 for prioritized
        assert (use_replay == 0) == (replay_size is None)

        self.env = env

        if use_replay == 1:
            self.memory = ReplayBuffer(replay_size)
        elif use_replay == 2:
            self.memory = PrioritizedReplayBuffer(replay_size)
        else:
            self.memory = None

        self.agent = AtariPlayer(env)
        self.target_q = AtariNetwork(self.agent.action_count).to(device)
        self.opt = torch.optim.RMSprop(self.agent.get_network().parameters())

        self.tb = SummaryWriter(log_dir=f'/data/rl/logs/{env.spec.id}')
        self.saver = tinder.saver.Saver('/data/rl/weights', env.spec.id)
        self.model = SimpleNamespace(
            net=self.agent.net,
            net_target=self.target_q,
            opt=self.opt,
            frame_i=0,
            episode_i=0,
        )

        self.saver.load_latest(self.model)

    def train(self, frame_total, render_episode_period, batch_size, episode_save_period=10):
        model = self.model

        frame_i = model.frame_i
        episode_i = model.episode_i

        bar_episode = tqdm.tqdm(initial=episode_i, leave=True)
        bar_frame = tqdm.tqdm(initial=frame_i, total=frame_total, leave=True)

        while frame_i < frame_total:  # new episode
            episode_i += 1
            bar_episode.update()

            is_playing = True
            obs = self.env.reset()
            self.agent.reset_episode()
            self.agent.push_obs(obs)

            s0 = self.agent.state

            while is_playing and frame_i < frame_total:
                frame_i += 1
                bar_frame.update()

                if frame_i < 1_000_000:
                    exploration_rate = 1 - frame_i/1_000_000*0.9
                else:
                    exploration_rate = 0.1

                # action = self.agent.get_next_action(exploration_rate=exploration_rate)
                action, q = self.agent.get_next_action_with_q(exploration_rate=exploration_rate)
                self.tb.add_scalar('Q', q, frame_i)
                obs, reward, is_done, _ = self.env.step(action)
                self.agent.push_obs(obs)
                if (render_episode_period is not None) and (episode_i % render_episode_period == 0):
                    self.env.render()

                if is_done:
                    is_playing = False

                self.memory.push({
                    's0': s0,
                    'a0': action,
                    'r0': reward,
                    's1': self.agent.state,
                    'done': is_done,
                })

                # train
                if self.memory is None:
                    pass
                else:
                    batch = self.memory.sample(batch_size)
                    if batch is None:
                        continue

                    self.opt.zero_grad()
                    loss = self.agent.loss_on_batch(batch)
                    loss.backward()
                    self.opt.step()

            if episode_i % episode_save_period == 0:
                model.episode_i = episode_i
                model.frame_i = frame_i
                self.saver.save(model, epoch=episode_i)

        bar_frame.close()
        bar_episode.close()
