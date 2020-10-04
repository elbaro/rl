from collections import deque
from PIL import Image
import torch
from torch.nn import functional as F
import gym
import numpy as np
from baselines.common.atari_wrappers import EpisodicLifeEnv


class Wrapper(gym.Wrapper):
    def __init__(
        self, env, stack=4, episodic_life=False, episodic_score=False, clip_reward=False
    ):
        """

        - grayscale
        - resize
        - crop
        - stack

        See Also
        --------
        baselines.common.atari_wrappers.LazyFrames
        """

        # env = EpisodicLifeEnv(env) # no effect on Pong

        if episodic_life:
            env = EpisodicLifeEnv(env)

        gym.Wrapper.__init__(self, env)
        self.stack = stack
        self.frames = deque([], maxlen=stack)

        shp = env.observation_space.shape
        # self.observation_space = gym.spaces.Box(low=0, high=1.0, shape=(
        #     shp[2] * stack, shp[0], shp[1]), dtype=np.float)
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(shp[2] * stack, shp[0], shp[1]), dtype=np.uint8
        )

        self.episodic_score = episodic_score
        self.clip_reward = clip_reward

    @staticmethod
    def convert_obs(obs):
        obs = np.asarray(
            Image.fromarray(obs).convert("L").resize((84, 110), Image.NEAREST)
        )
        obs = torch.from_numpy(obs[13 : 13 + 84, :])
        return obs

    def reset(self):
        obs = self.env.reset()
        obs = Wrapper.convert_obs(obs)
        for _ in range(self.stack):
            self.frames.append(obs)
        return self.make_stack()

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        obs = Wrapper.convert_obs(obs)
        self.frames.append(obs)

        if self.clip_reward:
            reward = float(np.sign(reward))

        if reward < 0 and self.episodic_score:
            done = True

        return self.make_stack(), reward, done, info

    def make_stack(self):
        return torch.stack(list(self.frames), dim=0)
