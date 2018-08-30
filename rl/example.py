from collections import deque
from types import SimpleNamespace
import torch
import tinder
import gym
import gym.spaces
from baselines.common.atari_wrappers import make_atari, wrap_deepmind
from .wrapper import Wrapper
from .dqn import AgentTrainer

tinder.bootstrap()

config = SimpleNamespace(
    name='',
    # env='PongNoFrameskip-v4',
    env='PongDeterministic-v4',
    replay_size=100_000,
    mode='train',
    gpu=tinder.config.Placeholder.STR,

    per=True,
    noisy=True,
    dueling=True,
)
tinder.config.override(config)

# make_atari does
# 1. noop 30 for randomness
# 2. repeat 'action' for 4 times
# env = make_atari(config.env)

# # env = gym.make('BreakoutDeterministic-v4')
# # env = gym.make('Breakout-v0')

# env = wrap_deepmind(  # includes 'FireResetEnv'
#     env,
#     episode_life=True,
#     clip_rewards=True,
#     scale=False,  # False for saving memory
#     frame_stack=False,  # this uses LazyFrame
# )

env = gym.make(config.env)
env = Wrapper(env, end_on_negative_score=False)
trainer = AgentTrainer(env, name=config.name, replay_size=config.replay_size,
                       use_prioritized=config.per, dueling=config.dueling, noisy=config.noisy)  # 1M -> >60GB

# PongNoFrameskip-v4: 1M
# Others: >50M
if config.mode == 'train':
    trainer.train(frame_total=2*1_000_000, render_episode_period=10, frame_target_update_period=10_000)
elif config.mode == 'play':
    trainer.play(sleep=0.001)
else:
    raise NotImplementedError

env.close()
