from collections import deque
from types import SimpleNamespace
import torch
import tinder
import gym
import gym.spaces

# from baselines.common.atari_wrappers import make_atari, wrap_deepmind
from .wrapper import Wrapper
from .dqn import AgentTrainer

tinder.bootstrap()

config = SimpleNamespace(
    name="",
    # env='PongDeterministic-v4',
    # env='SpaceInvadersDeterministic-v4',
    env="Hopper-v1",
    mode="train",
    gpu=tinder.config.Placeholder.STR,
    replay_size=300_000,
    frame_exploration=1 * 1_000_000,
    frame=10 * 1_000_000,
    per=True,
    noisy=True,
    dueling=True,
    clip=True,
    episodic_life=True,
    episodic_score=False,
    delay=0.01,
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
env = Wrapper(
    env,
    episodic_life=config.episodic_life,
    episodic_score=config.episodic_score,
    clip_reward=config.clip,
)
trainer = AgentTrainer(
    env,
    name=config.name,
    replay_size=config.replay_size,
    use_prioritized=config.per,
    dueling=config.dueling,
    noisy=config.noisy,
)  # 1M -> >60GB

# PongNoFrameskip-v4: 1M
# Others: >50M
if config.mode == "train":
    trainer.train(
        frame_total=config.frame,
        frame_exploration_count=config.frame_exploration,
        render_episode_period=10,
        frame_target_update_period=10_000,
    )
elif config.mode == "play":
    trainer.play(sleep=config.delay)
else:
    raise NotImplementedError

env.close()
