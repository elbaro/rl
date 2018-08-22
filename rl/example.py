import tinder
# import gym
# import gym.spaces
from baselines.common.atari_wrappers import make_atari, wrap_deepmind, WarpFrame
from .dqn import AtariPlayer, AgentTrainer

tinder.bootstrap()

"""[summary]
class NoopResetEnv(gym.Wrapper):
class FireResetEnv(gym.Wrapper):
class EpisodicLifeEnv(gym.Wrapper):
class MaxAndSkipEnv(gym.Wrapper):
class ClipRewardEnv(gym.RewardWrapper):
class WarpFrame(gym.ObservationWrapper):
class FrameStack(gym.Wrapper):
class ScaledFloatFrame(gym.ObservationWrapper):
class LazyFrames(object):
"""


# env = gym.make('Pong-v0')
env = make_atari('PongNoFrameskip-v0')
env = wrap_deepmind(
    env,
    episode_life=True,
    clip_rewards=False,
    scale=True,
    frame_stack=False,
)
trainer = AgentTrainer(env, replay_size=1_000_000)
trainer.train(frame_total=10*1_000_000, render_episode_period=10, batch_size=32)
