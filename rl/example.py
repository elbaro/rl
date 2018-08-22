import tinder
import gym
import gym.spaces
from .dqn import AtariPlayer, AgentTrainer

tinder.bootstrap()

env = gym.make('Pong-v0')
trainer = AgentTrainer(env, replay_size=1_000_000)
trainer.train(frame_total=10*1_000_000, render_episode_period=10, batch_size=32)
