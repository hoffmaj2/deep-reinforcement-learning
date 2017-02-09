import gym
from gym import wrappers
import universe  # register the universe environments
import tensorflow as tf
import numpy as np
from dqnAgent import DQNAgent

import timeit
# Possible mistakes:
#
# ordering of action, then update observation and reward
# Length of experience replay
#
# change sigmoid to linear, because there can be negatives, and because loss clipping is already there


# Notes:
# 
# take action space as a parameter?

# Thin the network down with striding?

# modify agents to access parameter server to be able to work in parallel
#	Will the experience replay have to be asynchronous as well?

# ability to do multiple actions?
#	This will include doing some other function than taking the argmax of the last layer of the network
#	This could just be a .50 threshold, may cause some instability in training with the mashing of all actions
def train():
	env = gym.make('PongDeterministic-v3') #Breakout-v0
	env = wrappers.Monitor(env, 'recordings/pong_experiment', force=True)
	# env.configure(remotes=1)  # automatically creates a local docker container

	print(env.action_space.n)


	observation = env.reset()
	reward = 0
	done = False

	print(np.array(observation).shape)

	agent = DQNAgent('2')

	epochs = 0
	timesteps = 0

	while True:
		while True:
			# start = timeit.timeit()
			# action_n = [[('KeyEvent', 'ArrowUp', True)] for ob in observation_n]  # your agent here

			# action = env.action_space.sample()
			# start = timeit.timeit()
			action = agent.step(observation, reward, done)
			# end = timeit.timeit()
			# print("agent step", end - start)

			# start = timeit.timeit()
			observation, reward, done, info = env.step(action)
			# end = timeit.timeit()
			# print("env step", end - start)
			# env.render(mode='rgb_array', close=False)
			timesteps += 1

			if done:
				# reset Agent
				score = agent.cumulatedReward
				observation = env.reset()
				agent.reset()
				reward = 0
				done = False
				print("Episode finished after {} timesteps with reward {}".format(timesteps, score))
				epochs += 1
				timesteps = 0
				break

if __name__ == "__main__":
	train()
