from model import baseCNN
from replayMemory import replayMemory
import numpy as np
import random
from scipy.misc import imresize

#
# something something experience replay something
#
# epsilon value and random action
# lifetime tune epsilon down?
#

class DQNAgent():
	def __init__(self):
		''' Load the Neural network into memory
			initialize replay memory (synchronized with other workers?)
			Have a rolling buffer for the last 4 frames	'''

		# print(np.array(initialobs).shape)
		self.screen_width  = 84
		self.screen_height = 84
		self.dims = (self.screen_width, self.screen_height)

		self.numActions = 6

		self.historyLength = 4
		self.history = np.zeros([self.historyLength, self.screen_height, self.screen_width])

		self.network = baseCNN(self.numActions)
		self.sess = self.network.getSess()
		self.memory = replayMemory()
		self.prevaction = 0
		self.epsilon = 0.1

		self.total_loss = 0
		self.total_q = 0
		self.update_count = 0

		self.learning_rate = 0.00025
		self.actionCount = 0

		self.trainFreq = 4
		self.target_q_update_step = 1 * 10000
		self.learn_start = 50000
		self.ep_start = 1
		self.ep_end = 0.1
		self.ep_end_t = 1000000

		self.discount = 0.99

		self.cumulatedReward = 0

	def reshape(self, img):
		return imresize(self.rgb2gray(img)/255., self.dims)

	def rgb2gray(self, img):
		return np.dot(img[...,:3], [0.299, 0.587, 0.114])

	def step(self, obs, reward, done, training = True):
		''' record the screen into memory
			use the network to compute an action to take
			take a random experience from memory and train the network'''
		screen = self.reshape(obs)
		self.history[:-1] = self.history[1:]
		self.history[-1] = screen

		self.cumulatedReward -= 1 + reward
		self.memory.addMemory(screen, self.cumulatedReward, self.prevaction, done)
		# Update from previous step
		if training and self.actionCount > 35 and self.actionCount % self.trainFreq == 0:
			self.q_learning_mini_batch()

		# every so often, update the target network for stability
		if self.actionCount % self.target_q_update_step == 0:
			print('updating target network, check to see if it is different?')
			self.network.update_target_network()
			


		# Take action
		action = self.predict(self.history)
		self.prevaction = action
		self.actionCount += 1
		return action


	def reset(self):
		''' On new episode, reset history and possible state of the NN '''
		self.history = np.zeros([self.historyLength, self.screen_height, self.screen_width])
		ep = (self.ep_end +
        max(0., (self.ep_start - self.ep_end) * (self.ep_end_t - max(0., self.actionCount - self.learn_start)) / self.ep_end_t))
		print("epsilon is",ep)
		

	def trainNet(self,):
		''' Take a random experience from the replay memory
			and use it update the network'''
		pass

	def predict(self, history):
		action = 0
		ep = (self.ep_end +
        max(0., (self.ep_start - self.ep_end) * (self.ep_end_t - max(0., self.actionCount - self.learn_start)) / self.ep_end_t))


		if random.random() < ep:
			action = random.randrange(self.numActions)
		else:
			action = self.network.q_action.eval({self.network.state: [history]},session=self.sess)[0]

		return action

	def randomAction(self, history):
		return random.choice(range(self.numActions))


	def q_learning_mini_batch(self):
		if self.memory.count < self.historyLength:
			return
		else:
			state_t, action, reward, state_t_plus_1, terminal = self.memory.getRandomMemory()

		# t = time.time()

		q_t_plus_1 = self.network.target_q.eval({self.network.target_state_t: state_t_plus_1},session=self.sess)

		terminal = np.array(terminal) + 0.
		max_q_t_plus_1 = np.max(q_t_plus_1, axis=1)
		target_q_t = (1. - terminal) * self.discount * max_q_t_plus_1 + reward

		_, q_t, loss, summary_str = self.sess.run([self.network.optim, self.network.q, self.network.loss, self.network.q_summary], {
			self.network.target_q_t: target_q_t,
			self.network.action: action,
			self.network.state: state_t,
			self.network.learning_rate_step: self.actionCount,
		})

		# self.writer.add_summary(summary_str, self.step)
		self.total_loss += loss
		self.total_q += q_t.mean()
		self.update_count += 1