import numpy as np
import random


class replayMemory():
	def __init__(self):
		# self.model_dir = model_dir
		self.memory_size = 1000000
		self.actions = np.empty(self.memory_size, dtype = np.uint8)
		self.rewards = np.empty(self.memory_size, dtype = np.integer)
		self.screens = np.empty((self.memory_size, 84, 84), dtype = np.float16)
		self.terminals = np.empty(self.memory_size, dtype = np.bool)
		self.history_length = 4
		self.dims = (84, 84)
		self.batch_size = 32
		self.count = 0
		self.current = 0

		# pre-allocate prestates and poststates for minibatch training
		self.prestates = np.empty((self.batch_size, self.history_length) + self.dims, dtype = np.float16)
		self.poststates = np.empty((self.batch_size, self.history_length) + self.dims, dtype = np.float16)
		
	def addMemory(self, screen, reward, action, terminal):
		assert screen.shape == self.dims
		# screen is post-state, after action and reward
		self.actions[self.current] = action
		self.rewards[self.current] = reward
		self.screens[self.current, ...] = screen
		self.terminals[self.current] = terminal
		self.count = max(self.count, self.current + 1)
		self.current = (self.current + 1) % self.memory_size

	def getState(self, index):
		assert self.count > 0, "replay memory is empy, use at least --random_steps 1"
		# normalize index to expected range, allows negative indexes
		index = index % self.count
		# if is not in the beginning of matrix
		if index >= self.history_length - 1:
			# use faster slicing
			return self.screens[(index - (self.history_length - 1)):(index + 1), ...]
		else:
			# otherwise normalize indexes and use slower list based access
			indexes = [(index - i) % self.count for i in reversed(range(self.history_length))]
			return self.screens[indexes, ...]

	def getRandomMemory(self):
		''' gets a batch of states and rewards from the memory 
			memory must include poststate, prestate and history '''
		
		assert self.count > self.history_length

		# sample random indexes
		indexes = []
		while len(indexes) < self.batch_size:
			# find random index 
			while True:
			# sample one index (ignore states wraping over)
				index = random.randint(self.history_length, self.count - 1)
				# if wraps over current pointer, then get new one
				if index >= self.current and index - self.history_length < self.current:
					continue
				# if wraps over episode end, then get new one
				# poststate (last screen) can be terminal state!
				if self.terminals[(index - self.history_length):index].any():
					continue
				# otherwise use this index
				break

			# NB! having index first is fastest in C-order matrices
			self.prestates[len(indexes), ...] = self.getState(index - 1)
			self.poststates[len(indexes), ...] = self.getState(index)
			indexes.append(index)

		actions = self.actions[indexes]
		rewards = self.rewards[indexes]
		terminals = self.terminals[indexes]

		# if self.cnn_format == 'NHWC':
		# 	return np.transpose(self.prestates, (0, 2, 3, 1)), actions, \
		# 	rewards, np.transpose(self.poststates, (0, 2, 3, 1)), terminals
		# else:
		# print(len(indexes))

		return self.prestates, actions, rewards, self.poststates, terminals

	def save(self):
		for idx, (name, array) in enumerate(
			zip(['actions', 'rewards', 'screens', 'terminals', 'prestates', 'poststates'],
				[self.actions, self.rewards, self.screens, self.terminals, self.prestates, self.poststates])):
			save_npy(array, os.path.join(self.model_dir, name))

	def load(self):
		for idx, (name, array) in enumerate(
			zip(['actions', 'rewards', 'screens', 'terminals', 'prestates', 'poststates'],
				[self.actions, self.rewards, self.screens, self.terminals, self.prestates, self.poststates])):
			array = load_npy(os.path.join(self.model_dir, name))