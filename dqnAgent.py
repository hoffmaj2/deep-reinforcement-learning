from model import baseCNN
from replayMemory import replayMemory
import numpy as np
import random
from scipy.misc import imresize
import scipy.signal
from collections import namedtuple
import tensorflow as tf
import timeit
import time
#
# something something experience replay something
#
# epsilon value and random action
# lifetime tune epsilon down?
#

def discount(x, gamma):
    return scipy.signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]

def process_rollout(rollout, gamma, lambda_=1.0):
    """
given a rollout, compute its returns and the advantage
"""
    batch_si = np.asarray(rollout.states)
    batch_a = np.asarray(rollout.actions)
    rewards = np.asarray(rollout.rewards)
    vpred_t = np.asarray(rollout.values + [rollout.r])

    rewards_plus_v = np.asarray(rollout.rewards + [rollout.r])
    batch_r = discount(rewards_plus_v, gamma)[:-1]
    delta_t = rewards + gamma * vpred_t[1:] - vpred_t[:-1]
    # this formula for the advantage comes "Generalized Advantage Estimation":
    # https://arxiv.org/abs/1506.02438
    batch_adv = discount(delta_t, gamma * lambda_)

    features = rollout.features[0]
    return Batch(batch_si, batch_a, batch_adv, batch_r, rollout.terminal, features)

Batch = namedtuple("Batch", ["si", "a", "adv", "r", "terminal", "features"])


class PartialRollout(object):
    """
a piece of a complete rollout.  We run our agent, and process its experience
once it has processed enough steps.
"""
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.r = 0.0
        self.terminal = False
        self.features = []

    def add(self, state, action, reward, value, terminal, features):
        self.states += [state]
        self.actions += [action]
        self.rewards += [reward]
        self.values += [value]
        self.terminal = terminal
        self.features += [features]

    def extend(self, other):
        assert not self.terminal
        self.states.extend(other.states)
        self.actions.extend(other.actions)
        self.rewards.extend(other.rewards)
        self.values.extend(other.values)
        self.r = other.r
        self.terminal = other.terminal
        self.features.extend(other.features)

class DQNAgent():
	def __init__(self, ID):
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
		self.value = 0

		self.network = baseCNN(self.numActions)
		self.memory = replayMemory()
		self.rollout = PartialRollout()
		self.prevaction = 0
		self.epsilon = 0.1

		self.total_loss = 0
		self.total_q = 0
		self.update_count = 0

		self.learning_rate = 0.00025
		self.actionCount = 0
		self.ID = ID
		self.trainFreq = 20
		self.target_q_update_step = 1 * 10000
		self.learn_start = 50000
		self.ep_start = 1
		self.ep_end = 0.1
		self.ep_end_t = 1000000

		self.discount = 0.99

		self.cumulatedReward = 0
		self.initMath()
		self.network.initialize()
		self.sess = self.network.getSess()

	def initMath(self):
		# self.env = env
		# self.task = task
		# worker_device = "/job:worker/task:{}/cpu:0".format(task)
		# with tf.device(tf.train.replica_device_setter(1, worker_device=worker_device)):
		#     with tf.variable_scope("global"):
		#         self.network = LSTMPolicy(env.observation_space.shape, env.action_space.n)
		#         self.global_step = tf.get_variable("global_step", [], tf.int32, initializer=tf.constant_initializer(0, dtype=tf.int32),
		#                                            trainable=False)

		# with tf.device(worker_device):
		#     with tf.variable_scope("local"):
		#         self.local_network = pi = LSTMPolicy(env.observation_space.shape, env.action_space.n)
		#         pi.global_step = self.global_step

		self.ac = tf.placeholder(tf.float32, [None, self.numActions], name="ac")
		self.adv = tf.placeholder(tf.float32, [None], name="adv")
		self.r = tf.placeholder(tf.float32, [None], name="r")

		log_prob_tf = tf.nn.log_softmax(self.network.out)
		prob_tf = tf.nn.softmax(self.network.out)

		# the "policy gradients" loss:  its derivative is precisely the policy gradient
		# notice that self.ac is a placeholder that is provided externally.
		# adv will contain the advantages, as calculated in process_rollout
		pi_loss = - tf.reduce_sum(tf.reduce_sum(log_prob_tf * self.ac, [1]) * self.adv)
		self.pi_loss = - tf.reduce_sum(tf.reduce_sum(log_prob_tf * self.ac, [1]) * self.adv)

		# loss of value function
		self.vf_loss = 0.5 * tf.reduce_sum(tf.square(self.network.value - self.r))
		entropy = - tf.reduce_sum(prob_tf * log_prob_tf)

		# bs = tf.to_float(tf.shape(pi.x)[0])
		self.loss = self.pi_loss + 0.5 * self.vf_loss - entropy * 0.01


		grads = tf.gradients(self.loss, self.network.var_list)

		grads, _ = tf.clip_by_global_norm(grads, 40.0)

		# copy weights from the parameter server to the local model
		# self.sync = tf.group(*[v1.assign(v2) for v1, v2 in zip(pi.var_list, self.network.var_list)])

		grads_and_vars = list(zip(grads, self.network.var_list))
		# inc_step = self.actionCount.assign_add(tf.shape(self.network.state)[0])
		inc_step = tf.Variable(20, trainable=False)
		# inc_step = 20

		# each worker has a different set of adam optimizer parameters
		opt = tf.train.AdamOptimizer(1e-4)
		# opt = tf.train.RMSPropOptimizer(learning_rate=7e-4,decay=0.99,epsilon=0.1)
		self.train_op = tf.group(opt.apply_gradients(grads_and_vars), inc_step)
		# self.summary_writer = None
		self.summary_writer = tf.summary.FileWriter("train_"+str(self.ID))
		self.local_steps = 0


		# if use_tf12_api:
		bs = 20.0
		tf.summary.scalar("model/policy_loss", self.pi_loss / bs)
		tf.summary.scalar("model/value_loss", self.vf_loss / bs)
		tf.summary.scalar("model/entropy", entropy / bs)
		# tf.summary.image("model/state", self.history[-1])
		tf.summary.scalar("model/grad_global_norm", tf.global_norm(grads))
		tf.summary.scalar("model/var_global_norm", tf.global_norm(self.network.var_list))
		self.summary_op = tf.summary.merge_all()


	def reshape(self, img):
		return imresize(self.rgb2gray(img)/255., self.dims)

	def rgb2gray(self, img):
		return np.dot(img[...,:3], [0.299, 0.587, 0.114])

	def step(self, obs, reward, done, training = True):
		''' record the screen into memory
			use the network to compute an action to take
			take a random experience from memory and train the network'''
		# start = timeit.timeit()
		screen = self.reshape(obs)
		self.history[:-1] = self.history[1:]
		self.history[-1] = screen

		# self.cumulatedReward = reward + self.discount*self.cumulatedReward
		self.cumulatedReward += reward
		# self.memory.addMemory(screen, self.cumulatedReward, self.prevaction, done)
		# self.rollout.add(screen, self.prevaction, self.cumulatedReward, self.value, done, [0]) # last_features
		# prevaction = tf.one_hot(self.prevaction, self.numActions).eval(session=self.sess)
		prevaction = np.zeros(self.numActions)
		prevaction[self.prevaction] = 1
		self.rollout.add(np.copy(self.history), prevaction, reward, self.value, done, [0]) # last_features
		# Update from previous step
		if training and self.actionCount > 35 and self.actionCount % self.trainFreq == 0 :
			# self.q_learning_mini_batch()
			self.advantageUpdate()

		# every so often, update the target network for stability
		# if self.actionCount % self.target_q_update_step == 0:
		# 	print('updating target network, check to see if it is different?')
		# 	self.network.update_target_network()
			

		action = 0
		value = 0
		# Take action
		action, value = self.predict(self.history)
		# print(value)
		self.value = value
		# value = self.value()
		self.prevaction = action
		self.actionCount += 1
		# end = timeit.timeit()
		# print(end-start)
		return action


	def reset(self):
		''' On new episode, reset history and possible state of the NN '''
		self.history = np.zeros([self.historyLength, self.screen_height, self.screen_width])
		self.cumulatedReward = 0
		ep = (self.ep_end +
        max(0., (self.ep_start - self.ep_end) * (self.ep_end_t - max(0., self.actionCount - self.learn_start)) / self.ep_end_t))
		# print("epsilon is",ep)
		

	def trainNet(self,):
		''' Take a random experience from the replay memory
			and use it update the network'''
		pass

	def predict(self, history):
		action = 0
		value = 0
		ep = (self.ep_end +
        max(0., (self.ep_start - self.ep_end) * (self.ep_end_t - max(0., self.actionCount - self.learn_start)) / self.ep_end_t))

		# start = time.time()
		fetches = [self.network.q, self.network.value]
		feed_dict = {self.network.state: [history]}
		fetched = self.sess.run(fetches, feed_dict=feed_dict)
		# print(fetched[0])
		action = np.random.choice(range(self.numActions), p=fetched[0][0])
		# print(action)
		value = fetched[1][0]
		# end = time.time()
		# print(end - start)
		# action = self.network.q_action.eval({self.network.state: [history]},session=self.sess)[0]
		# value = self.network.value.eval({self.network.state: [history]},session=self.sess)[0]

		ep = 0.1
		if random.random() < ep:
			action = random.randrange(self.numActions)

		return action, value

	def randomAction(self, history):
		return random.choice(range(self.numActions))

	def pull_batch(self):
		#TODO Change to a more asynch or random pull of batches?
		return self.rollout

	def advantageUpdate(self):
		# sess.run(self.sync)  # copy weights from shared to local
		rollout = self.pull_batch()
		action, value = self.predict(self.history)
		rollout.r = value
		# print(rollout.rewards)
		batch = process_rollout(rollout, gamma=0.99, lambda_=1.0)

		should_compute_summary = self.local_steps % 11 == 0
		# should_compute_summary = False
		if should_compute_summary:
			fetches = [self.summary_op, self.train_op, self.network.value]
		else:
			fetches = [self.train_op, self.network.value]#, self.actionCount]
		# batch_size = len(batch.a)
		# something = np.zeros([batch_size,6])
		# print(batch.a)
		# print("r",batch.r)
		# print("adv",batch.adv)
		feed_dict = {
			self.network.state: batch.si,
			self.ac: batch.a, # batch.a
			self.adv: batch.adv,
			self.r: batch.r
			# self.local_network.state_in[0]: batch.features[0],
			# self.local_network.state_in[1]: batch.features[1],
		}
		# fetches = [self.pi_loss]
		fetched = self.sess.run(fetches, feed_dict=feed_dict)

		# print("network value",fetched[-1])
		# print("r",batch.r)

		if should_compute_summary:
			self.summary_writer.add_summary(tf.Summary.FromString(fetched[0]), self.actionCount)
			self.summary_writer.flush()
		self.local_steps += 1
		self.rollout = PartialRollout()


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