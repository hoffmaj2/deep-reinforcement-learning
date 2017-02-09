import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import scale

class batch_norm(object):
    """Code modification of http://stackoverflow.com/a/33950177"""
    def __init__(self, epsilon=1e-5, momentum = 0.9, name="batch_norm"):
        with tf.variable_scope(name):
            self.epsilon = epsilon
            self.momentum = momentum

            self.ema = tf.train.ExponentialMovingAverage(decay=self.momentum)
            self.name = name

    def __call__(self, x, train=True):
        shape = x.get_shape().as_list()

        if train:
            with tf.variable_scope(self.name) as scope:
                self.beta = tf.get_variable("beta", [shape[-1]],
                                    initializer=tf.constant_initializer(0.))
                self.gamma = tf.get_variable("gamma", [shape[-1]],
                                    initializer=tf.random_normal_initializer(1., 0.02))
                
                try:
                    batch_mean, batch_var = tf.nn.moments(x, [0, 1, 2], name='moments')
                except:
                    batch_mean, batch_var = tf.nn.moments(x, [0, 1], name='moments')
                    
                ema_apply_op = self.ema.apply([batch_mean, batch_var])
                self.ema_mean, self.ema_var = self.ema.average(batch_mean), self.ema.average(batch_var)

                with tf.control_dependencies([ema_apply_op]):
                    mean, var = tf.identity(batch_mean), tf.identity(batch_var)
        else:
            mean, var = self.ema_mean, self.ema_var

        normed = tf.nn.batch_norm_with_global_normalization(
                x, mean, var, self.beta, self.gamma, self.epsilon, scale_after_normalization=True)

        return normed

def normalized_columns_initializer(std=1.0):
    def _initializer(shape, dtype=None, partition_info=None):
        out = np.random.randn(*shape).astype(np.float32)
        out *= std / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
        return tf.constant(out)
    return _initializer

def conv2d(input_, output_dim, 
           k_h=5, k_w=5, d_h=1, d_w=1, stddev=0.01,
           name="conv2d"):
    with tf.variable_scope(name):
        w = tf.get_variable('w', [k_h, k_w, input_.get_shape()[-1], output_dim],
                            initializer=tf.truncated_normal_initializer(stddev=stddev))
        # tf.get_variable_scope().reuse_variables()
        conv = tf.nn.conv2d(input_, w, strides=[1, d_h, d_w, 1], padding='SAME')

        biases = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0))
        print(conv.get_shape())
        theshape = [[-1], conv.get_shape().as_list()[1:]]
        newshape = [item for sublist in theshape for item in sublist]
        print(newshape)
        conv = tf.reshape(tf.nn.bias_add(conv, biases), newshape)#conv.get_shape()

        return conv

def linear(input_, output_size, scope=None, stddev=0.01, bias_start=0.0, with_w=False):
    shape = input_.get_shape().as_list()

    with tf.variable_scope(scope or "Linear"):
        matrix = tf.get_variable("Matrix", [shape[1], output_size], tf.float32,
                                 normalized_columns_initializer(stddev))
        bias = tf.get_variable("bias", [output_size],
            initializer=tf.constant_initializer(bias_start))
        if with_w:
            return tf.matmul(input_, matrix) + bias, matrix, bias
        else:
            return tf.matmul(input_, matrix) + bias

def lrelu(x, leak=0.2, name="lrelu"):
    return tf.maximum(x, leak*x)

def clipped_error(x):
	# Huber loss
	try:
		return tf.select(tf.abs(x) < 1.0, 0.5 * tf.square(x), tf.abs(x) - 0.5)
	except:
		return tf.where(tf.abs(x) < 1.0, 0.5 * tf.square(x), tf.abs(x) - 0.5)

class baseCNN():
	def __init__(self, numActions):
		''' define our basic convolutional network as described in the original paper
			This will include 4 screeens and basic actions '''

		# variable setup
		# --------------------------------------
		self.batch_size = None # 32
		self.numActions = numActions
		# X = tf.constant(images_valid.astype('float32'))
		# Y = tf.constant(labels_valid.astype('float32'))
		self.state = tf.placeholder(tf.float32,[self.batch_size,4,84,84])
		self.target_q_t = tf.placeholder('float32', [None], name='target_q_t')
		self.action = tf.placeholder('int64', [None], name='action')
		self.target_state_t = tf.placeholder('float32', [self.batch_size,4,84,84], name='target_s_t')

		# Online Network
		# --------------------------------------
		# c_bn0 = batch_norm(name='c_bn0')
		# c_bn1 = batch_norm(name='c_bn1')
		# c_bn2 = batch_norm(name='c_bn2')
		# c_bn3 = batch_norm(name='c_bn3')
		# c_bn4 = batch_norm(name='c_bn4')
		# c_bn5 = batch_norm(name='c_bn5')


		conv0 = lrelu(conv2d(self.state, 32, k_h=7, k_w=7, d_h=2, d_w=2,  name="c_conv0"))
		conv1 = lrelu(conv2d(conv0, 64, k_h=4, k_w=4, d_h=2, d_w=2, name="c_conv1"))
		conv2 = lrelu(conv2d(conv1, 64, k_h=4, k_w=4, name="c_conv2"))
		# conv3 = lrelu(c_bn4(conv2d(conv2, 32, k_h=7, k_w=7, name="c_conv3")))
		# flattened = tf.reshape(conv2, [self.batch_size, -1])
		shape = conv2.get_shape().as_list()[1:]

		# newshape = [item for sublist in [[-1], shape] for item in sublist]
		# print(newshape)
		newshape = 1
		for dimension in shape:
			newshape = newshape*dimension
		flattened = tf.reshape(conv2, [-1, newshape])
		print(flattened.get_shape())
		fc1 = lrelu(linear(flattened, 256, 'c_fully_connected'))
		self.out = tf.nn.sigmoid(linear(fc1, self.numActions, 'c_out_layer'))

		self.q = tf.nn.softmax(self.out)
		self.value = tf.reshape(linear(fc1, 1, 'c_value', stddev=1.0), [-1])

		# --------------------------------------

		# Other functios

		# --------------------------------------

		self.q_action = tf.argmax(self.q, dimension=1)
		

		avg_q = tf.reduce_mean(self.q, 0)

		action_one_hot = tf.one_hot(self.action, self.numActions, 1.0, 0.0, name='action_one_hot')
		q_acted = tf.reduce_sum(self.q * action_one_hot, reduction_indices=1, name='q_acted')

		# self.delta = self.target_q_t - q_acted
		self.global_step = tf.Variable(0, trainable=False)

		# self.loss = tf.reduce_mean(clipped_error(self.delta), name='loss')
		# self.learning_rate_step = tf.placeholder('int64', None, name='learning_rate_step')
		# self.learning_rate = 0.00025
		# self.learning_rate_minimum = 0.00025
		# self.learning_rate_decay = 0.96
		# self.learning_rate_decay_step = 5 * 10000
		# self.learning_rate_op = tf.maximum(self.learning_rate_minimum,
		# 	tf.train.exponential_decay(
		# 		self.learning_rate,
		# 		self.learning_rate_step,
		# 		self.learning_rate_decay_step,
		# 		self.learning_rate_decay,
		# 		staircase=True))
		# self.optim = tf.train.RMSPropOptimizer(
		# 	self.learning_rate_op, momentum=0.95, epsilon=0.01).minimize(self.loss)


		# Later for organizing graphs of q values
		# q_summary = []
		# avg_q = tf.reduce_mean(self.q, 0)
		# for idx in range(self.numActions):
		# 	q_summary.append(tf.summary.histogram('q/%s' % idx, avg_q[idx]))
		# self.q_summary = tf.summary.merge(q_summary, 'q_summary')


		# loss = tf.reduce_mean(fun)
		# #loss = tf.reduce_mean(tf.square(fun - Y))
		# t_vars = tf.trainable_variables()
		# c_vars = [var for var in t_vars if 'c_' in var.name]

		# for var in c_vars:
		#     print(var.name)

		# train = tf.train.AdamOptimizer(0.1).minimize(loss, var_list=c_vars)
		# mse = tf.reduce_mean(tf.square(tf.nn.softmax(h_0) - Y))
		# accuracy = tf.reduce_sum(tf.abs(tf.arg_max(h2_final,dimension=1) - tf.arg_max(y, dimension=1)))
		# self.initialize()
		self.var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, tf.get_variable_scope().name)
		# self.update_target_network()

	def initialize(self):
		init = tf.initialize_all_variables()
		config = tf.ConfigProto(log_device_placement=True)
		# config = tf.ConfigProto(device_filters=["/job:ps", "/gpu:0"])
		# config = tf.ConfigProto(device_filters=["/job:ps", "/job:worker/task:{}/gpu:0".format(args.task)])
		config.gpu_options.allow_growth = True
		self.sess = tf.Session(config=config)
		self.sess.run(init)

	def update_target_network(self):
		t_vars = tf.trainable_variables()
		c_vars = [var for var in t_vars if 'c_' in var.name]
		target_vars = [var for var in t_vars if 't_' in var.name]
		target_dict = {}
		for var in target_vars:
			target_dict[var.name] = var

		for var in c_vars:
			# print(var.assign())
			# print(var.name, 't' + var.name[1:])
			target_dict['t' + var.name[1:]].assign(var)

		# with tf.variable_scope('pred_to_target'):
		# 	self.t_w_input = {}
		# 	self.t_w_assign_op = {}

		# for name in self.w.keys():

		# 	self.t_w_input[name] = tf.placeholder('float32', self.t_w[name].get_shape().as_list(), name=name)
		# 	self.t_w_assign_op[name] = self.t_w[name].assign(self.t_w_input[name])

		# for name in self.w.keys():
		# 	self.t_w_assign_op[name].eval({self.t_w_input[name]: self.w[name].eval()})


	def getSess(self):
		return self.sess



	def predict(self,):
		pass

