import tensorflow as tf
import numpy as np


def weight_variable(shape, nm):
    # function to initialize weights
    initial = tf.truncated_normal(shape, stddev=0.1)
    tf.summary.histogram(nm, initial, collections=['always'])
    return tf.Variable(initial, name=nm)


class Net:
    # class to define state and output network

    def __init__(self, input_dim, state_dim, output_dim):
        # initialize weight and parameter

        self.EPSILON = 0.00000001

        self.input_dim = input_dim
        self.state_dim = state_dim
        self.output_dim = output_dim
        self.state_input = self.input_dim - 1 + state_dim  # removing the id_ dimension

        #### TO BE SET ON A SPECIFIC PROBLEM
        self.state_l1 = 5
        self.state_l2 = self.state_dim

        self.output_l1 = 5
        self.output_l2 = self.output_dim

    def netSt(self, inp):
        with tf.variable_scope('State_net'):

            layer1 = tf.layers.dense(inp, self.state_l1, activation=tf.nn.tanh)
            layer2 = tf.layers.dense(layer1, self.state_l2, activation=tf.nn.tanh)

            return layer2

    def netOut(self, inp):

            layer1 = tf.layers.dense(inp, self.output_l1, activation=tf.nn.tanh)
            layer2 = tf.layers.dense(layer1, self.output_l2, activation=tf.nn.softmax)

            return layer2

    def Loss(self, output, target, output_weight=None, mask=None):
        # method to define the loss function
        #lo = tf.losses.softmax_cross_entropy(target, output)
        output = tf.maximum(output, self.EPSILON, name="Avoiding_explosions")  # to avoid explosions
        xent = -tf.reduce_sum(target * tf.log(output), 1)

        mask = tf.cast(mask, dtype=tf.float32)
        mask /= tf.reduce_mean(mask)
        xent *= mask
        lo = tf.reduce_mean(xent)
        return lo

    def Metric(self, target, output, output_weight=None, mask=None):
        # method to define the evaluation metric

        correct_prediction = tf.equal(tf.argmax(output, 1), tf.argmax(target, 1))
        accuracy_all = tf.cast(correct_prediction, tf.float32)
        mask = tf.cast(mask, dtype=tf.float32)
        mask /= tf.reduce_mean(mask)
        accuracy_all *= mask

        return tf.reduce_mean(accuracy_all)
