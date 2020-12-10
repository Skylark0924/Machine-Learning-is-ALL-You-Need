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
        self.state_input = self.input_dim - 1 + state_dim

        # TO BE SET ON A SPECIFIC PROBLEM
        self.state_l1 = 15
        self.state_l2 = self.state_dim
        self.output_l1 = 10
        self.output_l2 = self.output_dim

        # list of weights
        # self.weights = {'State_L1': weight_variable([self.state_input, self.state_l1], "WEIGHT_STATE_L1"),
        #                 'State_L2': weight_variable([ self.state_l1, self.state_l2], "WEIGHT_STATE_L1"),
        #
        #                 'Output_L1':weight_variable([self.state_l2,self.output_l1], "WEIGHT_OUTPUT_L1"),
        #                 'Output_L2': weight_variable([self.output_l1, self.output_l2], "WEIGHT_OUTPUT_L2")
        #                 }
        #
        # # list of biases
        # self.biases = {'State_L1': weight_variable([self.state_l1],"BIAS_STATE_L1"),
        #                'State_L2': weight_variable([self.state_l2], "BIAS_STATE_L2"),
        #
        #                'Output_L1':weight_variable([self.output_l1],"BIAS_OUTPUT_L1"),
        #                'Output_L2': weight_variable([ self.output_l2], "BIAS_OUTPUT_L2")
        #                }

    # def netSt(self, inp):
    #     with tf.variable_scope('State_net'):
    #         # method to define the architecture of the state network
    #         layer1 = tf.nn.tanh(tf.add(tf.matmul(inp,self.weights["State_L1"]),self.biases["State_L1"]))
    #         layer2 = tf.nn.tanh(tf.add(tf.matmul(layer1, self.weights["State_L2"]), self.biases["State_L2"]))
    #
    #         return layer2
    #
    # def netOut(self, inp):
    #     # method to define the architecture of the output network
    #     with tf.variable_scope('Out_net'):
    #         layer1 = tf.nn.tanh(tf.add(tf.matmul(inp, self.weights["Output_L1"]), self.biases["Output_L1"]))
    #         layer2 = tf.nn.softmax(tf.add(tf.matmul(layer1, self.weights["Output_L2"]), self.biases["Output_L2"]))
    #
    #         return layer2

    def netSt(self, inp):
        with tf.variable_scope('State_net'):
            # # method to define the architecture of the state network
            # layer1 = tf.nn.tanh(tf.add(tf.matmul(inp, self.weights["State_L1"]), self.biases["State_L1"]))
            # layer2 = tf.nn.tanh(tf.add(tf.matmul(layer1, self.weights["State_L2"]), self.biases["State_L2"]))

            layer1 = tf.layers.dense(inp, self.state_l1, activation=tf.nn.tanh)
            layer2 = tf.layers.dense(layer1, self.state_l2, activation=tf.nn.tanh)

            return layer2

    def netOut(self, inp):
        # method to define the architecture of the output network
        # with tf.variable_scope('Out_net'):
        #     layer1 = tf.nn.tanh(tf.add(tf.matmul(inp, self.weights["Output_L1"]), self.biases["Output_L1"]))
        #     layer2 = tf.nn.softmax(tf.add(tf.matmul(layer1, self.weights["Output_L2"]), self.biases["Output_L2"]))
            layer1 = tf.layers.dense(inp, self.output_l1, activation=tf.nn.tanh)
            layer2 = tf.layers.dense(layer1, self.output_l2, activation=tf.nn.softmax)

            return layer2

    def Loss(self, output, target, output_weight=None):
        # method to define the loss function
        output = tf.maximum(output, self.EPSILON, name="Avoiding_explosions")  # to avoid explosions
        xent = -tf.reduce_sum(target * tf.log(output), 1)
        lo = tf.reduce_mean(xent)

        return lo

    def Metric(self, target, output, output_weight=None):
        # method to define the evaluation metric
        correct_prediction = tf.equal(tf.argmax(output, 1), tf.argmax(target, 1))
        metric = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        return metric