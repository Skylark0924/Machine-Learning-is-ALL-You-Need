import tensorflow as tf
import numpy as np
import datetime as time

# class for the core of the architecture
class GNN:
    def __init__(self, net,  input_dim, output_dim, state_dim, max_it=50, optimizer=tf.train.AdamOptimizer, learning_rate=0.01, threshold=0.01, graph_based=False,
                 param=str(time.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')), config=None, tensorboard=False, mask_flag=False):
        """
               create GNN instance. Feed this parameters:

               :net:  Net instance - it contains state network, output network, initialized weights, loss function and metric;
               :input_dim: dimension of the input
               :output_dim: dimension of the output
               :state_dim: dimension for the state
               :max_it:  maximum number of iteration of the state convergence procedure
               :optimizer:  optimizer instance
               :learning_rate: learning rate value
               :threshold:  value to establish the state convergence
               :graph_based: flag to denote a graph based problem
               :param: name of the experiment
               :config: ConfigProto protocol buffer object, to set configuration options for a session
               :tensorboard:  boolean flag to activate tensorboard
               """

        np.random.seed(0)
        tf.set_random_seed(0)
        self.tensorboard = tensorboard
        self.max_iter = max_it
        self.net = net
        self.optimizer = optimizer(learning_rate, name="optim")
        self.state_threshold = threshold
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.state_dim = state_dim
        self.graph_based = graph_based
        self.mask_flag = mask_flag
        self.build()

        self.session = tf.Session(config=config)
        #self.session = tf.Session()
        self.session.run(tf.global_variables_initializer())
        self.init_l = tf.local_variables_initializer()

        # parameter to monitor the learning via tensorboard and to save the model
        if self.tensorboard:
            self.merged_all = tf.summary.merge_all(key='always')
            self.merged_train = tf.summary.merge_all(key='train')
            self.merged_val = tf.summary.merge_all(key='val')
            self.writer = tf.summary.FileWriter('tmp/' + param, self.session.graph)
        # self.saver = tf.train.Saver()
        # self.save_path = "tmp/" + param + "saves/model.ckpt"

    def VariableState(self):
        '''Define placeholders for input, output, state, state_old, arch-node conversion matrix'''
        # placeholder for input and output

        self.comp_inp = tf.placeholder(tf.float32, shape=(None, self.input_dim), name="input")
        self.y = tf.placeholder(tf.float32, shape=(None, self.output_dim), name="target")

        if self.mask_flag:
            self.mask = tf.placeholder(tf.float32, name="mask")

        # state(t) & state(t-1)
        self.state = tf.placeholder(tf.float32, shape=(None, self.state_dim), name="state")
        self.state_old = tf.placeholder(tf.float32, shape=(None, self.state_dim), name="old_state")

        # arch-node conversion matrix
        self.ArcNode = tf.sparse_placeholder(tf.float32, name="ArcNode")

        # node-graph conversion matrix
        if self.graph_based:
            self.NodeGraph = tf.sparse_placeholder(tf.float32, name="NodeGraph")
        else:
            self.NodeGraph = tf.placeholder(tf.float32, name="NodeGraph")

    def build(self):
        '''build the architecture, setting variable, loss, training'''
        # network
        self.VariableState()
        self.loss_op = self.Loop()

        # loss
        with tf.variable_scope('loss'):
            if self.mask_flag:
                self.loss = self.net.Loss(self.loss_op[0], self.y, mask=self.mask)
                self.val_loss = self.net.Loss(self.loss_op[0], self.y, mask=self.mask)
            else:
                self.loss = self.net.Loss(self.loss_op[0], self.y)
                # val loss
                self.val_loss = self.net.Loss(self.loss_op[0], self.y)

            if self.tensorboard:
                self.summ_loss = tf.summary.scalar('loss', self.loss, collections=['train'])
                self.summ_val_loss = tf.summary.scalar('val_loss', self.val_loss, collections=['val'])

        # optimizer
        with tf.variable_scope('train'):
            self.grads = self.optimizer.compute_gradients(self.loss)
            self.train_op = self.optimizer.apply_gradients(self.grads, name='train_op')
            if self.tensorboard:
                for index, grad in enumerate(self.grads):
                    tf.summary.histogram("{}-grad".format(self.grads[index][1].name), self.grads[index],
                                         collections=['always'])

        # metrics
        with tf.variable_scope('metrics'):
            if self.mask_flag:
                self.metrics = self.net.Metric(self.y, self.loss_op[0], mask=self.mask)
            else:
                self.metrics = self.net.Metric(self.y, self.loss_op[0])

        # val metric
        with tf.variable_scope('val_metric'):
            if self.mask_flag:
                self.val_met = self.net.Metric(self.y, self.loss_op[0], mask=self.mask)
            else:
                self.val_met = self.net.Metric(self.y, self.loss_op[0])
            if self.tensorboard:
                self.summ_val_met = tf.summary.scalar('val_metric', self.val_met, collections=['always'])

    def convergence(self, a, state, old_state, k):
        with tf.variable_scope('Convergence'):
            # body of the while cicle used to iteratively calculate state

            # assign current state to old state
            old_state = state

            # grub states of neighboring node
            gat = tf.gather(old_state, tf.cast(a[:, 1], tf.int32))

            # slice to consider only label of the node and that of it's neighbor
            # sl = tf.slice(a, [0, 1], [tf.shape(a)[0], tf.shape(a)[1] - 1])
            # equivalent code
            sl = a[:, 2:]

            # concat with retrieved state
            inp = tf.concat([sl, gat], axis=1)

            # evaluate next state and multiply by the arch-node conversion matrix to obtain per-node states
            layer1 = self.net.netSt(inp)
            state = tf.sparse_tensor_dense_matmul(self.ArcNode, layer1)

            # update the iteration counter
            k = k + 1
        return a, state, old_state, k

    def condition(self, a, state, old_state, k):
        # evaluate condition on the convergence of the state
        with tf.variable_scope('condition'):
            # evaluate distance by state(t) and state(t-1)
            outDistance = tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(state, old_state)), 1) + 0.00000000001)
            # vector showing item converged or not (given a certain threshold)
            checkDistanceVec = tf.greater(outDistance, self.state_threshold)

            c1 = tf.reduce_any(checkDistanceVec)
            c2 = tf.less(k, self.max_iter)

        return tf.logical_and(c1, c2)


    def Loop(self):
        # call to loop for the state computation and compute the output
        # compute state
        with tf.variable_scope('Loop'):
            k = tf.constant(0)
            res, st, old_st, num = tf.while_loop(self.condition, self.convergence,
                                                 [self.comp_inp, self.state, self.state_old, k])
            if self.tensorboard:
                self.summ_iter = tf.summary.scalar('iteration', num, collections=['always'])

            if self.graph_based:
                # stf = tf.transpose(tf.matmul(tf.transpose(st), self.NodeGraph))

                stf = tf.sparse_tensor_dense_matmul(self.NodeGraph, st)
            else:
                stf = st
            out = self.net.netOut(stf)

        return out, num

    def Train(self, inputs, ArcNode, target, step, nodegraph=0.0, mask=None):
        ''' train methods: has to receive the inputs, arch-node matrix conversion, target,
        and optionally nodegraph indicator '''

        # Creating a SparseTEnsor with the feeded ArcNode Matrix
        arcnode_ = tf.SparseTensorValue(indices=ArcNode.indices, values=ArcNode.values,
                                        dense_shape=ArcNode.dense_shape)
        if self.graph_based:
            nodegraph = tf.SparseTensorValue(indices=nodegraph.indices, values=nodegraph.values,
                                        dense_shape=nodegraph.dense_shape)

        if self.mask_flag:
            fd = {self.NodeGraph: nodegraph, self.comp_inp: inputs,
                  self.state: np.zeros((ArcNode.dense_shape[0], self.state_dim)),
                  self.state_old: np.ones((ArcNode.dense_shape[0], self.state_dim)),
                  self.ArcNode: arcnode_, self.y: target, self.mask: mask}
        else:

            fd = {self.NodeGraph: nodegraph, self.comp_inp: inputs, self.state: np.zeros((ArcNode.dense_shape[0], self.state_dim)),
                  self.state_old: np.ones((ArcNode.dense_shape[0], self.state_dim)),
                  self.ArcNode: arcnode_, self.y: target}
        if self.tensorboard:
            _, loss, loop, merge_all, merge_tr = self.session.run(
                [self.train_op, self.loss, self.loss_op, self.merged_all, self.merged_train],
                feed_dict=fd)
            if step % 100 == 0:
                self.writer.add_summary(merge_all, step)
                self.writer.add_summary(merge_tr, step)
        else:
            _, loss, loop = self.session.run(
                [self.train_op, self.loss, self.loss_op],
                feed_dict=fd)

        return loss, loop[1]

    def Validate(self, inptVal, arcnodeVal, targetVal, step, nodegraph=0.0, mask=None):
        """ Takes care of the validation of the model - it outputs, regarding the set given as input,
         the loss value, the accuracy (custom defined in the Net file), the number of iteration
         in the convergence procedure """

        arcnode_ = tf.SparseTensorValue(indices=arcnodeVal.indices, values=arcnodeVal.values,
                                        dense_shape=arcnodeVal.dense_shape)
        if self.graph_based:
            nodegraph = tf.SparseTensorValue(indices=nodegraph.indices, values=nodegraph.values,
                                        dense_shape=nodegraph.dense_shape)

        if self.mask_flag:
            fd_val = {self.NodeGraph: nodegraph, self.comp_inp: inptVal,
                      self.state: np.zeros((arcnodeVal.dense_shape[0], self.state_dim)),
                      self.state_old: np.ones((arcnodeVal.dense_shape[0], self.state_dim)),
                      self.ArcNode: arcnode_,
                      self.y: targetVal,
                      self.mask: mask}
        else:

            fd_val = {self.NodeGraph: nodegraph, self.comp_inp: inptVal,
                      self.state: np.zeros((arcnodeVal.dense_shape[0], self.state_dim)),
                      self.state_old: np.ones((arcnodeVal.dense_shape[0], self.state_dim)),
                      self.ArcNode: arcnode_,
                      self.y: targetVal}

        if self.tensorboard:
            loss_val, loop, merge_all, merge_val, metr = self.session.run(
                [self.val_loss, self.loss_op, self.merged_all, self.merged_val, self.metrics], feed_dict=fd_val)
            self.writer.add_summary(merge_all, step)
            self.writer.add_summary(merge_val, step)
        else:
            loss_val, loop, metr = self.session.run(
                [self.val_loss, self.loss_op, self.metrics], feed_dict=fd_val)
        return loss_val, metr, loop[1]

    def Evaluate(self, inputs, st, st_old, ArcNode, target):
        '''evaluate method with initialized state -- not used for the moment: has to receive the inputs,
        initialization for state(t) and state(t-1),
        arch-node matrix conversion, target -- gives as output the accuracy on the set given as input'''

        arcnode_ = tf.SparseTensorValue(indices=ArcNode.indices, values=ArcNode.values,
                                        dense_shape=ArcNode.dense_shape)

        fd = {self.comp_inp: inputs, self.state: st, self.state_old: st_old,
              self.ArcNode: arcnode_, self.y: target}
        _ = self.session.run([self.init_l])
        met = self.session.run([self.metrics], feed_dict=fd)
        return met

    def Evaluate(self, inputs, ArcNode, target, nodegraph=0.0):
        '''evaluate methods: has to receive the inputs,  arch-node matrix conversion, target
         -- gives as output the accuracy on the set given as input'''

        arcnode_ = tf.SparseTensorValue(indices=ArcNode.indices, values=ArcNode.values,
                                        dense_shape=ArcNode.dense_shape)
        if self.graph_based:
            nodegraph = tf.SparseTensorValue(indices=nodegraph.indices, values=nodegraph.values,
                                        dense_shape=nodegraph.dense_shape)


        fd = {self.NodeGraph: nodegraph, self.comp_inp: inputs, self.state: np.zeros((ArcNode.dense_shape[0], self.state_dim)),
              self.state_old: np.ones((ArcNode.dense_shape[0], self.state_dim)),
              self.ArcNode: arcnode_, self.y: target}
        _ = self.session.run([self.init_l])
        met = self.session.run([self.metrics], feed_dict=fd)
        return met

    def Predict(self, inputs, st, st_old, ArcNode):
        ''' predict methods with initialized state -- not used for the moment:: has to receive the inputs,
         initialization for state(t) and state(t-1),
         arch-node matrix conversion -- gives as output the output values of the output function (all the nodes output
         for all the graphs (if node-based) or a single output for each graph (if graph based) '''

        arcnode_ = tf.SparseTensorValue(indices=ArcNode.indices, values=ArcNode.values,
                                        dense_shape=ArcNode.dense_shape)
        fd = {self.comp_inp: inputs, self.state: st, self.state_old: st_old,
              self.ArcNode: arcnode_}
        pr = self.session.run([self.loss_op], feed_dict=fd)
        return pr[0]

    def Predict(self, inputs, ArcNode, nodegraph=0.0):
        ''' predict methods: has to receive the inputs, arch-node matrix conversion -- gives as output the output
         values of the output function (all the nodes output
         for all the graphs (if node-based) or a single output for each graph (if graph based) '''

        arcnode_ = tf.SparseTensorValue(indices=ArcNode.indices, values=ArcNode.values,
                                        dense_shape=ArcNode.dense_shape)
        fd = {self.comp_inp: inputs, self.state: np.zeros((ArcNode.dense_shape[0], self.state_dim)),
              self.state_old: np.ones((ArcNode.dense_shape[0], self.state_dim)),
              self.ArcNode: arcnode_}
        pr = self.session.run([self.loss_op], feed_dict=fd)
        return pr[0]
