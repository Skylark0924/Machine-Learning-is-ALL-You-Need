import gym
import numpy as np
import pandas as pd
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR) # 隐藏warning

'''
连续动作空间
'''

class Skylark_PPO():
    def __init__(self, env, gamma = 0.9, epsilon = 0.1, kl_target = 0.01, t='ppo2'):
        self.t = t
        self.log = 'model/{}_log'.format(t)

        self.env = env
        self.bound = self.env.action_space.high[0]

        self.gamma = gamma
        self.A_LR = 0.0001
        self.C_LR = 0.0002
        self.A_UPDATE_STEPS = 10
        self.C_UPDATE_STEPS = 10

        # KL penalty, d_target、β for ppo1
        self.kl_target = kl_target
        self.beta = 0.5
        # ε for ppo2
        self.epsilon = epsilon

        self.sess = tf.Session()
        self.build_model()

    def _build_critic(self):
        """critic model.
        """
        with tf.variable_scope('critic'):
            x = tf.layers.dense(self.states, 100, tf.nn.relu)

            self.v = tf.layers.dense(x, 1)
            self.advantage = self.dr - self.v

    def _build_actor(self, name, trainable):
        """actor model.
        """
        with tf.variable_scope(name):
            x = tf.layers.dense(self.states, 100, tf.nn.relu, trainable=trainable)

            mu = self.bound * tf.layers.dense(x, 1, tf.nn.tanh, trainable=trainable)
            sigma = tf.layers.dense(x, 1, tf.nn.softplus, trainable=trainable)

            norm_dist = tf.distributions.Normal(loc=mu, scale=sigma)

        params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=name)

        return norm_dist, params

    def build_model(self):
        """build model with ppo loss.
        """
        # inputs
        self.states = tf.placeholder(tf.float32, [None, 3], 'states')
        self.action = tf.placeholder(tf.float32, [None, 1], 'action')
        self.adv = tf.placeholder(tf.float32, [None, 1], 'advantage')
        self.dr = tf.placeholder(tf.float32, [None, 1], 'discounted_r')

        # build model
        self._build_critic()
        nd, pi_params = self._build_actor('actor', trainable=True)
        old_nd, oldpi_params = self._build_actor('old_actor', trainable=False)

        # define ppo loss
        with tf.variable_scope('loss'):
            # critic loss
            self.closs = tf.reduce_mean(tf.square(self.advantage))

            # actor loss
            with tf.variable_scope('surrogate'):
                ratio = tf.exp(nd.log_prob(self.action) - old_nd.log_prob(self.action))
                surr = ratio * self.adv

            if self.t == 'ppo1':
                self.tflam = tf.placeholder(tf.float32, None, 'lambda')
                kl = tf.distributions.kl_divergence(old_nd, nd)
                self.kl_mean = tf.reduce_mean(kl)
                self.aloss = -(tf.reduce_mean(surr - self.tflam * kl))
            else: 
                self.aloss = -tf.reduce_mean(tf.minimum(
                    surr,
                    tf.clip_by_value(ratio, 1.- self.epsilon, 1.+ self.epsilon) * self.adv))

        # define Optimizer
        with tf.variable_scope('optimize'):
            self.ctrain_op = tf.train.AdamOptimizer(self.C_LR).minimize(self.closs)
            self.atrain_op = tf.train.AdamOptimizer(self.A_LR).minimize(self.aloss)

        with tf.variable_scope('sample_action'):
            self.sample_op = tf.squeeze(nd.sample(1), axis=0)

        # update old actor
        with tf.variable_scope('update_old_actor'):
            self.update_old_actor = [oldp.assign(p) for p, oldp in zip(pi_params, oldpi_params)]

        tf.summary.FileWriter(self.log, self.sess.graph)

        self.sess.run(tf.global_variables_initializer())

    def choose_action(self, state):
        """choice continuous action from normal distributions.

        Arguments:
            state: state.

        Returns:
           action.
        """
        state = state[np.newaxis, :]
        action = self.sess.run(self.sample_op, {self.states: state})[0]
        return np.clip(action, -self.bound, self.bound)

    def get_value(self, state):
        """get q value.

        Arguments:
            state: state.

        Returns:
           q_value.
        """
        if state.ndim < 2: state = state[np.newaxis, :]

        return self.sess.run(self.v, {self.states: state})

    def discount_reward(self, states, rewards, next_observation):
        """Compute target value.

        Arguments:
            states: state in episode.
            rewards: reward in episode.
            next_observation: state of last action.

        Returns:
            targets: q targets.
        """
        s = np.vstack([states, next_observation.reshape(-1, 3)])
        q_values = self.get_value(s).flatten()

        targets = rewards + self.gamma * q_values[1:]
        targets = targets.reshape(-1, 1)

        return targets

    def learn(self, states, action, dr):
        """update model.

        Arguments:
            states: states.
            action: action of states.
            dr: discount reward of action.
        """
        self.sess.run(self.update_old_actor)

        adv = self.sess.run(self.advantage,
                            {self.states: states,
                             self.dr: dr})

        # update actor
        if self.t == 'ppo1':
            # run ppo1 loss
            for _ in range(self.A_UPDATE_STEPS):
                _, kl = self.sess.run(
                    [self.atrain_op, self.kl_mean],
                    {self.states: states,
                     self.action: action,
                     self.adv: adv,
                     self.tflam: self.beta})

            if kl < self.kl_target / 1.5:
                self.beta /= 2
            elif kl > self.kl_target * 1.5:
                self.beta *= 2
        else:
            # run ppo2 loss
            for _ in range(self.A_UPDATE_STEPS):
                self.sess.run(self.atrain_op,
                              {self.states: states,
                               self.action: action,
                               self.adv: adv})

        # update critic
        for _ in range(self.C_UPDATE_STEPS):
            self.sess.run(self.ctrain_op,
                          {self.states: states,
                           self.dr: dr})

    def train(self, num_episodes, batch_size=32, num_steps = 1000):
        tf.reset_default_graph()

        for i in range(num_episodes):
            state = self.env.reset()

            states, actions, rewards = [], [], []
            steps, sum_rew = 0, 0
            done = False

            while not done and steps < num_steps:
                action = self.choose_action(state)
                next_state, reward, done, _ = self.env.step(action)
                states.append(state)
                actions.append(action)

                sum_rew += reward
                rewards.append((reward + 8) / 8)

                state = next_state
                steps += 1

                if steps % batch_size == 0:
                    states = np.array(states)
                    actions = np.array(actions)
                    rewards = np.array(rewards)
                    d_reward = self.discount_reward(states, rewards, next_state)

                    self.learn(states, actions, d_reward)

                    states, actions, rewards = [], [], []

            print('Episode: {} | Avg_reward: {} | Length: {}'.format(i, sum_rew/steps, steps))
        print("Training finished.")

if __name__ == "__main__":
    use_ray = False

    num_episodes = 1000
    env = gym.make("Pendulum-v0").env
    # env.render()

    if use_ray:
        import ray
        from ray import tune
        tune.run(
            'PPO',
            config={
                'env': "Taxi-v3",
                'num_workers': 1,
                # 'env_config': {}
            }
        )
    else:
        ppo_agent = Skylark_PPO(env)
        ppo_agent.train(num_episodes)