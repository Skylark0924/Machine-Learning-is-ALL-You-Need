import gym
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Reshape, Flatten, Conv2D
from keras.optimizers import Adam
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR) # 隐藏warning


'''
用于回合更新的离散控制
'''

class Skylark_VPG():
    def __init__(self, env, alpha = 0.1, gamma = 0.6, epsilon=0.1, update_freq = 200):
        self.obs_space = 80*80  # 视根据具体gym环境的state输出格式，具体分析
        self.act_space = env.action_space.n
        self.env = env
        self.alpha = alpha      # learning rate
        self.gamma = gamma      # discount rate
        self.states = []
        self.gradients = []
        self.rewards = []
        self.act_probs = []
        self.total_step = 0

        self.model = self._build_model()
        self.model.summary()

    def _build_model(self):
        model = Sequential()
        model.add(Reshape((1, 80, 80), input_shape=(self.obs_space,)))
        model.add(Conv2D(32, (6, 6), activation="relu", strides=(3, 3), 
                        padding="same", kernel_initializer="he_uniform"))
        model.add(Flatten())
        model.add(Dense(64, activation='relu', kernel_initializer='he_uniform'))
        model.add(Dense(32, activation='relu', kernel_initializer='he_uniform'))
        # softmax策略使用描述状态和行为的特征ϕ(s,a) 与参数\theta的线性组合来权衡一个行为发生的概率
        # 输出为每个动作的概率
        model.add(Dense(self.act_space, activation='softmax'))
        opt = Adam(lr=self.alpha)
        model.compile(loss='categorical_crossentropy', optimizer=opt)
        return model
    
    def choose_action(self, state):
        state = state.reshape([1, self.obs_space])
        act_prob = self.model.predict(state).flatten()
        prob = act_prob / np.sum(act_prob)
        self.act_probs.append(act_prob)
        # 按概率选取动作
        action = np.random.choice(self.act_space, 1, p=prob)[0]
        return action, prob
        
    def store_trajectory(self, s, a, r, prob):
        y = np.zeros([self.act_space])
        y[a] = 1 # 制作离散动作空间，执行了的置1
        self.gradients.append(np.array(y).astype('float32')-prob)
        self.states.append(s)
        self.rewards.append(r)

    def discount_rewards(self, rewards):
        '''
        从回合结束位置向前修正reward
        '''
        discounted_rewards = np.zeros_like(rewards)
        running_add = 0
        for t in reversed(range(0, rewards.size)):
            if rewards[t] != 0:
                running_add = 0
            running_add = running_add * self.gamma + rewards[t]
            discounted_rewards[t] = np.array(running_add)
        return discounted_rewards

    def learn(self):
        gradients = np.vstack(self.gradients)
        rewards = np.vstack(self.rewards)
        rewards = self.discount_rewards(rewards)
        # reward归一化
        rewards = (rewards - np.mean(rewards)) / (np.std(rewards) + 1e-7)
        gradients *= rewards
        X = np.squeeze(np.vstack([self.states]))
        Y = self.act_probs + self.alpha * np.squeeze(np.vstack([gradients]))
        self.model.train_on_batch(X, Y)
        self.states, self.act_probs, self.gradients, self.rewards = [], [], [], []

    def train(self, num_episodes, batch_size = 128, num_steps = 100):
        for i in range(num_episodes):
            state = self.env.reset()

            steps, sum_rew = 0, 0
            done = False
            while not done:
                # self.env.render()
                state = preprocess(state)
                action, prob = self.choose_action(state)
                # Interaction with Env
                next_state, reward, done, info = self.env.step(action) 
                
                self.store_trajectory(state, action, reward, prob)

                sum_rew += reward
                state = next_state
                steps += 1
                self.total_step += 1
            if done:
                self.learn()
                print('Episode: {} | Avg_reward: {} | Length: {}'.format(i, sum_rew/steps, steps))
        print("Training finished.")
    
    def load(self, path):
        self.model.load_weights(path)

    def save(self, path):
        self.model.save_weights(path)

def preprocess(I):
    '''
    根据具体gym环境的state输出格式，具体分析
    '''
    I = I[35:195]
    I = I[::2, ::2, 0]
    I[I == 144] = 0
    I[I == 109] = 0
    I[I != 0] = 1
    return I.astype(np.float).ravel()

if __name__ == "__main__":
    use_ray = True

    num_episodes = 1000
    env = gym.make("Pong-v0").env
    # env.render()

    if use_ray:
        import ray
        from ray import tune
        tune.run(
            'PG',
            config={
                'env': "Pong-v0",
                'num_workers': 1,
            }
        )
    else:
        pg_agent = Skylark_VPG(env)
        pg_agent.train(num_episodes)