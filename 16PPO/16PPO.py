import gym
import numpy as np
import torch
from torch import nn

class Network(nn.Module):
    def __init__(self, obs_space, act_space):
        super(Network, self).__init__()
        self.fc1 = nn.Linear(obs_space, 12)
        self.relu = nn.ReLU()
        self.out = nn.Linear(12, act_space)
    
    def forward(self, s):
        x = self.fc1(s)
        x = self.relu(x)
        Q_value = self.out(x)
        return Q_value

class Skylark_PPO():
    def __init__(self, env, alpha = 0.1, gamma = 0.6, epsilon=0.1, update_freq = 200):
        self.obs_space = env.observation_space.n
        self.act_space = env.action_space.n
        self.eval_net = Network(self.obs_space, self.act_space)
        self.target_net = Network(self.obs_space, self.act_space)
        self.env = env
        self.alpha = alpha      # learning rate
        self.gamma = gamma      # discount rate
        self.epsilon = epsilon  # epsilon-greedy 
        self.buffer_size = 1000 # total size of replay/memory buffer
        self.traj_count = 0   # count of trajectories in buffer
        self.replay_buffer = np.zeros((self.buffer_size, self.obs_space*2+2))
        self.total_step = 0
        self.update_freq = update_freq # Freq of target update

    def choose_action(self, state):
        state = torch.unsqueeze(torch.FloatTensor(state), 0)
        if np.random.uniform(0, 1) < self.epsilon:
            action = self.env.action_space.sample()     # Explore action space
        else:
            action = torch.argmax(self.eval_net(state)).numpy() # Exploit learned values
        return action
            
    def store_trajectory(self, s, a, r, s_):
        traj = np.hstack((s,[a,r],s_))
        index = self.traj_count % self.buffer_size # 记忆多了就覆盖旧数据
        self.replay_buffer[index, :] = traj
        self.traj_count += 1

    def learn(self, batch_size=128):
        if self.total_step % self.update_freq == 0:
             self.target_net.load_state_dict(self.eval_net.state_dict())
        
        # 抽取记忆库中的批数据
        sample_index = np.random.choice(self.buffer_size, batch_size)
        b_memory = self.replay_buffer[sample_index, :]
        b_s = torch.FloatTensor(b_memory[:, :self.obs_space])
        b_a = torch.LongTensor(b_memory[:, self.obs_space:self.obs_space+1].astype(int))
        b_r = torch.FloatTensor(b_memory[:, self.obs_space+1:self.obs_space+2])
        b_s_ = torch.FloatTensor(b_memory[:, -self.obs_space:])

        # 针对做过的动作b_a, 来选 q_eval 的值, (q_eval 原本有所有动作的值)
        q_eval = self.eval_net(b_s).gather(1, b_a)  # shape (batch, 1)
        q_next = self.target_net(b_s_).detach()     # q_next 不进行反向传递误差, 所以 detach
        q_target = b_r + self.gamma * q_next.max(1)[0]   # shape (batch, 1)
        loss = nn.MSELoss()(q_eval, q_target)

        # 计算, 更新 eval net
        optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=self.alpha)    # torch 的优化器
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    def train(self, num_episodes, batch_size = 128, num_steps = 100):
        for i in range(num_episodes):
            state = self.env.reset()

            steps, penalties, reward, sum_rew = 0, 0, 0, 0
            done = False
            while not done and steps < num_steps:
                action = self.choose_action(state)
                # Interaction with Env
                next_state, reward, done, info = self.env.step(action) 
                
                self.store_trajectory(state, action, reward, next_state)
                if self.traj_count > self.buffer_size:
                    self.learn(batch_size)

                if reward == -10:
                    penalties += 1

                sum_rew += reward
                state = next_state
                steps += 1
                self.total_step += 1
            print('Episode: {} | Avg_reward: {} | Length: {}'.format(i, sum_rew/steps, steps))
        print("Training finished.")

if __name__ == "__main__":
    use_ray = False

    num_episodes = 1000
    env = gym.make("Taxi-v3").env
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