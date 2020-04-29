# Q learning & Deep Q Network (DQN)
> Sequential Decision

## Q learning (Tabular)
### Principle 
根据$Q$表对下一时刻的动作进行选择，下图是$Q$表的更新方式

![](img\2019-04-10 19-14-32 的屏幕截图.png)

此时，$S_2$并未进行下一次的动作，而是预估了一下后果，由此来更新$S_1$的$Q$表。

![](img\2019-04-10 19-17-30 的屏幕截图.png)

其中，$\alpha$是**学习速率**，$\epsilon$是**选择$Q$表最大值的概率**。若$\epsilon=90\%$，则$90\%$概率选择$Q$表最大值即最优动作，$10\%$的概率随机动作。
$$
Q(s, a) \leftarrow Q(s, a)+\alpha\left[r+\gamma \max _{a^{\prime}} Q\left(s^{\prime}, a^{\prime}\right)-Q(s, a)\right]
$$
由于$Q(s',a')$是**下一次的动作**，会通过乘以**奖励衰减值**$\gamma$的方式影响前一次的$Q$表取值，因此很容易想到只要$\gamma\neq 0$，以后的每次动作得到的奖励值都会影响之前动作的$Q$表取值。

- Q估计：$s_1$状态最优动作$a$的Q值
- Q现实：在选择了动作$a$后，进入 $s'$状态。Q表中 $s'$状态对应的Q值的最大值加上执行动作$a$之后得到的奖励值$r$，即为Q现实。

### Implement
```
def train(self, num_epochs):
    for i in range(1, num_epochs):
        state = self.env.reset()

        epochs, penalties, reward, = 0, 0, 0
        done = False
        
        while not done:
            if np.random.uniform(0, 1) < self.epsilon:
                action = self.env.action_space.sample() # Explore action space
            else:
                action = np.argmax(self.q_table[state]) # Exploit learned values

            next_state, reward, done, info = self.env.step(action) 
            
            old_value = self.q_table[state, action]
            next_max = np.max(self.q_table[next_state])
            
            new_value = (1 - self.alpha) * old_value + self.alpha * (reward + self.gamma * next_max)
            self.q_table[state, action] = new_value

            if reward == -10:
                penalties += 1

            state = next_state
            epochs += 1
```

## DQN
### Principle 
1. 抛弃Q表这种Q值记录方式，使用神经网络生成Q值，在状态较多的情况下格外有效率

![DQN](img\DQN3.png)

2. - Q估计：通过NN预测出的$Q(s_2, a_1), Q(s_2,a_2)$的最大值
   - Q现实：Q 估计中最大值的动作来换取环境中的奖励 reward+$\gamma*$下一步$s'$中通过NN预测出的$Q(s', a_1), Q(s',a_2)$的最大值
   
3. **DQN两大利器**：
   - Experience replay: 作为一种离线学习，每次 DQN 更新的时候，我们都可以随机抽取一些之前的经历进行学习。随机抽取这种做法打乱了经历之间的相关性，也使得神经网络更新更有效率。
   - Fixed Q-target: 在 DQN 中使用到两个结构相同但参数不同的神经网络, 预测 Q 估计的神经网络具备最新的参数, 而预测 Q 现实的神经网络使用的参数则是很久以前的.
   
4. **算法**：
$$
Q(s, a) \leftarrow Q(s, a)+\alpha\left[r+\gamma \max _{a^{\prime}} \hat{Q}\left(s^{\prime}, a^{\prime}| \hat\theta\right)-Q(s, a|\theta)\right]
$$
  ![DQN 算法更新 (img\4-1-1-1554948278323.jpg)](https://morvanzhou.github.io/static/results/reinforcement-learning/4-1-1.jpg)

- 记忆库 (用于重复学习)
- 神经网络计算 Q 值
- 暂时冻结 `q_target` 参数 (切断相关性)


### Implement
```
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

class Skylark_DQN():
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
```