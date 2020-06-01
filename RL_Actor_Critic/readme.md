# Actor-Critic
> - on-policy
> - Actor-Critic structure Sequential Decision

## Vanilla Actor-Critic
### Principle

- a Critic that measures how good the action taken is (value-based)
- an Actor that controls how our agent behaves (policy-based)

Instead of waiting until the end of episode as we do in Monte Carlo REINFORCE, we make an update at each step(**TD Learning**)

![image-20191205104741531](../img/image-20191205104741531.png)

结合了 Policy Gradient (Actor) 和 Function Approximation (Critic) 的方法. `Actor` 基于概率选行为, `Critic` 基于 `Actor` 的行为评判行为的得分, `Actor` 根据 `Critic` 的评分修改选行为的概率，输入的单次奖赏变成了critic输出的总奖赏增量td-error。critic建立s-Q的网络，然后根据[s, r, s_]来训练，并返回td-error。
![](../img/image-20191205105318993.png)
 
![](./../img/ac.jpg)

**优势**：可以进行单步更新, 比传统的 Policy Gradient 要快.

**劣势**：取决于 Critic 的价值判断, 但是 Critic 难收敛, 再加上 Actor 的更新, 就更难收敛. 为了解决收敛问题, Google Deepmind 提出了 `Actor Critic` 升级版 `Deep Deterministic Policy Gradient`. 后者融合了 DQN 的优势, 解决了收敛难的问题. 

## Advantage Actor-Critic (A2C)
### Principle
与Actor-Critic唯一不同的是，使用了优势函数
![](../img/image-20191205110708645.png)
以解决 value-based methods 存在的 **high variability** 问题。

优势函数告诉我们**与该状态下采取的任意行动得到的平均value相比，所取得的提升。**

- $A(s, a)>0$：our gradient is pushed in that direction.
- $A(s, a)<0$：(our action does worse than the average value of that state) our gradient is pushed in the opposite direction.

然而这样做却需要做两个value网络，不划算。实际上可以将上式转换为一个 state value function $V(s)$ 的计算：

![](../img/image-20191205111303995.png)

代码中即为：
```
for r in self.model.rewards[::-1]:
    # calculate the discounted value
    R = r + self.gamma * R
    returns.insert(0, R)

returns = torch.tensor(returns)
returns = (returns - returns.mean()) / (returns.std() + self.eps)

for (log_prob, value), R in zip(saved_actions, returns):
    advantage = R - value.item()
```

## Implement
```
'''A2C Implement'''
class Policy(nn.Module):
    """
    implements both actor and critic in one model
    """
    def __init__(self):
        super(Policy, self).__init__()
        self.affine1 = nn.Linear(4, 128)

        # actor's layer
        self.action_head = nn.Linear(128, 2)

        # critic's layer
        self.value_head = nn.Linear(128, 1)

        # action & reward buffer
        self.saved_actions = []
        self.rewards = []

    def forward(self, x):
        """
        forward of both actor and critic
        """
        x = F.relu(self.affine1(x))

        # actor: choses action to take from state s_t 
        # by returning probability of each action
        action_prob = F.softmax(self.action_head(x), dim=-1)

        # critic: evaluates being in the state s_t
        state_values = self.value_head(x)

        # return values for both actor and critic as a tuple of 2 values:
        # 1. a list with the probability of each action over the action space
        # 2. the value from state s_t 
        return action_prob, state_values

class Skylark_A2C():
    def __init__(self, env):
        self.model = Policy()
        self.env = env
        self.optimizer = optim.Adam(self.model.parameters(), lr = 3e-2)
        self.eps = np.finfo(np.float32).eps.item()
        self.SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])
        self.render = False
        self.log_interval = 1
        self.gamma = 0.99
    
    def select_action(self, state):
        state = torch.from_numpy(state).float()
        probs, state_value = self.model(state)

        # create a categorical distribution over the list of probabilities of actions
        m = Categorical(probs)

        # and sample an action using the distribution
        action = m.sample()

        # save to action buffer
        self.model.saved_actions.append(self.SavedAction(m.log_prob(action), state_value))

        # the action to take (left or right)
        return action.item()

    def finish_episode(self):
        """
        Training code. Calculates actor and critic loss and performs backprop.
        """
        R = 0
        saved_actions = self.model.saved_actions
        policy_losses = [] # list to save actor (policy) loss
        value_losses = [] # list to save critic (value) loss
        returns = [] # list to save the true values

        # calculate the true value using rewards returned from the environment
        for r in self.model.rewards[::-1]:
            # calculate the discounted value
            R = r + self.gamma * R
            returns.insert(0, R)

        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + self.eps)

        for (log_prob, value), R in zip(saved_actions, returns):
            advantage = R - value.item()

            # calculate actor (policy) loss 
            policy_losses.append(-log_prob * advantage)

            # calculate critic (value) loss using L1 smooth loss
            value_losses.append(F.smooth_l1_loss(value, torch.tensor([R])))

        # reset gradients
        self.optimizer.zero_grad()

        # sum up all the values of policy_losses and value_losses
        loss = torch.stack(policy_losses).sum() + torch.stack(value_losses).sum()

        # perform backprop
        loss.backward()
        self.optimizer.step()

        # reset rewards and action buffer
        del self.model.rewards[:]
        del self.model.saved_actions[:]
    
    def train(self, num_episodes):
        running_reward = 10

        # run inifinitely many episodes
        for i in range(1, num_episodes):

            # reset environment and episode reward
            state = self.env.reset()
            ep_reward = 0

            # for each episode, only run 9999 steps so that we don't 
            # infinite loop while learning
            for t in range(1, 10000):
                # select action from policy
                action = self.select_action(state)

                # take the action
                state, reward, done, _ = self.env.step(action)

                if self.render:
                    self.env.render()

                self.model.rewards.append(reward)
                ep_reward += reward
                if done:
                    break

            # update cumulative reward (smooth)
            running_reward = 0.05 * ep_reward + (1 - 0.05) * running_reward

            # perform backprop
            self.finish_episode()

            # log results
            if i % self.log_interval == 0:
                print('Episode {}\tLast reward: {:.2f}\tAverage reward: {:.2f}'.format(
                    i, ep_reward, running_reward))

            # check if we have "solved" the cart pole problem
            if running_reward > self.env.spec.reward_threshold:
                print("Solved! Running reward is now {} and "
                    "the last episode runs to {} time steps!".format(running_reward, t))
                break
```

## Reference 
1. [Pytorch Actor-Critic](https://github.com/pytorch/examples/blob/master/reinforcement_learning/actor_critic.py)