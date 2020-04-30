# PG & TRPO & PPO & DPPO
> Policy based Sequential Decision

> 别看底下有英文，真的很简单，不信你读😉

做Reinforcement Learning方向的，要明确其目标: **找到可以让agent获得最优回报的最优行为策略 $\pi^*$**，所以对策略直接进行建模并按照梯度提升就是一个很自然的想法了。

## Vanilla Policy Gradient
> - on-policy
> - either discrete or continuous action spaces

Policy gradient输出不是 action 的 value, 而是具体的那一个 action, 这样 policy gradient 就跳过了 value 评估这个阶段, 对策略本身进行评估。

### Theory

$$
\pi_\theta(a|s)=P[a|s]
$$

We must find the best parameters (θ) to maximize a score function, J(θ).
$$
J(\theta)=E_{\pi_\theta}[\sum\gamma r]
$$
There are two steps:

- Measure the quality of a π (policy) with a **policy score function** J(θ) (策略评估)
- Use **policy gradient ascent** to find the best parameter θ that improves our π. (策略提升)

### Policy score function

- episode environment with same start state $s_1$
   $$
   J_1(\theta)=E_\pi[G_1=R_1+\gamma R_2+\gamma^2 R_3+\dots]=E_\pi (V(s_1))
   $$

- continuous environment (use the average value) 

	$$\begin{aligned}
        J_{avgv}(\theta)&=E_{\pi}(V(s))=\sum_{s\in \mathcal{S}} d^\pi (s)V^\pi(s)\\&=\sum_{s\in \mathcal{S}} d^\pi (s) \sum_{a\in \mathcal{A}} \pi_\theta(a|s)Q^\pi(s,a)
    \end{aligned}
    $$

	where $d^\pi (s)=\dfrac{N(s)}{\sum_{s'}N(s')}$, $N(s)$ means Number of occurrences of the state, $\sum_{s'}N(s')$ represents Total number of occurrences of all state. So $d^\pi (s)$ 代表在策略 $\pi_\theta$ 下马尔科夫链的平稳分布 (on-policy state distribution under π), 详见[Policy Gradient Algorithms - lilianweng's blog](https://lilianweng.github.io/lil-log/2018/04/08/policy-gradient-algorithms.html)👍
	
- use the average reward per time step. The idea here is that we want to get the most reward per time step.

   ![image-20191205092649257](../img/Reinforcement%20Learning%20Notes.assets/image-20191205092649257.png)

### Policy gradient asscent
与我们惯用的梯度下降相反，这里用的是**梯度上升**！
$$
\theta\leftarrow \theta + \alpha\nabla_\theta J(\theta)
$$

$$
\theta^*=\arg\max_\theta \underbrace{E_{\pi \theta}[\sum_t R(s_t,a_t)]}_{J(\theta)}
$$

Our score function J(θ) can be also defined as:

![image-20191205102211427](../img/Reinforcement%20Learning%20Notes.assets/image-20191205102211427.png)

Since $J(θ)$ is composed of state distribution and action distribution, when we gradient with respect to $\theta$, the effect of action is simple to find but the state effect is much more complicated due to the unknown environment. The solution is to use **Policy Gradient Theorem**:

我们将上一节的三种policy score function归纳为：

![image-20191205103636621](../img/Reinforcement%20Learning%20Notes.assets/image-20191205103636621.png)


It provides a nice reformation of the derivative of the objective function to not involve the derivative of the state distribution $d_π(.)$ and simplify the gradient computation $∇_θJ(θ)$ a lot.

$$
\begin{aligned}
\nabla_\theta J(\theta)&=\nabla_\theta \sum_{s \in \mathcal{S}} d^{\pi}(s)\sum_\tau \pi(\tau;\theta)R(\tau)\\
&\propto\sum_{s \in \mathcal{S}} d^{\pi}(s)\sum_\tau \nabla_\theta \pi(\tau;\theta)R(\tau)
\end{aligned}
$$

**Proof:** [Policy Gradient Algorithms - lilianweng's blog](https://lilianweng.github.io/lil-log/2018/04/08/policy-gradient-algorithms.html)👍

It is also hard to differentiating $\pi$, unless we can transform it into a **logarithm**. ([likelihood ratio trick](http://blog.shakirm.com/2015/11/machine-learning-trick-of-the-day-5-log-derivative-trick/))

![image-20191205103234091](../img/Reinforcement%20Learning%20Notes.assets/image-20191205103234091.png)

分解 $log\pi_\theta(\tau)$, 去掉不影响偏导的无关项, 就可以得到只与当前动作-状态对有关的[最大似然估计](https://zhuanlan.zhihu.com/p/26614750).

![](../img/Reinforcement%20Learning%20Notes.assets/微信截图_20200430093636.png)



那么这个log的偏导怎么求呢?

![](../img/Reinforcement%20Learning%20Notes.assets/微信截图_20200430100118.png)

在Coding的时候就是这段:

```
y = np.zeros([self.act_space])
y[act] = 1 # 制作离散动作空间，执行了的置1
self.gradients.append(np.array(y).astype('float32')-prob)
```

最后, 我们得到了VPG的更新方法:

![image-20191205103810941](../img/Reinforcement%20Learning%20Notes.assets/image-20191205103810941.png)

对应的code就是, 这里对reward做了归一化:

```
def learn(self):
    gradients = np.vstack(self.gradients)
    rewards = np.vstack(self.rewards)
    rewards = self.discount_rewards(rewards)
    # reward归一化
    rewards = (rewards - np.mean(rewards)) / (np.std(rewards) + 1e-7)
    gradients *= rewards
    X = np.squeeze(np.vstack([self.states]))
    Y = self.act_probs + self.alpha * np.squeeze(np.vstack([gradients]))
```

### Pseudocode

**REINFORCE**: 一种基于整条回合数据的更新, remember that? Monte-Carlo method!

![Policy Gradients 算法更新 (./img/5-1-1.png)](https://morvanzhou.github.io/static/results/reinforcement-learning/5-1-1.png)

> 其中，$\nabla log \pi_{\theta}(s_t,a_t)v_t$可以理解为在状态 $s$对所选动作的 $a$ 的吃惊度，$\pi_{\theta}(s_t,a_t)$概率越小，反向的 $log(Policy(s,a))$(即 `-log(P)`) 反而越大. 如果在 `Policy(s,a)` 很小的情况下, 拿到了一个大的 `R`, 也就是大的 `V`, 那 $\nabla log \pi_{\theta}(s_t,a_t)v_t$ 就更大, 表示更吃惊, (**我选了一个不常选的动作, 却发现原来它能得到了一个好的 reward, 那我就得对我这次的参数进行一个大幅修改**). 这就是吃惊度的物理意义.

**VPG** ([OpenAI SpinningUp](https://spinningup.openai.com/en/latest/algorithms/vpg.html#documentation-pytorch-version)的定义)

![](https://spinningup.openai.com/en/latest/_images/math/262538f3077a7be8ce89066abbab523575132996.svg)

可以发现引入了值函数/优势函数，这是后期改进之后的版本，使其可以用于非回合制的环境。


### Implement
```
'''
用于回合更新的离散控制 REINFORCE
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

            steps, penalties, reward, sum_rew = 0, 0, 0, 0
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
```

### Feature
**Advantages**

1. 输出的这个 action 可以是一个**连续值**, 之前我们说到的 value-based 方法输出的都是不连续的值, 然后再选择值最大的 action. 而 policy gradient 可以在一个连续分布上选取 action.

2. Convergence: The problem with value-based methods is that they can have a big oscillation while training. This is because the choice of action may change dramatically for an arbitrarily small change in the estimated action values.

   On the other hand, with policy gradient, we just follow the gradient to find the best parameters. We see a smooth update of our policy at each step.

   Because we follow the gradient to find the best parameters, we’re guaranteed to converge on a local maximum (worst case) or global maximum (best case).

3. Policy gradients can learn stochastic policies

   - we don’t need to implement an exploration/exploitation trade off.

   -  get rid of the problem of perceptual aliasing.

**Disadvantages**

1. A lot of the time, they converge on a local maximum rather than on the global optimum.
2. In a situation of Monte Carlo, waiting until the end of episode to calculate the reward.

## TRPO (Trust Region Policy Optimization)
> - on-policy
> - either discrete or continuous action spaces

### Principle
TRPO译为**信赖域策略优化**，TRPO的出现是要解决VPG存在的问题的：**VPG的更新步长 $\alpha$ 是个固定值，很容易产生从一个不好的策略'提升'到另一个更差的策略上。**

这让我想起了优化中对步长的估计：Armijo-Goldstein准则、Wolfe-Powell准则等。当然和TRPO关系不大。

TRPO有一个大胆的想法，要**让更新后的策略回报函数单调不减**。一个自然的想法是，**将新策略所对应的回报函数表示成旧策略所对应的回报函数+其他项**。下式就是TRPO的起手式：

$$\eta(\hat{\pi})=\eta(\pi)+E_{s_{0}, a_{0}, \cdots \hat{\pi}}\left[\sum_{t=0}^{\infty} \gamma^{t} A_{\pi}\left(s_{t}, a_{t}\right)\right]$$

- $\eta$会被用作代价函数，毕竟在PG的梯度上升中，代价函数和回报函数等价
- $A_\pi$为优势函数([这个会在A2C的章节讲到]())

$$\begin{aligned}
A_{\pi}(s, a)&=Q_{\pi}(s, a)-V_{\pi}(s) \\
&=E_{s^{\prime}\sim P\left(s^{\prime}| s, a\right)}  \left[r(s)+\gamma V^{\pi}\left(s^{\prime}\right)-V^{\pi}(s)\right]
\end{aligned}$$

> **Proof:**  (也可以通过构造法反推)
> $$\begin{aligned}
E_{\tau | \hat{\pi}}\left[\sum_{t=0}^{\infty} \gamma^{t} A_{\pi}\left(s_{t}, a_{t}\right)\right] 
&=E_{\tau | \hat{\pi}}\left[\sum_{t=0}^{\infty} \gamma^{t}\left(r(s)+\gamma V^{\pi}\left(s_{t+1}\right)-V^{\pi}\left(s_{t}\right)\right)\right] \\
&=E_{\tau | \hat{\pi}}\left[\sum_{t=0}^{\infty} \gamma^{t}\left(r\left(s_{t}\right)\right)+\sum_{t=0}^{\infty} \gamma^{t}\left(\gamma V^{\pi}\left(s_{t+1}\right)-V^{\pi}\left(s_{t}\right)\right)\right] \\
&=E_{\tau | \hat{\pi}}\left[\sum_{t=0}^{\infty} \gamma^{t}\left(r\left(s_{t}\right)\right)\right]+E_{s_{0}}\left[-V^{\pi}\left(s_{0}\right)\right] \\
&=\eta(\hat{\pi})-\eta(\pi)
\end{aligned}$$


由此，我们就实现了将新策略的回报表示为旧策略回报的目标。

有了起手式，我们在实际操作时候具体怎么计算呢？尤其是优势函数外那个期望的怎么处理？


将其分解成state和action的求和：

$$\eta(\hat{\pi})=\eta(\pi)+\sum_{t=0}^{\infty} \sum_{s} P\left(s_{t}=s | \hat{\pi}\right) \sum_{a} \hat{\pi}(a | s) \gamma^{t} A_{\pi}(s, a)$$

- $\sum_{a} \hat{\pi}(a | s) \gamma^{t} A_{\pi}(s, a)$ 是在状态 s 时，计算动作 a 的边际分布
- $\sum_{s} P\left(s_{t}=s | \hat{\pi}\right)$ 是在时间 t 时，求状态 s 的边际分布
- $\sum_{t=0}^{\infty} \sum_{s} P\left(s_{t}=s | \hat{\pi}\right)$ 对整个时间序列求和。

定义 

$\rho_{\pi}(s)=P\left(s_{0}=s\right)+\gamma P\left(s_{1}=s\right)+\gamma^{2} P\left(s_{2}=s\right)+\cdots$ 

即有 
$$\eta(\hat{\pi})=\eta(\pi)+\sum_{s} \rho_{\hat{\pi}}(s) \sum_{a} \hat{\pi}(a | s) A^{\pi}(s, a)$$

这个公式在应用时也无法使用，因为状态 s 是根据新策略的分布产生的，而新策略又是我们要求的，这就导致了含有 $\hat{\pi}$ 的项我们都无从得知。

### Tricks
1. 一个简单的想法：用旧策略代替上式中的新策略
2. **重要性采样**来处理动作分布，也是TRPO的关键：
   $$\sum_{a} \hat{\pi}_{\theta}\left(a | s_{n}\right) A_{\theta_{o l d}}\left(s_{n}, a\right)=E_{a \sim q}\left[\frac{\hat{n}_{\theta}\left(a | s_{n}\right)}{\pi_{\theta_{o l d}}\left(a | s_{n}\right)} A_{\theta_{o l d}}\left(s_{n}, a\right)\right]$$
    得到 $\hat{\pi}$ 的一阶近似，**替代回报函数 $L_\pi(\hat{\pi})$**

    $$L_{\pi}(\hat{\pi})=\eta(\pi)+E_{s \sim \rho_{\theta_{o l d}}, a \sim \pi_{\theta_{o l d}}}\left[\frac{\hat{\pi}_{\theta}\left(a | s_{n}\right)}{\pi_{\theta_{o l d}}\left(a | s_{n}\right)} A_{\theta_{o l d}}\left(s_{n}, a\right)\right]$$

    ![](https://pic4.zhimg.com/80/v2-f8e72c18f3cb17bcd3c828c71c842d3f_1440w.jpg)

    说完了损失函数的构建，那么步长到底怎么定呢？

    $$\eta(\hat{\pi}) \geq L_{\pi}(\hat{\pi})-C D_{K L}^{\max }(\pi, \hat{\pi})$$

    - 惩罚因子 $C=\frac{2 \epsilon_{V}}{(1-V)^{2}}$
    - $D_{K L}^{\max }(\pi, \hat{\pi})$ 为每个状态下动作分布的最大值

    得到一个单调递增的策略序列：

    $$M_{i}(\pi)=L_{\pi_{i}}(\tilde{\pi})-C D_{K L}^{\max }\left(\pi_{i}, \tilde{\pi}\right)$$

    可知

    $$\eta\left(\pi_{i+1}\right) \geq M_{i}\left(\pi_{i+1}\right), and \quad \eta\left(\pi_{i}\right)=M_{i}\left(\pi_{i}\right)$$

    $$\eta\left(\pi_{i+1}\right)-\eta\left(\pi_{i}\right) \geq M_{i}\left(\pi_{i+1}\right)-M\left(\pi_{i}\right)$$

    我们只需要在新策略序列中找到一个使 $M_i$ 最大的策略即可，对策略的搜寻就变成了优化问题：

    $$\max _{\hat{\theta}}\left[L_{\theta_{o l d}}-C D_{K L}^{\max }\left(\theta_{o l d}, \hat{\theta}\right)\right]$$

    在TRPO原文中写作了约束优化问题：

    $$\max _{\theta} E_{s \sim \rho_{\theta_{o l d}}, a \sim \pi_{o_{o l d}}}\left[\frac{\tilde{\pi}_{\theta}\left(a | s_{n}\right)}{\pi_{\theta_{o l d}}\left(a | s_{n}\right)} A_{\theta_{o l d}}\left(s_{n}, a\right)\right] \\ s.t. \quad  D_{K L}^{\max }\left(\theta_{o l d} || \theta\right) \leq \delta$$
3. 利用平均KL散度代替最大KL散度
4. 对约束问题二次近似，非约束问题一次近似，这是凸优化的一种常见改法。最后TRPO利用共轭梯度的方法进行最终的优化。

> Q: 为什么觉得TRPO的叙述方式反了？私以为应该是在约束新旧策略的散度的前提下，找到使替代回报函数$L_\pi(\hat{\pi})$ 最大的 $\theta$ -> 转化为约束优化问题，这样就自然多了嘛。所以那一步惩罚因子的作用很让人迷惑，**烦请大佬们在评论区解惑**。

### Implement



### Reference 
1. [TRPO - Medium](https://medium.com/@jonathan_hui/rl-trust-region-policy-optimization-trpo-explained-a6ee04eeeee9)
2. [TRPO - OpenAI SpinningUp](https://spinningup.openai.com/en/latest/algorithms/trpo.html)
3. [TRPO与PPO](https://zhuanlan.zhihu.com/p/58765380)
4. [强化学习进阶 第七讲 TRPO](https://zhuanlan.zhihu.com/p/26308073)

## PPO (Proximal Policy Optimization)

### Theory

**The central idea of Proximal Policy Optimization is to avoid having too large policy update.** To do that, we use a ratio that will tells us the difference between our new and old policy and clip this ratio from 0.8 to 1.2. Doing that will ensure **that our policy update will not be too large.**

The problem comes from the step size of gradient ascent:

- Too small, **the training process was too slow**
- Too high, **there was too much variability in the training.**

The idea is that PPO improves the stability of the Actor training by limiting the policy update at each training step.

To be able to do that PPO introduced a new objective function called “**Clipped surrogate objective function**” that **will constraint the policy change in a small range using a clip.**

Instead of using log pi to trace the impact of the actions, we can use **the ratio between the probability of action under current policy divided by the probability of the action under previous policy.**
$$
r_t(\theta)=\dfrac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}, \text{so } r(\theta_{old})=1
$$

- If $r_t(θ)$ >1, it means that the **action is more probable in the current policy than the old policy.**
- If $r_t(θ)$ is between 0 and 1: it means that the **action is less probable for current policy than for the old one.**

As consequence, our new objective function could be:
$$
L^{CPI}(\theta)=\hat{\mathbb{E}}_t\lbrack\dfrac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}\hat{A}_t\rbrack=\hat{\mathbb{E}}_t[r_t(\theta)\hat{A}_t]
$$
**By doing that we’ll ensure that not having too large policy update because the new policy can’t be too different from the older one.**

To do that we have two solutions:

- TRPO (Trust Region Policy Optimization) uses KL divergence constraints outside of the objective function to constraint the policy update. But this method **is much complicated to implement and it takes more computation time.**
- PPO clip probability ratio directly in the objective function with its Clipped surrogate objective function.

![image-20191205121930328](../img/Reinforcement%20Learning%20Notes.assets/image-20191205121930328.png)

The final Clipped Surrogate(代理) Objective Loss:

![image-20191205190844049](../img/Reinforcement%20Learning%20Notes.assets/image-20191205190844049.png)

### Feature
**Advantage**

It can be used in both discrete and continuous control.

**Disadvantage**

on-policy -> data inefficient



## Reference 
1. [Policy Gradients - 莫烦](https://morvanzhou.github.io/tutorials/machine-learning/reinforcement-learning/5-1-A-PG/)
2. [Policy Gradient Algorithms - lilianweng's blog](https://lilianweng.github.io/lil-log/2018/04/08/policy-gradient-algorithms.html)👍
3. [An introduction to Policy Gradients with Cartpole and Doom](https://www.freecodecamp.org/news/an-introduction-to-policy-gradients-with-cartpole-and-doom-495b5ef2207f/)
4. [OpenAI spinningup](https://spinningup.openai.com/en/latest/algorithms/trpo.html)
5. [Policy gradient - Berkeley](http://rail.eecs.berkeley.edu/deeprlcourse-fa17/f17docs/lecture_4_policy_gradient.pdf)