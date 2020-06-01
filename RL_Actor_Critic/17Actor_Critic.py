import argparse
import gym
import numpy as np
from itertools import count
from collections import namedtuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

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

class Skylark_Actor_Critic():
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

            # update cumulative reward
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

if __name__ == "__main__":
    use_ray = False

    num_episodes = 1000
    env = gym.make("CartPole-v0").env

    if use_ray:
        import ray
        from ray import tune
        tune.run(
            'A2C', 
            config={
                'env': "CartPole-v0",
                'num_workers': 1,
                # 'env_config': {}
            }
        )
    else:
        ac_agent = Skylark_Actor_Critic(env)
        ac_agent.train(num_episodes)
