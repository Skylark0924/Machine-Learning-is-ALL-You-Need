from env import *
import gym
import numpy as np

class Skylark_Qlearning():
    def __init__(self, env, alpha = 0.1, gamma = 0.6, epsilon=0.1):
        self.q_table = np.zeros([env.observation_space.n, env.action_space.n])
        self.env = env
        self.alpha = alpha      # learning rate
        self.gamma = gamma      # discount rate
        self.epsilon = epsilon  # epsilon-greedy 
    
    def train(self, num_epochs):
        for i in range(1, num_episodes):
            state = self.env.reset()

            epochs, penalties, reward, sum_rew = 0, 0, 0, 0
            done = False
            
            while not done:
                if np.random.uniform(0, 1) < self.epsilon:
                    action = self.env.action_space.sample() # Explore action space
                else:
                    action = np.argmax(self.q_table[state]) # Exploit learned values

                next_state, reward, done, info = self.env.step(action) # Interaction with Env
                
                old_value = self.q_table[state, action]
                next_max = np.max(self.q_table[next_state])
                
                new_value = (1 - self.alpha) * old_value + self.alpha * (reward + self.gamma * next_max)
                self.q_table[state, action] = new_value

                if reward == -10:
                    penalties += 1

                sum_rew += reward
                state = next_state
                epochs += 1
            print('Episode: {} | Avg_reward: {} | Length: {}'.format(i, sum_rew/epochs, epochs))
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
            'DQN',
            config={
                'env': "Taxi-v3",
                'num_workers': 1,
                # 'env_config': {}
            }
        )
    else:
        ql_agent = Skylark_Qlearning(env)
        ql_agent.train(num_episodes)