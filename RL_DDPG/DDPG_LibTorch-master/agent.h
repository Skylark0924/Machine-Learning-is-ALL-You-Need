
#ifndef PROJECT_AGENT_H
#define PROJECT_AGENT_H

#include <torch/torch.h>
#include "boost/shared_ptr.hpp"
#include "ddpg_model.h"
#include "ReplayBuffer.h"

class OUNoise;

class Agent {
public:
    int numOfThisAgent;
    static int totalNumberOfAgents; 

    Agent(int state_size, int action_size, int random_seed );
    Agent(const Agent &other) = delete;
    Agent() = delete;

    void step(std::vector<float> state, std::vector<float> action, float reward, std::vector<float> next_state, bool done);
    std::vector<float> act(std::vector<float> state, bool add_noise = true);
    void reset();
    void learn(std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> experiences, double gamma);

    void saveCheckPoints(int e);
    void loadCheckPoints(int e);

    std::shared_ptr<Actor> actor_local;
    std::shared_ptr<Actor> actor_target;
    torch::optim::Adam actor_optimizer;

    std::shared_ptr<Critic> critic_local;
    std::shared_ptr<Critic> critic_target;
    torch::optim::Adam critic_optimizer;


private:
    std::string getExecutablePath();

    void soft_update(std::shared_ptr<torch::nn::Module> local, std::shared_ptr<torch::nn::Module> target, double tau);
    void hard_copy_weights( std::shared_ptr<torch::nn::Module> local, std::shared_ptr<torch::nn::Module> target );

    int stateSize;
    int actionSize;
    int seed;
    OUNoise* noise;
    torch::Device device; 
    ReplayBuffer memory;
};


class OUNoise {
//"""Ornstein-Uhlenbeck process."""
private:
    size_t size;
    std::vector<float> mu;
    std::vector<float> state;
    float theta=0.15;
    float sigma=0.1;

public:
    OUNoise (size_t size_in) {
        size = size_in;
        mu = std::vector<float>(size, 0);
        reset();
    }

    void reset() {
        state = mu;
    }

    std::vector<float> sample(std::vector<float> action) {
    //"""Update internal state and return it as a noise sample."""
        for (size_t i = 0; i < state.size(); i++) {
            auto random = ((float) rand() / (RAND_MAX));
            float dx = theta * (mu[i] - state[i]) + sigma * random;
            state[i] = state[i] + dx;
            action[i] += state[i];
        }
        return action;
    }
};

#endif //PROJECT_AGENT_H

