#ifndef PROJECT_DDPG_MODEL_H
#define PROJECT_DDPG_MODEL_H

#include <torch/torch.h>

class Actor : public torch::nn::Module {
public:
    Actor(int64_t state_size, int64_t action_size, int64_t seed = 0, int64_t fc1_units=400, int64_t fc2_units=300);
    void reset_parameters();

    torch::Tensor forward(torch::Tensor state);
    torch::nn::BatchNormOptions bn_options(int64_t features);
    std::pair<double,double> hidden_init(torch::nn::Linear& layer);


private:
    torch::nn::Linear fc1{nullptr}, fc2{nullptr}, fc3{nullptr};
    torch::nn::BatchNorm bn1{nullptr};
};


/******************* Critic *****************/

class Critic : public torch::nn::Module {
public:
    Critic(int64_t state_size, int64_t action_size, int64_t seed = 0, int64_t fcs1_units=400, int64_t fc2_units=300);

    void reset_parameters();

    torch::Tensor forward(torch::Tensor x, torch::Tensor action);
    std::pair<double,double> hidden_init(torch::nn::Linear& layer);


private:
    torch::nn::Linear fcs1{nullptr}, fc2{nullptr}, fc3{nullptr};
    torch::nn::BatchNorm bn1{nullptr};
};

#endif //PROJECT_DDPG_MODEL_H
