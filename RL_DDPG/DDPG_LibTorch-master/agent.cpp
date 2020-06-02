
#include "agent.h"
#include "boost/make_shared.hpp"
#include "ReplayBuffer.h"

int BUFFER_SIZE = int(1e5);  // replay buffer size
int BATCH_SIZE = 128;        // minibatch size
double GAMMA = 0.99;            // discount factor
double TAU = 1e-3;              // for soft update of target parameters
double LR_ACTOR = 1e-4;         // learning rate of the actor
double LR_CRITIC = 1e-3;        // learning rate of the critic
double WEIGHT_DECAY = 0;        // L2 weight decay

int Agent::totalNumberOfAgents = 0;

Agent::Agent(int state_size, int action_size, int random_seed )
: actor_local(std::make_shared<Actor>(state_size, action_size, random_seed)),
actor_target(std::make_shared<Actor>(state_size, action_size, random_seed)),
actor_optimizer(actor_local->parameters(), /*lr=*/LR_ACTOR),
critic_local(std::make_shared<Critic>(state_size, action_size, random_seed)),
critic_target(std::make_shared<Critic>(state_size, action_size, random_seed)),
critic_optimizer(critic_local->parameters(), /*lr=*/LR_CRITIC),
device(torch::kCPU)
{
    numOfThisAgent = ++totalNumberOfAgents;

    torch::DeviceType device_type;
    if (torch::cuda::is_available()) {
        device_type = torch::kCUDA;
        std::cout << "Agent - Cuda available" << std::endl;
    } else {
        device_type = torch::kCPU;
        std::cout << "Agent - CPU used" << std::endl;

    }
    device = torch::Device(device_type);

    stateSize = state_size;
    actionSize = action_size;
    seed = random_seed;

//  Actor Network (w/ Target Network)
    actor_local->to(device);
    actor_target->to(device);

//  Critic Network (w/ Target Network)
    critic_local->to(device);
    critic_target->to(device);

    critic_optimizer.options.weight_decay_ = WEIGHT_DECAY;

    hard_copy_weights(actor_target, actor_local);
    hard_copy_weights(critic_target, critic_local);
    noise = new OUNoise(static_cast<size_t>(action_size));
}

void Agent::hard_copy_weights( std::shared_ptr<torch::nn::Module> local, std::shared_ptr<torch::nn::Module> target)
{
    
    for (size_t i = 0; i < target->parameters().size(); i++) {
        target->parameters()[i] = local->parameters()[i];
    }
}

std::vector<float> Agent::act(std::vector<float> state, bool add_noise )
{
    torch::Tensor torchState = torch::tensor(state, torch::dtype(torch::kFloat)).to(device);
    actor_local->eval();

    torch::NoGradGuard guard;
    auto action = actor_local->forward(torchState).to(torch::kCPU);
    actor_local->train();
    std::vector<float> v(action.data<float>(), action.data<float>() + action.numel());
    if (add_noise)
        noise->sample(v);
    for (size_t i =0; i < v.size(); i++) {
        v[i] = std::fmin(std::fmax(v[i],-1.f), 1.f);
    }
    return v;
}

void Agent::reset()
{
    noise.reset();
}

void Agent::step(std::vector<float> state, std::vector<float> action, float reward, std::vector<float> next_state, bool done)
{
    torch::Tensor state_t = torch::tensor(state, torch::dtype(torch::kFloat)).to(device);
    torch::Tensor action_t = torch::tensor(action, torch::dtype(torch::kFloat)).to(device);
    torch::Tensor reward_t = torch::tensor(reward, torch::dtype(torch::kFloat)).to(device);
    torch::Tensor next_state_t = torch::tensor(next_state, torch::dtype(torch::kFloat)).to(device);
    torch::Tensor done_t = torch::tensor(done, torch::dtype(torch::kFloat)).to(device);

    memory.addExperienceState(state_t, action_t, reward_t, next_state_t, done_t);
    // Learn, if enough samples are available in memory
    if (memory.getLength() > size_t(BATCH_SIZE))
    {
        learn(memory.sample(), GAMMA);
    }
}

void Agent::learn(std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> experiences, double gamma)
{
//    Update policy and value parameters using given batch of experience tuples.
//    Q_targets = r + γ * critic_target(next_state, actor_target(next_state))

    auto& [state, action, reward, next_state, done] = experiences;

// ---------------------------- update critic ---------------------------- #

    auto actions_next = actor_target->forward(next_state);
    auto Q_targets_next = critic_target->forward(next_state, actions_next);
    auto Q_targets = reward + (gamma * Q_targets_next * (1 - done));
    auto Q_expected = critic_local->forward(state, action); 

    torch::Tensor critic_loss = torch::mse_loss(Q_expected, Q_targets.detach());
    critic_optimizer.zero_grad();
    critic_loss.backward();

    critic_optimizer.step();

// ---------------------------- update actor ---------------------------- #

    auto actions_pred = actor_local->forward(state);
    auto actor_loss = -critic_local->forward(state, actions_pred).mean();

    actor_optimizer.zero_grad();
    actor_loss.backward();
    actor_optimizer.step();

// ----------------------- update target networks ----------------------- #
    soft_update(critic_local, critic_target, TAU);
    soft_update(actor_local, actor_target, TAU);

}

void Agent::soft_update(std::shared_ptr<torch::nn::Module> local, std::shared_ptr<torch::nn::Module> target, double tau)
{
//    Soft update model parameters.
//    θ_target = τ*θ_local + (1 - τ)*θ_target
    torch::NoGradGuard no_grad;
    for (size_t i = 0; i < target->parameters().size(); i++) {
        target->parameters()[i].copy_(tau * local->parameters()[i] + (1.0 - tau) * target->parameters()[i]);
    }
}

void Agent::saveCheckPoints(int eps)
{
    std::string path = getExecutablePathCopy();
    auto fileActor (path + "checkpoints/ckp_actor_agent" + std::to_string(numOfThisAgent) +"_" + std::to_string(eps) + ".pt");
    auto fileCritic(path + "checkpoints/ckp_critic_agent"+ std::to_string(numOfThisAgent) +"_" + std::to_string(eps) + ".pt");
    torch::save(std::dynamic_pointer_cast<torch::nn::Module>(actor_local) , fileActor);
    torch::save(std::dynamic_pointer_cast<torch::nn::Module>(critic_local) , fileCritic);
}

void Agent::loadCheckPoints(int eps)
{
    std::string path = getExecutablePathCopy();
    auto fileActor (path + "checkpoints/ckp_actor_agent" + std::to_string(numOfThisAgent) +"_" + std::to_string(eps) + ".pt");
    auto fileCritic(path + "checkpoints/ckp_critic_agent"+ std::to_string(numOfThisAgent) +"_" + std::to_string(eps) + ".pt");
    torch::load(actor_local, fileActor);
    torch::load(critic_local, fileCritic);
}

std::string Agent::getExecutablePath() 
{
    char buff[PATH_MAX];
    ssize_t len = ::readlink("/proc/self/exe", buff, sizeof(buff) - 1);
    if (len != -1) {
        buff[len] = '\0';
        std::string path = std::string(buff);
        std::size_t found = path.find_last_of("/");
        path = path.substr(0,found)+"/";
        return path;
    }
    std::cout << "Could not determine path of executable" << std::endl;
    return "";
}
