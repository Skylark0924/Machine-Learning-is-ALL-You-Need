// C++ API - OPENAIGYM
// #include "binding-cpp/include/gym/gym.h"


#include "iostream"

#include <boost/make_shared.hpp>
#include "ddpg_model.h"
#include <torch/torch.h>
#include "agent.h"

std::string getTimeString(double remaining) {
    int hours = int(remaining / 3600);         // Round down
    remaining -= 3600 * hours;
    int min = int(remaining / 60);
    remaining -= 60 * min;
    double sec = remaining;
    std::stringstream ss;
    ss << hours << ":" << std::setw(2) << std::setfill('0') << min << ":" << std::setw(2) << sec;
    std::string retVal(ss.str());
    return retVal;
}

/**************     TESTING    **********************/

void test_environment(const boost::shared_ptr<Gym::Environment>& env, Agent& agent,  int checkPointNumber )
{
    agent.loadCheckPoints(checkPointNumber);

    Gym::State s;
    env->reset(&s);

    float total_reward = 0;
    while (1) {
        auto oldState = s;
        auto action = agent.act(s.observation, false);
        env->step(action, /*render=*/true, &s);
//        assert(s.observation.size()==observation_space->sample().size());
        total_reward += s.reward;
        if (s.done){
            break;
        }
        std::cout << "Average Score:\t" << total_reward << std::endl;
    }
}


/**************     TRAINING    **********************/

void train_environment(const boost::shared_ptr<Gym::Environment> env, Agent& agent,
                  bool renderEnv, int episodes_to_run = 1)
{

    auto startTime = std::chrono::system_clock::now();
    auto episodeTime = startTime;

    boost::circular_buffer<float> scoreBuffer{100};
    for (int e=1; e <= episodes_to_run; e++)
    {
        Gym::State s;
        env->reset(&s);

        float total_reward = 0;
        int total_steps = 0;
        auto time_now = std::chrono::system_clock::now();
        for (int i = 0; i < 300; i++) {
            total_steps++;
            auto oldState = s;
            auto action = agent.act(s.observation, true);
            env->step(action, renderEnv, &s);
//            assert(s.observation.size()==observation_space->sample().size());
            total_reward += s.reward;
            agent.step(oldState.observation, action, s.reward, s.observation, s.done);
            if (s.done) {
                break;
            }
        }

        scoreBuffer.push_back(total_reward);

        if (e % (episodes_to_run/5) == 0) {
            std::cout << "****************** Checkpoint saved: " << e << "Episodes *******************" << std::endl;
            agent.saveCheckPoints(e);
        }

        if (e % 10 == 0) {
            auto avg_mean = std::accumulate( scoreBuffer.end()-scoreBuffer.size(), scoreBuffer.end(), 0.0)/ scoreBuffer.size();
            std::cout << "Episode:\t" << e << "\t\tAverage Score:\t" << avg_mean <<  "\t\tCurrent Score:\t" << total_reward <<
            "\t\tEnv steps:\t" << total_steps << std::endl;
            auto total_time = getTimeString(std::chrono::duration_cast<std::chrono::seconds>(time_now-startTime).count());
            auto episode_time = getTimeString(std::chrono::duration_cast<std::chrono::seconds>(time_now-episodeTime).count());
            std::cout << "Total:\t" << total_time << "\t\t100 Steps:\t" << episode_time << std::endl;
            episodeTime  = time_now;
        }
    }
}


int main() {
    bool training = true;

    try {
        boost::shared_ptr<Gym::Client> client = Gym::client_create("127.0.0.1", 5000);
        boost::shared_ptr<Gym::Environment> env = client->make("Pendulum-v0");
        auto action_space = env->action_space();
        auto observation_space = env->observation_space();

        // Get Action_Size
        int action_size = 0;
        if (env->action_space()->type == Gym::Space::SpaceType::DISCRETE) {
            action_size = env->action_space()->discreet_n;
        } else { // CONTINUOUS
            action_size = env->action_space()->box_shape[0]; 
        }

        // Get State Size
        int state_size = 0;
        if (env->observation_space()->type == Gym::Space::SpaceType::DISCRETE) {
            state_size = env->observation_space()->discreet_n;
        } else { // CONTINUOUS
            state_size = env->observation_space()->box_shape[0];
        }

        auto agent = Agent(state_size , action_size, 2);
        std::cout << "(main.cpp) state_size = " << state_size << std::endl;
        std::cout << "(main.cpp) action_size = " << action_size << std::endl;


        if (training)
            train_environment(env, agent, /*render*/ false, 10000);
        else
            test_environment(env, agent, 100);

    } catch (const std::exception& e) {
        fprintf(stderr, "ERROR: %s\n", e.what());
        return 1;
    }

    return 0;

}
