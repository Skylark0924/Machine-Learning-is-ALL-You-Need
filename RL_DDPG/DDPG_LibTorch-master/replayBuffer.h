#ifndef PROJECT_REPLAYBUFFER_H
#define PROJECT_REPLAYBUFFER_H

#include <boost/circular_buffer.hpp>
#include "torch/torch.h"

using Experience = std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>;

class ReplayBuffer {
public:
    ReplayBuffer() {}

    void addExperienceState(torch::Tensor state, torch::Tensor action, torch::Tensor reward, torch::Tensor next_state, torch::Tensor done)
    {
        addExperienceState(std::make_tuple(state, action, reward, next_state, done));
    }

    void addExperienceState(Experience experience) {
        circular_buffer.push_back(experience);
    }

    std::vector<Experience> sample(int num_agent) {
        std::vector<Experience> experiences;
        for (int i = 0; i < num_agent; i++) {
            experiences.push_back(sample());
        }
        return experiences;
    }

    Experience sample() {
            return circular_buffer.at(static_cast<size_t>(rand() % static_cast<int>(circular_buffer.size())));
    }

    size_t getLength() {
        return circular_buffer.size();
    }



private:
    boost::circular_buffer<Experience> circular_buffer{10000};
};

#endif //PROJECT_REPLAYBUFFER_H

