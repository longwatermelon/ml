#include "deepnn.h"
#include <impl.h>
#include <cstddef>

void nn::Model::fit(const std::vector<std::vector<float>> &mx,
         const std::vector<float> &y, size_t epochs)
{
    for (size_t i = 0; i < epochs; ++i)
    {
        forward_prop(mx);
        back_prop();
    }
}

void nn::Model::forward_prop(const std::vector<std::vector<float>> &mx)
{
    auto solve_va = [&](const std::vector<float> &va_prev){
    };

    for (size_t i = 0; i <= m_layers.size(); ++i)
    {
    }
}

void nn::Model::back_prop()
{
}

