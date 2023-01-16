#include "deepnn.h"
#include <reg.h>
#include <cstddef>

void nn::Model::compile()
{
    for (size_t i = 1; i < m_layers.size(); ++i)
    {
        int prev_nunits = m_layers[i - 1].nunits;
        for (auto &vw : m_layers[i].mw)
            vw = std::vector<float>(prev_nunits, 0.f);
    }
}

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
    for (size_t i = 0; i < m_layers.size(); ++i)
    {
    }
}

void nn::Model::forward_prop_solve(nn::Layer &curr, const nn::Layer &prev)
{
}

void nn::Model::back_prop()
{
}

