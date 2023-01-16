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
}

void nn::Model::forward_prop(const std::vector<std::vector<float>> &mx)
{
}

void nn::Model::back_prop()
{
}

