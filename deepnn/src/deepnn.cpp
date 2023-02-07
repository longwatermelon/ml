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
        forward_prop_solve(m_layers[i],
            i == 0 ? Layer(mx.size(), Activation::None) : m_layers[i - 1]);
    }
}

void nn::Model::forward_prop_solve(nn::Layer &curr, const nn::Layer &prev)
{
    curr.vz = reg::vec::matmul(curr.mw, prev.va);
    curr.vz = reg::vec::apply_fn(curr.vz,
            [prev](std::vector<float> &v, size_t i){
        v[i] += prev.vb[i];
    });

    curr.va = activation_fn(curr.activation, curr.vz);
}

void nn::Model::back_prop()
{
}

std::vector<float> nn::Model::activation_fn(Activation type, const std::vector<float> &vz)
{
    switch (type)
    {
    case Activation::Linear:
        return vz;
    case Activation::Relu:
        return reg::vec::apply_fn(vz, [](float z){ return std::max(0.f, z); });
    case Activation::Sigmoid:
        return reg::vec::apply_fn(vz, [](float z){ return 1.f / (1.f + std::exp(-z)); });
    case Activation::None:
        return vz;
    }
}

