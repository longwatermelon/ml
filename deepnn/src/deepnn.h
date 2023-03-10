#pragma once
#include <vector>
#include <cstddef>
#include <glm/glm.hpp>

namespace nn
{
    enum class Activation
    {
        None,
        Relu,
        Sigmoid,
        Linear
    };

    enum class Losses
    {
        SparseCategoricalCrossentropy
    };

    struct Layer
    {
        int nunits{ 0 };
        Activation activation{ Activation::None };

        std::vector<std::vector<float>> mw;
        std::vector<float> vb, vz, va;

        Layer(int nunits, Activation activation)
            : nunits(nunits), activation(activation)
        {
            mw.resize(nunits);
            vb.resize(nunits);
            vz.resize(nunits);
            va.resize(nunits);
        }
    };

    class Model
    {
    public:
        Model(const std::vector<Layer> &layers)
            : m_layers(layers) {}
        ~Model() = default;

        // First hidden layer will not have its column vectors initialized until Model::fit
        void compile();

        void fit(const std::vector<std::vector<float>> &mx,
                 const std::vector<float> &y, size_t epochs);

        void set_loss(Losses loss);

    private:
        void forward_prop(const std::vector<std::vector<float>> &mx);
        void forward_prop_solve(nn::Layer &curr, const nn::Layer &prev);
        void back_prop();

        std::vector<float> activation_fn(Activation type, const std::vector<float> &vz);

    private:
        std::vector<Layer> m_layers;
        Losses m_loss{ Losses::SparseCategoricalCrossentropy };
    };
}

