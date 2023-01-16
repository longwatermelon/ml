#pragma once
#include <vector>

namespace nn
{
    enum class Activation
    {
        ReLU,
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
        Activation activation{ Activation::ReLU };

        Layer(int nunits, Activation activation)
            : nunits(nunits), activation(activation) {}
    };

    class Model
    {
    public:
        Model(const std::vector<Layer> &layers)
            : m_layers(layers) {}
        ~Model() = default;

        void compile(Losses loss);

    private:
        std::vector<Layer> m_layers;
        Losses m_loss{ Losses::SparseCategoricalCrossentropy };
    };
}

