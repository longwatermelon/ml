#pragma once
#include <vector>

namespace nn
{
    enum class Activation
    {
        RELU,
        LINEAR,
        SIGMOID
    };

    struct Layer
    {
        int nunits{ 0 };
        Activation activation{ Activation::RELU };

        Layer(int nunits, Activation activation)
            : nunits(nunits), activation(activation) {}
    };

    class Model
    {
    public:
        Model(const std::vector<Layer> &layers)
            : m_layers(layers) {}
        ~Model() = default;

        void compile();

    private:
        std::vector<Layer> m_layers;
    };
}

