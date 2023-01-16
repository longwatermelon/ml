#pragma once
#include <vector>
#include <cstddef>

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

        void fit(const std::vector<std::vector<float>> &mx,
                 const std::vector<float> &y, size_t epochs);

        void set_loss(Losses loss);

    private:
        void forward_prop(const std::vector<std::vector<float>> &mx);
        void back_prop();

    private:
        std::vector<Layer> m_layers;
        Losses m_loss{ Losses::SparseCategoricalCrossentropy };
    };
}

