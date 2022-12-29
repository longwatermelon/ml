#pragma once
#include <vector>
#include <cstddef>
#include <array>

namespace nn
{
    // N: num features
    template <size_t N>
    struct Neuron
    {
        Neuron()
        {
            weights.fill(0.f);
        }

        Neuron(const std::array<float, N> &weights, float bias)
            : weights(weights), bias(bias) {}

        std::array<float, N> weights;
        float bias{ 0.f };
    };

    // Nn: num neurons
    // Nf: num features
    template <size_t Nn, size_t Nf>
    struct Dense
    {
        Dense()
            : neurons() {}

        Dense(const std::array<Neuron<Nf>, Nn> &neurons)
            : neurons(neurons) {}

        std::array<Neuron<Nf>, Nn> neurons;
    };
}

