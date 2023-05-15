#pragma once
#include <vector>
#include <cstddef>
#include <Eigen/Dense>

namespace deepnn
{
    enum class Activation
    {
        Sigmoid,
        Linear,
        Relu
    };

    struct Layer
    {
        Layer(int n, Activation a_fn)
            : n(n), a_fn(a_fn) {}

        int n;
        Eigen::MatrixXf W;
        Eigen::VectorXf vb;
        Activation a_fn;
    };

    class Model
    {
    public:
        Model(const std::vector<Layer> &layers);
        ~Model() = default;

    private:
        std::vector<Layer> m_layers;
    };
}
