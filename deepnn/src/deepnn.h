#pragma once
#include "matrix.h"
#include <vector>
#include <cstddef>

namespace nn
{
    enum class Activation
    {
        Sigmoid,
        Linear,
        Relu,
        Tanh
    };

    struct Layer
    {
        Layer(int n, Activation a_fn)
            : n(n), a_fn(a_fn) {}

        int n;
        mt::mat W;
        mt::vec vb;
        Activation a_fn;

        mt::mat A, Z;
    };

    class Model
    {
    public:
        Model(const std::vector<Layer> &layers);
        ~Model() = default;

        void train(const mt::mat &X, const mt::mat &Y, int epochs);

    private:
        void forward_prop(const mt::mat &X);
        void back_prop(const mt::mat &Y);
        void apply_diffs(int l,
                const mt::mat &dW,
                const mt::vec &db
        );
        mt::mat gprime(int l, const mt::mat &Z);

    private:
        std::vector<Layer> m_layers;
    };
}
