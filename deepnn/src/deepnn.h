#pragma once
#include "matrix.h"
#include <vector>
#include <cstddef>

namespace nn
{
    enum class Activation
    {
        Sigmoid = 0,
        Linear = 1,
        Relu = 2,
        Tanh = 3
    };

    struct Layer
    {
        Layer() = default;
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
        Model(const std::string &src);
        ~Model() = default;

        void train(const mt::mat &X, const mt::mat &Y, int epochs, float a);
        std::vector<float> predict(const mt::mat &X);

        void save_params(const std::string &fp);

        Layer layer(int i) const { return m_layers[i]; }

    private:
        void forward_prop(const mt::mat &X);
        void back_prop(const mt::mat &Y, float a);
        void apply_diffs(int l,
                const mt::mat &dW,
                const mt::vec &db,
                float a
        );
        mt::mat gprime(int l, const mt::mat &Z);

        float cost(const mt::mat &Y);

    public:
        std::vector<Layer> m_layers;
    };
}
