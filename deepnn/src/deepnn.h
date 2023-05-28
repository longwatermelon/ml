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

        // Forward prop
        mt::mat A, Z;

        // Back prop
        mt::mat dZ;
    };

    void dense_forward_prop(Layer &l, Layer &back_l, int m);
    // If front_l == nullptr it is assumed l is the last layer.
    // Y and back_l are only used if front_l == nullptr.
    // Returns dW, d_vb
    std::pair<mt::mat, mt::vec> dense_back_prop(Layer &l, Layer *back_l, Layer *front_l,
                         const mt::mat &Y = mt::mat(0, 0));

    class Model
    {
    public:
        // Simple neural network
        Model(const std::vector<Layer> &layers, float random_init_range = 1.f);
        Model(const std::string &src);
        ~Model() = default;

        void forward_prop(const mt::mat &X);
        void back_prop(const mt::mat &Y, float a);

        void train(const mt::mat &X, const mt::mat &Y, int epochs, float a, int print_intervals = 100);
        std::vector<float> predict(const mt::mat &X);

        void save_params(const std::string &fp);

        Layer layer(int i) const { return m_layers[i]; }

    private:
        void apply_diffs(int l,
                const mt::mat &dW,
                const mt::vec &db,
                float a
        );

        float cost(const mt::mat &Y);

    public:
        std::vector<Layer> m_layers;
    };
}
