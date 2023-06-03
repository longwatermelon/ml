#pragma once
#include "matrix.h"
#include <vector>
#include <cstddef>
#include <memory>

namespace nn
{
    enum class Activation
    {
        Sigmoid = 0,
        Linear = 1,
        Relu = 2,
        Tanh = 3
    };

    enum class LayerType
    {
        DENSE,
        CONV
    };

    using shape4 = std::vector<std::vector<mt::mat>>;

    struct ParamDiff
    {
        LayerType type;

        // Dense
        mt::mat dense_dW;
        mt::vec dense_db;

        // Conv
        shape4 conv_dW;
        mt::vec conv_db;
    };

    struct Input
    {
        mt::mat dense;
        shape4 conv;
    };

    struct Layer
    {
        virtual ~Layer() = default;
        virtual LayerType type() const = 0;
    };

    struct Dense : Layer
    {
        Dense() = default;
        Dense(int n)
            : n(n) {}
        Dense(int n, Activation a_fn)
            : n(n), a_fn(a_fn) {}

        int n;
        mt::mat W;
        mt::vec vb;
        Activation a_fn{ Activation::Linear };

        // Forward prop
        mt::mat A, Z;

        // Back prop
        mt::mat dZ;

        void forward_prop(const Dense *back_l, int m);
        // If front_l == nullptr it is assumed l is the last layer.
        // Y and back_l are only used if front_l == nullptr.
        // Returns dW, d_vb
        ParamDiff back_prop(const Dense *behind, const Dense *front,
                     const mt::mat &Y = mt::mat(0, 0));

        LayerType type() const override { return LayerType::DENSE; }
    };

    struct Conv : Layer
    {
        Conv() = default;
        Conv(int filters, int filterh, int filterw, int stride, Activation a_fn)
            : nc(filters), fh(filterh), fw(filterw), s(stride), a_fn(a_fn) {}

        int nc;
        int fh, fw;
        int s;

        // First index i gives W matrix for the ith filter
        // Second index j gives jth channel of the ith filter
        shape4 W;
        mt::vec b;

        // First index i gives the 3D A matrix for the ith example (out of m)
        // Second index j gives the jth channel of the ith A matrix, dimensions n_h by n_w
        shape4 A, Z;

        // Back prop
        shape4 dZ;

        Activation a_fn{ Activation::Linear };

        void forward_prop(const Conv *back_l, int m);
        ParamDiff back_prop(const Conv *behind, const Conv *front,
                     const mt::mat &Y = mt::mat(0, 0));

        LayerType type() const override { return LayerType::CONV; }

        // TODO Account for nonzero padding
        int nh(int prev_nh) const { return (prev_nh - fh) / s + 1; }
        int nw(int prev_nw) const { return (prev_nw - fw) / s + 1; }
    };

    class Model
    {
    public:
        // Simple neural network
        Model() = default;
        Model(const std::string &src);
        ~Model() = default;

        void forward_prop(const Input &X);
        void back_prop(const mt::mat &Y, float a);

        void train(const Input &X, const mt::mat &Y, int epochs, float a, int print_intervals = 100);
        std::vector<float> predict(const Input &X);

        void save_params(const std::string &fp);

        template <typename T>
        void add(std::unique_ptr<T> l)
        {
            m_layers.emplace_back(std::move(l));
        }

        const Layer *layer(int i) const { return m_layers[i].get(); }

    private:
        void apply_diffs(int l, const ParamDiff &diff, float a);
        float cost(const mt::mat &Y);

        void prepare_layer0(const Input &X);

    public:
        std::vector<std::unique_ptr<Layer>> m_layers;
    };
}
