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
        DENSE
    };

    struct ParamDiff
    {
        LayerType type;

        // Dense
        mt::mat dense_dW;
        mt::vec dense_db;
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


    class Model
    {
    public:
        // Simple neural network
        Model() = default;
        Model(const std::string &src);
        ~Model() = default;

        void forward_prop(const mt::mat &X);
        void back_prop(const mt::mat &Y, float a);

        void train(const mt::mat &X, const mt::mat &Y, int epochs, float a, int print_intervals = 100);
        std::vector<float> predict(const mt::mat &X);

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

    public:
        std::vector<std::unique_ptr<Layer>> m_layers;
    };
}
