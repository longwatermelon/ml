#pragma once
#include "tensor.h"
#include "autograd.h"
#include "opt.h"
#include <type_traits>
#include <utility>

namespace nn {

struct Module {
    // cleanup derived modules through base pointers
    virtual ~Module() = default;
    // forward pass, X as input
    virtual GTensor forward(const GTensor &X) = 0;
    // return the parameters of the model
    virtual vec<GTensor*> params() = 0;
};

// train a model
void train(Module &model, const Tensor &X, const Tensor &Y, int epochs, Loss loss, Optimizer &opt, int batch_size = 32);

struct Linear : Module {
    // params
    GTensor W,b;

    // ctor
    Linear(int n_prev, int n);

    // forward pass
    GTensor forward(const GTensor &A_prev) override;
    // params
    vec<GTensor*> params() override;
};

struct Relu : Module {
    // ctor
    Relu() = default;

    // forward pass
    GTensor forward(const GTensor &A_prev) override;
    // params
    vec<GTensor*> params() override;
};

struct Softmax : Module {
    // ctor
    Softmax() = default;

    // forward pass
    GTensor forward(const GTensor &A_prev) override;
    // params
    vec<GTensor*> params() override;
};

struct Sequential : Module {
    vec<std::unique_ptr<Module>> layers;

    // add an owned layer to the sequence
    template <typename T>
    void add(const T &layer) {
        static_assert(std::is_base_of<Module, T>::value, "Layer must derive from nn::Module");

        auto layer_ptr = std::make_unique<T>(layer);
        layers.push_back(std::move(layer_ptr));
    }

    // forward pass
    GTensor forward(const GTensor &A_prev) override;
    // params
    vec<GTensor*> params() override;
};

struct Conv2d : Module {
    // params
    GTensor W,b;

    // hyperparams
    int k;

    // ctor
    Conv2d(int in_channels, int out_channels, int kernel_size);

    // forward pass
    GTensor forward(const GTensor &A_prev) override;
    // params
    vec<GTensor*> params() override;
};

struct Flatten : Module {
    // ctor
    Flatten() = default;

    // forward pass
    GTensor forward(const GTensor &A_prev) override;
    // params
    vec<GTensor*> params() override;
};

struct Attention : Module {
    // params
    GTensor W_Q, W_K, W_V;

    // hyperparams
    int d, d_k, d_v;
    bool causal_mask = false;

    // ctor
    Attention(int d, int d_k, int d_v);
    Attention &with_causal_mask() { causal_mask = true; return *this; }

    // forward pass
    GTensor forward(const GTensor &A_prev) override;
    // params
    vec<GTensor*> params() override;
};

struct MultiHeadAttention : Module {
    // params
    vec<Attention> heads;
    vec<GTensor> W_O;

    // hyperparams
    int d, h;

    // ctor (d must be divisible by h)
    MultiHeadAttention(int d, int h);
    MultiHeadAttention &with_causal_mask() {
        for (Attention &head : heads) {
            head.with_causal_mask();
        }
        return *this;
    }

    // forward pass
    GTensor forward(const GTensor &A_prev) override;
    // params
    vec<GTensor*> params() override;
};

struct LayerNorm : Module {
    // params
    GTensor gamma, beta;

    // hyperparams
    int d;

    // ctor
    LayerNorm(int d);

    // forward pass
    GTensor forward(const GTensor &A_prev) override;
    // params
    vec<GTensor*> params() override;
};

} // namespace nn
