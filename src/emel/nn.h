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
    template <typename Layer, typename... Args>
    Layer &add(Args&&... args) {
        static_assert(std::is_base_of<Module, Layer>::value, "Layer must derive from nn::Module");

        auto layer = std::make_unique<Layer>(std::forward<Args>(args)...);
        Layer &ref = *layer;
        layers.push_back(std::move(layer));
        return ref;
    }

    // forward pass
    GTensor forward(const GTensor &A_prev) override;
    // params
    vec<GTensor*> params() override;
};

struct Conv2d : Module {
    // params
    GTensor W,b;

    // ctor
    Conv2d() = default;

    // forward pass
    GTensor forward(const GTensor &A_prev) override;
    // params
    vec<GTensor*> params() override;
};

// train a model
void train(Module &model, const Tensor &X, const Tensor &Y, int epochs, Loss loss, Optimizer &opt, int batch_size = 32);

} // namespace nn
