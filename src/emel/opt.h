#pragma once
#include "autograd.h"

// ---- loss ----

enum class Loss {
    CrossEntropy,
    CrossEntropyLogits,
    // Y holds class indices (one axis fewer than Yhat) instead of one-hot rows
    CrossEntropyLogitsSparse,
};

// apply loss function
GTensor apply_loss(const GTensor &Yhat, const GTensor &Y, Loss loss);
// apply loss function, but return scalar
float apply_loss_scalar(const GTensor &Yhat, const GTensor &Y, Loss loss);

struct Optimizer {
    // cleanup derived optimizers through base pointers
    virtual ~Optimizer() = default;
    // update parameters
    virtual void step() = 0;
};

struct Sgd : Optimizer {
    vec<GTensor*> params;
    float alpha;

    // ctor
    Sgd(const vec<GTensor*> &params, float alpha);

    // update parameters
    void step() override;
};

struct Adam : Optimizer {
    vec<GTensor*> params;
    float alpha, beta1, beta2, eps;
    vec<Tensor> m, v; // first/second moment estimates per param
    int t; // timestep

    // ctor
    Adam(const vec<GTensor*> &params, float alpha, float beta1 = 0.9f, float beta2 = 0.999f, float eps = 1e-8f);

    // update parameters
    void step() override;
};
