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
double apply_loss_scalar(const GTensor &Yhat, const GTensor &Y, Loss loss);

struct Optimizer {
    // cleanup derived optimizers through base pointers
    virtual ~Optimizer() = default;
    // update parameters
    virtual void step() = 0;
};

struct Sgd : Optimizer {
    vec<GTensor*> params;
    double alpha;

    // ctor
    Sgd(const vec<GTensor*> &params, double alpha);

    // update parameters
    void step() override;
};

struct Adam : Optimizer {
    vec<GTensor*> params;
    double alpha, beta1, beta2, eps;
    vec<Tensor> m, v; // first/second moment estimates per param
    int t; // timestep

    // ctor
    Adam(const vec<GTensor*> &params, double alpha, double beta1 = 0.9, double beta2 = 0.999, double eps = 1e-8);

    // update parameters
    void step() override;
};
