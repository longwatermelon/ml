#pragma once
#include "autograd.h"

// ---- loss ----

enum class Loss {
    CrossEntropy,
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
