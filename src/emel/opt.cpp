#include "opt.h"

// ---- loss helpers ----

// apply loss to nn output
GTensor apply_loss(const GTensor &Yhat, const GTensor &Y, Loss loss) {
    switch (loss) {
    case Loss::CrossEntropy: {
        GTensor log_Yhat = Yhat.log();
        GTensor YlogYhat = Y.hadamard(log_Yhat);
        GTensor sum_per_example = YlogYhat.sum_reduce(1, false);
        GTensor sum_batch = sum_per_example.sum_reduce(0, true);
        int batch_sz = Y.get_tensor().shape[0];
        GTensor avg_batch = sum_batch.ediv(GTensor({1}, batch_sz));
        return -avg_batch;
    } break;
    }

    __builtin_unreachable();
}

// apply loss function, but return scalar
double apply_loss_scalar(const GTensor &Yhat, const GTensor &Y, Loss loss) {
    return apply_loss(Yhat, Y, loss).get_tensor().at({0});
}

// apply loss function
GTensor apply_loss(Loss loss, const GTensor &Y, const GTensor &Yhat) {
    switch (loss) {
    case Loss::CrossEntropy: {
        GTensor log_Yhat = Yhat.log();
        GTensor YlogYhat = Y.hadamard(log_Yhat);
        GTensor sum_per_example = YlogYhat.sum_reduce(1, false);
        GTensor sum_batch = sum_per_example.sum_reduce(0, true);
        int batch_sz = Y.get_tensor().shape[0];
        GTensor avg_batch = sum_batch.ediv(GTensor({1}, batch_sz));
        return -avg_batch;
    } break;
    }

    __builtin_unreachable();
}

// ---- sgd ----

// ctor
Sgd::Sgd(const vec<GTensor*> &params, double alpha) {
    this->params = params;
    this->alpha = alpha;
}

// update parameters
void Sgd::step() {
    for (GTensor *p : params) {
        // \theta -= \theta \odot grad
        p->get_tensor_ref() -= Tensor({1},alpha).hadamard(p->get_grad());
    }
}
