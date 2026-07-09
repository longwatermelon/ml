#include "opt.h"
#include <cmath>

// ---- loss helpers ----

// compute cross entropy from probabilities
static GTensor cross_entropy_probs(const GTensor &Yhat, const GTensor &Y) {
    GTensor log_Yhat = Yhat.log();
    GTensor YlogYhat = Y.hadamard(log_Yhat);
    GTensor sum_per_example = YlogYhat.sum_reduce(1, false);
    GTensor sum_batch = sum_per_example.sum_reduce(0, true);
    int batch_sz = Y.get_tensor().shape[0];
    GTensor avg_batch = sum_batch.ediv(GTensor({1}, batch_sz));
    return -avg_batch;
}

// compute cross entropy from logits
static GTensor cross_entropy_logits(const GTensor &Yhat, const GTensor &Y) {
    int batch_sz = Y.get_tensor().shape[0];
    GTensor row_max = Yhat.max_reduce(1, false);
    GTensor shifted = Yhat - row_max.reshape({batch_sz, 1});
    GTensor log_sum_exp = shifted.exp().sum_reduce(1, false).log() + row_max;
    GTensor true_logits = Y.hadamard(Yhat).sum_reduce(1, false);
    GTensor loss_per_example = log_sum_exp - true_logits;
    GTensor sum_batch = loss_per_example.sum_reduce(0, true);
    return sum_batch.ediv(GTensor({1}, batch_sz));
}

// apply loss to nn output
GTensor apply_loss(const GTensor &Yhat, const GTensor &Y, Loss loss) {
    switch (loss) {
    case Loss::CrossEntropy:
        return cross_entropy_probs(Yhat, Y);
    case Loss::CrossEntropyLogits:
        return cross_entropy_logits(Yhat, Y);
    }

    __builtin_unreachable();
}

// apply loss function, but return scalar
double apply_loss_scalar(const GTensor &Yhat, const GTensor &Y, Loss loss) {
    return apply_loss(Yhat, Y, loss).get_tensor().at({0});
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

// ---- adam ----

// ctor
Adam::Adam(const vec<GTensor*> &params, double alpha, double beta1, double beta2, double eps) {
    this->params = params;
    this->alpha = alpha;
    this->beta1 = beta1;
    this->beta2 = beta2;
    this->eps = eps;
    this->t = 0;

    // zero-init moment estimates matching param shapes
    for (GTensor *p : params) {
        m.push_back(Tensor(p->get_tensor().shape, 0));
        v.push_back(Tensor(p->get_tensor().shape, 0));
    }
}

// update parameters
void Adam::step() {
    t++;
    // bias correction denominators
    double bc1 = 1 - pow(beta1, t);
    double bc2 = 1 - pow(beta2, t);

    for (int i = 0; i < sz(params); ++i) {
        const Tensor &g = params[i]->get_grad();

        m[i] = m[i].apply(g, [&](double mm, double gg) { return beta1*mm + (1-beta1)*gg; });
        v[i] = v[i].apply(g, [&](double vv, double gg) { return beta2*vv + (1-beta2)*gg*gg; });

        params[i]->get_tensor_ref() -= m[i].apply(v[i], [&](double mm, double vv) {
            return alpha * (mm/bc1) / (sqrt(vv/bc2) + eps);
        });
    }
}
