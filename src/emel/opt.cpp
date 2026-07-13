#include "opt.h"
#include <cmath>

// ---- loss helpers ----

// average every entry and keep the scalar as shape [1]
static GTensor mean_all(const GTensor &values) {
    int cnt = values.get_tensor().num_el();
    GTensor sum = values;
    int axes = sz(values.get_tensor().shape);
    for (int axis = axes-1; axis > 0; --axis) {
        sum = sum.sum_reduce(axis, false);
    }
    sum = sum.sum_reduce(0, true);
    return sum.ediv(GTensor({1}, cnt));
}

// compute numerically stable log-softmax along an axis; positions values as Q in H(P,Q)
static GTensor log_softmax(const GTensor &values, int axis) {
    GTensor shifted = values - values.max_reduce(axis, true);
    return shifted - shifted.exp().sum_reduce(axis, true).log();
}

// compute cross entropy from probabilities
static GTensor cross_entropy_probs(const GTensor &Yhat, const GTensor &Y) {
    int last_axis = sz(Y.get_tensor().shape) - 1;
    return -mean_all(Y.hadamard(Yhat.log()).sum_reduce(last_axis, false));
}

// compute cross entropy from logits
static GTensor cross_entropy_logits(const GTensor &Yhat, const GTensor &Y) {
    int last_axis = sz(Y.get_tensor().shape) - 1;
    GTensor log_probs = log_softmax(Yhat, last_axis);
    return -mean_all(Y.hadamard(log_probs).sum_reduce(last_axis, false));
}

// compute cross entropy from logits, where Y holds class indices (one axis fewer than Yhat)
static GTensor cross_entropy_logits_sparse(const GTensor &Yhat, const GTensor &Y) {
    const vec<int> &yhat_shape = Yhat.get_tensor().shape;
    const vec<int> &y_shape = Y.get_tensor().shape;
    int logits_axis = sz(yhat_shape) - 1;
    GTensor logprobs = log_softmax(Yhat, logits_axis);

    // source distribution probs should be one-hot on target logits
    // so just run logprobs.gather using class indices in Y
    // flatten because non-logit axis is the same for both
    int logit_count = yhat_shape.back();
    int numel = Yhat.get_tensor().num_el();
    GTensor flat_logprobs = logprobs.reshape({numel});
    Tensor I(y_shape, 0.);
    vec<int> cur(sz(y_shape), 0);
    do {
        int prod = 1;
        for (int i = 0; i < sz(cur); ++i) {
            prod *= y_shape[i];
        }
        prod *= logit_count;

        int block_st = 0;
        for (int i = 0; i < sz(cur); ++i) {
            prod /= y_shape[i];
            block_st += prod * cur[i];
        }

        I.at(cur) = block_st + Y.get_tensor().at(cur);
    } while (advance_ind(cur, y_shape));

    return -mean_all(flat_logprobs.gather_flat(I));
}

// apply loss to nn output
GTensor apply_loss(const GTensor &Yhat, const GTensor &Y, Loss loss) {
    switch (loss) {
    case Loss::CrossEntropy:
        return cross_entropy_probs(Yhat, Y);
    case Loss::CrossEntropyLogits:
        return cross_entropy_logits(Yhat, Y);
    case Loss::CrossEntropyLogitsSparse:
        return cross_entropy_logits_sparse(Yhat, Y);
    }

    __builtin_unreachable();
}

// apply loss function, but return scalar
float apply_loss_scalar(const GTensor &Yhat, const GTensor &Y, Loss loss) {
    return apply_loss(Yhat, Y, loss).get_tensor().at({0});
}

// ---- sgd ----

// ctor
Sgd::Sgd(const vec<GTensor*> &params, float alpha) {
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
Adam::Adam(const vec<GTensor*> &params, float alpha, float beta1, float beta2, float eps) {
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
    float bc1 = 1 - pow(beta1, (float)t);
    float bc2 = 1 - pow(beta2, (float)t);

    for (int i = 0; i < sz(params); ++i) {
        const Tensor &g = params[i]->get_grad();

        m[i] = m[i].apply(g, [&](float mm, float gg) { return beta1*mm + (1-beta1)*gg; });
        v[i] = v[i].apply(g, [&](float vv, float gg) { return beta2*vv + (1-beta2)*gg*gg; });

        params[i]->get_tensor_ref() -= m[i].apply(v[i], [&](float mm, float vv) {
            return alpha * (mm/bc1) / (sqrt(vv/bc2) + eps);
        });
    }
}
