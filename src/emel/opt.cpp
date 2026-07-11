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

// compute cross entropy from probabilities
static GTensor cross_entropy_probs(const GTensor &Yhat, const GTensor &Y) {
    GTensor log_Yhat = Yhat.log();
    GTensor YlogYhat = Y.hadamard(log_Yhat);
    int last_axis = sz(Y.get_tensor().shape) - 1;
    GTensor sum_per_example = YlogYhat.sum_reduce(last_axis, false);

    return -mean_all(sum_per_example);
}

// compute cross entropy from logits
static GTensor cross_entropy_logits(const GTensor &Yhat, const GTensor &Y) {
    int last_axis = sz(Y.get_tensor().shape) - 1;
    GTensor row_max = Yhat.max_reduce(last_axis, false);
    vec<int> new_shape = Y.get_tensor().shape;
    new_shape.back() = 1;
    GTensor shifted = Yhat - row_max.reshape(new_shape);
    GTensor log_sum_exp = shifted.exp().sum_reduce(last_axis, false).log() + row_max;
    GTensor true_logits = Y.hadamard(Yhat).sum_reduce(last_axis, false);
    GTensor loss_per_example = log_sum_exp - true_logits;

    return mean_all(loss_per_example);
}

// compute cross entropy from logits, where Y holds class indices (one axis fewer than Yhat)
static GTensor cross_entropy_logits_sparse(const GTensor &Yhat, const GTensor &Y) {
    const Tensor &Yi = Y.get_tensor();
    const vec<int> &yhat_shape = Yhat.get_tensor().shape;
    assert(sz(Yi.shape) + 1 == sz(yhat_shape));

    // logsumexp over the class axis, max-shifted for stability
    int last_axis = sz(yhat_shape) - 1;
    GTensor row_max = Yhat.max_reduce(last_axis, false);
    vec<int> keep_shape = yhat_shape;
    keep_shape.back() = 1;
    GTensor shifted = Yhat - row_max.reshape(keep_shape);
    GTensor log_sum_exp = shifted.exp().sum_reduce(last_axis, false).log() + row_max;

    // gather the true-class logits: I[ind] = {ind..., Yi[ind]}
    vec<int> I_shape = Yi.shape;
    I_shape.push_back(sz(yhat_shape));
    Tensor I(I_shape, 0.f);
    vec<int> cur(sz(Yi.shape), 0);
    do {
        vec<int> icur = cur;
        icur.push_back(0);
        for (int a = 0; a < sz(cur); ++a) {
            icur.back() = a;
            I.at(icur) = cur[a];
        }
        icur.back() = sz(cur);
        I.at(icur) = Yi.at(cur);
    } while (advance_ind(cur, Yi.shape));
    GTensor true_logits = Yhat.gather(I);

    return mean_all(log_sum_exp - true_logits);
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
