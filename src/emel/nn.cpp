#include "nn.h"
#include <random>
#include <algorithm>

namespace nn {

// ---- linear module ----

// ctor
Linear::Linear(int n_prev, int n) {
    W = GTensor({n_prev, n}, 0.);
    b = GTensor({n}, 0.);

    // random init to [-0.5, 0.5]
    W.get_tensor_ref().apply_inplace([](double x){return (double)(rand() % 100) / 99 - 0.5;});
}

// forward pass
GTensor Linear::forward(const GTensor &A_prev) {
    return A_prev * W + b;
}

// params
vec<GTensor*> Linear::params() {
    return {&W, &b};
}

// ---- relu module ----

// forward pass
GTensor Relu::forward(const GTensor &A_prev) {
    return A_prev.relu();
}

// params
vec<GTensor*> Relu::params() {
    return {};
}

// ---- softmax module ----

// forward pass
GTensor Softmax::forward(const GTensor &A_prev) {
    GTensor argmax = A_prev.max_reduce(1, true);
    GTensor numerator = (A_prev - argmax).exp();
    GTensor sum = numerator.sum_reduce(1, true);
    GTensor result = numerator.ediv(sum);
    return result;
}

// params
vec<GTensor*> Softmax::params() {
    return {};
}

// ---- sequential module ----

// forward pass
GTensor Sequential::forward(const GTensor &A_prev) {
    GTensor in = A_prev;
    for (int i = 0; i < sz(layers); ++i) {
        GTensor out = layers[i]->forward(in);
        in = out;
    }

    return in;
}

// params
vec<GTensor*> Sequential::params() {
    vec<GTensor*> out;
    for (int i = 0; i < sz(layers); ++i) {
        vec<GTensor*> layer_params = layers[i]->params();
        out.insert(end(out), all(layer_params));
    }

    return out;
}

// train a model
void train(Module &model, const Tensor &X, const Tensor &Y, int epochs, Loss loss, Optimizer &opt, int batch_size) {
    assert(batch_size > 0);

    int m = X.shape[0];
    mt19937 g(0);

    for (int epoch = 0; epoch < epochs; ++epoch) {
        // eval
        GTensor Yhat = model.forward(GTensor(X));
        double loss_fullbatch = apply_loss_scalar(Yhat, Y, loss);
        printf("\repoch %d/%d... | loss = %.6f", epoch+1, epochs, loss_fullbatch);
        fflush(stdout);

        // minibatching - process in chunks of batch_size
        // shuffle data order first
        vec<int> inds(m);
        iota(all(inds), 0);
        shuffle(all(inds), g);

        for (int st = 0; st < m; st += batch_size) {
            int cur_batch = min(batch_size, m-st);

            // copy minibatch over
            Tensor Xb({cur_batch, X.shape[1]}, 0.);
            Tensor Yb({cur_batch, Y.shape[1]}, 0.);
            for (int i = 0; i < cur_batch; ++i) {
                int ind = inds[st+i];
                for (int j = 0; j < Xb.shape[1]; ++j) {
                    Xb.at({i,j}) = X.at({ind,j});
                }
                for (int j = 0; j < Yb.shape[1]; ++j) {
                    Yb.at({i,j}) = Y.at({ind,j});
                }
            }

            // forward pass
            GTensor Yhatb = model.forward(Xb);

            // backward pass
            GTensor g_loss = apply_loss(Yhatb, Yb, loss);
            g_loss.compute_all_grads();
            opt.step();
        }
    }
    putchar('\n');
}

} // namespace nn
