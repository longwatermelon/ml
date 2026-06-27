#include "nn.h"
#include <numeric>
#include <random>
#include <algorithm>
namespace ag = autograd;

// neuron count, input feature count, activation fn
Layer::Layer(int n, int n_prev, Activation act) : act(act), n(n) {
    this->W = ag::fns::leaf(Tensor({n, n_prev}, 0.));
    this->b = ag::fns::leaf(Tensor({n, 1}, 0.));

    // random init to [-0.5, 0.5]
    this->W->result.apply_inplace([](double x){return (double)(rand() % 100) / 99 - 0.5;});
}

// applies the activation to Z column-wise, returning A of the same shape
static ag::ValuePtr apply_act(Activation act, ag::ValuePtr Z) {
    switch (act) {
    case Activation::Linear:
        return Z;
    case Activation::Relu:
        return ag::fns::relu(Z);
    case Activation::Softmax: {
        ag::ValuePtr argmax = ag::fns::max_reduce(Z, 0, true);
        ag::ValuePtr numerator = ag::fns::exp(ag::fns::add(Z, ag::fns::hadamard(ag::fns::leaf(Tensor({1}, -1.)), argmax)));
        ag::ValuePtr sum = ag::fns::sum_reduce(numerator, 0, true);
        ag::ValuePtr result = ag::fns::ediv(numerator, sum);
        return result;
    }
    }

    __builtin_unreachable();
}

// forward pass using prev layer's output --- updates Z, A
void Layer::forward(ag::ValuePtr A_prev) {
    this->Z = ag::fns::add(ag::fns::matmul(this->W, A_prev), this->b);
    this->A = apply_act(this->act, this->Z);
}

// ---- public API ----

// construct with (neuron count, activation) info, plus input layer's # features
Nn::Nn(int input_features, const vec<pair<int, Activation>> &layers) {
    // placeholder input layer; doesn't matter, it's only a placeholder for A=X.
    m_layers = {Layer(input_features,1,Activation::Linear)};

    // push hidden layers / output layer
    int n_prev = m_layers[0].n;
    for (auto &[n, act] : layers) {
        m_layers.push_back(Layer(n, n_prev, act));
        n_prev = n;
    }
}

// train nn over epochs (minibatching), with learning rate alpha and a loss
void Nn::train(const Tensor &X, const Tensor &Y, int epochs, int batch_size, double alpha, Loss loss) {
    assert(batch_size > 0);

    int m = X.shape[1];
    random_device rd;
    mt19937 g(rd());

    for (int epoch = 0; epoch < epochs; ++epoch) {
        if (epoch % 10 == 0) {
            printf("\repoch %d/%d...", epoch+1, epochs);
            fflush(stdout);
        }

        // minibatching - process in chunks of batch_size
        // shuffle data order first
        vec<int> inds(m);
        iota(all(inds), 0);
        shuffle(all(inds), g);

        for (int st = 0; st < m; st += batch_size) {
            int cur_batch = min(batch_size, m-st);

            // copy minibatch over
            Tensor Xb({X.shape[0], cur_batch}, 0.);
            Tensor Yb({Y.shape[0], cur_batch}, 0.);
            for (int j = 0; j < cur_batch; ++j) {
                int ind = inds[st+j];
                for (int i = 0; i < Xb.shape[0]; ++i) {
                    Xb.at({i,j}) = X.at({i,ind});
                }
                for (int i = 0; i < Yb.shape[0]; ++i) {
                    Yb.at({i,j}) = Y.at({i,ind});
                }
            }

            forward(Xb);
            backward(loss, Yb, alpha);
        }
    }
}

// ---- internals ----

// forward prop
void Nn::forward(const Tensor &X) {
    // receive input into network
    m_layers[0].A = ag::fns::leaf(X);

    // forward
    for (int i = 1; i < sz(m_layers); ++i) {
        m_layers[i].forward(m_layers[i-1].A);
    }
}

// apply loss to nn output
static ag::ValuePtr apply_loss(Loss loss, ag::ValuePtr Y, ag::ValuePtr Yhat) {
    switch (loss) {
    case Loss::CrossEntropy: {
        ag::ValuePtr log_Yhat = ag::fns::log(Yhat);
        ag::ValuePtr YlogYhat = ag::fns::hadamard(Y, log_Yhat);
        ag::ValuePtr sum_per_example = ag::fns::sum_reduce(YlogYhat, 0, false);
        ag::ValuePtr sum_batch = ag::fns::sum_reduce(sum_per_example, 0, false);
        int batch_sz = Y->result.shape[1];
        ag::ValuePtr avg_batch = ag::fns::ediv(sum_batch, ag::fns::leaf(Tensor({1},batch_sz)));
        ag::ValuePtr neg_avg_batch = ag::fns::hadamard(ag::fns::leaf(Tensor({1},-1.)), avg_batch);
        return neg_avg_batch;
    } break;
    }

    __builtin_unreachable();
}

// back prop, labels y, learning rate alpha
void Nn::backward(Loss loss, const Tensor &Y, double alpha) {
    // apply cost
    ag::ValuePtr Yhat = m_layers.back().A;
    ag::ValuePtr cost = apply_loss(loss, ag::fns::leaf(Y), Yhat);

    // autograd gradients
    ag::compute_all_grads(cost);

    // update parameters
    auto update = [alpha](Tensor &value, Tensor &grad) {
        value -= Tensor({1},alpha).hadamard(grad);
    };

    for (int i = 1; i < sz(m_layers); ++i) {
        update(m_layers[i].W->result, m_layers[i].W->grad);
        update(m_layers[i].b->result, m_layers[i].b->grad);
    }
}
