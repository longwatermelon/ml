#include "nn.h"
#include <numeric>
#include <random>
#include <algorithm>

// neuron count, input feature count, activation fn
Layer::Layer(int n, int n_prev, Activation act) : act(act), n(n) {
    this->W = GTensor({n, n_prev}, 0.);
    this->b = GTensor({n, 1}, 0.);

    // random init to [-0.5, 0.5]
    this->W.get_tensor_ref().apply_inplace([](double x){return (double)(rand() % 100) / 99 - 0.5;});
}

// applies the activation to Z column-wise, returning A of the same shape
static GTensor apply_act(Activation act, GTensor Z) {
    switch (act) {
    case Activation::Linear:
        return Z;
    case Activation::Relu:
        return Z.relu();
    case Activation::Softmax: {
        GTensor argmax = Z.max_reduce(0, true);
        GTensor numerator = (Z - argmax).exp();
        GTensor sum = numerator.sum_reduce(0, true);
        GTensor result = numerator.ediv(sum);
        return result;
    }
    }

    __builtin_unreachable();
}

// forward pass using prev layer's output --- updates Z, A
void Layer::forward(GTensor A_prev) {
    Z = W * A_prev + b;
    A = apply_act(act, Z);
}

// ---- loss helpers ----

// apply loss to nn output
static GTensor apply_loss(Loss loss, GTensor Y, GTensor Yhat) {
    switch (loss) {
    case Loss::CrossEntropy: {
        GTensor log_Yhat = Yhat.log();
        GTensor YlogYhat = Y.hadamard(log_Yhat);
        GTensor sum_per_example = YlogYhat.sum_reduce(0, false);
        GTensor sum_batch = sum_per_example.sum_reduce(0, true);
        int batch_sz = Y.get_tensor().shape[1];
        GTensor avg_batch = sum_batch.ediv(GTensor({1}, batch_sz));
        return -avg_batch;
    } break;
    }

    __builtin_unreachable();
}

double calc_loss(const Tensor &Yhat, const Tensor &Y, Loss loss) {
    GTensor out = apply_loss(loss, GTensor(Y), GTensor(Yhat));
    return out.get_tensor().at({0});
}

// ---- nn ctors ----

// construct with (neuron count, activation) info, plus input layer's # features
Nn::Nn(int input_features, const vec<pair<int, Activation>> &layers) {
    // placeholder input layer; doesn't matter, it's only a placeholder for A=X.
    m_layers = {Layer(input_features, 1, Activation::Linear)};

    // push hidden layers / output layer
    int n_prev = m_layers[0].n;
    for (auto &[n, act] : layers) {
        m_layers.push_back(Layer(n, n_prev, act));
        n_prev = n;
    }
}

// ---- standard nn ops ----

// train nn over epochs (minibatching), with learning rate alpha and a loss
void Nn::train(const Tensor &X, const Tensor &Y, int epochs, int batch_size, double alpha, Loss loss) {
    assert(batch_size > 0);

    int m = X.shape[1];
    mt19937 g(0);

    for (int epoch = 0; epoch < epochs; ++epoch) {
        // eval
        const Tensor &Yhat = predict(X);
        printf("\repoch %d/%d... | loss = %.6f", epoch+1, epochs, calc_loss(Yhat, Y, loss));
        fflush(stdout);

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
    putchar('\n');
}

// forward pass, returning activations of last layer
Tensor Nn::predict(const Tensor &X) {
    forward(X);
    return m_layers.back().A.get_tensor();
}

// ---- nn internals ----

// forward prop
void Nn::forward(const Tensor &X) {
    // receive input into network
    m_layers[0].A = GTensor(X);

    // forward
    for (int i = 1; i < sz(m_layers); ++i) {
        m_layers[i].forward(m_layers[i-1].A);
    }
}

// back prop, labels y, learning rate alpha
void Nn::backward(Loss loss, const Tensor &Y, double alpha) {
    // apply cost
    GTensor cost = apply_loss(loss, GTensor(Y), GTensor(m_layers.back().A));

    // autograd gradients
    cost.compute_all_grads();

    // update parameters
    auto update = [alpha](Tensor &value, const Tensor &grad) {
        value -= Tensor({1},alpha).hadamard(grad);
    };

    for (int i = 1; i < sz(m_layers); ++i) {
        update(m_layers[i].W.get_tensor_ref(), m_layers[i].W.get_grad());
        update(m_layers[i].b.get_tensor_ref(), m_layers[i].b.get_grad());
    }
}
