#include "nn.h"
namespace ag = autograd;

// neuron count, input feature count, activation fn
Layer::Layer(int n, int n_prev, Activation act) : act(act), n(n) {
    this->W = ag::fns::leaf(Tensor({n, n_prev}, 0.));
    this->b = ag::fns::leaf(Tensor({n, 1}, 0.));
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
    default: __builtin_unreachable();
    }
}

// forward pass using prev layer's output --- updates Z, A
void Layer::forward(ag::ValuePtr A_prev) {
    this->Z = ag::fns::add(ag::fns::matmul(this->W, A_prev), this->b);
    this->A = apply_act(this->act, this->Z);
}

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
static ag::ValuePtr apply_loss(Loss loss, ag::ValuePtr y, ag::ValuePtr yhat) {
    switch (loss) {
    case Loss::CrossEntropy: {
        ag::ValuePtr log_yhat = ag::fns::log(yhat);
        ag::ValuePtr ylogyhat = ag::fns::hadamard(y, log_yhat);
        ag::ValuePtr sum_per_example = ag::fns::sum_reduce(ylogyhat, 0, false);
        ag::ValuePtr sum_batch = ag::fns::sum_reduce(sum_per_example, 0, false);
        int batch_sz = y->result.shape[1];
        ag::ValuePtr avg_batch = ag::fns::ediv(sum_batch, ag::fns::leaf(Tensor({1},batch_sz)));
        ag::ValuePtr neg_avg_batch = ag::fns::hadamard(ag::fns::leaf(Tensor({1},-1.)), avg_batch);
        return neg_avg_batch;
    } break;
    }
}

// back prop, labels y, learning rate alpha
void Nn::backward(Loss loss, const Tensor &y, double alpha) {
    // apply cost
    ag::ValuePtr yhat = m_layers.back().A;
    ag::ValuePtr cost = apply_loss(loss, ag::fns::leaf(y), yhat);

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
