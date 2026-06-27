#pragma once
#include "tensor.h"
#include "autograd.h"

enum class Activation {
    Linear,
    Relu,
    Softmax,
};

struct Layer {
    Activation act;
    int n;
    autograd::ValuePtr W,Z,A,b;

    // neuron count, input feature count, activation fn
    Layer(int n, int n_prev, Activation act);

    // forward pass using prev layer's output --- updates Z, A
    void forward(autograd::ValuePtr A_prev);
};

// ---- loss ----

enum class Loss {
    CrossEntropy,
};

// compute loss
double calc_loss(const Tensor &Yhat, const Tensor &Y, Loss loss);

class Nn {
    vec<Layer> m_layers;

public:
    // ---- nn ctors ----

    // construct with (neuron count, activation) info, plus input layer's # features
    Nn(int input_features, const vec<pair<int, Activation>> &layers);

    // ---- standard nn ops ----

    // train nn over epochs (minibatching), with learning rate alpha and a loss
    void train(const Tensor &X, const Tensor &Y, int epochs, int batch_size, double alpha, Loss loss);
    // forward pass, returning activations of last layer
    Tensor predict(const Tensor &X);

private:
    // ---- nn internals ----

    // forward prop
    void forward(const Tensor &X);
    // back prop, labels y, learning rate alpha
    void backward(Loss loss, const Tensor &Y, double alpha);
};
