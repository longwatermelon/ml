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

enum class Loss {
    CrossEntropy,
};

class Nn {
    vec<Layer> m_layers;

public:
    // ---- public API ----

    // construct with (neuron count, activation) info, plus input layer's # features
    Nn(int input_features, const vec<pair<int, Activation>> &layers);

    // train nn over epochs (minibatching), with learning rate alpha and a loss
    void train(const Tensor &X, const Tensor &Y, int epochs, int batch_size, double alpha, Loss loss);

private:
    // ---- internals ----

    // forward prop
    void forward(const Tensor &X);
    // back prop, labels y, learning rate alpha
    void backward(Loss loss, const Tensor &Y, double alpha);
};
