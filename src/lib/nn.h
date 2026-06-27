#pragma once
#include "tensor.h"

enum class Activation {
    Linear,
    Relu,
    Softmax,
};

struct Layer {
    Activation act;
    int n;
    Tensor W,Z,A,b;

    Layer(int n, int n_prev, Activation act);

    // forward pass using prev layer's output --- updates Z, A
    void forward(const Tensor &A_prev);
};

class Nn {
    vec<Layer> m_layers;

public:
    Nn(const vec<pair<int, Activation>> &layers);

    // forward prop
    void forward(Tensor X);
};
