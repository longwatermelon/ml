#pragma once
#include "matrix.h"

enum class Activation {
    Linear,
    Relu,
    Softmax,
};

struct Layer {
    Activation act;
    int n;
    Matrix W,Z,A,b;

    Layer(int n, int n_prev, Activation act);

    // updates Z, A
    void forward(const Matrix &A_prev);
};

class Nn {
    vec<Layer> m_layers;

public:
    Nn(const vec<pair<int, Activation>> &layers);

    void forward(Matrix X);
};
