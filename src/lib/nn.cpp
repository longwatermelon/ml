#include "nn.h"

Layer::Layer(int n, int n_prev, Activation act) : act(act), n(n) {
    this->W = Tensor({n, n_prev}, 0.);
    this->b = Tensor({n, 1}, 0.);
}

// applies the activation to Z column-wise, returning A of the same shape
static Tensor apply_act(Activation act, const Tensor &Z) {
    switch (act) {
    case Activation::Linear:
        return Z;
    case Activation::Relu:
        return Z.apply([](double z){ return max(0., z); });
    case Activation::Softmax: {
        // // each column is one example's pre-activation vector, normalized independently
        // Tensor A(Z.rows, Z.cols);
        // for (int j = 0; j < Z.cols; j++) {
        //     // subtract the column max for numerical stability before exp
        //     Tensor col = Z.get_col(j);
        //     double mx = col.max();
        //     Tensor e = col.apply([mx](double z){ return exp(z - mx); });
        //     double s = e.sum();
        //     for (int i = 0; i < Z.rows; i++)
        //         A(i, j) = e(i, 0) / s;
        // }
        return Z;
    }
    default: __builtin_unreachable();
    }
}


// forward pass using prev layer's output --- updates Z, A
void Layer::forward(const Tensor &A_prev) {
    this->Z = this->W * A_prev + b;
    this->A = apply_act(this->act, this->Z);
}

Nn::Nn(const vec<pair<int, Activation>> &layers) {
    // placeholder n_prev for layer 0, it doesn't matter; A = X in forward pass.
    int n_prev = 1;
    for (auto &[n, act] : layers) {
        m_layers.push_back(Layer(n, n_prev, act));
        n_prev = n;
    }
}

// forward prop
void Nn::forward(Tensor X) {
    // receive input into network
    m_layers[0].A = X;

    // forward
    for (int i = 1; i < sz(m_layers); ++i) {
        m_layers[i].forward(m_layers[i-1].A);
    }
}
