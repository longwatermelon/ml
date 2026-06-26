#include "tensor.h"

Tensor::Tensor(const vec<int> &data_1d) {
}

Tensor::Tensor(const vec2<int> &data_2d) {
}

Tensor Tensor::zeros(const vec<int> &shape) {
}

Tensor Tensor::ones(const vec<int> &shape) {
}

// element access
double &Tensor::operator()(int i, int j) {
}

double Tensor::operator()(int i, int j) const {
}

// element-wise arithmetic with another matrix
Tensor Tensor::operator+(const Tensor &o) const {
}

Tensor Tensor::operator-(const Tensor &o) const {
}

Tensor &Tensor::operator+=(const Tensor &o) {
}

Tensor &Tensor::operator-=(const Tensor &o) {
}

Tensor Tensor::hadamard(const Tensor &o) const {
}

Tensor Tensor::ediv(const Tensor &o) const {
}

// matrix ops --- operates on last two dims, parallelized across all previous dims
Tensor Tensor::operator*(const Tensor &o) const {
}

Tensor Tensor::transpose() const {
}

// element-wise function application
Tensor Tensor::apply(const std::function<double(double)> &f) const {
}

Tensor &Tensor::apply_inplace(const std::function<double(double)> &f) {
}

// reductions
double Tensor::sum() const {
}

double Tensor::min() const {
}

double Tensor::max() const {
}
