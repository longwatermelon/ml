#include "tensor.h"
#include <cassert>
#include <algorithm>

// 1d tensor from a flat vector of ints
Tensor::Tensor(const vec<int> &data_1d)
    : shape({sz(data_1d)}), stride({1}), data(sz(data_1d)) {
    for (int i = 0; i < sz(data_1d); i++)
        data[i] = data_1d[i];
}

// 2d tensor from a 2d vector of ints
Tensor::Tensor(const vec2<int> &data_2d) {
}

// tensor of all zeros with the given shape
Tensor Tensor::zeros(const vec<int> &shape) {
}

// tensor of all ones with the given shape
Tensor Tensor::ones(const vec<int> &shape) {
}

// ---- shape ops ----

// change shape without changing data, materializes if non-contiguous
void Tensor::reshape(const vec<int> &new_shape) {
}

// broadcast dims of size 1 to match new_shape by zeroing their strides
void Tensor::broadcast(const vec<int> &new_shape) {
}

// sum-reduce along axes where target has size 1
void Tensor::unbroadcast(const vec<int> &target) {
}

// permute dimensions: new shape[i] = old shape[p[i]]
void Tensor::permute(const vec<int> &p) {
}

// ---- element access ----

// mutable element access by multi-dim index
double &Tensor::at(const vec<int> &ind) {
}

// const element access by multi-dim index
double Tensor::at(const vec<int> &ind) const {
}

// ---- element-wise arithmetic ----

// element-wise sum
Tensor Tensor::operator+(const Tensor &o) const {
}

// element-wise difference
Tensor Tensor::operator-(const Tensor &o) const {
}

// in-place element-wise addition
Tensor &Tensor::operator+=(const Tensor &o) {
}

// in-place element-wise subtraction
Tensor &Tensor::operator-=(const Tensor &o) {
}

// element-wise product
Tensor Tensor::hadamard(const Tensor &o) const {
}

// element-wise division
Tensor Tensor::ediv(const Tensor &o) const {
}

// negate every element
Tensor Tensor::operator-() const {
    return apply([](double x) { return -x; });
}

// ---- matrix ops ----

// matmul on last two dims, batched over all leading dims
Tensor Tensor::operator*(const Tensor &o) const {
}

// transpose last two dims
Tensor Tensor::transpose() const {
}

// ---- element-wise function application ----

// return a new tensor with f applied to every element
Tensor Tensor::apply(const std::function<double(double)> &f) const {
}

// apply f to every element in place
Tensor &Tensor::apply_inplace(const std::function<double(double)> &f) {
}

// ---- reductions ----

// sum along an axis, result has shape[axis]=1 (keepdims)
Tensor Tensor::sum(int axis) const {
}

// index of the minimum along an axis, result has shape[axis]=1
Tensor Tensor::argmin(int axis) const {
}

// index of the maximum along an axis, result has shape[axis]=1
Tensor Tensor::argmax(int axis) const {
}
