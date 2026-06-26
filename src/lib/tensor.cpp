#include "tensor.h"
#include <cassert>

// 1d tensor from a vector
Tensor::Tensor(const vec<double> &data_1d) {
    shape = {sz(data_1d)};
    stride = {1};
    data = data_1d;
}

// 2d tensor from a 2d vector
Tensor::Tensor(const vec2<double> &data_2d) {
    shape = {sz(data_2d), sz(data_2d[0])};
    stride = {sz(data_2d[0]), 1};
    data.resize(shape[0] * shape[1]);
    for (int i = 0; i < shape[0]; ++i) {
        for (int j = 0; j < shape[1]; ++j) {
            at({i,j}) = data_2d[i][j];
        }
    }
}

// tensor of all zeros with the given shape
Tensor Tensor::zeros(const vec<int> &shape) {
}

// tensor of all ones with the given shape
Tensor Tensor::ones(const vec<int> &shape) {
}

// ---- shape ops ----

// change shape without changing data
void Tensor::reshape(const vec<int> &new_shape) {
}

// broadcast dims of size 1 to match new_shape
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
    assert(sz(ind) == sz(shape));

    int flat_ind = 0;
    for (int i = 0; i < sz(ind); ++i) {
        flat_ind += stride[i] * ind[i];
    }

    assert(0 <= flat_ind && flat_ind < sz(data));
    return data[flat_ind];
}

// const element access by multi-dim index
double Tensor::at(const vec<int> &ind) const {
    assert(sz(ind) == sz(shape));

    int flat_ind = 0;
    for (int i = 0; i < sz(ind); ++i) {
        flat_ind += stride[i] * ind[i];
    }

    assert(0 <= flat_ind && flat_ind < sz(data));
    return data[flat_ind];
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
