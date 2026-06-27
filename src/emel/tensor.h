#pragma once
#include "util.h"
#include <cassert>
#include <functional>

/* 
 * General policy regarding tensors: before any operations that modify the tensor itself, first materialize it.
 * This avoids annoyances with accidentally modifying multiple entries in the non-contiguous tensor when we're
 * only trying to modify one.
*/

struct Tensor {
    // think of stride[i] denoting the size taken up in the data vector by a sub-shape below it.
    // stride is how we index by sub-shapes as we go left in axes instead of individual entries;
    // individual entries are only sub-shapes for the least significant axis (the rightmost one).
    vec<int> shape, stride;
    vec<double> data;

    // ---- ctors ----

    Tensor() = default;

    Tensor(const vec<double> &data_1d);
    Tensor(const vec2<double> &data_2d);
    Tensor(const vec<int> &shape, double value);

    // ---- shape ops ----

    // reinterpret same data with different shape
    void reshape(const vec<int> &new_shape);
    // broadcast shape to match new_shape
    void broadcast(const vec<int> &new_shape);
    // sum-reduce along all axes i where shape[i]=1
    void unbroadcast(const vec<int> &shape);
    // re-order axis arg order while maintaining semantic meaning of axes
    Tensor permute(const vec<int> &p) const;
    // left-pad axes of current shape to match target shape's dimension cnt
    void pad_shape(const vec<int> &target);
    // consolidate data, become contiguous again
    Tensor materialize() const;
    // return if tensor is contiguous
    bool is_contiguous() const;

    // ---- element access ----

    // lvalue ref
    double &at(const vec<int> &ind);
    // rvalue
    double at(const vec<int> &ind) const;

    // ---- arithmetic / operations ----

    Tensor operator+(const Tensor &o) const;
    Tensor operator-(const Tensor &o) const;
    Tensor &operator+=(const Tensor &o);
    Tensor &operator-=(const Tensor &o);
    // element-wise prod
    Tensor hadamard(const Tensor &o) const;
    // element-wise div
    Tensor ediv(const Tensor &o) const;
    // unary negation
    Tensor operator-() const;
    // matmul on least significant two axes, parallelized across the rest
    Tensor operator*(const Tensor &o) const;
    // transpose on least significant two axes, parallelized across the rest
    Tensor transpose() const;
    // sum-reduce along an axis. keepdims = if reduced axis remains as len 1 or gets deleted.
    Tensor sum(int axis, bool keepdims) const;
    // return tensor of one-hot encoded argmaxes along axis arrays
    Tensor argmax(int axis) const;

    // ---- functionals ----

    // apply to copy of this
    Tensor apply(const std::function<double(double)> &f) const;
    // applies function between two tensors, auto-broadcasts both tensors as needed
    Tensor apply(const Tensor &o, const std::function<double(double, double)> &f) const;
    // apply to this, return ref to this
    Tensor &apply_inplace(const std::function<double(double)> &f);
    // applies function between two tensors, store result in this, auto-broadcast both tensors as needed
    Tensor &apply_inplace(const Tensor &o, const std::function<double(double, double)> &f);

    // ---- getters ----

    // # elements that exist in the tensor
    int num_el() const;

    // ---- save/load ----

    // serialize to bytes
    vec<uint8_t> serialize() const;
    // deserialize from bytes
    static Tensor deserialize(const vec<uint8_t> &bytes);
};
