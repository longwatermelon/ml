#pragma once
#include "util.h"
#include <cassert>
#include <functional>
#include <initializer_list>

struct Tensor {
    // think of stride[i] denoting the size taken up in the data vector by a sub-shape below it.
    // stride is how we index by sub-shapes as we go left in axes instead of individual entries;
    // individual entries are only sub-shapes for the least significant axis (the rightmost one).
    vec<int> shape, stride;
    vec<double> data;

    Tensor() = default;

    Tensor(const vec<double> &data_1d);
    Tensor(const vec2<double> &data_2d);
    Tensor(const vec<int> &shape, double value);

    // shape ops
    void reshape(const vec<int> &new_shape);
    void broadcast(const vec<int> &new_shape);
    // sum-reduce along all axes i where shape[i]=1
    void unbroadcast(const vec<int> &shape);
    // p[i]: shape[i] := shape[p[i]]
    void permute(const vec<int> &p);
    void pad_shape(const vec<int> &target);
    // consolidate data, become contiguous again
    Tensor make_contiguous() const;

    // element access
    double &at(const vec<int> &ind);
    double at(const vec<int> &ind) const;

    // element-wise arithmetic with another tensor
    Tensor operator+(const Tensor &o) const;
    Tensor operator-(const Tensor &o) const;
    Tensor &operator+=(const Tensor &o);
    Tensor &operator-=(const Tensor &o);
    // element-wise prod
    Tensor hadamard(const Tensor &o) const;
    // element-wise div
    Tensor ediv(const Tensor &o) const;
    Tensor operator-() const;

    // matrix ops --- operates on last two dims, parallelized across all previous dims
    Tensor operator*(const Tensor &o) const;
    Tensor transpose() const;

    // element-wise function application
    Tensor apply(const std::function<double(double)> &f) const;
    // applies function between two tensors, auto-broadcasts up to one tensor if needed
    Tensor apply(const Tensor &o, const std::function<double(double, double)> &f) const;
    Tensor &apply_inplace(const std::function<double(double)> &f);

    // reductions
    Tensor sum(int axis, bool keepdims = true) const;
    Tensor argmin(int axis) const;
    Tensor argmax(int axis) const;
};
