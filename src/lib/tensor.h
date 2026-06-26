#pragma once
#include "util.h"
#include <cassert>
#include <functional>
#include <initializer_list>

struct Tensor {
    // shape, stride must have same len
    vec<int> shape;
    vec<int> stride;
    vec<double> data;

    Tensor() = default;

    Tensor(const vec<int> &data_1d);
    Tensor(const vec2<int> &data_2d);
    static Tensor zeros(const vec<int> &shape);
    static Tensor ones(const vec<int> &shape);

    // shape ops
    void reshape(const vec<int> &new_shape);
    void broadcast(const vec<int> &new_shape);
    // sum-reduce along all axes i where shape[i]=1
    void unbroadcast(const vec<int> &shape);
    // p[i]: shape[i] := shape[p[i]]
    void permute(const vec<int> &p);

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
    Tensor &apply_inplace(const std::function<double(double)> &f);

    // reductions
    Tensor sum(int axis) const;
    Tensor argmin(int axis) const;
    Tensor argmax(int axis) const;
};
