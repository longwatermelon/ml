#pragma once
#include "util.h"
#include <cassert>
#include <functional>
#include <initializer_list>

struct Tensor {
    // shape, stride must have same len
    vec<int> shape;
    vec<int> stride;
    // data has prod(shape) elements
    vec<double> data;

    Tensor() = default;

    Tensor(const vec<int> &data_1d);
    Tensor(const vec2<int> &data_2d);
    static Tensor zeros(const vec<int> &shape);
    static Tensor ones(const vec<int> &shape);

    // element access
    double &operator()(int i, int j);
    double operator()(int i, int j) const;

    // element-wise arithmetic with another matrix
    Tensor operator+(const Tensor &o) const;
    Tensor operator-(const Tensor &o) const;
    Tensor &operator+=(const Tensor &o);
    Tensor &operator-=(const Tensor &o);
    Tensor hadamard(const Tensor &o) const; // element-wise product
    Tensor ediv(const Tensor &o) const;     // element-wise division

    // matrix ops --- operates on last two dims, parallelized across all previous dims
    Tensor operator*(const Tensor &o) const;
    Tensor transpose() const;

    // element-wise function application
    Tensor apply(const std::function<double(double)> &f) const;
    Tensor &apply_inplace(const std::function<double(double)> &f);

    // reductions
    double sum() const;
    double min() const;
    double max() const;
};
