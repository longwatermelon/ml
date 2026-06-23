#pragma once
#include "util.h"
#include <cassert>
#include <functional>
#include <initializer_list>

struct Matrix {
    int rows, cols;
    vec2<double> data;

    Matrix() = default;

    Matrix(int rows, int cols)
        : rows(rows), cols(cols),
          data((assert(rows >= 0 && cols >= 0), vec2<double>(rows, cols))) {}

    // construct directly from a 2d buffer, which must be rectangular
    Matrix(const vec2<double> &data)
        : rows(sz(data)), cols(data.empty() ? 0 : sz(data[0])), data(data) {
        for (int i = 0; i < rows; i++)
            assert(sz(this->data[i]) == cols && "all rows must have the same length");
    }

    // construct from a brace literal, e.g. {{1, 2}, {3, 4}}
    Matrix(std::initializer_list<std::initializer_list<double>> init);

    // factory helpers for common matrices
    static Matrix zeros(int rows, int cols);
    static Matrix ones(int rows, int cols);
    static Matrix identity(int n);
    static Matrix from_diagonal(const vec<double> &d);

    // element access
    double &operator()(int i, int j);
    double operator()(int i, int j) const;

    // shape helpers
    bool same_shape(const Matrix &o) const;
    bool is_square() const;

    // element-wise arithmetic with another matrix
    Matrix operator+(const Matrix &o) const;
    Matrix operator-(const Matrix &o) const;
    Matrix &operator+=(const Matrix &o);
    Matrix &operator-=(const Matrix &o);
    Matrix hadamard(const Matrix &o) const; // element-wise product
    Matrix ediv(const Matrix &o) const;     // element-wise division

    // unary negation
    Matrix operator-() const;

    // scalar arithmetic (scalar broadcast to every entry)
    Matrix operator+(double s) const;
    Matrix operator-(double s) const;
    Matrix operator*(double s) const;
    Matrix operator/(double s) const;
    Matrix &operator+=(double s);
    Matrix &operator-=(double s);
    Matrix &operator*=(double s);
    Matrix &operator/=(double s);

    // matrix multiplication
    Matrix operator*(const Matrix &o) const;

    // comparison
    bool operator==(const Matrix &o) const;
    bool operator!=(const Matrix &o) const;

    // structural operations
    Matrix transpose() const;
    Matrix submatrix(int skip_row, int skip_col) const; // remove one row and column
    Matrix get_row(int i) const;
    Matrix get_col(int j) const;

    // element-wise function application
    Matrix apply(const std::function<double(double)> &f) const;
    Matrix &apply_inplace(const std::function<double(double)> &f);

    // reductions
    double sum() const;
    double trace() const;
    double min() const;
    double max() const;

    // linear algebra
    double determinant() const;
    Matrix inverse() const;
    Matrix pow(int e) const; // repeated matrix multiplication for square matrices
};

// scalar on the left
Matrix operator+(double s, const Matrix &m); // s + each entry
Matrix operator-(double s, const Matrix &m); // s - each entry
Matrix operator*(double s, const Matrix &m); // s * each entry
Matrix operator/(double s, const Matrix &m); // s / each entry

// pretty-print to a stream
std::ostream &operator<<(std::ostream &os, const Matrix &m);
