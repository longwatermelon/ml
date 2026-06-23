#pragma once
#include "util.h"

struct Matrix {
    int rows, cols;
    vec2<double> data;

    Matrix(int rows, int cols)
        : rows(rows), cols(cols), data(vec2<double>(rows, cols)) {}
};
