#pragma once
#include "common.h"
#include <cstdio>
#include <vector>
#include <array>
#include <sstream>
#include <functional>
#include <cmath>

namespace reg
{
    using DataPoint = common::DataPoint<float, float>;

    namespace general
    {
        float calc_mean(const Eigen::VectorXf &values);
        float calc_sd(const Eigen::VectorXf &values);
        void zscore_normalize(Eigen::VectorXf &features, float &sd, float &mean);

        // All vectors have NF length
        void feature_scale(std::vector<DataPoint> &data, Eigen::VectorXf &sd, Eigen::VectorXf &mean);

        // All vectors have NF length
        float cost(const std::vector<DataPoint> &data,
                   const Eigen::VectorXf &vw,
                   const std::function<float(const DataPoint&)> &err_f,
                   float lambda = 0.f);

        // func parameters: vw, vx, b
        // All vectors have NF length
        void descend(Eigen::VectorXf &vw, float &b, float a,
                     const std::vector<DataPoint> &data,
                     const std::function<float(const Eigen::VectorXf&,
                                               const Eigen::VectorXf&,
                                               float)> &prediction_func,
                     float lambda = 0.f);
    }

    namespace linear
    {
        float f_wb(float w, float x, float b);
    }

    namespace multilinear
    {
        float f_wb(const Eigen::VectorXf &vw, const Eigen::VectorXf &features,
                   float b);
    }

    namespace logistic
    {
        float g(float z);
        float f_wb(float w, float x, float b);
        float loss(float prediction, float data_y);
    }

    namespace multilogistic
    {
        float f_wb(const Eigen::VectorXf &vw, const Eigen::VectorXf &vx, float b);
        float g(float z);
    }

    namespace softmax
    {
        float g(float z, const Eigen::VectorXf &vz);
        Eigen::VectorXf solve_va(Eigen::VectorXf vz);
    }
}

