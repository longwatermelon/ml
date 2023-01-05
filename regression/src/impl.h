#pragma once
#include <cstdio>
#include <vector>
#include <array>
#include <sstream>
#include <functional>
#include <glm/glm.hpp>

namespace reg
{
    // N: num features
    struct DataPoint
    {
        DataPoint() = default;
        DataPoint(const std::vector<float> &features, float y)
            : features(features), y(y) {}

        std::vector<float> features;
        float y{ 0.f };
    };

    namespace vec
    {
        std::vector<float> apply_fn(std::vector<float> v, const std::function<float(float)> &fn);
        float dot(const std::vector<float> &a, const std::vector<float> &b);
        float sum(const std::vector<float> &v);
        std::string to_string(const std::vector<float> &v);
    }

    namespace general
    {
        float calc_mean(const std::vector<float> &values);
        float calc_sd(const std::vector<float> &values);
        void zscore_normalize(std::vector<float> &features, float &sd, float &mean);

        // All vectors have NF length
        void feature_scale(std::vector<DataPoint> &data, std::vector<float> &sd, std::vector<float> &mean);

        // All vectors have NF length
        float cost(const std::vector<DataPoint> &data,
                   const std::vector<float> &vw,
                   const std::function<float(const DataPoint&)> &err_f,
                   float lambda = 0.f);

        // N: num features
        // func parameters: vw, vx, b
        // All vectors have NF length
        void descend(std::vector<float> &vw, float &b, float a,
                     const std::vector<DataPoint> &data,
                     const std::function<float(const std::vector<float>&,
                                               const std::vector<float>&,
                                               float)> &prediction_func,
                     float lambda = 0.f);
    }

    namespace linear
    {
        float f_wb(float w, float x, float b);
    }

    namespace multilinear
    {
        float f_wb(const std::vector<float> &vw, const std::vector<float> &features,
                   float b);
    }

    namespace logistic
    {
        float f_wb(float w, float x, float b);
        float loss(float prediction, float data_y);
    }

    namespace multilogistic
    {
        float f_wb(const std::vector<float> &vw, const std::vector<float> &vx, float b);
        float g(float z);
    }

    namespace softmax
    {
        float g(float z, const std::vector<float> &vz);
        std::vector<float> solve_va(std::vector<float> vz);
    }
}

