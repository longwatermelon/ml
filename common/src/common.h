#pragma once
#include <eigen3/Eigen/Dense>
#include <vector>
#include <string>
#include <functional>

namespace common
{
    // N: num features
    template <typename T, typename U>
    struct DataPoint
    {
        DataPoint() = default;
        DataPoint(size_t n)
            : features(n) {}
        DataPoint(const Eigen::VectorX<U> &features, U y)
            : features(features), y(y) {}

        Eigen::VectorX<U> features;
        U y{};
    };
}
 