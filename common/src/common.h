#pragma once
#include <vector>
#include <string>
#include <functional>

namespace common
{
    // N: num features
    struct DataPoint
    {
        DataPoint() = default;
        DataPoint(size_t n)
            : features(n) {}
        DataPoint(const std::vector<float> &features, float y)
            : features(features), y(y) {}

        std::vector<float> features;
        float y{ 0.f };
    };

    namespace vec
    {
        // Automatic assign
        std::vector<float> apply_fn(std::vector<float> v, const std::function<float(float)> &fn);
        // Manual assign
        std::vector<float> apply_fn(std::vector<float> v,
                const std::function<void(std::vector<float> &v, size_t i)> &fn);

        float dot(const std::vector<float> &a, const std::vector<float> &b);
        float sum(const std::vector<float> &v);
        std::string to_string(const std::vector<float> &v);

        std::vector<float> matmul(const std::vector<std::vector<float>> &m, const std::vector<float> &v);
    }
}
 