#pragma once
#include <vector>
#include <array>
#include <glm/glm.hpp>
#include <graph2.h>

template <size_t N>
struct DataPoint
{
    DataPoint()
    {
        features.fill(0.f);
    }

    DataPoint(const std::array<float, N> &features, float y)
        : features(features), y(y) {}

    std::array<float, N> features;
    float y{ 0.f };
};

namespace general
{
    float calc_mean(const std::vector<float> &values);
    float calc_sd(const std::vector<float> &values);
    void zscore_normalize(std::vector<float> &features, float &sd, float &mean);

    float cost(float w, float b, const std::vector<glm::vec2> &data,
            const std::function<float(glm::vec2)> &err);

    template <size_t N>
    void feature_scale(std::vector<DataPoint<N>> &data, std::array<float, N> &sd, std::array<float, N> &mean)
    {
        for (size_t c = 0; c < N; ++c)
        {
            std::vector<float> features;
            features.reserve(data.size());

            for (size_t r = 0; r < data.size(); ++r)
                features.emplace_back(data[r].features[c]);

            zscore_normalize(features, sd[c], mean[c]);

            for (size_t r = 0; r < data.size(); ++r)
                data[r].features[c] = features[r];
        }
    }

    template <size_t N>
    void descend(std::array<float, N> &vw, float &b, float a,
                 const std::vector<DataPoint<N>> &data,
                 const std::function<float(const std::array<float, N>&,
                                           const DataPoint<N>&,
                                           float)> &func)
    {
        // Calculate new b
        float db_j = 0.f;
        for (size_t i = 0; i < data.size(); ++i)
            db_j += func(vw, data[i], b) - data[i].y;
        db_j /= data.size();
        float b_new = b - a * db_j;

        // Calculate new vw
        std::array<float, N> vw_new;
        for (size_t j = 0; j < N; ++j)
        {
            float dw_j = 0.f;
            for (size_t i = 0; i < data.size(); ++i)
                dw_j += (func(vw, data[i], b) - data[i].y) * data[i].features[j];
            dw_j /= data.size();

            vw_new[j] = vw[j] - a * dw_j;
        }

        b = b_new;
        vw = vw_new;
    }
}

namespace linear
{
    float f_wb(float w, float b, float x);
}

namespace multilinear
{
    template <size_t N>
    float f_wb(const std::array<float, N> &vw, const DataPoint<N> &vx,
               float b)
    {
        // w \dot x + b
        float res = b;
        for (size_t i = 0; i < N; ++i)
            res += vw[i] * vx.features[i];
        return res;
    }
}

namespace logistic
{
    float f_wb(float w, float b, float x);
    float loss(float w, float b, float prediction, float data_y);
}

