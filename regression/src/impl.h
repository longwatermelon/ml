#pragma once
#include <vector>
#include <array>
#include <glm/glm.hpp>
#include <graph2.h>

namespace linear
{
    void descend(float &w, float &b, float a, const std::vector<glm::vec2> &data);
}

namespace multilinear
{
    float calc_mean(const std::vector<float> &values);
    float calc_sd(const std::vector<float> &values);
    void zscore_normalize(std::vector<float> &features, float &sd, float &mean);
    void feature_scale(Graph2 &g, float &sd, float &mean);

    template <size_t N>
    float f_wb(const std::array<float, N>& vw, const std::array<float, N> &vx,
               float b)
    {
        // w \dot x + b
        float res = b;
        for (size_t i = 0; i < N; ++i)
            res += vw[i] * vx[i];
        return res;
    }

    template <size_t N>
    void descend(std::array<float, N> &vw, float &b, float a,
                 const std::vector<std::array<float, N>> mx,
                 const std::vector<float> vy)
    {
        // Calculate new b
        float db_j = 0.f;
        for (size_t i = 0; i < mx.size(); ++i)
            db_j += f_wb(vw, mx[i], b) - vy[i];
        db_j /= mx.size();
        float b_new = b - a * db_j;

        // Calculate new vw
        std::array<float, N> vw_new;
        for (size_t j = 0; j < N; ++j)
        {
            float dw_j = 0.f;
            for (size_t i = 0; i < mx.size(); ++i)
                dw_j += (f_wb(vw, mx[i], b) - vy[i]) * mx[i][j];
            dw_j /= mx.size();

            vw_new[j] = vw[j] - a * dw_j;
        }

        b = b_new;
        vw = vw_new;
    }
}

namespace logistic
{
    float f_wb(float w, float b, float x);
    void descend(float &w, float &b, float a, const std::vector<glm::vec2> &data);

    float loss(float w, float b, float prediction, float data_y);
    float cost(float w, float b, const std::vector<glm::vec2> &data);
}

