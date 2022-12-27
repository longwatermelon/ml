#pragma once
#include <vector>
#include <graph2.h>

namespace multilinear
{
    float calc_mean(const std::vector<float> &values);
    float calc_sd(const std::vector<float> &values);
    std::vector<float> zscore_normalize(std::vector<float> features);
    void feature_scale(Graph2 &g, const std::string &out_fp);
}

