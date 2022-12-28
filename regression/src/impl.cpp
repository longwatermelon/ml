#include "impl.h"
#include <sstream>

//// GENERAL
float general::calc_mean(const std::vector<float> &values)
{
    float sum = 0.f;
    for (const auto &e : values)
        sum += e;

    return sum / values.size();
}

float general::calc_sd(const std::vector<float> &values)
{
    float mean = calc_mean(values);
    float sd = 0.f;

    for (const auto &e : values)
        sd += std::pow(e - mean, 2);

    return std::sqrt(sd / values.size());
}

void general::zscore_normalize(std::vector<float> &features, float &sd, float &mean)
{
    mean = calc_mean(features);
    sd = calc_sd(features);

    std::string data;
    for (size_t i = 0; i < features.size(); ++i)
        features[i] = (features[i] - mean) / sd;
}

//// LINEAR
float linear::f_wb(const std::array<float, 1> &w, const std::array<float, 1> &features, float b)
{
    return w[0] * features[0] + b;
}

//// LOGISTIC
float logistic::f_wb(const std::array<float, 1> w, const std::array<float, 1> &features, float b)
{
    return 1.f / (1.f + std::exp(-(w[0] * features[0] + b)));
}

float logistic::loss(float w, float b, float prediction, float data_y)
{
    return ((int)data_y == 1 ? -data_y * std::log(prediction) : -(1.f - data_y) * std::log(1.f - prediction));
}

