#include "reg.h"
#include <sstream>

//// GENERAL
float reg::general::calc_mean(const std::vector<float> &values)
{
    float sum = 0.f;
    for (const auto &e : values)
        sum += e;

    return sum / values.size();
}

float reg::general::calc_sd(const std::vector<float> &values)
{
    float mean = calc_mean(values);
    float sd = 0.f;

    for (const auto &e : values)
        sd += std::pow(e - mean, 2);

    return std::sqrt(sd / values.size());
}

void reg::general::zscore_normalize(std::vector<float> &features, float &sd, float &mean)
{
    mean = calc_mean(features);
    sd = calc_sd(features);

    std::string data;
    for (size_t i = 0; i < features.size(); ++i)
        features[i] = (features[i] - mean) / sd;
}

void reg::general::feature_scale(std::vector<DataPoint> &data, std::vector<float> &sd, std::vector<float> &mean)
{
    for (size_t c = 0; c < sd.size() /* feature num */; ++c)
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

float reg::general::cost(const std::vector<DataPoint> &data,
           const std::vector<float> &vw,
           const std::function<float(const DataPoint&)> &err_f,
           float lambda)
{
    float cost = 0.f;
    for (const auto &p : data)
        cost += err_f(p);
    cost /= 2.f * data.size();

    float reg_term = 0.f;
    for (const auto &w : vw)
        reg_term += w * w;
    reg_term *= lambda / (2.f * data.size());

    return cost + reg_term;
}

void reg::general::descend(std::vector<float> &vw, float &b, float a,
             const std::vector<DataPoint> &data,
             const std::function<float(const std::vector<float>&,
                                       const std::vector<float>&,
                                       float)> &prediction_func,
             float lambda)
{
    // Calculate new b
    float db_j = 0.f;
    for (size_t i = 0; i < data.size(); ++i)
        db_j += prediction_func(vw, data[i].features, b) - data[i].y;
    db_j /= data.size();
    float b_new = b - a * db_j;

    // Calculate new vw
    std::vector<float> vw_new(vw.size());
    for (size_t j = 0; j < vw.size(); ++j)
    {
        float dw_j = 0.f;
        for (size_t i = 0; i < data.size(); ++i)
            dw_j += (prediction_func(vw, data[i].features, b) - data[i].y) * data[i].features[j];
        dw_j /= data.size();
        dw_j += lambda / data.size() * vw[j];

        vw_new[j] = vw[j] - a * dw_j;
    }

    b = b_new;
    vw = vw_new;
}


//// LINEAR
float reg::linear::f_wb(float w, float x, float b)
{
    return w * x + b;
}

//// MULTILINEAR
float reg::multilinear::f_wb(const std::vector<float> &vw, const std::vector<float> &features,
           float b)
{
    // w \dot x + b
    float res = b;
    for (size_t i = 0; i < vw.size(); ++i)
        res += vw[i] * features[i];
    return res;
}

//// LOGISTIC
float reg::logistic::g(float z)
{
    return 1.f / (1.f + std::exp(-z));
}

float reg::logistic::f_wb(float w, float x, float b)
{
    return w * x + b;
}

float reg::logistic::loss(float prediction, float data_y)
{
    return ((int)data_y == 1 ? -data_y * std::log(prediction) : -(1.f - data_y) * std::log(1.f - prediction));
}

// MULTILOGISTIC
float reg::multilogistic::f_wb(const std::vector<float> &vw, const std::vector<float> &vx, float b)
{
    return vec::dot(vw, vx) + b;
}

float reg::multilogistic::g(float z)
{
    return 1.f / (1.f + std::exp(-z));
}

//// SOFTMAX
float reg::softmax::g(float z, const std::vector<float> &vz)
{
    return std::exp(z) / vec::sum(vec::apply_fn(vz, [](float z){ return std::exp(z); }));
}

std::vector<float> reg::softmax::solve_va(std::vector<float> vz)
{
    vz = vec::apply_fn(vz, [](float z){ return std::exp(z); });
    float sum = vec::sum(vz);
    vz = vec::apply_fn(vz, [sum](float z){ return z / sum; });
    return vz;
}

