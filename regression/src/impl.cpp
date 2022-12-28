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

void general::feature_scale(Graph2 &g, float &sd, float &mean)
{
    std::vector<float> features;
    for (const auto &e : g.data())
        features.emplace_back(e.x);

    zscore_normalize(features, sd, mean);

    std::string data;
    float min = std::numeric_limits<float>::max(),
          max = std::numeric_limits<float>::min();
    for (size_t i = 0; i < features.size(); ++i)
    {
        float x = features[i];
        if (x < min) min = x;
        if (x > max) max = x;
        data += "data " + std::to_string(x) + " " + std::to_string(g.data()[i].y) + "\n";
    }

    std::stringstream ss;
    ss << "min " << min << ' ' << g.min().y << "\n"
       << "max " << max << ' ' << g.max().y << "\n"
       << "step " << std::abs(max - min) / 5.f << ' ' << g.step().y << "\n"
       << data;

    g.load(ss.str());
}

float general::cost(float w, float b, const std::vector<glm::vec2> &data,
        const std::function<float(glm::vec2)> &err)
{
    float cost = 0.f;
    for (const auto &p : data)
        cost += err(p);
    return cost / data.size();
}

//// LINEAR
float linear::f_wb(float w, float b, float x)
{
    return w * x + b;
}

void linear::descend(float &w, float &b, float a, const std::vector<glm::vec2> &data)
{
    float dw_j = 0.f,
          db_j = 0.f;
    for (const auto &p : data)
    {
        dw_j += (f_wb(w, b, p.x) - p.y) * p.x;
        db_j += f_wb(w, b, p.x) - p.y;
    }

    dw_j *= 1.f / data.size();
    db_j *= 1.f / data.size();

    w = w - a * dw_j;
    b = b - a * db_j;
}

//// LOGISTIC
float logistic::f_wb(float w, float b, float x)
{
    return 1.f / (1.f + std::exp(-(w * x + b)));
}

void logistic::descend(float &w, float &b, float a, const std::vector<glm::vec2> &data)
{
    float dw_j = 0.f,
          db_j = 0.f;
    for (const auto &p : data)
    {
        dw_j += (f_wb(w, b, p.x) - p.y) * p.x;
        db_j += f_wb(w, b, p.x) - p.y;
    }

    dw_j /= data.size();
    db_j /= data.size();

    w -= a * dw_j;
    b -= a * db_j;
}

float logistic::loss(float w, float b, float prediction, float data_y)
{
    return ((int)data_y == 1 ? -data_y * std::log(prediction) : -(1.f - data_y) * std::log(1.f - prediction));
}

