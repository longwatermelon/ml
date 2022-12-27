#include "linear.h"
#include "multilinear.h"
#include <fstream>

void linear::descend(float &w, float &b, float a, const std::vector<glm::vec2> &data)
{
    float dw_j = 0.f,
          db_j = 0.f;
    for (const auto &p : data)
    {
        dw_j += (w * p.x + b - p.y) * p.x;
        db_j += w * p.x + b - p.y;
    }

    dw_j *= 1.f / data.size();
    db_j *= 1.f / data.size();

    w = w - a * dw_j;
    b = b - a * db_j;
}

float multilinear::calc_mean(const std::vector<float> &values)
{
    float sum = 0.f;
    for (const auto &e : values)
        sum += e;

    return sum / values.size();
}

float multilinear::calc_sd(const std::vector<float> &values)
{
    float mean = calc_mean(values);
    float sd = 0.f;

    for (const auto &e : values)
        sd += std::pow(e - mean, 2);

    return std::sqrt(sd / values.size());
}

std::vector<float> multilinear::zscore_normalize(std::vector<float> features)
{
    float mean = calc_mean(features);
    float sd = calc_sd(features);

    std::string data;
    for (size_t i = 0; i < features.size(); ++i)
        features[i] = (features[i] - mean) / sd;

    return features;
}

void multilinear::feature_scale(Graph2 &g, const std::string &out_fp)
{
    std::vector<float> features;
    for (const auto &e : g.data())
        features.emplace_back(e.x);

    features = zscore_normalize(features);

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

    std::ofstream ofs(out_fp);
    ofs << "min " << min << ' ' << g.min().y << "\n"
        << "max " << max << ' ' << g.max().y << "\n"
        << "step " << std::abs(max - min) / 5.f << ' ' << g.step().y << "\n"
        << data;
    ofs.close();

    g.load(out_fp);
}

