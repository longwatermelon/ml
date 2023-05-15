#include "reg.h"
#include <sstream>

namespace reg
{
namespace general
{
    float calc_mean(const Eigen::VectorXf &values)
    {
        float sum = 0.f;
        for (const auto &e : values)
            sum += e;

        return sum / values.size();
    }

    float calc_sd(const Eigen::VectorXf &values)
    {
        float mean = calc_mean(values);
        float sd = 0.f;

        for (const auto &e : values)
            sd += std::pow(e - mean, 2);

        return std::sqrt(sd / values.size());
    }

    void zscore_normalize(Eigen::VectorXf &features, float &sd, float &mean)
    {
        mean = calc_mean(features);
        sd = calc_sd(features);

        std::string data;
        for (long int i = 0; i < features.size(); ++i)
            features[i] = (features[i] - mean) / sd;
    }

    void feature_scale(std::vector<DataPoint> &data, Eigen::VectorXf &sd, Eigen::VectorXf &mean)
    {
        for (long int c = 0; c < sd.size() /* feature num */; ++c)
        {
            Eigen::VectorXf features(data.size());

            for (size_t r = 0; r < data.size(); ++r)
                features[r] = data[r].features[c];

            zscore_normalize(features, sd[c], mean[c]);

            for (size_t r = 0; r < data.size(); ++r)
                data[r].features[c] = features[r];
        }
    }

    float cost(const std::vector<DataPoint> &data,
            const Eigen::VectorXf &vw,
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

    void descend(Eigen::VectorXf &vw, float &b, float a,
                const std::vector<DataPoint> &data,
                const std::function<float(const Eigen::VectorXf&,
                                        const Eigen::VectorXf&,
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
        Eigen::VectorXf vw_new(vw.size());
        for (long int j = 0; j < vw.size(); ++j)
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
}
namespace linear
{
    float f_wb(float w, float x, float b)
    {
        return w * x + b;
    }
}
namespace multilinear
{
    float f_wb(const Eigen::VectorXf &vw, const Eigen::VectorXf &features,
            float b)
    {
        // w \dot x + b
        float res = b;
        for (long int i = 0; i < vw.size(); ++i)
            res += vw[i] * features[i];
        return res;
    }
}
namespace logistic
{
    float g(float z)
    {
        return 1.f / (1.f + std::exp(-z));
    }

    float f_wb(float w, float x, float b)
    {
        return w * x + b;
    }

    float loss(float prediction, float data_y)
    {
        return ((int)data_y == 1 ? -data_y * std::log(prediction) : -(1.f - data_y) * std::log(1.f - prediction));
    }
}
namespace multilogistic
{
    float f_wb(const Eigen::VectorXf &vw, const Eigen::VectorXf &vx, float b)
    {
        return vw.dot(vx) + b;
        // return common::vec::dot(vw, vx) + b;
    }

    float g(float z)
    {
        return 1.f / (1.f + std::exp(-z));
    }
}
namespace softmax
{
    float g(float z, const Eigen::VectorXf &vz)
    {
        // return std::exp(z) / common::vec::sum(common::vec::apply_fn(vz, [](float z){ return std::exp(z); }));
        return std::exp(z) / vz.unaryExpr([](float z){ return std::exp(z); }).sum();
    }

    Eigen::VectorXf solve_va(Eigen::VectorXf vz)
    {
        vz = vz.unaryExpr([](float z){ return std::exp(z); });
        // vz = common::vec::apply_fn(vz, [](float z){ return std::exp(z); });
        // float sum = common::vec::sum(vz);
        float sum = vz.sum();
        vz = vz.unaryExpr([sum](float z){ return z / sum; });
        // vz = common::vec::apply_fn(vz, [sum](float z){ return z / sum; });
        return vz;
    }
}
}
