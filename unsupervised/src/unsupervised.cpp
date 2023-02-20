#include "unsupervised.h"
#include <cstddef>
#include <cmath>
#include <stdexcept>

namespace kmeans
{
    void move_centroids(const std::vector<std::vector<float>> &data,
                        std::vector<std::vector<float>> &centroids)
    {
        std::vector<std::vector<std::vector<float>>> points;
        assign_points_to_centroids(data, centroids, points);

        std::vector<std::vector<float>> averages(centroids.size());
        for (size_t i = 0; i < averages.size(); ++i)
        {
            if (!points[i].empty())
            {
                averages[i] = points_average(points[i]);
                centroids[i] = averages[i];
            }
        }
    }

    void assign_points_to_centroids(
            const std::vector<std::vector<float>> &data,
            const std::vector<std::vector<float>> &centroids,
            std::vector<std::vector<std::vector<float>>> &out_points)
    {
        out_points.resize(centroids.size());
        for (const auto &dp : data)
        {
            size_t nearest = 0;
            float nearest_dist = INFINITY;
            for (size_t i = 0; i < centroids.size(); ++i)
            {
                float dist = distance(centroids[i], dp);
                if (dist < nearest_dist)
                {
                    nearest_dist = dist;
                    nearest = i;
                }
            }

            out_points[nearest].emplace_back(dp);
        }
    }

    float cost(const std::vector<std::vector<float>> &centroids,
               const std::vector<std::vector<float>> &points)
    {
        float error = 0.f;
        for (size_t c = 0; c < centroids.size(); ++c)
            error += distance(centroids[c], points[c]);

        return error / points.size();
    }

    float distance(const std::vector<float> &va, const std::vector<float> &vb)
    {
        if (va.size() != vb.size())
            throw std::runtime_error("[kmeans::distance] va and vb are not the same length.");

        float dist = 0.f;
        for (size_t i = 0; i < va.size(); ++i)
            dist += std::pow(va[i] - vb[i], 2);
        return std::sqrt(dist);
    }

    std::vector<float> points_average(const std::vector<std::vector<float>> &mx)
    {
        std::vector<float> average(mx[0].size());
        for (const auto &vx : mx)
        {
            for (size_t i = 0; i < vx.size(); ++i)
                average[i] += vx[i];
        }

        for (size_t i = 0; i < average.size(); ++i)
            average[i] /= mx.size();

        return average;
    }
}
