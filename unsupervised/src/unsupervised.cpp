#include "unsupervised.h"
#include <eigen3/Eigen/Dense>
#include <cstddef>
#include <cmath>
#include <eigen3/Eigen/src/Core/Matrix.h>
#include <stdexcept>

namespace kmeans
{
    void move_centroids(const std::vector<Eigen::Vector2f> &data,
                        std::vector<Eigen::Vector2f> &centroids)
    {
        std::vector<std::vector<Eigen::Vector2f>> points;
        assign_points_to_centroids(data, centroids, points);

        std::vector<Eigen::Vector2f> averages(centroids.size());
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
            const std::vector<Eigen::Vector2f> &data,
            const std::vector<Eigen::Vector2f> &centroids,
            std::vector<std::vector<Eigen::Vector2f>> &out_points)
    {
        out_points.resize(centroids.size());
        for (const auto &dp : data)
        {
            size_t nearest = 0;
            float nearest_dist = INFINITY;
            for (size_t i = 0; i < centroids.size(); ++i)
            {
                float dist = (centroids[i] - dp).norm();
                if (dist < nearest_dist)
                {
                    nearest_dist = dist;
                    nearest = i;
                }
            }

            out_points[nearest].emplace_back(dp);
        }
    }

    float cost(const std::vector<Eigen::Vector2f> &centroids,
               const std::vector<Eigen::Vector2f> &points)
    {
        float error = 0.f;
        for (size_t c = 0; c < centroids.size(); ++c)
            error += (centroids[c] - points[c]).norm();

        return error / points.size();
    }

    Eigen::Vector2f points_average(const std::vector<Eigen::Vector2f> &mx)
    {
        Eigen::Vector2f average;
        average.setZero();

        for (const auto &vx : mx)
            average += vx;

        return average / mx.size();
    }
}
