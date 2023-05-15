#pragma once
#include <vector>
#include <eigen3/Eigen/Dense>

namespace kmeans
{
    void move_centroids(const std::vector<Eigen::Vector2f> &data,
                        std::vector<Eigen::Vector2f> &centroids);
    void assign_points_to_centroids(
            const std::vector<Eigen::Vector2f> &data,
            const std::vector<Eigen::Vector2f> &centroids,
            std::vector<std::vector<Eigen::Vector2f>> &out_points);
    float cost(const std::vector<Eigen::Vector2f> &centroids,
               const std::vector<Eigen::Vector2f> &points);
    Eigen::Vector2f points_average(const std::vector<Eigen::Vector2f> &mx);
}
