#pragma once
#include <vector>

namespace kmeans
{
    void move_centroids(const std::vector<std::vector<float>> &data,
                        std::vector<std::vector<float>> &centroids);
    void assign_points_to_centroids(
            const std::vector<std::vector<float>> &data,
            const std::vector<std::vector<float>> &centroids,
            std::vector<std::vector<std::vector<float>>> &out_points);
    float cost(const std::vector<std::vector<float>> &centroids,
               const std::vector<std::vector<float>> &points);

    float distance(const std::vector<float> &va, const std::vector<float> &vb);
    std::vector<float> points_average(const std::vector<std::vector<float>> &mx);
}
