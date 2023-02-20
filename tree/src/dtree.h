#pragma once
#include "common.h"
#include <memory>
#include <vector>

namespace dtree
{
    using DataPoint = common::DataPoint<bool, bool>;

    struct DTree
    {
        std::unique_ptr<DTree> yes, no;
        int feature{ 0 };
        bool decision{ true };

        bool predict(const std::vector<bool> &features);
    };

    std::unique_ptr<DTree> create_dtree(const std::vector<DataPoint> &data, const std::vector<int> &used_indices = {});
    void split(const std::vector<DataPoint> &data, int feature,
               std::vector<DataPoint> &no, std::vector<DataPoint> &yes);

    float H(float p);
}
