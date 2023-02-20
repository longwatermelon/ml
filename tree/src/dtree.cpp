#include "dtree.h"
#include <algorithm>
#include <cmath>
#include <memory>

static void check_uniformity(const std::vector<dtree::DataPoint> &yes,
                             const std::vector<dtree::DataPoint> &no,
                             bool &uniform_yes, bool &uniform_no)
{
    uniform_yes = true;
    uniform_no = true;

    if (yes.empty()) uniform_yes = false;
    if (no.empty()) uniform_no = false;

    for (size_t i = 1; i < yes.size(); ++i)
        if (yes[i].y != yes[i - 1].y)
            uniform_yes = false;
    for (size_t i = 1; i < no.size(); ++i)
        if (no[i].y != no[i - 1].y)
            uniform_no = false;
}

namespace dtree
{
    bool DTree::predict(const std::vector<bool> &features)
    {
        if (!this->no && !this->yes)
            return this->decision;

        bool f = features[this->feature];
        return (f ? this->yes : this->no)->predict(features);
    }

    std::unique_ptr<DTree> create_dtree(const std::vector<DataPoint> &data, const std::vector<int> &used_indices)
    {
        if (data.size() == 0)
            return std::make_unique<DTree>();

        int nf = data[0].features.size();
        float max_h = 0.f;
        int best_feature_split = 0;
        std::unique_ptr<DTree> tree = std::make_unique<DTree>();

        for (int i = 0; i < nf; ++i)
        {
            if (std::find(used_indices.begin(), used_indices.end(), i) != used_indices.end())
                continue;

            std::vector<DataPoint> yes, no;
            split(data, i, no, yes);

            bool uniform_yes, uniform_no;
            check_uniformity(yes, no, uniform_yes, uniform_no);

            if (uniform_yes)
            {
                tree->yes = std::make_unique<DTree>();
                tree->yes->decision = yes[0].y;
            }

            if (uniform_no)
            {
                tree->no = std::make_unique<DTree>();
                tree->no->decision = no[0].y;
            }

            if (uniform_no || uniform_yes)
            {
                best_feature_split = i;

                if (uniform_no && uniform_yes)
                    break;

                max_h = 2.f;
            }

            float h = H((float)yes.size() / data.size());
            if (h > max_h)
            {
                max_h = h;
                best_feature_split = i;
            }
        }

        std::vector<DataPoint> yes, no;
        split(data, best_feature_split, no, yes);

        tree->feature = best_feature_split;
        std::vector<int> used_indices_new = used_indices;
        used_indices_new.emplace_back(best_feature_split);
        if (!tree->yes)
            tree->yes = create_dtree(yes, used_indices_new);
        if (!tree->no)
            tree->no = create_dtree(no, used_indices_new);
        return tree;
    }

    void split(const std::vector<DataPoint> &data, int feature,
                std::vector<DataPoint> &no, std::vector<DataPoint> &yes)
    {
        for (const auto &dp : data)
        {
            if (dp.features[feature])
                yes.emplace_back(dp);
            else
                no.emplace_back(dp);
        }
    }

    float H(float p)
    {
        if (p == 0.f || p == 1.f) return 0.f;
        return -p * std::log2(p) - (1.f - p) * std::log2(1.f - p);
    }
}
