#include "common.h"
#include <sstream>

//// VEC
std::vector<float> common::vec::apply_fn(std::vector<float> v,
              const std::function<float(float)> &fn)
{
    for (auto &f : v)
        f = fn(f);
    return v;
}

std::vector<float> common::vec::apply_fn(std::vector<float> v,
        const std::function<void(std::vector<float> &v, size_t i)> &fn)
{
    for (size_t i = 0; i < v.size(); ++i)
        fn(v, i);
    return v;
}

float common::vec::dot(const std::vector<float> &a, const std::vector<float> &b)
{
    if (a.size() != b.size())
        throw std::runtime_error("[common::vec::dot] a.size() != b.size().");

    float sum = 0.f;
    for (size_t i = 0; i < a.size(); ++i)
        sum += a[i] * b[i];
    return sum;
}

float common::vec::sum(const std::vector<float> &v)
{
    float total = 0.f;
    for (auto &f : v)
        total += f;
    return total;
}

std::string common::vec::to_string(const std::vector<float> &v)
{
    std::stringstream ss;
    ss.precision(2);
    for (auto &f : v)
        ss << std::fixed << f << ", ";
    return "[" + ss.str().substr(0, ss.str().size() - 2) + "]";
}

std::vector<float> common::vec::matmul(const std::vector<std::vector<float>> &m, const std::vector<float> &v)
{
    if (v.size() != m[0].size())
    {
        fprintf(stderr, "[common::vec::matmul] Matrix is shaped differently from vector.\n");
        exit(EXIT_FAILURE);
    }

    std::vector<float> res(v.size());

    for (size_t r = 0; r < m.size(); ++r)
    {
        res[r] = 0.f;
        for (size_t c = 0; c < m[r].size(); ++c)
            res[r] += v[c] * m[r][c];
    }

    return res;
}

