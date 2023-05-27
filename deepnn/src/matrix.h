#pragma once
#include <cctype>
#include <vector>
#include <cstdio>
#include <iostream>

namespace mt
{
    class mat
    {
    public:
        mat() = default;
        mat(int r, int c)
        {
            m_data.resize(c);
            for (int i = 0; i < m_data.size(); ++i)
                m_data[i].resize(r);
        }

        void set(float s)
        {
            for (int r = 0; r < rows(); ++r)
                for (int c = 0; c < cols(); ++c)
                    atref(r, c) = s;
        }

        mat operator*(mat m) const
        {
            mat res(rows(), m.cols());

            for (int i = 0; i < rows(); ++i)
            {
                for (int j = 0; j < m.cols(); ++j)
                {
                    res.atref(i, j) = 0;
                    for (int k = 0; k < cols(); ++k)
                        res.atref(i, j) += at(i, k) * m.at(k, j);
                }
            }

            return res;
        }

        mat operator*(float s) const
        {
            mat res(rows(), cols());
            for (int r = 0; r < rows(); ++r)
            {
                for (int c = 0; c < cols(); ++c)
                {
                    res.atref(r, c) = at(r, c) * s;
                }
            }

            return res;
        }

        mat operator+(mat other) const
        {
            mat res = *this;
            for (int r = 0; r < other.rows(); ++r)
            {
                for (int c = 0; c < other.cols(); ++c)
                    res.atref(r, c) += other.at(r, c);
            }

            return res;
        }

        mat transpose() const
        {
            mat res(cols(), rows());

            for (int r = 0; r < rows(); ++r)
            {
                for (int c = 0; c < cols(); ++c)
                {
                    res.atref(c, r) = at(r, c);
                }
            }

            return res;
        }

        void print() const
        {
            for (int r = 0; r < rows(); ++r)
            {
                for (int c = 0; c < cols(); ++c)
                {
                    printf("%f ", at(r, c));
                }
                printf("\n");
            }
        }

        void print_dims() const
        {
            printf("%dx%d\n", rows(), cols());
        }

        float at(int r, int c) const { check(r, c); return m_data[c][r]; }
        float& atref(int r, int c) { check(r, c); return m_data[c][r]; }
        void check(int r, int c) const
        {
            if (r < 0 || r >= rows() || c < 0 || c >= cols())
            {
                std::cerr << "Error in indexing mt::mat\n";
                exit(EXIT_FAILURE);
            }
        }

        int rows() const { return m_data[0].size(); }
        int cols() const { return m_data.size(); }

        std::vector<std::vector<float>> m_data;
    };

    class vec : public mat
    {
    public:
        vec() = default;
        vec(int n)
            : mat(n, 1) {}
    };
}

