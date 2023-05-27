#include "deepnn.h"
#include <cstddef>
#include <cmath>
#include <functional>
#include <cstdio>
#include <fstream>
#include <sstream>

static const auto sigmoid = [](float z){ return 1.f / (1.f + std::exp(-z)); };

namespace nn
{
    Model::Model(const std::vector<Layer> &layers)
        : m_layers(layers)
    {
        // Input layer
        m_layers.insert(m_layers.begin(), Layer(0, Activation::Linear));
        m_layers[0].n = 1;
        for (size_t i = 1; i < m_layers.size(); ++i)
        {
            m_layers[i].W = mt::mat(m_layers[i].n, m_layers[i - 1].n);

            for (int r = 0; r < m_layers[i].W.rows(); ++r)
                for (int c = 0; c < m_layers[i].W.cols(); ++c)
                    m_layers[i].W.atref(r, c) = (float)(rand() % 100) / 100.f - .5f;

            m_layers[i].vb = mt::vec(m_layers[i].n);
        }
    }

    Model::Model(const std::string &src)
    {
        std::ifstream ifs(src);
        std::string line;
        while (std::getline(ifs, line))
        {
            std::stringstream ss(line);
            std::string first;
            ss >> first;

            if (first == "l")
                m_layers.emplace_back(Layer());

            if (first == "n")
                ss >> m_layers.back().n;

            if (first == "W")
            {
                int rows, cols;
                ss >> rows >> cols;
                m_layers.back().W = mt::mat(rows, cols);
                for (int r = 0; r < rows; ++r)
                {
                    for (int c = 0; c < cols; ++c)
                        ss >> m_layers.back().W.atref(r, c);
                }
            }

            if (first == "b")
            {
                int len;
                ss >> len;
                m_layers.back().vb = mt::vec(len);
                for (int i = 0; i < len; ++i)
                    ss >> m_layers.back().vb.atref(i, 0);
            }

            if (first == "afn")
            {
                int afn;
                ss >> afn;
                m_layers.back().a_fn = (Activation)afn;
            }
        }
    }

    void Model::train(const mt::mat &X, const mt::mat &Y, int epochs, float a)
    {
        m_layers[0].n = X.rows();
        m_layers[1].W = mt::mat(m_layers[1].W.rows(), m_layers[0].n);

        for (int i = 0; i < epochs; ++i)
        {
            forward_prop(X);
            if ((i + 1) % 100 == 0)
                printf("Iteration %d: %f\n", i + 1, cost(Y));
            back_prop(Y, a);
        }
    }

    std::vector<float> Model::predict(const mt::mat &X)
    {
        forward_prop(X);

        std::vector<float> res;
        for (int i = 0; i < m_layers.back().A.rows(); ++i)
            res.emplace_back(m_layers.back().A.at(i, 0));
        return res;
    }

    void Model::forward_prop(const mt::mat &X)
    {
        m_layers[0].A = X;
        int m = X.cols();

        for (size_t i = 1; i < m_layers.size(); ++i)
        {
            mt::mat A(m_layers[i].n, X.cols());

            std::function<float(float)> activation_fn;
            switch (m_layers[i].a_fn)
            {
            case Activation::Linear:
                activation_fn = [](float z){ return z; };
                break;
            case Activation::Sigmoid:
                activation_fn = sigmoid;
                break;
            case Activation::Relu:
                activation_fn = [](float z){ return std::max(0.f, z); };
                break;
            case Activation::Tanh:
                activation_fn = [](float z){ return (std::exp(z) - std::exp(-z)) / (std::exp(z) + std::exp(-z)); };
                break;
            }

            mt::mat Z = m_layers[i].W * m_layers[i - 1].A;
            for (int c = 0; c < m; ++c)
                for (int r = 0; r < m_layers[i].vb.rows(); ++r)
                {
                    Z.atref(r, c) += m_layers[i].vb.at(r, 0);
                    A.atref(r, c) = activation_fn(Z.at(r, c));
                }

            m_layers[i].Z = Z;
            m_layers[i].A = A;
        }
    }

    void Model::back_prop(const mt::mat &Y, float a)
    {
        std::vector<mt::mat> dWs;
        std::vector<mt::vec> d_vbs;
        mt::mat dZ_prev = m_layers.back().A,
                dW_prev;

        // Output layer
        for (int r = 0; r < m_layers.back().A.rows(); ++r)
        {
            for (int c = 0; c < m_layers.back().A.cols(); ++c)
            {
                dZ_prev.atref(r, c) = m_layers.back().A.atref(r, c) - Y.at(r, c);
            }
        }
        dW_prev = (dZ_prev * m_layers[m_layers.size() - 2].A.transpose()) * (1.f / Y.cols());

        mt::vec d_vb(dZ_prev.rows());
        d_vb.set(0.f);
        for (int i = 0; i < dZ_prev.cols(); ++i)
        {
            for (int j = 0; j < dZ_prev.rows(); ++j)
                d_vb.atref(j, 0) += dZ_prev.at(j, i);
        }

        for (int i = 0; i < d_vb.rows(); ++i)
            d_vb.atref(i, 0) /= Y.cols();

        dWs.emplace_back(dW_prev);
        d_vbs.emplace_back(d_vb);

        // Only hidden layers
        for (int i = m_layers.size() - 2; i > 0; --i)
        {
            mt::mat left = m_layers[i + 1].W.transpose() * dZ_prev;
            mt::mat right = gprime(i, m_layers[i].Z);
            // Element-wise product left * right
            mt::mat dZ(left.rows(), left.cols());
            for (int r = 0; r < left.rows(); ++r)
            {
                for (int c = 0; c < left.cols(); ++c)
                {
                    dZ.atref(r, c) = left.at(r, c) * right.at(r, c);
                }
            }

            mt::mat dW = (dZ * m_layers[i - 1].A.transpose()) * (1.f / Y.cols());

            d_vb = mt::vec(dZ.rows());
            d_vb.set(0.f);
            for (int j = 0; j < dZ.cols(); ++j)
            {
                for (int k = 0; k < dZ.rows(); ++k)
                    d_vb.atref(k, 0) += dZ.at(k, j);
            }
            for (int j = 0; j < d_vb.rows(); ++j)
                d_vb.atref(j, 0) /= dZ.cols();

            dWs.insert(dWs.begin(), dW);
            d_vbs.insert(d_vbs.begin(), d_vb);
            /* apply_diffs(i, dW, d_vb); */
            dW_prev = dW;
            dZ_prev = dZ;
        }

        for (size_t i = 1; i < m_layers.size(); ++i)
            apply_diffs(i, dWs[i - 1], d_vbs[i - 1], a);
    }

    void Model::save_params(const std::string &fp)
    {
        std::ofstream ofs(fp);
        for (size_t i = 0; i < m_layers.size(); ++i)
        {
            const Layer &l = m_layers[i];
            ofs << "l\n";
            ofs << "n " << l.n << '\n';

            if (i > 0)
            {
                ofs << "W " << l.W.rows() << ' ' << l.W.cols() << ' ';
                for (int r = 0; r < l.W.rows(); ++r)
                    for (int c = 0; c < l.W.cols(); ++c)
                        ofs << l.W.at(r, c) << ' ';
                ofs << '\n';

                ofs << "b " << l.vb.rows() << ' ';
                for (int j = 0; j < l.vb.rows(); ++j)
                    ofs << l.vb.at(j, 0) << ' ';
                ofs << '\n';

                ofs << "afn " << (int)l.a_fn << '\n';
            }
        }
    }

    void Model::apply_diffs(int l,
            const mt::mat &dW,
            const mt::vec &db,
            float a
        )
    {
        m_layers[l].W = m_layers[l].W + dW * (-a);

        for (int i = 0; i < m_layers[l].vb.rows(); ++i)
            m_layers[l].vb.atref(i, 0) -= a * db.at(i, 0);
    }

    mt::mat Model::gprime(int l, const mt::mat &Z)
    {
        switch (m_layers[l].a_fn)
        {
        case Activation::Linear:
        {
            mt::mat res = Z;
            res.set(1.f);
            return res;
        } break;
        case Activation::Sigmoid:
        {
            mt::mat g = Z;
            g.set(0.f);
            for (int r = 0; r < Z.rows(); ++r)
            {
                for (int c = 0; c < Z.cols(); ++c)
                {
                    float g_rc = sigmoid(Z.at(r, c));
                    g.atref(r, c) = g_rc * (1.f - g_rc);
                }
            }

            return g;
        } break;
        case Activation::Relu:
        {
            mt::mat res = Z;
            res.set(0.f);
            for (int r = 0; r < Z.rows(); ++r)
            {
                for (int c = 0; c < Z.cols(); ++c)
                {
                    if (Z.at(r, c) >= 0.f)
                        res.atref(r, c) = 1.f;
                }
            }

            return res;
        } break;
        case Activation::Tanh:
        {
            mt::mat res = Z;
            for (int r = 0; r < Z.rows(); ++r)
            {
                for (int c = 0; c < Z.cols(); ++c)
                {
                    float z = Z.at(r, c);
                    float tanh = (std::exp(z) - std::exp(-z)) / (std::exp(z) + std::exp(-z));
                    res.atref(r, c) = 1.f - tanh * tanh;
                }
            }

            return res;
        } break;
        }

        mt::mat tmp;
        tmp.set(0.f);
        return tmp;
    }

    float Model::cost(const mt::mat &Y)
    {
        float sum = 0.f;
        for (int i = 0; i < Y.cols(); ++i)
        {
            for (int j = 0; j < Y.rows(); ++j)
            {
                sum += Y.at(j, i) * std::log(m_layers.back().A.at(j, i));
            }
        }

        return -sum;
    }
}

