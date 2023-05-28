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
    static std::function<float(float)> get_afn(Activation a)
    {
        std::function<float(float)> fn;
        switch (a)
        {
            case Activation::Linear:
                fn = [](float z){ return z; };
                break;
            case Activation::Sigmoid:
                fn = sigmoid;
                break;
            case Activation::Relu:
                fn = [](float z){ return std::max(0.f, z); };
                break;
            case Activation::Tanh:
                fn = [](float z){ return (std::exp(z) - std::exp(-z)) / (std::exp(z) + std::exp(-z)); };
                break;
        }
        return fn;
    }

    static mt::mat gprime(const Layer &l)
    {
        switch (l.a_fn)
        {
        case Activation::Linear:
        {
            mt::mat res = l.Z;
            res.set(1.f);
            return res;
        } break;
        case Activation::Sigmoid:
        {
            mt::mat g = l.Z;
            g.set(0.f);
            l.Z.foreach([&](int r, int c){
                float g_rc = sigmoid(l.Z.at(r, c));
                g.atref(r, c) = g_rc * (1.f - g_rc);
            });
            return g;
        } break;
        case Activation::Relu:
        {
            mt::mat res = l.Z;
            res.set(0.f);
            l.Z.foreach([&](int r, int c){
                if (l.Z.at(r, c) >= 0.f)
                    res.atref(r, c) = 1.f;
            });
            return res;
        } break;
        case Activation::Tanh:
        {
            mt::mat res = l.Z;
            l.Z.foreach([&](int r, int c){
                float z = l.Z.at(r, c);
                float tanh = (std::exp(z) - std::exp(-z)) / (std::exp(z) + std::exp(-z));
                res.atref(r, c) = 1.f - tanh * tanh;
            });
            return res;
        } break;
        }

        mt::mat tmp;
        tmp.set(0.f);
        return tmp;
    }


    void dense_forward_prop(Layer &l, Layer &back_l, int m)
    {
        l.A = mt::mat(l.n, m);
        std::function<float(float)> afn = get_afn(l.a_fn);

        l.Z = l.W * back_l.A;
        l.Z.foreach([&](int r, int c){
            l.Z.atref(r, c) += l.vb.at(r, 0);
            l.A.atref(r, c) = afn(l.Z.at(r, c));
        });
    }

    std::pair<mt::mat, mt::vec> dense_back_prop(Layer &l, Layer *bl, Layer *fl, const mt::mat &Y)
    {
        if (!fl)
            l.dZ = l.A - Y;
        else
        {
            mt::mat left = fl->W.transpose() * fl->dZ;
            mt::mat right = gprime(l);
            l.dZ = left.element_wise_mul(right);
        }

        mt::mat dW = l.dZ * bl->A.transpose() * (1.f / Y.cols());
        mt::vec d_vb(l.dZ.rows());
        l.dZ.foreach([&l, &d_vb](int r, int c){ d_vb.atref(r, 0) += l.dZ.at(r, c); });

        return { dW, d_vb };
    }

    Model::Model(const std::vector<Layer> &layers, float random_init_range)
        : m_layers(layers)
    {
        // Input layer
        m_layers.insert(m_layers.begin(), Layer(0, Activation::Linear));
        m_layers[0].n = 1;
        for (size_t i = 1; i < m_layers.size(); ++i)
        {
            m_layers[i].W = mt::mat(m_layers[i].n, m_layers[i - 1].n);
            m_layers[i].W.foreach([random_init_range](float &elem){ elem = ((float)(rand() % 1000) / 1000.f - .5f) * random_init_range; });
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

    void Model::forward_prop(const mt::mat &X)
    {
        m_layers[0].A = X;

        for (size_t i = 1; i < m_layers.size(); ++i)
        {
            dense_forward_prop(m_layers[i], m_layers[i - 1], X.cols());
        }
    }

    void Model::back_prop(const mt::mat &Y, float a)
    {
        std::vector<std::pair<mt::mat, mt::vec>> diffs;
        for (size_t i = m_layers.size() - 1; i > 0; --i)
        {
            std::pair<mt::mat, mt::vec> diff = dense_back_prop(
                m_layers[i], &m_layers[i - 1],
                i == m_layers.size() - 1 ? nullptr : &m_layers[i + 1],
                Y
            );
            diffs.insert(diffs.begin(), diff);
        }

        for (size_t i = 1; i < m_layers.size(); ++i)
            apply_diffs(i, diffs[i - 1].first, diffs[i - 1].second, a);
    }

    void Model::train(const mt::mat &X, const mt::mat &Y, int epochs, float a, int print_intervals)
    {
        m_layers[0].n = X.rows();
        m_layers[1].W = mt::mat(m_layers[1].W.rows(), m_layers[0].n);

        for (int i = 0; i < epochs; ++i)
        {
            forward_prop(X);
            if ((i + 1) % print_intervals == 0)
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


    float Model::cost(const mt::mat &Y)
    {
        float sum = 0.f;
        Y.foreach([&](int r, int c){
            sum += Y.at(r, c) * std::log(m_layers.back().A.at(r, c));
        });

        return -sum;
    }
}

