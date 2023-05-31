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

    static mt::mat gprime(const Layer *l)
    {
        if (l->type() == LayerType::DENSE)
        {
            const Dense *ll = dynamic_cast<const Dense*>(l);
            switch (ll->a_fn)
            {
            case Activation::Linear:
            {
                mt::mat res = ll->Z;
                res.set(1.f);
                return res;
            } break;
            case Activation::Sigmoid:
            {
                mt::mat g = ll->Z;
                g.set(0.f);
                ll->Z.foreach([&](int r, int c){
                    float g_rc = sigmoid(ll->Z.at(r, c));
                    g.atref(r, c) = g_rc * (1.f - g_rc);
                });
                return g;
            } break;
            case Activation::Relu:
            {
                mt::mat res = ll->Z;
                res.set(0.f);
                ll->Z.foreach([&](int r, int c){
                    if (ll->Z.at(r, c) >= 0.f)
                        res.atref(r, c) = 1.f;
                });
                return res;
            } break;
            case Activation::Tanh:
            {
                mt::mat res = ll->Z;
                ll->Z.foreach([&](int r, int c){
                    float z = ll->Z.at(r, c);
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

        return mt::mat(0, 0);
    }

    void Dense::forward_prop(const Dense *back_l, int m)
    {
        A = mt::mat(n, m);
        std::function<float(float)> afn = get_afn(a_fn);

        Z = W * back_l->A;
        Z.foreach([&](int r, int c){
            Z.atref(r, c) += vb.at(r, 0);
            A.atref(r, c) = afn(Z.at(r, c));
        });
    }

    ParamDiff Dense::back_prop(const Dense *behind, const Dense *front,
                 const mt::mat &Y)
    {
        if (!front)
            dZ = A - Y;
        else
        {
            mt::mat left = front->W.transpose() * front->dZ;
            mt::mat right = gprime(this);
            dZ = left.element_wise_mul(right);
        }

        mt::mat dW = dZ * behind->A.transpose() * (1.f / Y.cols());
        mt::vec db(dZ.rows());
        dZ.foreach([this, &db](int r, int c){ db.atref(r, 0) += dZ.at(r, c); });

        return ParamDiff{
            .dense_dW = dW,
            .dense_db = db
        };
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

            if (first == "dense")
                m_layers.emplace_back(std::make_unique<Dense>());

            switch (m_layers.back()->type())
            {
            case LayerType::DENSE:
            {
                Dense *l = dynamic_cast<Dense*>(m_layers.back().get());
                if (first == "n")
                    ss >> l->n;

                if (first == "W")
                {
                    int rows, cols;
                    ss >> rows >> cols;
                    l->W = mt::mat(rows, cols);
                    for (int r = 0; r < rows; ++r)
                    {
                        for (int c = 0; c < cols; ++c)
                            ss >> l->W.atref(r, c);
                    }
                }

                if (first == "b")
                {
                    int len;
                    ss >> len;
                    l->vb = mt::vec(len);
                    for (int i = 0; i < len; ++i)
                        ss >> l->vb.atref(i, 0);
                }

                if (first == "afn")
                {
                    int afn;
                    ss >> afn;
                    l->a_fn = (Activation)afn;
                }
            } break;
            }
        }
    }

    void Model::forward_prop(const mt::mat &X)
    {
        dynamic_cast<Dense*>(m_layers[0].get())->A = X;

        for (size_t i = 1; i < m_layers.size(); ++i)
        {
            dynamic_cast<Dense*>(m_layers[i].get())->forward_prop(
                dynamic_cast<Dense*>(m_layers[i - 1].get()),
                X.cols()
            );
        }
    }

    void Model::back_prop(const mt::mat &Y, float a)
    {
        std::vector<ParamDiff> diffs;
        for (size_t i = m_layers.size() - 1; i > 0; --i)
        {
            ParamDiff diff = dynamic_cast<Dense*>(m_layers[i].get())->back_prop(
                dynamic_cast<Dense*>(m_layers[i - 1].get()),
                i == m_layers.size() - 1 ? nullptr : dynamic_cast<Dense*>(m_layers[i + 1].get()),
                Y
            );
            diffs.insert(diffs.begin(), diff);
        }

        for (size_t i = 1; i < m_layers.size(); ++i)
            apply_diffs(i, diffs[i - 1], a);
    }

    void Model::train(const mt::mat &X, const mt::mat &Y, int epochs, float a, int print_intervals)
    {
        for (size_t i = 1; i < m_layers.size(); ++i)
        {
            if (m_layers[i]->type() == LayerType::DENSE)
            {
                // TODO Generalize for cases when layer l-1 isn't dense
                Dense *l = dynamic_cast<Dense*>(m_layers[i].get());
                l->W = mt::mat(l->n, dynamic_cast<Dense*>(m_layers[i - 1].get())->n);
                l->W.foreach([](float &elem){ elem = ((float)(rand() % 1000) / 1000.f - .5f); });
                l->vb = mt::vec(l->n);
            }
        }

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
        switch (m_layers.back()->type())
        {
        case LayerType::DENSE:
            Dense *l = dynamic_cast<Dense*>(m_layers.back().get());
            for (int i = 0; i < l->A.rows(); ++i)
                res.emplace_back(l->A.at(i, 0));
            break;
        }

        return res;
    }

    void Model::save_params(const std::string &fp)
    {
        std::ofstream ofs(fp);
        for (size_t i = 0; i < m_layers.size(); ++i)
        {
            const Layer *l = m_layers[i].get();
            if (l->type() == LayerType::DENSE)
            {
                const Dense *ll = dynamic_cast<const Dense*>(l);
                ofs << "l\n";
                ofs << "n " << ll->n << '\n';

                if (i > 0)
                {
                    ofs << "W " << ll->W.rows() << ' ' << ll->W.cols() << ' ';
                    for (int r = 0; r < ll->W.rows(); ++r)
                        for (int c = 0; c < ll->W.cols(); ++c)
                            ofs << ll->W.at(r, c) << ' ';
                    ofs << '\n';

                    ofs << "b " << ll->vb.rows() << ' ';
                    for (int j = 0; j < ll->vb.rows(); ++j)
                        ofs << ll->vb.at(j, 0) << ' ';
                    ofs << '\n';

                    ofs << "afn " << (int)ll->a_fn << '\n';
                }
            }
        }
    }

    void Model::apply_diffs(int l, const ParamDiff &diff, float a)
    {
        switch (m_layers[l]->type())
        {
        case LayerType::DENSE:
        {
            Dense *ll = dynamic_cast<Dense*>(m_layers[l].get());
            ll->W = ll->W + diff.dense_dW * (-a);

            for (int i = 0; i < ll->vb.rows(); ++i)
                ll->vb.atref(i, 0) -= a * diff.dense_db.at(i, 0);
        } break;
        }
    }


    float Model::cost(const mt::mat &Y)
    {
        // TODO Generalize for non-dense layers
        float sum = 0.f;
        Y.foreach([&](int r, int c){
            sum += Y.at(r, c) * std::log(dynamic_cast<Dense*>(m_layers.back().get())->A.at(r, c));
        });

        return -sum;
    }
}

