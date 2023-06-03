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
        switch (a)
        {
        case Activation::Linear:
            return [](float z){ return z; };
        case Activation::Sigmoid:
            return sigmoid;
        case Activation::Relu:
            return [](float z){ return std::max(0.f, z); };
        case Activation::Tanh:
            return [](float z){ return (std::exp(z) - std::exp(-z)) / (std::exp(z) + std::exp(-z)); };
        }
    }

    static std::function<float(float)> get_afn_derivative(Activation a)
    {
        switch (a)
        {
        case Activation::Linear:
            return [](float z){ return 1.f; };
        case Activation::Sigmoid:
            return [](float z){ return sigmoid(z) * (1.f - sigmoid(z)); };
        case Activation::Relu:
            return [](float z){ return z > 0.f ? z : 0.f; };
        case Activation::Tanh:
            return [](float z){
                float tanh = (std::exp(z) - std::exp(-z)) / (std::exp(z) + std::exp(-z));
                return 1.f - tanh * tanh;
            };
        }
    }

    shape4 shape4_from(int blocks, int length, int rows, int cols, float init = 0.f)
    {
        shape4 res(blocks);
        for (size_t i = 0; i < res.size(); ++i)
        {
            res[i].resize(length);
            for (size_t j = 0; j < res[i].size(); ++j)
            {
                res[i][j] = mt::mat(rows, cols);
                res[i][j].set(init);
            }
        }

        return res;
    }

    shape4 shape4_match(const shape4 &other, float init = 0.f)
    {
        shape4 res = other;
        for (auto &block : res)
        {
            for (auto &layer : block)
                layer.set(init);
        }

        return res;
    }

    // Front: W, dZ
    // Back: A
    static std::unique_ptr<Conv> dense2conv(const Dense *l, bool front)
    {
        std::unique_ptr<Conv> res = std::make_unique<Conv>();

        if (front)
        {
        }
        else
        {
        }

        return res;
    }

    // Front: W, dZ
    // Back: A
    static std::unique_ptr<Dense> conv2dense(const Conv *l, bool front)
    {
        std::unique_ptr<Dense> res = std::make_unique<Dense>();

        if (front)
        {
        }
        else
        {
        }

        return res;
    }

    static std::unique_ptr<Dense> to_dense(const Layer *l, bool front)
    {
        switch (l->type())
        {
        case LayerType::DENSE:
            return std::make_unique<Dense>(*dynamic_cast<const Dense*>(l));
        case LayerType::CONV:
            return conv2dense(dynamic_cast<const Conv*>(l), front);
        }
    }

    static std::unique_ptr<Conv> to_conv(const Layer *l, bool front)
    {
        switch (l->type())
        {
        case LayerType::DENSE:
            return dense2conv(dynamic_cast<const Dense*>(l), front);
        case LayerType::CONV:
            return std::make_unique<Conv>(*dynamic_cast<const Conv*>(l));
        }
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
            mt::mat right = this->Z;
            right.foreach([this](float &z){
                z = get_afn_derivative(this->a_fn)(z);
            });
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

    static mt::mat convolve(const mt::mat &filter, const mt::mat &input, int s,
                            int resh, int resw)
    {
        mt::mat res(resh, resw);

        res.foreach([&](int r, int c){
            res.atref(r, c) = 0.f;
            for (int i = 0; i < filter.rows(); i += s)
            {
                for (int j = 0; j < filter.cols(); j += s)
                    res.atref(r, c) += filter.at(i, j) *
                                       input.at(r + i, c + j);
            }
        });

        return res;
    }

    void Conv::forward_prop(const Conv *back_l, int m)
    {
        // ith example, jth channel (from filter)
        // shape m n_c fh fw
        shape4 wa = shape4_from(m, this->W.size(), this->fh, this->fw);

        for (size_t ex = 0; ex < back_l->A.size(); ++ex)/* const auto &example : back_l->A */
        {
            // Iter over filters
            for (size_t i = 0; i < this->W.size(); ++i)
            {
                // Sum convolutions of each filter channel on its corresponding input channel
                // Iter over channels
                for (size_t j = 0; j < this->W[0].size(); ++j)
                {
                    // Filter i channel j convolve example channel j
                    // W[i][j] * example[j]
                    wa[ex][i] = wa[ex][i] + convolve(
                        this->W[i][j], back_l->A[ex][j], this->s,
                        this->nh(back_l->A[0][0].rows()),
                        this->nw(back_l->A[0][0].cols())
                    );
                }
            }
        }

        // z = wa + b
        // b is a vector in the direction of the channels
        // Each element in b is added to a different channel
        // Example, row and column in each layer of wa are constant in adding b
        for (auto &ex : wa)
        {
            // Iter over channels
            for (size_t c = 0; c < ex.size(); ++c)
            {
                ex[c].foreach([&ex, c, this](int row, int col){
                    ex[c].atref(row, col) += this->b.atref(row, 0);
                });
            }
        }

        // Set Z and A
        this->Z = wa;
        this->A = shape4_match(this->Z);

        std::function<float(float)> fn = get_afn(this->a_fn);
        // Iter over examples
        for (size_t i = 0; i < this->A.size(); ++i)
        {
            // Iter over channels
            for (size_t j = 0; j < this->A[0].size(); ++j)
            {
                this->A[i][j].foreach([this, &fn, i, j](int r, int c){
                    this->A[i][j].atref(r, c) = fn(this->Z[i][j].at(r, c));
                });
            }
        }
    }

    ParamDiff Conv::back_prop(const Conv *behind, const Conv *front,
                 const mt::mat &Y)
    {
        /* this->dZ = shape4_match(this->A); */

        /* // shape m, n_c^l, n_h^l, n_w^l */
        /* std::function<float(float)> gprime = get_afn_derivative(this->a_fn); */
        /* shape4 gprime_z = shape4_match(this->dZ); */
        /* for (size_t ex = 0; ex < gprime_z.size(); ++ex) */
        /* { */
        /*     for (size_t ch = 0; ch < gprime_z[0].size(); ++ch) */
        /*         gprime_z[ex][ch].foreach([&](float &z){ z = gprime(z); }); */
        /* } */

        /* for (size_t ex = 0; ex < this->dZ.size(); ++ex) */
        /* { */
        /*     // Reverse convolution with front variables */
        /*     // Iterate over filters */
        /*     for (size_t i = 0; i < front->W.size(); ++i) */
        /*     { */
        /*         // Iterate over channels */
        /*         for (size_t j = 0; j < front->W[0].size(); ++j) */
        /*         { */
        /*             this->dZ[ex][j] = this->dZ[ex][j] + */
        /*                 convolve(front->W[i][j].rot180(), front->dZ[ex][j], */
        /*                          this->s, */
        /*                          this->nh(behind->A[0][0].rows()), */
        /*                          this->nw(behind->A[0][0].cols()) */
        /*                 ); */
        /*         } */
        /*     } */

        /*     // Element wise multiply with gprime */
        /*     for (size_t c = 0; c < this->dZ[0].size(); ++c) */
        /*         this->dZ[ex][c].element_wise_mul(gprime_z[ex][c]); */
        /* } */

        /* shape4 dW = shape4_match(this->W); */

        /* // n_c^{[l]} */
        /* for (size_t ch = 0; ch < dW.size(); ++ch) */
        /* { */
        /*     // n_c^{[l-1]} */
        /*     for (size_t chb = 0; chb < dW[0].size(); ++chb) */
        /*     { */
        /*     } */
        /* } */

        /* mt::mat dW = this->dZ * behind->A.transpose() * (1.f / Y.cols()); */
        /* mt::vec db(this->dZ.rows()); */
        /* this->dZ.foreach([this, &db](int r, int c){ db.atref(r, 0) += this->dZ.at(r, c); }); */

        /* return ParamDiff{ */
        /*     .dense_dW = dW, */
        /*     .dense_db = db */
        /* }; */
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

    void Model::forward_prop(const Input &X)
    {
        prepare_layer0(X);

        for (size_t i = 1; i < m_layers.size(); ++i)
        {
            switch (m_layers[i]->type())
            {
            case LayerType::DENSE:
                dynamic_cast<Dense*>(m_layers[i].get())->forward_prop(
                    dynamic_cast<Dense*>(m_layers[i - 1].get()),
                    X.dense.cols()
                );
                break;
            case LayerType::CONV:
                dynamic_cast<Conv*>(m_layers[i].get())->forward_prop(
                    dynamic_cast<Conv*>(m_layers[i - 1].get()),
                    X.conv.size()
                );
                break;
            }
        }
    }

    void Model::back_prop(const mt::mat &Y, float a)
    {
        std::vector<ParamDiff> diffs;
        for (size_t i = m_layers.size() - 1; i > 0; --i)
        {
            ParamDiff diff;
            switch (m_layers[i]->type())
            {
            case LayerType::DENSE:
                diff = dynamic_cast<Dense*>(m_layers[i].get())->back_prop(
                    dynamic_cast<Dense*>(m_layers[i - 1].get()),
                    i == m_layers.size() - 1 ? nullptr : dynamic_cast<Dense*>(m_layers[i + 1].get()),
                    Y
                );
                break;
            case LayerType::CONV:
                diff = dynamic_cast<Conv*>(m_layers[i].get())->back_prop(
                    dynamic_cast<Conv*>(m_layers[i - 1].get()),
                    i == m_layers.size() - 1 ? nullptr : dynamic_cast<Conv*>(m_layers[i + 1].get()),
                    Y
                );
                break;
            }
            diffs.insert(diffs.begin(), diff);
        }

        for (size_t i = 1; i < m_layers.size(); ++i)
            apply_diffs(i, diffs[i - 1], a);
    }

    void Model::train(const Input &X, const mt::mat &Y, int epochs, float a, int print_intervals)
    {
        // Preconditions
        // Must have hidden layers
        // Last layer must be dense
        // LayerData layer must have same type as first hidden layer
        if (m_layers.size() < 3)
            throw std::runtime_error("[Model::train] Hidden layers are required.");

        if (m_layers.back()->type() != LayerType::DENSE)
            throw std::runtime_error("[Model::train] Last layer must be dense.");

        if (m_layers[0]->type() != m_layers[1]->type())
            throw std::runtime_error(
                "LayerData layer must have same type as first hidden layer."
            );

        prepare_layer0(X);

        for (size_t i = 1; i < m_layers.size(); ++i)
        {
            switch (m_layers[i]->type())
            {
            case LayerType::DENSE:
            {
                // TODO Turn prev_n calculation into function if necessary
                int prev_n;
                switch (m_layers[i - 1]->type())
                {
                case LayerType::DENSE:
                    prev_n = dynamic_cast<Dense*>(m_layers[i - 1].get())->n;
                    break;
                case LayerType::CONV:
                {
                    Conv *conv = dynamic_cast<Conv*>(m_layers[i - 1].get());
                    // Flattened A dims (guaranteed to be set)
                    prev_n = conv->A.size() * conv->A[0].size() *
                        conv->A[0][0].rows() * conv->A[0][0].cols();
                } break;
                }

                Dense *l = dynamic_cast<Dense*>(m_layers[i].get());
                l->W = mt::mat(l->n, prev_n);
                l->W.foreach([](float &elem){ elem = ((float)(rand() % 1000) / 1000.f - .5f); });
                l->vb = mt::vec(l->n);
            } break;
            case LayerType::CONV:
            {
                // Filter count in a conv layer determines the layer's channel count
                int prev_nc;
                int prev_nw, prev_nh;
                switch (m_layers[i - 1]->type())
                {
                case LayerType::DENSE:
                    prev_nh = dynamic_cast<Dense*>(m_layers[i - 1].get())->n;
                    prev_nw = 1;
                    prev_nc = 1;
                    break;
                case LayerType::CONV:
                {
                    Conv *prev = dynamic_cast<Conv*>(m_layers[i - 1].get());
                    prev_nc = prev->nc;
                    prev_nh = prev->A[0][0].rows();
                    prev_nw = prev->A[0][0].cols();
                } break;
                }

                Conv *l = dynamic_cast<Conv*>(m_layers[i].get());
                l->W = shape4_from(l->nc, prev_nc, l->fh, l->fw, 2.f);
                l->b = mt::vec(l->nc);

                l->A = shape4_from(X.conv.size(), l->nc, l->nh(prev_nh), l->nw(prev_nw));
            } break;
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

    std::vector<float> Model::predict(const Input &X)
    {
        forward_prop(X);

        // Last layer is always dense, model requires at least one output regression unit
        std::vector<float> res;
        Dense *l = dynamic_cast<Dense*>(m_layers.back().get());
        for (int i = 0; i < l->A.rows(); ++i)
            res.emplace_back(l->A.at(i, 0));

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

    void Model::prepare_layer0(const Input &X)
    {
        switch (m_layers[0]->type())
        {
        case LayerType::DENSE:
            dynamic_cast<Dense*>(m_layers[0].get())->A = X.dense;
            break;
        case LayerType::CONV:
            dynamic_cast<Conv*>(m_layers[0].get())->A = X.conv;
            break;
        }
    }

    float Model::cost(const mt::mat &Y)
    {
        // Output layer can only have dense layers
        float sum = 0.f;
        Y.foreach([&](int r, int c){
            sum += Y.at(r, c) * std::log(dynamic_cast<Dense*>(m_layers.back().get())->A.at(r, c));
        });

        return -sum;
    }
}

