#include "deepnn.h"
#include <reg.h>
#include <cstddef>

namespace deepnn
{
    Model::Model(const std::vector<Layer> &layers)
        : m_layers(layers)
    {
        for (size_t i = 1; i < m_layers.size(); ++i)
        {
            m_layers[i].W = Eigen::MatrixXf(m_layers[i].n, m_layers[i - 1].n);
            m_layers[i].vb = Eigen::VectorXf(m_layers[i].n);
        }
    }
}
