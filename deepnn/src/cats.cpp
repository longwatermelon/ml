#include "deepnn.h"
#include <iostream>

int main()
{
    nn::Model model({
        nn::Layer(4, nn::Activation::Tanh),
        nn::Layer(2, nn::Activation::Sigmoid)
    });

    mt::mat X(3, 1);
    X.atref(0, 0) = 0.f;
    X.atref(1, 0) = 1.f;
    X.atref(2, 0) = 2.f;
    /* Eigen::MatrixXf X(1, 3); */
    /* X << 0.f, 1.f, 2.f; */

    /* Eigen::VectorXf vy(3); */
    /* vy << 0.f, 1.f, 2.f; */
    mt::vec vy(1);
    vy.atref(0, 0) = 1.f;

    model.train(X, vy, 1000);

    return 0;
}

