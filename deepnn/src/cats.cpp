#include "deepnn.h"

int main()
{
    nn::Model model({
        nn::Layer(25, nn::Activation::Relu),
        nn::Layer(15, nn::Activation::Relu),
        nn::Layer(10, nn::Activation::Linear)
    });

    model.compile();

    return 0;
}

