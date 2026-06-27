#include "emel/nn.h"

int main() {
    Nn nn(784, {
        {128, Activation::Relu},
        {64, Activation::Relu},
        {10, Activation::Softmax},
    });
}
