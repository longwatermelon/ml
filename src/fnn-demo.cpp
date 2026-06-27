#include "emel/nn.h"

int main() {
    Nn nn(784, {
        {128, Activation::Relu},
        {64, Activation::Relu},
        {10, Activation::Softmax},
    });

    Tensor Xtrain = Tensor::deserialize(read_file_bytes("data/mnist-digits/train_X.tensor"));
    Tensor Ytrain = Tensor::deserialize(read_file_bytes("data/mnist-digits/train_Y.tensor"));
    Tensor Xtest = Tensor::deserialize(read_file_bytes("data/mnist-digits/test_X.tensor"));
    Tensor Ytest = Tensor::deserialize(read_file_bytes("data/mnist-digits/test_Y.tensor"));

    nn.train(Xtrain, Ytrain, 10, 32, 0.001, Loss::CrossEntropy);

    Tensor Yhat = nn.predict(Xtest);
    printf("test loss: %.6f\n", loss(Yhat, Ytest, Loss::CrossEntropy));
}
