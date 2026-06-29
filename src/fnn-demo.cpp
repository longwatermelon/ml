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

    nn.train(Xtrain, Ytrain, 10, 32, 0.15, Loss::CrossEntropy);

    Tensor Yhat = nn.predict(Xtest);
    printf("test loss: %.6f\n", calc_loss(Yhat, Ytest, Loss::CrossEntropy));

    Tensor Yhat_argmax = Yhat.argmax(1);
    Tensor Ytest_argmax = Ytest.argmax(1);
    Tensor share_mask = Yhat_argmax.hadamard(Ytest_argmax);
    double count = share_mask.sum(0, false).sum(0, true).at({0});
    double accuracy = count / Ytest.shape[0];
    printf("accuracy: %f\n", accuracy);
}
