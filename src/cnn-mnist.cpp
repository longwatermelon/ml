#include "emel/nn.h"
#include "emel/opt.h"

int main() {
    nn::Sequential model;
    model.add<nn::Conv2d>(1, 4, 5);
    model.add<nn::Relu>();
    model.add<nn::Conv2d>(4, 8, 5);
    model.add<nn::Relu>();
    model.add<nn::Flatten>();
    model.add<nn::Linear>(8*20*20, 64);
    model.add<nn::Linear>(64, 10);

    Tensor Xtrain = Tensor::deserialize(read_file_bytes("data/mnist-digits/train_X.tensor"));
    Tensor Ytrain = Tensor::deserialize(read_file_bytes("data/mnist-digits/train_Y.tensor"));
    Tensor Xtest = Tensor::deserialize(read_file_bytes("data/mnist-digits/test_X.tensor"));
    Tensor Ytest = Tensor::deserialize(read_file_bytes("data/mnist-digits/test_Y.tensor"));

    vec<int> shape = Xtrain.shape;
    vprint(all(shape));

    Sgd opt(model.params(), 0.15);
    nn::train(model, Xtrain, Ytrain, 10, Loss::CrossEntropyLogits, opt, 32);

    Tensor Ylogits = model.forward(Xtest).get_tensor();
    printf("test loss: %.6f\n", apply_loss_scalar(Ylogits, Ytest, Loss::CrossEntropyLogits));

    Tensor Yhat_argmax = Ylogits.argmax(1);
    Tensor Ytest_argmax = Ytest.argmax(1);
    Tensor share_mask = Yhat_argmax.hadamard(Ytest_argmax);
    double count = share_mask.sum(0, false).sum(0, true).at({0});
    double accuracy = count / Ytest.shape[0];
    printf("accuracy: %f\n", accuracy);
}
