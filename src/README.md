# README

The `.cpp` files here outside of `emel/` are examples, and this file documents what each of them does.

`fnn-mnist.cpp`: Demo of a simple FNN trained on an MNIST digits dataset.
  * MNIST digits are `28x28` from `data/mnist-digits/` --- the network is a standard `784` (input) --> `128` (relu) --> `64` (relu) --> `10` (softmax), evaluated with cross-entropy loss. Training run is over `10` epochs with a batch size of `32` and a learning rate of `0.15`.
  * The accuracy eval part, for each example, replaces the predicted probability vector and the ground truth probability vector both with one-hot argmax masks, then element-wise multiplies the two together so that only examples which have the same output prediction as ground truth pass with a one-hot vector --- others get sent to all zeros. Sum-reduce along both axes to obtain the number of examples which match prediction and ground truth.

`cnn-mnist.cpp`: Demo of a simple CNN trained on an MNIST digits dataset.
  * Data comes in as `[N,1,28,28]` where `N` is batch size. The network is `conv5 Cout=4` ---> `cont5 Cout=8` ---> flatten ---> `64` linear ---> `10` linear.

