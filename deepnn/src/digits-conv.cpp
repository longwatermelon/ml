#include "deepnn.h"
#include "deps/stb_image.h"
#include <vector>
#include <string>

#define ZEROES 10
#define ONES 10

std::vector<float> from_image(const std::string &f)
{
    std::vector<float> values;
    int x, y, n;
    unsigned char* data = stbi_load(f.c_str(), &x, &y, &n, 0);

    if (data)
    {
        for (int i = 0; i < x * y * n; i += n)
        {
            values.emplace_back((float)data[i] / 255.f);
        }
    }

    stbi_image_free(data);
    return values;
}

int main()
{
    srand(1);

    // One RGB image of size 5x5, completely red
    nn::Input X;
    X.conv.emplace_back(std::vector<mt::mat>(3));
    X.conv[0][0] = mt::mat(6, 6);
    X.conv[0][0].set(1.f);
    X.conv[0][1] = mt::mat(6, 6);
    X.conv[0][1].set(0.f);
    X.conv[0][2] = mt::mat(6, 6);
    X.conv[0][2].set(0.f);

    mt::mat Y(1, 1);
    Y.atref(0, 0) = 1.f;

    nn::Model model;
    model.add(std::make_unique<nn::Conv>(3, 1, 1, 1, nn::Activation::Linear));
    model.add(std::make_unique<nn::Conv>(2, 3, 3, 1, nn::Activation::Relu));
    model.add(std::make_unique<nn::Dense>(1, nn::Activation::Sigmoid));

    model.train(X, Y, 1000, 1.f);

    return 0;
}

