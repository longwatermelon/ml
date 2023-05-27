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

    nn::Model model({
        nn::Layer(25, nn::Activation::Sigmoid),
        nn::Layer(15, nn::Activation::Sigmoid),
        nn::Layer(2, nn::Activation::Sigmoid)
    });

    std::vector<std::vector<float>> images;

    for (int i = 0; i < ZEROES; ++i)
        images.emplace_back(from_image("data/digits/" + std::to_string(i) + "0.png"));
    for (int i = 0; i < ONES; ++i)
        images.emplace_back(from_image("data/digits/" + std::to_string(i) + "1.png"));

    int nf = images[0].size();
    int m = images.size();

    mt::mat X(nf, m);
    for (int r = 0; r < nf; ++r)
    {
        for (int c = 0; c < m; ++c)
        {
            X.atref(r, c) = images[c][r];
        }
    }

    mt::mat Y(model.m_layers.back().n, m);
    for (int c = 0; c < m; ++c)
    {
        if (c <= ZEROES - 1) // Zero
        {
            Y.atref(0, c) = 1.f;
            Y.atref(1, c) = 0.f;
        }
        else
        {
            Y.atref(0, c) = 0.f;
            Y.atref(1, c) = 1.f;
        }
    }

    model.train(X, Y, 2000, 2.f);

    std::vector<float> test0 = from_image("data/digits/test0.png");
    std::vector<float> test1 = from_image("data/digits/test1.png");

    mt::mat Xtest0(nf, 1), Xtest1(nf, 1);
    for (int r = 0; r < nf; ++r)
    {
        Xtest0.atref(r, 0) = test0[r];
        Xtest1.atref(r, 0) = test1[r];
    }

    std::vector<float> zero = model.predict(Xtest0);
    std::vector<float> one = model.predict(Xtest1);

    printf("Zero: %f %f\nOne: %f %f\n", zero[0], zero[1], one[0], one[1]);

    return 0;
}

