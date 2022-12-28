#include "impl.h"
#include <fstream>
#include <vector>
#include <sstream>

#define NF 4

int main(int argc, char **argv)
{
    // Read data
    std::vector<DataPoint<NF>> data;

    std::ifstream ifs("data/multilinear/data");
    std::string buf;

    while (std::getline(ifs, buf))
    {
        DataPoint<NF> p;
        std::stringstream ss(buf);

        for (size_t i = 0; i < NF; ++i)
            ss >> p.features[i];

        ss >> p.y;
        data.emplace_back(p);
    }

    ifs.close();

    // Scale
    std::array<float, NF> vsd, vmean;
    general::feature_scale(data, vsd, vmean);

    // Gradient descent
    std::array<float, NF> vw;
    vw.fill(0.f);
    float b = 0.f;

    for (int i = 0; i < 500; ++i)
    {
        if ((i + 1) % 100 == 0)
            printf("Iteration %d: w = [%f, %f, %f, %f], b = %f\n",
                    i + 1, vw[0], vw[1], vw[2], vw[3], b);
        multilinear::descend(vw, b, .1f, data);
    }

    // Predict
    float input[NF] = {
        1200, // size (sqft)
        3, // bedrooms
        1, // floors
        40 // age
    };

    // z-score normalization on input
    float prediction = b;
    for (int i = 0; i < NF; ++i)
    {
        input[i] = (input[i] - vmean[i]) / vsd[i];
        prediction += input[i] * vw[i];
    }

    printf("Price prediction of house with size = 1200, bedrooms = 3, floors = 1, age = 40: $%.2f\n", prediction * 1000.f);

    return 0;
}

