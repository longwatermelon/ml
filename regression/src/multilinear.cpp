#include "reg.h"
#include <fstream>
#include <vector>
#include <sstream>

#define NF 4
using namespace reg;

int main(int argc, char **argv)
{
    // Read data
    // 4 feature data points
    std::vector<DataPoint> data;

    std::ifstream ifs("data/multilinear/data");
    std::string buf;

    while (std::getline(ifs, buf))
    {
        DataPoint p(NF);
        std::stringstream ss(buf);

        for (size_t i = 0; i < NF; ++i)
            ss >> p.features[i];

        ss >> p.y;
        data.emplace_back(p);
    }

    ifs.close();

    // Scale
    std::vector<float> vsd(NF), vmean(NF);
    general::feature_scale(data, vsd, vmean);

    // Gradient descent
    std::vector<float> vw(NF);
    float b = 0.f;

    for (int i = 0; i < 500; ++i)
    {
        if ((i + 1) % 100 == 0)
            printf("Iteration %d: w = [%f, %f, %f, %f], b = %f\n",
                    i + 1, vw[0], vw[1], vw[2], vw[3], b);
        general::descend(vw, b, .1f, data, multilinear::f_wb);
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

