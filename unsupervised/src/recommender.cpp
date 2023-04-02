#include "reg.h"
#include "unsupervised.h"

int main()
{
    std::vector<reg::DataPoint> mx = {
        // romance | action | horror
        reg::DataPoint({ 1.f, .1f, 0.f }, 1.f), // romance movie
        reg::DataPoint({ 0.f, 1.f, 0.f }, 0.f), // action movie
        reg::DataPoint({ .5f, 0.f, 0.f }, .5f), // comedy movie
        reg::DataPoint({ 0.f, 1.f, 1.f }, 0.f)  // horror movie
    };

    std::vector<float> vw(3);
    float b = 0.f;

    for (std::size_t i = 0; i < 1000; ++i)
    {
        reg::general::descend(vw, b, .1f, mx,
            [](const std::vector<float> &vw,
                const std::vector<float> &vx,
                float b){
                return common::vec::dot(vw, vx) + b;
            }
        );

        if ((i + 1) % 100 == 0)
            printf("Iteration %zu | vw = %s\n", i + 1, common::vec::to_string(vw).c_str());
    }

    std::vector<float> romance = { 1.f, 0.f, 0.f };
    std::vector<float> action = { 0.f, 1.f, 0.f };
    std::vector<float> comedy = { 0.f, 0.f, 0.f };
    std::vector<float> horror = { .3f, 1.f, 1.f };
    printf("User ratings:\nRomance %.1f stars\nAction %.1f stars\nComedy %.1f stars\nHorror %.1f stars\n",
            (common::vec::dot(vw, romance) + b) * 5.f,
            (common::vec::dot(vw, action) + b) * 5.f,
            (common::vec::dot(vw, comedy) + b) * 5.f,
            (common::vec::dot(vw, horror) + b) * 5.f);

    return 0;
}
