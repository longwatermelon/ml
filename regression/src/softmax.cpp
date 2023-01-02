#include "impl.h"

using namespace reg;

int main()
{
    std::array<float, 4> vz{ 1.f, 2.f, 3.f, 4.f };
    std::array<float, 4> va = softmax::solve_va(vz);

    printf("va = [%.2f, %.2f, %.2f, %.2f]\n", va[0], va[1], va[2], va[3]);

    return 0;
}

