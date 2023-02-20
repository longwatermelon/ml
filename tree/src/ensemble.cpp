#include "dtree.h"
#include "data.h"
#include <iostream>

#define NTREES 100

int main()
{
    srand(time(0));

    std::vector<std::unique_ptr<dtree::DTree>> ensemble(NTREES);
    for (int i = 0; i < NTREES; ++i)
    {
        std::vector<dtree::DataPoint> random_select(data.size());
        for (size_t j = 0; j < data.size(); ++j)
            random_select[j] = data[rand() % data.size()];

        ensemble[i] = dtree::create_dtree(random_select);
    }

    bool pointy_ears, round_face, whiskers;
    char ans;

    std::cout << "Pointy ears? (y/n): ";
    std::cin >> ans;
    pointy_ears = ans == 'y';

    std::cout << "Round face? (y/n): ";
    std::cin >> ans;
    round_face = ans == 'y';

    std::cout << "Whiskers? (y/n): ";
    std::cin >> ans;
    whiskers = ans == 'y';

    int yes_count = 0;
    for (auto &tree : ensemble)
    {
        if (tree->predict({ pointy_ears, round_face, whiskers }))
            ++yes_count;
    }

    printf(yes_count >= NTREES / 2 ? "Cat\n" : "Not a cat\n");

    return 0;
}
