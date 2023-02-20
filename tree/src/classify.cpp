#include "dtree.h"

int main(int argc, char **argv)
{
    if (argc != 4)
        printf("Incorrect number of args. Usage: ./a.out [pointy ears] [round face] [whiskers]\n");

    std::vector<dtree::DataPoint> data(10);
    // pointy ears, round face, whiskers | cat
    data[0] = dtree::DataPoint({ 1, 1, 1 }, 1);
    data[1] = dtree::DataPoint({ 0, 0, 1 }, 1);
    data[2] = dtree::DataPoint({ 0, 1, 0 }, 0);
    data[3] = dtree::DataPoint({ 1, 0, 1 }, 0);
    data[4] = dtree::DataPoint({ 1, 1, 1 }, 1);
    data[5] = dtree::DataPoint({ 1, 1, 0 }, 1);
    data[6] = dtree::DataPoint({ 0, 0, 0 }, 0);
    data[7] = dtree::DataPoint({ 1, 1, 0 }, 1);
    data[8] = dtree::DataPoint({ 0, 1, 0 }, 0);
    data[9] = dtree::DataPoint({ 0, 1, 0 }, 0);

    std::unique_ptr<dtree::DTree> tree = dtree::create_dtree(data);
    printf("%d\n", tree->predict({ (bool)std::stoi(argv[1]), (bool)std::stoi(argv[2]), (bool)std::stoi(argv[3]) }));

    return 0;
}
