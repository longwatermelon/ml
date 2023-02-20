#include "dtree.h"
#include "data.h"

int main(int argc, char **argv)
{
    if (argc != 4)
        printf("Incorrect number of args. Usage: ./a.out [pointy ears] [round face] [whiskers]\n");

    std::unique_ptr<dtree::DTree> tree = dtree::create_dtree(data);
    printf("%s\n", tree->predict({ (bool)std::stoi(argv[1]), (bool)std::stoi(argv[2]), (bool)std::stoi(argv[3]) }) ? "Cat" : "Not a cat");

    return 0;
}
