#include "graph.h"
#include <fstream>
#include <sstream>

Graph::Graph(const std::string &data_fp)
{
    std::ifstream ifs(data_fp);
    std::string buf;

    while (std::getline(ifs, buf))
    {
    }
}

Graph::~Graph()
{
}

void Graph::render(SDL_Renderer *rend, SDL_Rect r)
{
}

