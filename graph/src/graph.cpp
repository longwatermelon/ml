#include "graph.h"
#include <fstream>
#include <sstream>

Graph::Graph(const std::string &data_fp)
{
    std::ifstream ifs(data_fp);
    std::string buf;

    while (std::getline(ifs, buf))
    {
        std::stringstream ss(buf);
        std::string field;
        ss >> field;

        if (field == "xlabel") ss >> m_xlabel;
        if (field == "ylabel") ss >> m_ylabel;
        if (field == "xmax") ss >> m_xmax;
        if (field == "ymax") ss >> m_ymax;
        if (field == "xstep") ss >> m_xstep;
        if (field == "ystep") ss >> m_ystep;
        if (field == "data")
        {
            SDL_Point p;
            ss >> p.x >> p.y;
            m_data.emplace_back(p);
        }
    }
}

Graph::~Graph()
{
}

void Graph::render(SDL_Renderer *rend, SDL_Rect r)
{
    // White bg
    SDL_SetRenderDrawColor(rend, 255, 255, 255, 255);
    SDL_RenderFillRect(rend, &r);

    // Axes
    SDL_SetRenderDrawColor(rend, 0, 0, 0, 255);
    SDL_RenderDrawLine(rend, r.x + 20, r.y + 20, r.x + 20, r.y + r.h - 20);
    SDL_RenderDrawLine(rend, r.x + 20, r.y + r.h - 20, r.x + r.w - 20, r.y + r.h - 20);
}

