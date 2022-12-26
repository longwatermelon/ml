#include "graph.h"
#include <fstream>
#include <sstream>
#include <iostream>

Graph::Graph(const std::string &data_fp)
{
    std::ifstream ifs(data_fp);
    std::string buf;

    while (std::getline(ifs, buf))
    {
        std::stringstream ss(buf);
        std::string field;
        ss >> field;

        if (field == "min") ss >> m_min.x >> m_min.y;
        if (field == "max") ss >> m_max.x >> m_max.y;
        if (field == "step") ss >> m_step.x >> m_step.y;
        if (field == "data")
        {
            glm::vec2 p;
            ss >> p.x >> p.y;
            m_data.emplace_back(p);
        }
    }
}

Graph::~Graph()
{
}

void Graph::render(SDL_Renderer *rend, SDL_Rect r, const std::function<float(float)> &func) const
{
    // White bg
    SDL_SetRenderDrawColor(rend, 255, 255, 255, 255);
    SDL_RenderFillRect(rend, &r);

    // Axes
    SDL_SetRenderDrawColor(rend, 0, 0, 0, 255);
    SDL_RenderDrawLine(rend, r.x + 20, r.y + 20, r.x + 20, r.y + r.h - 20);
    SDL_RenderDrawLine(rend, r.x + 20, r.y + r.h - 20, r.x + r.w - 20, r.y + r.h - 20);

    // Ticks
    float xstep = (r.w - 40) / ((m_max.x - m_min.x) / m_step.x);
    for (float x = r.x + 20 + xstep; x <= r.x + r.w - 20; x += xstep)
        SDL_RenderDrawLine(rend, (int)x, r.y + r.h - 25, (int)x, r.y + r.h - 15);

    float ystep = (r.h - 40) / ((m_max.y - m_min.y) / m_step.y);
    for (float y = r.y + 20; y < r.y + r.h - 20; y += ystep)
        SDL_RenderDrawLine(rend, r.x + 15, (int)y, r.x + 25, (int)y);

    // Data points
    SDL_SetRenderDrawColor(rend, 255, 0, 0, 255);
    for (const auto &p : m_data)
    {
        float x = r.x + 20 + ((p.x - m_min.x) / (m_max.x - m_min.x)) * (r.w - 40);
        float y = r.y + 20 + (1.f - (p.y - m_min.y) / (m_max.y - m_min.y)) * (r.h - 40);
        SDL_RenderDrawLineF(rend, x - 3.f, y - 3.f, x + 3.f, y + 3.f);
        SDL_RenderDrawLineF(rend, x - 3.f, y + 3.f, x + 3.f, y - 3.f);
    }

    // Line
    SDL_SetRenderDrawColor(rend, 0, 0, 255, 255);
    for (int x = r.x + 20; x < r.x + r.w - 20; ++x)
    {
        float x1 = (float)(x - (r.x + 20)) / (r.w - 40) * (m_max.x - m_min.x) + m_min.x;
        float x2 = (float)(x + 1 - (r.x + 20)) / (r.w - 40) * (m_max.x - m_min.x) + m_min.x;
        float y1 = r.y + r.h - 20 - (((func(x1) - m_min.y) / (m_max.y - m_min.y) + m_min.y) * (r.h - 40));
        float y2 = r.y + r.h - 20 - (((func(x2) - m_min.y) / (m_max.y - m_min.y) + m_min.y) * (r.h - 40));
        SDL_RenderDrawLineF(rend, x, y1, x + 1, y2);
    }
    /* int b = (1.f - m_b / m_ymax) * (r.h - 40) + 20; */
    /* int y2 = b + (-m_w * m_xmax / m_ymax) * (r.h - 40); */
    /* SDL_RenderDrawLine(rend, r.x + 20, b, r.x + r.w - 20, y2); */
}

void Graph::set_line(float w, float b)
{
    m_w = w;
    m_b = b;
}

