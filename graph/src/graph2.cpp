#include "graph2.h"
#include <fstream>
#include <sstream>
#include <iostream>

graph::Graph2::Graph2(const std::string &data_fp)
{
    std::ifstream ifs(data_fp);
    std::stringstream ss;
    std::string buf;

    while (std::getline(ifs, buf))
        ss << buf << "\n";

    load(ss.str());
}

graph::Graph2::~Graph2()
{
}

void graph::Graph2::render(SDL_Renderer *rend, SDL_Rect r, const std::function<float(float)> &func) const
{
    // White bg
    SDL_SetRenderDrawColor(rend, 255, 255, 255, 255);
    SDL_RenderFillRect(rend, &r);

    // Axes
    SDL_SetRenderDrawColor(rend, 0, 0, 0, 255);
    SDL_RenderDrawLine(rend, r.x + m_espace, r.y + m_espace, r.x + m_espace, r.y + r.h - m_espace);
    SDL_RenderDrawLine(rend, r.x + m_espace, r.y + r.h - m_espace, r.x + r.w - m_espace, r.y + r.h - m_espace);

    // Ticks
    float xstep = (r.w - (m_espace * 2.f)) / ((m_max.x - m_min.x) / m_step.x);
    for (float x = r.x + m_espace + xstep; x <= r.x + r.w - m_espace; x += xstep)
        SDL_RenderDrawLine(rend, (int)x, r.y + r.h - m_espace - 5, (int)x, r.y + r.h - m_espace + 5);

    float ystep = (r.h - (m_espace * 2.f)) / ((m_max.y - m_min.y) / m_step.y);
    for (float y = r.y + m_espace; y < r.y + r.h - m_espace; y += ystep)
        SDL_RenderDrawLine(rend, r.x + m_espace - 5, (int)y, r.x + m_espace + 5, (int)y);

    // Data points
    SDL_SetRenderDrawColor(rend, 255, 0, 0, 255);
    for (const auto &p : m_data)
    {
        float x = gx2scr(p.x, r);
        float y = r.y + (r.h - (gy2scr(p.y, r) - r.y));
        SDL_RenderDrawLineF(rend, x - 3.f, y - 3.f, x + 3.f, y + 3.f);
        SDL_RenderDrawLineF(rend, x - 3.f, y + 3.f, x + 3.f, y - 3.f);
    }

    // Line
    SDL_SetRenderDrawColor(rend, 0, 0, 255, 255);
    for (int x = r.x + m_espace; x < r.x + r.w - m_espace; ++x)
    {
        float x1 = (float)(x - (r.x + m_espace)) / (r.w - (m_espace * 2.f)) * (m_max.x - m_min.x) + m_min.x;
        float x2 = (float)(x + 1 - (r.x + m_espace)) / (r.w - (m_espace * 2.f)) * (m_max.x - m_min.x) + m_min.x;
        float y1 = r.y + (r.h - (gy2scr(func(x1), r) - r.y));
        float y2 = r.y + (r.h - (gy2scr(func(x2), r) - r.y));
        SDL_RenderDrawLineF(rend, x, y1, x + 1, y2);
    }
}

void graph::Graph2::load(const std::string &config)
{
    m_data.clear();
    std::istringstream ss(config);
    std::string buf;

    while (std::getline(ss, buf))
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

float graph::Graph2::gx2scr(float x, SDL_Rect r) const
{
    return (x - m_min.x) / (m_max.x - m_min.x) * (r.w - (m_espace * 2.f)) + r.x + m_espace;
}

float graph::Graph2::gy2scr(float y, SDL_Rect r) const
{
    return (y - m_min.y) / (m_max.y - m_min.y) * (r.h - (m_espace * 2.f)) + r.y + m_espace;
}

