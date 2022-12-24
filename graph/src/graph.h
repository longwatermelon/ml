#pragma once
#include <string>
#include <vector>
#include <SDL2/SDL.h>

class Graph
{
public:
    Graph(const std::string &data_fp);
    ~Graph();

    void render(SDL_Renderer *rend, SDL_Rect r);
    void set_line(float w, float b);

private:
    std::string m_xlabel, m_ylabel;
    float m_xmax{ -1.f }, m_ymax{ -1.f };
    float m_xstep{ -1.f }, m_ystep{ -1.f };
    std::vector<SDL_FPoint> m_data;

    float m_w{ 0.f }, m_b{ 0.f };
};

