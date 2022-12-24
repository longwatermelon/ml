#pragma once
#include <string>
#include <SDL2/SDL.h>

class Graph
{
public:
    Graph(const std::string &data_fp);
    ~Graph();

    void render(SDL_Renderer *rend, SDL_Rect r);

private:
    std::string m_xlabel, m_ylabel;
    int m_xmax{ -1 }, m_ymax{ -1 };
};

