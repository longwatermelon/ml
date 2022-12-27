#pragma once
#include <string>
#include <vector>
#include <functional>
#include <SDL2/SDL.h>
#include <glm/glm.hpp>

class Graph
{
public:
    Graph(const std::string &data_fp);
    ~Graph();

    void render(SDL_Renderer *rend, SDL_Rect r, const std::function<float(float)> &func) const;

    const std::vector<glm::vec2> &data() const { return m_data; }

private:
    float gx2scr(float x, SDL_Rect r) const;
    float gy2scr(float y, SDL_Rect r) const;

private:
    glm::vec2 m_min{ 0.f }, m_max{ 0.f };
    glm::vec2 m_step{ 0.f };
    std::vector<glm::vec2> m_data;

    float m_espace{ 20.f };
};

