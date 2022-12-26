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

    void set_line(float w, float b);

    const std::vector<glm::vec2> &data() const { return m_data; }

private:
    float m_xmax{ -1.f }, m_ymax{ -1.f };
    float m_xstep{ -1.f }, m_ystep{ -1.f };
    std::vector<glm::vec2> m_data;

    float m_w{ 0.f }, m_b{ 0.f };
};

