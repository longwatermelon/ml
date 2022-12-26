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
    glm::vec2 m_min{ 0.f }, m_max{ 0.f };
    glm::vec2 m_step{ 0.f };
    std::vector<glm::vec2> m_data;

    float m_w{ 0.f }, m_b{ 0.f };
};

