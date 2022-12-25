#pragma once
#include <string>
#include <functional>
#include <glm/glm.hpp>
#include <SDL2/SDL.h>

class Graph3
{
public:
    Graph3(const std::string &data_fp);
    ~Graph3();

    void render(SDL_Renderer *rend, SDL_Rect rect, const std::function<float(float, float)> &func);
    void rot(glm::vec3 rotation) { m_angle += rotation; }

private:
    glm::vec3 m_max{ 0.f }, m_step{ 0.f };
    glm::vec3 m_angle{ 0.f };
};

