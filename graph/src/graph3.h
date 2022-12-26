#pragma once
#include <string>
#include <functional>
#include <glm/glm.hpp>
#include <SDL2/SDL.h>

class Graph3
{
public:
    Graph3(const std::string &data_fp, const std::function<float(float, float)> &func = [](float, float){ return 0.f; });
    ~Graph3();

    void render(SDL_Renderer *rend, SDL_Rect rect) const;
    void rot(glm::vec3 rotation) { m_angle += rotation; }

    void set_point(float x, float z);

private:
    void find_y_minmax();

private:
    glm::vec3 m_min{ 0.f }, m_max{ 0.f }, m_step{ 0.f };
    glm::vec3 m_angle{ 0.f, 0.f, 0.f };

    std::function<float(float, float)> m_func{};
    float m_px{ 0.f }, m_pz{ 0.f };
};

