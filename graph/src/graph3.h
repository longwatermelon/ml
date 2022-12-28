#pragma once
#include <string>
#include <functional>
#include <glm/glm.hpp>
#include <SDL2/SDL.h>

class Graph3
{
public:
    Graph3() = default;
    Graph3(const std::string &data_fp, const std::function<float(float, float)> &func = [](float, float){ return 0.f; });
    ~Graph3();

    void render(SDL_Renderer *rend, SDL_Rect rect) const;
    void rot(glm::vec3 rotation) { m_angle += rotation; }

    void add_point(float x, float z);

    void load(const std::string &config);

private:
    void find_y_minmax();

    float gx2world(float x) const;
    float gy2world(float y) const;
    float gz2world(float z) const;

private:
    glm::vec3 m_min{ 0.f }, m_max{ 0.f }, m_step{ 0.f };
    glm::vec3 m_angle{ 0.f, 0.f, 0.f };

    std::function<float(float, float)> m_func{};
    std::vector<std::pair<int, int>> m_points;

    float m_axis_len{ 100.f };
};

