#pragma once
#include <string>
#include <vector>
#include <functional>
#include <SDL2/SDL.h>
#include <glm/glm.hpp>

namespace graph
{
    class Graph2
    {
    public:
        Graph2() = default;
        Graph2(const std::string &data_fp);
        ~Graph2();

        void render(SDL_Renderer *rend, SDL_Rect r, const std::function<float(float)> &func) const;

        void load(const std::string &config);

        const std::vector<glm::vec2> &data() const { return m_data; }
        glm::vec2 min() const { return m_min; }
        glm::vec2 max() const { return m_max; }
        glm::vec2 step() const { return m_step; }

    private:
        float gx2scr(float x, SDL_Rect r) const;
        float gy2scr(float y, SDL_Rect r) const;

    private:
        glm::vec2 m_min{ 0.f }, m_max{ 0.f };
        glm::vec2 m_step{ 0.f };
        std::vector<glm::vec2> m_data;

        float m_espace{ 20.f };
    };
}

