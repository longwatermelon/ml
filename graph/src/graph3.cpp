#include "graph3.h"
#include <fstream>
#include <sstream>

Graph3::Graph3(const std::string &data_fp)
{
    std::ifstream ifs(data_fp);
    std::string buf;

    while (std::getline(ifs, buf))
    {
        std::stringstream ss(buf);
        std::string field;
        ss >> field;

        if (field == "max") ss >> m_max.x >> m_max.y >> m_max.z;
        if (field == "step") ss >> m_step.x >> m_step.y >> m_step.z;
    }
}

Graph3::~Graph3()
{
}

static glm::ivec2 project(glm::vec3 p, SDL_Rect rect)
{
    return {
        ((p.x / p.z) + .5f) * 600.f,
        ((-p.y / p.z) + .5f) * 600.f
    };
}

static glm::vec3 rotate(glm::vec3 p, glm::vec3 orig, glm::vec3 angle)
{
    glm::mat3 rotx = {
        1, 0, 0,
        0, cosf(angle.y), -sinf(angle.y),
        0, sinf(angle.y), cosf(angle.y)
    };

    glm::mat3 roty =  {
        cosf(angle.x), 0, sinf(angle.x),
        0, 1, 0,
        -sinf(angle.x), 0, cosf(angle.x)
    };

    return orig + (rotx * roty * (p - orig));
}

void Graph3::render(SDL_Renderer *rend, SDL_Rect rect, const std::function<float(float, float)> &func)
{
    // Black bg
    SDL_SetRenderDrawColor(rend, 0, 0, 0, 255);
    SDL_RenderFillRect(rend, &rect);

    // Draw axes
    glm::ivec2 oproj, xproj, yproj, zproj;
    {
        glm::vec3 orig = { 0.f, 0.f, 200.f };
        glm::vec3 graph_orig = glm::vec3(0.f, 0.f, 200.f) + glm::vec3(-m_max.x / 2.f, -m_max.y / 2.f, m_max.z / 2.f);
        glm::vec3 x = graph_orig,
                  y = graph_orig,
                  z = graph_orig;
        x.x += m_max.x;
        y.y += m_max.y;
        z.z -= m_max.z;

        graph_orig = rotate(graph_orig, orig, m_angle);
        x = rotate(x, orig, m_angle);
        y = rotate(y, orig, m_angle);
        z = rotate(z, orig, m_angle);

        oproj = project(graph_orig, rect);
        xproj = project(x, rect);
        yproj = project(y, rect);
        zproj = project(z, rect);
    }

    SDL_SetRenderDrawColor(rend, 255, 255, 255, 255);
    SDL_RenderDrawLine(rend, oproj.x, oproj.y, xproj.x, xproj.y);
    SDL_RenderDrawLine(rend, oproj.x, oproj.y, yproj.x, yproj.y);
    SDL_RenderDrawLine(rend, oproj.x, oproj.y, zproj.x, zproj.y);

    // Draw points
    for (float x = 0.f; x < m_max.x; x += m_step.x)
    {
        for (float z = 0.f; z < m_max.z; z += m_step.z)
        {
        }
    }
}

