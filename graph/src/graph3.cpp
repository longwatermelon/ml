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
    float min = rect.w < rect.h ? rect.w : rect.h;
    float max = rect.w < rect.h ? rect.h : rect.w;

    glm::ivec2 proj = {
        rect.x + ((p.x / p.z) + .5f) * min,
        rect.y + ((-p.y / p.z) + .5f) * min
    };

    if (rect.w > rect.h)
        proj.x += (max - rect.h) / 2;
    else
        proj.y += (max - rect.w) / 2;

    return proj;
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
    float axis_len = 100.f;
    glm::vec3 orig = { 0.f, 0.f, 200.f };
    glm::vec3 graph_orig = orig - glm::vec3(axis_len / 2.f);
    glm::ivec2 oproj, xproj, yproj, zproj;
    {
        glm::vec3 x = graph_orig,
                  y = graph_orig,
                  z = graph_orig;
        x.x += axis_len;
        y.y += axis_len;
        z.z += axis_len;

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
            float y = (func(x, z) / m_max.y) * axis_len;
            SDL_SetRenderDrawColor(rend, y > axis_len ? 0 : (1.f - y / axis_len) * 255.f, 0, 0, 255);
            glm::vec3 p((x / m_max.x) * axis_len, y, (z / m_max.z) * axis_len);
            p = graph_orig + rotate(p, glm::vec3{ 0.f }, m_angle);
            glm::ivec2 proj = project(p, rect);
            SDL_RenderDrawPoint(rend, proj.x, proj.y);
        }
    }
}

