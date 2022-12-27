#include "graph3.h"
#include <fstream>
#include <sstream>

Graph3::Graph3(const std::string &data_fp, const std::function<float(float, float)> &func)
{
    std::ifstream ifs(data_fp);
    std::string buf;

    while (std::getline(ifs, buf))
    {
        std::stringstream ss(buf);
        std::string field;
        ss >> field;

        if (field == "min") ss >> m_min.x >> m_min.z;
        if (field == "max") ss >> m_max.x >> m_max.z;
        if (field == "step") ss >> m_step.x >> m_step.z;
    }

    m_func = func;
    find_y_minmax();
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

void Graph3::render(SDL_Renderer *rend, SDL_Rect rect) const
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
    for (float x = m_min.x; x < m_max.x; x += m_step.x)
    {
        for (float z = m_min.z; z < m_max.z; z += m_step.z)
        {
            float y = ((m_func(x, z) - m_min.y) / (m_max.y - m_min.y)) * axis_len;
            SDL_SetRenderDrawColor(rend, y > axis_len ? 0 : (1.f - y / axis_len) * 255.f, 0, 0, 255);
            glm::vec3 p(((x - m_min.x) / (m_max.x - m_min.x)) * axis_len, y, ((z - m_min.z) / (m_max.z - m_min.z)) * axis_len);
            p = graph_orig + rotate(p, glm::vec3{ 0.f }, m_angle);
            glm::ivec2 proj = project(p, rect);
            SDL_RenderDrawPoint(rend, proj.x, proj.y);
        }
    }

    // Draw point history
    SDL_SetRenderDrawColor(rend, 0, 255, 0, 255);
    for (size_t i = 0; i < m_points.size(); ++i)
    {
        float x = m_points[i].first;
        float z = m_points[i].second;

        float ax = (x - m_min.x) / (m_max.x - m_min.x) * axis_len;
        float az = (z - m_min.z) / (m_max.z - m_min.z) * axis_len;

        glm::vec2 pbot = project(graph_orig + rotate({ ax, 0.f, az }, glm::vec3{ 0.f }, m_angle), rect);
        if (i == 0 || i == m_points.size() - 1)
        {
            glm::vec2 ptop = project(graph_orig + rotate({ ax, axis_len, az }, glm::vec3{ 0.f }, m_angle), rect);
            SDL_RenderDrawLine(rend, ptop.x, ptop.y, pbot.x, pbot.y);
        }

        if (i != m_points.size() - 1)
        {
            float nax = (m_points[i + 1].first - m_min.x) / (m_max.x - m_min.x) * axis_len;
            float naz = (m_points[i + 1].second - m_min.z) / (m_max.z - m_min.z) * axis_len;
            glm::vec2 npbot = project(graph_orig + rotate({ nax, 0.f, naz }, glm::vec3{ 0.f }, m_angle), rect);
            SDL_RenderDrawLine(rend, pbot.x, pbot.y, npbot.x, npbot.y);
        }
    }
}

void Graph3::add_point(float x, float z)
{
    m_points.emplace_back(x, z);
}

void Graph3::find_y_minmax()
{
    m_min.y = std::numeric_limits<float>::max();
    m_max.y = std::numeric_limits<float>::min();

    for (float x = m_min.x; x < m_max.x; x += m_step.x)
    {
        for (float z = m_min.z; z < m_max.z; z += m_step.z)
        {
            float y = m_func(x, z);
            if (y < m_min.y) m_min.y = y;
            if (y > m_max.y) m_max.y = y;
        }
    }
}


