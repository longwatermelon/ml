#include "common.h"
#include <iostream>
#include <graph.h>
#include <graph3.h>
#include <SDL2/SDL.h>
#include <glm/glm.hpp>

float cost(float w, float b, const std::vector<glm::vec2> &data)
{
    float sum = 0.f;
    for (const auto &p : data)
        sum += std::pow(w * p.x + b - p.y, 2);
    return (sum * (1.f / (2.f * data.size())));
}

void descend(float &w, float &b, const std::vector<glm::vec2> &data)
{
    float dw_j = 0.f,
          db_j = 0.f;
    for (const auto &p : data)
    {
        dw_j += (w * p.x + b - p.y) * p.x;
        db_j += w * p.x + b - p.y;
    }

    dw_j *= 1.f / data.size();
    db_j *= 1.f / data.size();

    float a = .0001f;
    w = w - a * dw_j;
    b = b - db_j;
    /* printf("%f %f %f %f %f\n", w, b, dw_j, db_j, cost(w, b, data)); */
}

int main()
{
    SDL_Window *win;
    SDL_Renderer *rend;
    common::init_sdl(&win, &rend);

    bool running = true;
    SDL_Event evt;

    Graph g("graph");
    float w = 0.f,
          b = 0.f;
    g.set_line(w, b);

    Graph3 g3("graph3");

    bool mouse_down = false;

    while (running)
    {
        while (SDL_PollEvent(&evt))
        {
            switch (evt.type)
            {
            case SDL_QUIT:
                running = false;
                break;
            case SDL_KEYDOWN:
                switch (evt.key.keysym.sym)
                {
                case SDLK_SPACE:
                    descend(w, b, g.data());
                    g.set_line(w, b);
                    break;
                }
                break;
            case SDL_MOUSEBUTTONDOWN:
                mouse_down = true;
                break;
            case SDL_MOUSEBUTTONUP:
                mouse_down = false;
                break;
            case SDL_MOUSEMOTION:
                if (mouse_down)
                    g3.rot({ evt.motion.xrel / 200.f, evt.motion.yrel / 200.f, 0.f });
                break;
            }
        }

        SDL_RenderClear(rend);

        g.render(rend, { 0, 0, 600, 300 });
        g3.render(rend, { 0, 300, 600, 300 }, [&](float x, float z){
            float sum = 0.f;
            for (const auto &p : g.data())
                sum += std::pow((x - 3.f) * p.x + z - p.y, 2);
            return (sum * (1.f / (2.f * g.data().size())));
        });

        SDL_SetRenderDrawColor(rend, 0, 0, 0, 255);
        SDL_RenderPresent(rend);
    }

    common::quit_sdl(win, rend);
    return 0;
}

