#include "linear.h"
#include "common.h"
#include <iostream>
#include <graph2.h>
#include <graph3.h>
#include <SDL2/SDL.h>
#include <glm/glm.hpp>

void linear::descend(float &w, float &b, float a, const std::vector<glm::vec2> &data)
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

    w = w - a * dw_j;
    b = b - a * db_j;
}

int main(int argc, char **argv)
{
    INIT_SDL("Linear regression")

    bool running = true;
    SDL_Event evt;

    Graph2 g("data-linear/graph");
    float w = -500.f,
          b = -200.f;

    Graph3 g3("data-linear/graph3", [&](float x, float z){
        float sum = 0.f;
        for (const auto &p : g.data())
            sum += std::pow(x * p.x + z - p.y, 2);
        return (sum * (1.f / (2.f * g.data().size())));
    });

    g3.add_point(w, b);

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
                    linear::descend(w, b, .2f, g.data());
                    g3.add_point(w, b);
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

        g.render(rend, { 0, 0, 600, 300 }, [w, b](float x){ return w * x + b; });
        g3.render(rend, { 0, 300, 600, 300 });

        SDL_SetRenderDrawColor(rend, 0, 0, 0, 255);
        SDL_RenderPresent(rend);
    }

    QUIT_SDL
    return 0;
}

