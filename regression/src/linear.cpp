#include "reg.h"
#include "common.h"
#include <iostream>
#include <graph2.h>
#include <graph3.h>
#include <SDL2/SDL.h>
#include <glm/glm.hpp>

using namespace reg;

int main(int argc, char **argv)
{
    SDL_Init(SDL_INIT_VIDEO);
    SDL_Window *win = SDL_CreateWindow("Linear regression",
            SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED,
            600, 600, SDL_WINDOW_SHOWN);
    SDL_Renderer *rend = SDL_CreateRenderer(win, -1,
            SDL_RENDERER_ACCELERATED | SDL_RENDERER_PRESENTVSYNC);

    bool running = true;
    SDL_Event evt;

    graph::Graph2 g("data/linear/graph");
    g.add_cross_shape();

    // 1 feature data
    std::vector<DataPoint> data;
    for (const auto &p : g.data())
        data.emplace_back(DataPoint(Eigen::VectorXf({{ p.p.x }}), p.p.y));

    Eigen::VectorXf vw({{ -500.f }});
    float b = -200.f;

    graph::Graph3 g3("data/linear/graph3", [&](float x, float z){
        return general::cost(data, vw, [x, z](const DataPoint &p){
            return std::pow(linear::f_wb(x, p.features[0], z) - p.y, 2);
        });
    });

    g3.add_history(vw[0], b);

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

        const Uint8 *keystates = SDL_GetKeyboardState(0);
        if (keystates[SDL_SCANCODE_SPACE])
        {
            general::descend(vw, b, .2f, data,
                    [](const Eigen::VectorXf &vw,
                       const Eigen::VectorXf &vx,
                       float b){
                return linear::f_wb(vw[0], vx[0], b);
            });
            g3.add_history(vw[0], b);
        }

        SDL_RenderClear(rend);

        g.render(rend, { 0, 0, 600, 300 });
        g.render_line(rend, { 0, 0, 600, 300 }, [vw, b](float x){ return vw[0] * x + b; }, { 0.f, 0.f, 1.f });
        g3.render(rend, { 0, 300, 600, 300 });

        SDL_SetRenderDrawColor(rend, 0, 0, 0, 255);
        SDL_RenderPresent(rend);
    }

    SDL_DestroyRenderer(rend);
    SDL_DestroyWindow(win);
    SDL_Quit();
    return 0;
}

