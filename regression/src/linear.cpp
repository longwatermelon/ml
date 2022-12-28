#include "impl.h"
#include <iostream>
#include <graph2.h>
#include <graph3.h>
#include <SDL2/SDL.h>
#include <glm/glm.hpp>

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

    Graph2 g("data/linear/graph");
    float w = -500.f,
          b = -200.f;

    Graph3 g3("data/linear/graph3", [&](float x, float z){
        return general::cost(x, z, g.data(), [x, z](glm::vec2 datap){
            return std::pow(linear::f_wb(x, z, datap.x) - datap.y, 2);
        });
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
            linear::descend(w, b, .2f, g.data());
            g3.add_point(w, b);
        }

        SDL_RenderClear(rend);

        g.render(rend, { 0, 0, 600, 300 }, [w, b](float x){ return w * x + b; });
        g3.render(rend, { 0, 300, 600, 300 });

        SDL_SetRenderDrawColor(rend, 0, 0, 0, 255);
        SDL_RenderPresent(rend);
    }

    SDL_DestroyRenderer(rend);
    SDL_DestroyWindow(win);
    SDL_Quit();
    return 0;
}

