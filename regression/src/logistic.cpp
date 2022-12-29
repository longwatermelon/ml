#include "impl.h"
#include <SDL2/SDL.h>
#include <graph2.h>
#include <graph3.h>

using namespace reg;

int main(int argc, char **argv)
{
    SDL_Init(SDL_INIT_VIDEO);
    SDL_Window *win = SDL_CreateWindow("Logistic regression",
            SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED,
            600, 600, SDL_WINDOW_SHOWN);
    SDL_Renderer *rend = SDL_CreateRenderer(win, -1,
            SDL_RENDERER_ACCELERATED | SDL_RENDERER_PRESENTVSYNC);

    bool running = true;
    SDL_Event evt;

    Graph2 graph("data/logistic/graph");

    std::vector<DataPoint<1>> data;
    data.reserve(graph.data().size());
    for (const auto &p : graph.data())
        data.emplace_back(DataPoint<1>({ p.x }, p.y));

    std::array<float, 1> vw = { 0.f };
    float b = 0.f;

    Graph3 graph3("data/logistic/graph3", [&](float x, float z){
        return general::cost<1>(data, [x, z](const DataPoint<1> &p){
            return logistic::loss(x, z, logistic::f_wb({ x }, p.features, z), p.y);
        });
    });

    graph3.add_point(vw[0], b);

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
                case SDLK_p:
                    printf("%fx + %f\n", vw[0], b);
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
                    graph3.rot({ evt.motion.xrel / 200.f, evt.motion.yrel / 200.f, 0.f });
                break;
            }
        }

        const Uint8 *keystates = SDL_GetKeyboardState(0);
        if (keystates[SDL_SCANCODE_SPACE])
        {
            general::descend<1>(vw, b, 1.f, data, logistic::f_wb);
            graph3.add_point(vw[0], b);
        }

        SDL_RenderClear(rend);

        graph.render(rend, { 0, 0, 600, 300 }, [vw, b](float x){
            return logistic::f_wb(vw, { x }, b);
        });

        graph3.render(rend, { 0, 300, 600, 300 });

        SDL_SetRenderDrawColor(rend, 0, 0, 0, 255);
        SDL_RenderPresent(rend);
    }

    SDL_DestroyRenderer(rend);
    SDL_DestroyWindow(win);
    SDL_Quit();
    return 0;
}

