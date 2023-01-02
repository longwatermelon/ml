#include "impl.h"
#include "common.h"
#include <graph2.h>

using namespace reg;

int main()
{
    SDL_Init(SDL_INIT_VIDEO);
    SDL_Window *win = SDL_CreateWindow("Softmax regression",
            SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED,
            600, 600, SDL_WINDOW_SHOWN);
    SDL_Renderer *rend = SDL_CreateRenderer(win, -1,
            SDL_RENDERER_ACCELERATED | SDL_RENDERER_PRESENTVSYNC);

    graph::Graph2 graph("data/softmax/data");
    common::add_cross_shape(graph);
    common::add_tri_shape(graph);
    common::add_square_shape(graph);

    std::array<float, 2> vw;
    vw.fill(0.f);
    float b = 0.f;

    std::vector<DataPoint<2>> data;
    for (const auto &p : graph.data())
        data.emplace_back(DataPoint<2>({ p.p.x, p.p.y }, p.shape));

    for (size_t i = 0; i < 1000; ++i)
    {
        general::descend(vw, b, .1f, data);
    }

    bool running = true;
    SDL_Event evt;

    while (running)
    {
        while (SDL_PollEvent(&evt))
        {
            switch (evt.type)
            {
            case SDL_QUIT:
                running = false;
                break;
            }
        }

        SDL_RenderClear(rend);

        graph.render(rend, { 0, 0, 600, 600 }, [vw, b](float x){
            return 0.f;
            /* return (.5f - b - vw[0] * x) / vw[1]; */
        });

        SDL_SetRenderDrawColor(rend, 0, 0, 0, 255);
        SDL_RenderPresent(rend);
    }

    SDL_DestroyRenderer(rend);
    SDL_DestroyWindow(win);
    SDL_Quit();
    return 0;
}

