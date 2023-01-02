#include "impl.h"
#include "common.h"
#include <graph2.h>

using namespace reg;

int main()
{
    SDL_Init(SDL_INIT_VIDEO);
    SDL_Window *win = SDL_CreateWindow("Multiple feature logistic regression",
            SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED,
            600, 600, SDL_WINDOW_SHOWN);
    SDL_Renderer *rend = SDL_CreateRenderer(win, -1,
            SDL_RENDERER_ACCELERATED | SDL_RENDERER_PRESENTVSYNC);

    graph::Graph2 graph("data/multilogistic/data");
    common::add_cross_shape(graph);
    common::add_tri_shape(graph);

    std::array<float, 2> vw;
    vw.fill(0.f);
    float b = 0.f;

    std::vector<DataPoint<2>> data;
    for (const auto &p : graph.data())
        data.emplace_back(DataPoint<2>({ p.p.x, p.p.y }, p.shape));

    for (size_t i = 0; i < 10000; ++i)
    {
        general::descend<2>(vw, b, .1f, data,
                [](const std::array<float, 2> &vw,
                   const std::array<float, 2> &vx,
                   float b){
            return multilogistic::g(multilogistic::f_wb(vw, vx, b));
        });
        if ((i + 1) % 1000 == 0)
            printf("Iteration %zu: w = [%f, %f], b = %f, cost = %f\n",
                    i + 1, vw[0], vw[1], b,
                    2.f * general::cost<2>(data, vw, [vw, b](const DataPoint<2> &dp){
                        return logistic::loss(multilogistic::f_wb(vw, dp.features, b), dp.y);
                    }
            ));
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

        graph.render(rend, { 0, 0, 600, 600 });

        graph.render_line(rend, { 0, 0, 600, 600 }, [vw, b](float x){
            return (.5f - b - vw[0] * x) / vw[1];
        }, { 0.f, 0.f, 1.f });

        SDL_SetRenderDrawColor(rend, 0, 0, 0, 255);
        SDL_RenderPresent(rend);
    }

    SDL_DestroyRenderer(rend);
    SDL_DestroyWindow(win);
    SDL_Quit();
    return 0;
}

