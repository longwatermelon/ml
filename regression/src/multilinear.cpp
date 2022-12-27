#include "multilinear.h"
#include "linear.h"
#include "common.h"
#include <fstream>
#include <graph2.h>

int main(int argc, char **argv)
{
    INIT_SDL("Multiple linear regression")

    bool running = true;
    SDL_Event evt;

    std::array<std::string, 4> data_paths = {
        "data-multilinear/age-graph",
        "data-multilinear/bedrooms-graph",
        "data-multilinear/floors-graph",
        "data-multilinear/size-graph"
    };

    std::array<Graph2, 4> graphs;
    for (int i = 0; i < 4; ++i)
        graphs[i] = Graph2(data_paths[i]);

    std::array<SDL_Rect, 4> rects = {
        SDL_Rect{ 0, 0, 300, 300 },
        { 300, 0, 300, 300 },
        { 0, 300, 300, 300 },
        { 300, 300, 300, 300 }
    };

    std::array<std::pair<float, float>, 4> wb;
    wb.fill({ 0.f, 0.f });

    std::array<Graph2, 4> scaled_graphs = graphs;

    for (int i = 0; i < 4; ++i)
        multilinear::feature_scale(scaled_graphs[i], data_paths[i] + "-scaled");

    bool flag = false;
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
                case SDLK_s:
                    flag = !flag;
                    break;
                }
            }
        }

        const Uint8 *keystates = SDL_GetKeyboardState(0);
        if (keystates[SDL_SCANCODE_SPACE])
        {
            flag = true;
            for (int i = 0; i < 4; ++i)
                linear::descend(wb[i].first, wb[i].second, .1f, scaled_graphs[i].data());
        }

        SDL_RenderClear(rend);

        for (int i = 0; i < 4; ++i)
        {
            if (flag)
                scaled_graphs[i].render(rend, rects[i],
                    [wb, i](float x){ return wb[i].first * x + wb[i].second; });
            else
                graphs[i].render(rend, rects[i], [](float x){ return 0.f; });
        }

        SDL_SetRenderDrawColor(rend, 0, 0, 0, 255);
        SDL_RenderPresent(rend);
    }

    QUIT_SDL
    return 0;
}

