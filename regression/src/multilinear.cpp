#include "common.h"
#include <graph2.h>

int main(int argc, char **argv)
{
    INIT_SDL("Multiple linear regression")

    bool running = true;
    SDL_Event evt;

    Graph2 graphs[4] = {
        Graph2("data-multilinear/age-graph"),
        Graph2("data-multilinear/bedrooms-graph"),
        Graph2("data-multilinear/floors-graph"),
        Graph2("data-multilinear/size-graph")
    };

    SDL_Rect rects[4] = {
        { 0, 0, 300, 300 },
        { 300, 0, 300, 300 },
        { 0, 300, 300, 300 },
        { 300, 300, 300, 300 }
    };

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

        for (int i = 0; i < 4; ++i)
            graphs[i].render(rend, rects[i], [](float x){ return 0.f; });

        SDL_SetRenderDrawColor(rend, 0, 0, 0, 255);
        SDL_RenderPresent(rend);
    }

    QUIT_SDL
    return 0;
}

