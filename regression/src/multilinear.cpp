#include "common.h"
#include "graph.h"

int main(int argc, char **argv)
{
    INIT_SDL("Multiple linear regression")

    bool running = true;
    SDL_Event evt;

    Graph graphs[4] = {
        Graph("data-multilinear/age-graph"),
        Graph("data-multilinear/bedrooms-graph"),
        Graph("data-multilinear/floors-graph"),
        Graph("data-multilinear/size-graph")
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

