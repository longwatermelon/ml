#include "common.h"

int main()
{
    SDL_Window *win;
    SDL_Renderer *rend;
    common::init_sdl(&win, &rend);

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

        SDL_SetRenderDrawColor(rend, 0, 0, 0, 255);
        SDL_RenderPresent(rend);
    }

    common::quit_sdl(win, rend);
    return 0;
}

