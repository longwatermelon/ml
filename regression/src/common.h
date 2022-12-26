#pragma once
#include <SDL2/SDL.h>

namespace common
{
    void init_sdl(SDL_Window **w, SDL_Renderer **r);
    void quit_sdl(SDL_Window *w, SDL_Renderer *r);
}

