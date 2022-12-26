#include "common.h"

void common::init_sdl(SDL_Window **w, SDL_Renderer **r)
{
    SDL_Init(SDL_INIT_VIDEO);
    *w = SDL_CreateWindow("Window",
        SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED,
        600, 600, SDL_WINDOW_SHOWN);
    *r = SDL_CreateRenderer(*w, -1, SDL_RENDERER_ACCELERATED | SDL_RENDERER_PRESENTVSYNC);
}

void common::quit_sdl(SDL_Window *w, SDL_Renderer *r)
{
    SDL_DestroyRenderer(r);
    SDL_DestroyWindow(w);
    SDL_Quit();
}

