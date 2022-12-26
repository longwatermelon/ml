#pragma once
#include <SDL2/SDL.h>

#define INIT_SDL(title) \
    SDL_Init(SDL_INIT_VIDEO); \
    SDL_Window *win = SDL_CreateWindow(title, \
        SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED, \
        600, 600, SDL_WINDOW_SHOWN); \
    SDL_Renderer *rend = SDL_CreateRenderer(win, -1, SDL_RENDERER_ACCELERATED | SDL_RENDERER_PRESENTVSYNC);

#define QUIT_SDL \
    SDL_DestroyRenderer(rend); \
    SDL_DestroyWindow(win); \
    SDL_Quit();

