#include <iostream>
#include <graph.h>
#include <SDL2/SDL.h>

void descend(float &w, float &b, const std::vector<SDL_FPoint> &data)
{
    float dw_j = 0.f;
    for (const auto &p : data)
        dw_j += (w * p.x + b - p.y) * p.x;
    dw_j *= 1.f / data.size();

    float db_j = 0.f;
    for (const auto &p : data)
        db_j += w * p.x + b - p.y;
    db_j *= 1.f / data.size();

    float a = .0001f;
    w = w - a * dw_j;
    b = b - a * db_j;
}

int main()
{
    SDL_Init(SDL_INIT_VIDEO);
    SDL_Window *win = SDL_CreateWindow("Linear regression",
        SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED,
        600, 600,
        SDL_WINDOW_SHOWN);
    SDL_Renderer *rend = SDL_CreateRenderer(win, -1,
        SDL_RENDERER_ACCELERATED | SDL_RENDERER_PRESENTVSYNC);

    bool running = true;
    SDL_Event evt;

    Graph g("graph");
    float w = 0.f,
          b = 0.f;
    g.set_line(w, b);

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
                case SDLK_SPACE:
                    descend(w, b, g.data());
                    g.set_line(w, b);
                    break;
                }
            }
        }

        SDL_RenderClear(rend);

        g.render(rend, { 0, 0, 600, 600 });

        SDL_SetRenderDrawColor(rend, 0, 0, 0, 255);
        SDL_RenderPresent(rend);
    }

    SDL_DestroyRenderer(rend);
    SDL_DestroyWindow(win);
    SDL_Quit();
    return 0;
}

