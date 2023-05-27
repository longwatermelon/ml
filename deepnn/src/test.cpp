#include "deepnn.h"
#include <iostream>
#include <SDL2/SDL.h>
#include <graph2.h>

int main()
{
    SDL_Init(SDL_INIT_VIDEO);
    SDL_Window *w = SDL_CreateWindow("test", SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED,
            800, 800, SDL_WINDOW_SHOWN);
    SDL_Renderer *r = SDL_CreateRenderer(w, -1, SDL_RENDERER_ACCELERATED | SDL_RENDERER_PRESENTVSYNC);

    graph::Graph2 graph("data/planar");
    graph.add_cross_shape();
    graph.add_tri_shape();

    nn::Model model({
        nn::Layer(4, nn::Activation::Tanh),
        nn::Layer(1, nn::Activation::Sigmoid)
    });

    /* mt::mat X(2, 400); */
    /* mt::mat Y(1, 400); */

    /* for (int i = 0; i < graph.data().size(); ++i) */
    /* { */
    /*     X.atref(0, i) = graph.data()[i].p.x; */
    /*     X.atref(1, i) = graph.data()[i].p.y; */
    /*     Y.atref(0, i) = graph.data()[i].shape; */
    /* } */

    mt::mat X(2, 3);
    X.m_data = {
        { 1.62434536, -0.61175641, -0.52817175 },
        { -1.07296862,  0.86540763, -2.3015387 }
    };

    X = X.transpose();

    mt::mat Y(1, 3);
    Y.m_data = {
        { 1, 0, 1 }
    };

    Y = Y.transpose();

    model.m_layers[1].W.m_data = {
        { 1, 1, 1, 1 },
        { 2, 2, 2, 2 }
    };

    model.m_layers[2].W.m_data = {
        { 1 },
        { 2 },
        { 3 },
        { 4 }
    };

    model.train(X, Y, 10000, 1.2f);

    bool running = true;
    SDL_Event evt;
    while (running)
    {
        while (SDL_PollEvent(&evt))
        {
            if (evt.type == SDL_QUIT)
                running = false;
        }

        SDL_RenderClear(r);

        graph.render(r, { 0, 0, 800, 800 });

        SDL_SetRenderDrawColor(r, 0, 0, 0, 255);
        SDL_RenderPresent(r);
    }

    SDL_DestroyRenderer(r);
    SDL_DestroyWindow(w);
    SDL_Quit();

    return 0;
}

