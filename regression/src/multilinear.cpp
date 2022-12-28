#include "impl.h"
#include <fstream>
#include <vector>
#include <sstream>
#include <graph2.h>

#define NFEATURES 4

int main(int argc, char **argv)
{
    bool graphics = (argc == 2 && strcmp(argv[1], "render") == 0);
    SDL_Window *win;
    SDL_Renderer *rend;

    if (graphics)
    {
        SDL_Init(SDL_INIT_VIDEO);
        win = SDL_CreateWindow("Multiple linear regression",
            SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED,
            600, 600, SDL_WINDOW_SHOWN);
        rend = SDL_CreateRenderer(win, -1, SDL_RENDERER_ACCELERATED | SDL_RENDERER_PRESENTVSYNC);
    }

    bool running = true;
    SDL_Event evt;

    std::array<std::string, NFEATURES> data_paths = {
        "data/multilinear/age-graph",
        "data/multilinear/bedrooms-graph",
        "data/multilinear/floors-graph",
        "data/multilinear/size-graph"
    };

    std::array<Graph2, NFEATURES> graphs;
    for (int i = 0; i < NFEATURES; ++i)
        graphs[i] = Graph2(data_paths[i]);

    std::array<SDL_Rect, NFEATURES> rects = {
        SDL_Rect{ 0, 0, 300, 300 },
        { 300, 0, 300, 300 },
        { 0, 300, 300, 300 },
        { 300, 300, 300, 300 }
    };

    std::array<Graph2, NFEATURES> scaled_graphs = graphs;
    std::array<float, NFEATURES> vsd, vmean;
    for (int i = 0; i < NFEATURES; ++i)
        multilinear::feature_scale(scaled_graphs[i], vsd[i], vmean[i]);

    std::array<float, NFEATURES> vw;
    vw.fill(0.f);
    float b = 0.f;

    std::vector<std::array<float, NFEATURES>> mx;
    std::vector<float> vy;
    for (size_t i = 0; i < scaled_graphs[0].data().size(); ++i)
    {
        std::array<float, NFEATURES> arr;
        for (size_t j = 0; j < NFEATURES; ++j)
            arr[j] = scaled_graphs[j].data()[i].x;
        mx.emplace_back(arr);
        // y is same between all data points at index i
        vy.emplace_back(scaled_graphs[0].data()[i].y);
    }

    for (int i = 0; i < 500; ++i)
    {
        multilinear::descend(vw, b, .1f, mx, vy);
        if (i == 0 || (i + 1) % 50 == 0)
            printf("Iteration %d: vw = [%f, %f, %f, %f], b = %f\n",
                    i + 1, vw[0], vw[1], vw[2], vw[3], b);
    }

    float input[NFEATURES] = {
        40, // age
        3, // bedrooms
        1, // floors
        1200 // size (sqft)
    };

    float prediction = b;
    for (int i = 0; i < NFEATURES; ++i)
    {
        input[i] = (input[i] - vmean[i]) / vsd[i];
        prediction += input[i] * vw[i];
    }

    printf("Price prediction of house with age = 40, bedrooms = 3, floors = 1, size = 1200: $%.2f\n", prediction * 1000.f);

    if (graphics)
    {
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

            SDL_RenderClear(rend);

            for (int i = 0; i < NFEATURES; ++i)
            {
                if (flag)
                    scaled_graphs[i].render(rend, rects[i], [](float x){ return 0.f; });
                else
                    graphs[i].render(rend, rects[i], [](float x){ return 0.f; });
            }

            SDL_SetRenderDrawColor(rend, 0, 0, 0, 255);
            SDL_RenderPresent(rend);
        }

        SDL_DestroyRenderer(rend);
        SDL_DestroyWindow(win);
        SDL_Quit();
    }
    return 0;
}

