#include "unsupervised.h"
#include <SDL2/SDL_events.h>
#include <SDL2/SDL_keycode.h>
#include <SDL2/SDL_render.h>
#include <SDL2/SDL_video.h>
#include <common.h>
#include <graph2.h>
#include <ctime>

int main()
{
    srand(time(0));

    SDL_Init(SDL_INIT_VIDEO);
    SDL_Window *win = SDL_CreateWindow("K-means clustering",
        SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED,
        600, 600, SDL_WINDOW_SHOWN);
    SDL_Renderer *rend = SDL_CreateRenderer(win, -1,
        SDL_RENDERER_ACCELERATED | SDL_RENDERER_PRESENTVSYNC);

    graph::Graph2 graph("data/kmeans");
    graph.add_cross_shape();
    graph.add_tri_shape();
    graph.add_square_shape();

    std::vector<Eigen::Vector2f> mx;
    for (const auto &dp : graph.data())
        mx.emplace_back(Eigen::Vector2f({{dp.p.x, dp.p.y}}));

    std::vector<Eigen::Vector2f> centroids = {
        Eigen::Vector2f({{ (float)(rand() % 50), (float)(rand() % 50) }}),
        Eigen::Vector2f({{ (float)(rand() % 50), (float)(rand() % 50) }})
    };
    graph.push_datapoint(graph::DataPoint2{ .p = { centroids[0][0], centroids[0][1] }, .shape = 2 });
    graph.push_datapoint(graph::DataPoint2{ .p = { centroids[1][0], centroids[1][1] }, .shape = 2 });

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
            case SDL_KEYDOWN:
            {
                switch (evt.key.keysym.sym)
                {
                case SDLK_SPACE:
                {
                    kmeans::move_centroids(mx, centroids);
                    std::vector<std::vector<Eigen::Vector2f>> points;
                    kmeans::assign_points_to_centroids(mx, centroids, points);
                    graph.clear_data();

                    for (size_t i = 0; i < centroids.size(); ++i)
                    {
                        for (const auto &dp : points[i])
                            graph.push_datapoint(graph::DataPoint2{ .p = { dp[0], dp[1] }, .shape = i });
                        graph.push_datapoint(graph::DataPoint2{ .p = { centroids[i][0], centroids[i][1] }, .shape = 2 });
                    }
                } break;
                }
            } break;
            }
        }

        SDL_RenderClear(rend);

        graph.render(rend, { 0, 0, 600, 600 });

        SDL_SetRenderDrawColor(rend, 0, 0, 0, 255);
        SDL_RenderPresent(rend);
    }

    SDL_DestroyRenderer(rend);
    SDL_DestroyWindow(win);
    SDL_Quit();

    return 0;
}
