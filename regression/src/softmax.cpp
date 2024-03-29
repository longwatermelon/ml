#include "reg.h"
#include "common.h"
#include <graph2.h>
#include <iostream>

using namespace reg;

#define NF 2
#define NY 4

int main()
{
    SDL_Init(SDL_INIT_VIDEO);
    SDL_Window *win = SDL_CreateWindow("Softmax regression",
            SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED,
            600, 600, SDL_WINDOW_SHOWN);
    SDL_Renderer *rend = SDL_CreateRenderer(win, -1,
            SDL_RENDERER_ACCELERATED | SDL_RENDERER_PRESENTVSYNC);

    graph::Graph2 graph("data/softmax/data");
    graph.add_cross_shape();
    graph.add_tri_shape();
    graph.add_square_shape();
    graph.add_plus_shape();

    // mw is NY x NF
    Eigen::MatrixXf mw(NY, NF);
    mw.setZero();
    Eigen::VectorXf vb(NY);
    vb.setZero();

    std::vector<DataPoint> data;
    for (const auto &p : graph.data())
        data.emplace_back(DataPoint(Eigen::VectorXf({{ p.p.x, p.p.y }}), p.shape));

    for (size_t i = 0; i < 1000; ++i)
    {
        for (int j = 0; j < NY; ++j)
        {
            std::vector<DataPoint> data_copy = data;
            for (auto &dp : data_copy)
                dp.y = dp.y == j ? 1 : 0;

            Eigen::VectorXf vw = mw.row(j);
            general::descend(vw, vb[j], .1f, data_copy,
                    [mw, vb](const Eigen::VectorXf &vw,
                       const Eigen::VectorXf &vx,
                       float b){
                Eigen::VectorXf vz(NY);
                for (size_t i = 0; i < NY; ++i)
                    vz[i] = mw.row(i).transpose().dot(vx) + vb[i];

                return softmax::g(vw.dot(vx) + b, vz);
            });
            mw.row(j) = vw;
        }

        if ((i + 1) % 100 == 0)
        {
            std::cout << "Iteration " << i << ": mw = \n[" << mw;
            // printf("Iteration %zu: mw = [ ", i);
                // printf("%s ", common::vec::to_string(vw).c_str());
            std::cout << "]\nvb = [" << vb.transpose() << "]\n";
            // printf("]| vb = %s\n", common::vec::to_string(vb).c_str());
        }
    }

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

        SDL_Rect graph_r = { 0, 0, 600, 600 };
        graph.render(rend, graph_r);
        std::array<glm::vec3, NY> colors = {
            glm::vec3{ 1.f, 0.f, 0.f },
            { 0.f, 0.f, 1.f },
            { .6f, .6f, 0.f },
            { 0.f, 1.f, 0.f }
        };

        graph.render_trend(rend, graph_r,
                [colors, mw, vb](glm::vec2 p){
            Eigen::VectorXf vz(NY);
            for (int i = 0; i < NY; ++i)
                vz[i] = mw.row(i).dot(Eigen::VectorXf({{ p.x, p.y }})) + vb[i];
            Eigen::VectorXf va = softmax::solve_va(vz);
            return va[0] * colors[0] + va[1] * colors[1] + va[2] * colors[2] + va[3] * colors[3];
        });

        /* for (int i = 0; i < NY; ++i) */
        /* { */
        /*     std::array<float, NF> vw = mw[i]; */
        /*     float b = vb[i]; */

        /*     graph.render_line(rend, graph_r, */
        /*             [vw, b](float x){ */
        /*         return (.5f - b - vw[0] * x) / vw[1]; */
        /*     }, colors[i]); */
        /* } */

        SDL_SetRenderDrawColor(rend, 0, 0, 0, 255);
        SDL_RenderPresent(rend);
    }

    SDL_DestroyRenderer(rend);
    SDL_DestroyWindow(win);
    SDL_Quit();
    return 0;
}

