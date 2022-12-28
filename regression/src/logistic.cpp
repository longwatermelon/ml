#include <SDL2/SDL.h>
#include <graph2.h>
#include <graph3.h>

namespace logistic
{
    float f_wb(float w, float b, float x);
    void descend(float &w, float &b, float a, const std::vector<glm::vec2> &data);

    float loss(float w, float b, float prediction, float data_y);
    float cost(float w, float b, const std::vector<glm::vec2> &data);
}

float logistic::f_wb(float w, float b, float x)
{
    return 1.f / (1.f + std::exp(-(w * x + b)));
}

void logistic::descend(float &w, float &b, float a, const std::vector<glm::vec2> &data)
{
    float dw_j = 0.f,
          db_j = 0.f;
    for (const auto &p : data)
    {
        dw_j += (f_wb(w, b, p.x) - p.y) * p.x;
        db_j += f_wb(w, b, p.x) - p.y;
    }

    dw_j /= data.size();
    db_j /= data.size();

    w -= a * dw_j;
    b -= a * db_j;
}

float logistic::loss(float w, float b, float prediction, float data_y)
{
    return ((int)data_y == 1 ? -data_y * std::log(prediction) : -(1.f - data_y) * std::log(1.f - prediction));
}

float logistic::cost(float w, float b, const std::vector<glm::vec2> &data)
{
    float cost = 0.f;
    for (const auto &p : data)
        cost += loss(w, b, f_wb(w, b, p.x), p.y);
    cost /= data.size();
    return cost;
}

int main(int argc, char **argv)
{
    SDL_Init(SDL_INIT_VIDEO);
    SDL_Window *win = SDL_CreateWindow("Logistic regression",
            SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED,
            600, 600, SDL_WINDOW_SHOWN);
    SDL_Renderer *rend = SDL_CreateRenderer(win, -1,
            SDL_RENDERER_ACCELERATED | SDL_RENDERER_PRESENTVSYNC);

    bool running = true;
    SDL_Event evt;

    Graph2 graph("data/logistic/graph");
    float w = 0.f,
          b = 0.f;

    Graph3 graph3("data/logistic/graph3", [graph](float x, float z){
        return logistic::cost(x, z, graph.data());
    });
    graph3.add_point(w, b);

    printf("%f\n", logistic::cost(7.f, 0.f, graph.data()));

    bool mouse_down = false;

    /* for (int i = 0; i < 8000; ++i) */
    /* { */
    /*     logistic::descend(w, b, .1f, graph.data()); */
    /*     graph3.add_point(w, b); */
    /* } */

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
                case SDLK_p:
                    printf("%fx + %f\n", w, b);
                    break;
                }
                break;
            case SDL_MOUSEBUTTONDOWN:
                mouse_down = true;
                break;
            case SDL_MOUSEBUTTONUP:
                mouse_down = false;
                break;
            case SDL_MOUSEMOTION:
                if (mouse_down)
                    graph3.rot({ evt.motion.xrel / 200.f, evt.motion.yrel / 200.f, 0.f });
                break;
            }
        }

        const Uint8 *keystates = SDL_GetKeyboardState(0);
        if (keystates[SDL_SCANCODE_SPACE])
        {
            logistic::descend(w, b, 1.f, graph.data());
            graph3.add_point(w, b);
        }

        SDL_RenderClear(rend);

        graph.render(rend, { 0, 0, 600, 300 }, [w, b](float x){
            return 1.f / (1 + std::exp(-(w * x + b)));
        });

        graph3.render(rend, { 0, 300, 600, 300 });

        SDL_SetRenderDrawColor(rend, 0, 0, 0, 255);
        SDL_RenderPresent(rend);
    }

    SDL_DestroyRenderer(rend);
    SDL_DestroyWindow(win);
    SDL_Quit();
    return 0;
}

