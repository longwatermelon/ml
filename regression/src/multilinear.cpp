#include "common.h"
#include <fstream>
#include <graph2.h>

float calc_mean(const std::vector<float> &values)
{
    float sum = 0.f;
    for (const auto &e : values)
        sum += e;

    return sum / values.size();
}

float calc_sd(const std::vector<float> &values)
{
    float mean = calc_mean(values);
    float sd = 0.f;

    for (const auto &e : values)
        sd += std::pow(e - mean, 2);

    return std::sqrt(sd / values.size());
}

std::vector<float> zscore_normalize(std::vector<float> features)
{
    float mean = calc_mean(features);
    float sd = calc_sd(features);

    std::string data;
    for (size_t i = 0; i < features.size(); ++i)
        features[i] = (features[i] - mean) / sd;

    return features;
}

void feature_scale(Graph2 &g, const std::string &out_fp)
{
    std::vector<float> features;
    for (const auto &e : g.data())
        features.emplace_back(e.x);

    features = zscore_normalize(features);

    std::string data;
    float min = std::numeric_limits<float>::max(),
          max = std::numeric_limits<float>::min();
    for (size_t i = 0; i < features.size(); ++i)
    {
        float x = features[i];
        if (x < min) min = x;
        if (x > max) max = x;
        data += "data " + std::to_string(x) + " " + std::to_string(g.data()[i].y) + "\n";
    }

    std::ofstream ofs(out_fp);
    ofs << "min " << min << ' ' << g.min().y << "\n"
        << "max " << max << ' ' << g.max().y << "\n"
        << "step " << std::abs(max - min) / 5.f << ' ' << g.step().y << "\n"
        << data;
    ofs.close();

    g.load(out_fp);
}

int main(int argc, char **argv)
{
    INIT_SDL("Multiple linear regression")

    bool running = true;
    SDL_Event evt;

    std::array<std::string, 4> data_paths = {
        "data-multilinear/age-graph",
        "data-multilinear/bedrooms-graph",
        "data-multilinear/floors-graph",
        "data-multilinear/size-graph"
    };

    std::array<Graph2, 4> graphs;
    for (int i = 0; i < 4; ++i)
        graphs[i] = Graph2(data_paths[i]);

    std::array<SDL_Rect, 4> rects = {
        SDL_Rect{ 0, 0, 300, 300 },
        { 300, 0, 300, 300 },
        { 0, 300, 300, 300 },
        { 300, 300, 300, 300 }
    };

    std::array<Graph2, 4> scaled_graphs = graphs;

    for (int i = 0; i < 4; ++i)
        feature_scale(scaled_graphs[i], data_paths[i] + "-scaled");

    float min = std::numeric_limits<float>::max();
    float max = std::numeric_limits<float>::min();
    for (size_t i = 0; i < graphs[3].data().size(); ++i)
    {
        float x = graphs[3].data()[i].x;
        if (x < min) min = x;
        if (x > max) max = x;
    }

    printf("%f %f\n", min, max);

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
                case SDLK_SPACE:
                    flag = !flag;
                    break;
                }
            }
        }

        SDL_RenderClear(rend);

        for (int i = 0; i < 4; ++i)
            (flag ? graphs[i] : scaled_graphs[i]).render(rend, rects[i], [](float x){ return 0.f; });

        SDL_SetRenderDrawColor(rend, 0, 0, 0, 255);
        SDL_RenderPresent(rend);
    }

    QUIT_SDL
    return 0;
}

