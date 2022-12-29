#include "common.h"

void common::add_cross_shape(graph::Graph2 &g)
{
    g.add_shape(graph::Graph2Shape(
        {
            { 0.f, 0.f }, { 1.f, 1.f },
            { 1.f, 0.f }, { 0.f, 1.f }
        },
        { 1.f, 0.f, 0.f }
    ));
}

void common::add_tri_shape(graph::Graph2 &g)
{
    g.add_shape(graph::Graph2Shape(
        {
            { .5f, 0.f }, { 0.f, 1.f },
            { .5f, 0.f }, { 1.f, 1.f },
            { 0.f, 1.f }, { 1.f, 1.f }
        },
        { 0.f, .5f, 1.f }
    ));
}

