#pragma once
#include "dtree.h"

inline std::vector<dtree::DataPoint> data = {
    dtree::DataPoint({ 1, 1, 1 }, 1),
    dtree::DataPoint({ 0, 0, 1 }, 1),
    dtree::DataPoint({ 0, 1, 0 }, 0),
    dtree::DataPoint({ 1, 0, 1 }, 0),
    dtree::DataPoint({ 1, 1, 1 }, 1),
    dtree::DataPoint({ 1, 1, 0 }, 1),
    dtree::DataPoint({ 0, 0, 0 }, 0),
    dtree::DataPoint({ 1, 1, 0 }, 1),
    dtree::DataPoint({ 0, 1, 0 }, 0),
    dtree::DataPoint({ 0, 1, 0 }, 0)
};
