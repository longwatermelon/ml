#pragma once
#include <vector>
#include <glm/glm.hpp>

namespace linear
{
    void descend(float &w, float &b, float a, const std::vector<glm::vec2> &data);
}

