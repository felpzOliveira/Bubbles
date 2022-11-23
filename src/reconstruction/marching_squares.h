#pragma once
#include <geometry.h>
#include <grid.h>

constexpr int kDirectionLeft = 1 << 0;
constexpr int kDirectionRight = 1 << 1;
constexpr int kDirectionDown = 1 << 2;
constexpr int kDirectionUp = 1 << 3;
constexpr int kDirectionBack = 1 << 4;
constexpr int kDirectionFront = 1 << 5;

void MarchingSquares(FieldGrid2f *grid, Float isovalue,
                     std::vector<vec3f> *triangles);
