/* date = October 4th 2021 18:30 */
#pragma once
#include <grid.h>
#include <obj_loader.h>

constexpr int kDirectionNone = 0;
constexpr int kDirectionLeft = 1 << 0;
constexpr int kDirectionRight = 1 << 1;
constexpr int kDirectionDown = 1 << 2;
constexpr int kDirectionUp = 1 << 3;
constexpr int kDirectionBack = 1 << 4;
constexpr int kDirectionFront = 1 << 5;
constexpr int kDirectionAll = kDirectionLeft | kDirectionRight |
                              kDirectionDown | kDirectionUp | kDirectionBack |
                              kDirectionFront;

void MarchingCubes(FieldGrid3f *grid, const vec3f& gridSize, const vec3f& origin,
                   HostTriangleMesh3* mesh, Float isoValue=0, int bndClose=kDirectionAll,
                   int bndConnectivity=kDirectionNone);
