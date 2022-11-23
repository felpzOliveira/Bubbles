/* date = October 4th 2021 18:30 */
#pragma once
#include <util.h>
#include <grid.h>
#include <obj_loader.h>
#include <functional>

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

void MarchingCubes(FieldGrid3f *grid, HostTriangleMesh3* mesh, Float isoValue,
                   std::function<void(vec3ui u)> fn, int bndClose=kDirectionAll,
                   int bndConnectivity=kDirectionNone);
