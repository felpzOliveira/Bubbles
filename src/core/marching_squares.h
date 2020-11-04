#pragma once
#include <geometry.h>
#include <grid.h>

void MarchingSquares(FieldGrid2f *grid, Float isovalue, 
                     std::vector<vec3f> *triangles);