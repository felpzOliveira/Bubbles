/* date = October 4th 2021 18:30 */
#pragma once
#include <util.h>
#include <grid.h>
#include <obj_loader.h>
#include <functional>

void MarchingCubes(FieldGrid3f *grid, HostTriangleMesh3* mesh, Float isoValue,
                   bool rotate_faces=false);
