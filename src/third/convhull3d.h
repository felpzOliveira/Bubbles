/* date = September 27th 2021 18:20 */
#pragma once
#include <geometry.h>
#include <functional>

/*
 * Wrap convex hull routines for 3D outside sandim method so we can actually
 * check its instabilities.
 */

template<typename T> struct IndexedParticle{
    int pId;
    T p;
};

void ConvexHullPrepare();
void ConvexHullFinish();

void ConvexHull3D(IndexedParticle<vec3f> *ips, int maxLen,
                  int len, std::function<void(int)> reporter);
