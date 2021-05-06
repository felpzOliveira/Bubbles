/* date = May 5th 2021 13:4 */
#pragma once
#include <shape.h>
#include <geometry.h>
#include <cutil.h>

/*
* Defines utility sdf functions for generating sdf shapes. These can be used
* for generating coliders/emitters with complex shapes. If you wish to combine
* sdfs and make more complex shapes you can add your own sdf computation here.
*/

#define SDF_Sphere(center, radius) GPU_LAMBDA(vec3f point, Shape *) -> Float{\
    return Distance(point, center) - radius;\
}

#define SDF_Torus(center, radius) GPU_LAMBDA(vec3f point, Shape *) -> Float{\
    vec3f p = point - center;\
    vec2f xz(p.x, p.z);\
    vec2f q(xz.Length() - radius.x, p.y);\
    return q.Length() - radius.y;\
}

#define SDF_RoundBox(center, length, radius) GPU_LAMBDA(vec3f point, Shape *) -> Float{\
    vec3f p = point - center;\
    vec3f q = Abs(p) - length;\
    vec3f a(Max(q.x, 0), Max(q.y, 0), Max(q.z, 0));\
    Float f = Max(q.x, Max(q.y, q.z));\
    return a.Length() + Min(f, 0.0) - radius;\
}
