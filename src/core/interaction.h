#pragma once

#include <geometry.h>
#include <transform.h>

class Shape2;
class Shape;

class SurfaceInteraction2{
    public:
    vec2f p, n;
    vec2f pError;
    const Shape2 *shape = nullptr;

    bb_cpu_gpu SurfaceInteraction2(){ shape = nullptr; }
    bb_cpu_gpu SurfaceInteraction2(const vec2f &p): p(p), n(vec2f(0)), pError(vec2f(0))
    { shape = nullptr; }
    bb_cpu_gpu SurfaceInteraction2(const vec2f &po, const vec2f &no,
                                     const vec2f &pErr): p(po), n(no), pError(pErr)
    { shape = nullptr; }

    bb_cpu_gpu SurfaceInteraction2(const vec2f &po, const vec2f &no,
                                     const vec2f &pErr, const Shape2 *sh)
        : p(po), n(no), pError(pErr), shape(sh){}

};

class SurfaceInteraction{
    public:
    vec3f p;
    Normal3f n;
    vec3f pError;
    const Shape *shape = nullptr;

    bb_cpu_gpu SurfaceInteraction(){ shape = nullptr; }
    bb_cpu_gpu SurfaceInteraction(const vec3f &p) : p(p), n(Normal3f(0)), pError(vec3f(0))
    { shape = nullptr; }

    bb_cpu_gpu SurfaceInteraction(const vec3f &po, const Normal3f &no,
                                    const vec3f &pErr): p(po), n(no), pError(pErr)
    { shape = nullptr; }

    bb_cpu_gpu SurfaceInteraction(const vec3f &po, const Normal3f &no,
                                    const vec3f &pErr, const Shape *sh)
        : p(po), n(no), pError(pErr), shape(sh){}
};

inline bb_cpu_gpu
Ray SpawnRayInDirection(const SurfaceInteraction &isect, const vec3f &dir){
    vec3f o = OffsetRayOrigin(isect.p, isect.pError, isect.n, dir);
    //vec3f o = isect.p + Epsilon * dir;
    return Ray(o, dir, Infinity);
}
