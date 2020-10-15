#pragma once
#include <geometry.h>
#include <vector>
#include <functional>

/*
* These generators provide a callback which allways accept points.
*/

/* 2D Point generation over a domain */
class PointGenerator2{
    public:
    __host__ PointGenerator2();
    __host__ void Generate(const Bounds2f &domain, Float spacing, 
                           std::vector<vec2f> *points) const;
    __host__ virtual void ForEach(const Bounds2f &domain, Float spacing,
                                  const std::function<bool(const vec2f &)> &callback) 
        const = 0;
};

class TrianglePointGenerator : public PointGenerator2{
    public:
    __host__ TrianglePointGenerator();
    __host__ virtual void ForEach(const Bounds2f &domain, Float spacing,
                                  const std::function<bool(const vec2f &)> &callback) 
        const override;
};

class TrianglePointGeneratorDevice{
    public:
    __bidevice__ TrianglePointGeneratorDevice();
    __bidevice__ int Generate(const Bounds2f &domain, Float spacing,
                              vec2f *points, int maxn);
};

/* 3D Point generation over a domain */
class PointGenerator3{
    public:
    __host__ PointGenerator3();
    __host__ void Generate(const Bounds3f &domain, Float spacing,
                           std::vector<vec3f> *points) const;
    __host__ virtual void ForEach(const Bounds3f &domain, Float spacing,
                                  const std::function<bool(const vec3f &)> &callback)
        const = 0;
};

/*
* This guy has a complicated name and I don't know how to translate the word 'lattice'
* but you can think of this as a Generator that divides the domain in Voxels and loops
* through each important point in each Voxel. Almost like a uniform method but carefully
* choosing points.
*/
class BccLatticePointGenerator : public PointGenerator3{
    public:
    __host__ BccLatticePointGenerator();
    __host__ virtual void ForEach(const Bounds3f &domain, Float spacing,
                                  const std::function<bool(const vec3f &)> &callback)
        const override;
};

class BccLatticePointGeneratorDevice{
    public:
    __bidevice__ BccLatticePointGeneratorDevice();
    __bidevice__ int Generate(const Bounds3f &domain, Float spacing, 
                              vec3f *points, int maxn);
};