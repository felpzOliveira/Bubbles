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
    PointGenerator2();
    void Generate(const Bounds2f &domain, Float spacing, std::vector<vec2f> *points) const;
    virtual void ForEach(const Bounds2f &domain, Float spacing,
                        const std::function<bool(const vec2f &)> &callback) const = 0;
};

class TrianglePointGenerator : public PointGenerator2{
    public:
    TrianglePointGenerator();
    virtual void ForEach(const Bounds2f &domain, Float spacing,
                         const std::function<bool(const vec2f &)> &callback) const override;
};

class TrianglePointGeneratorDevice{
    public:
    bb_cpu_gpu TrianglePointGeneratorDevice();
    bb_cpu_gpu int Generate(const Bounds2f &domain, Float spacing,
                            vec2f *points, int maxn);
};

/* 3D Point generation over a domain */
class PointGenerator3{
    public:
    PointGenerator3();
    void Generate(const Bounds3f &domain, Float spacing, std::vector<vec3f> *points) const;
    virtual void ForEach(const Bounds3f &domain, Float spacing,
                         const std::function<bool(const vec3f &)> &callback) const = 0;
};

/*
* This guy has a complicated name and I don't know how to translate it
* but you can think of this as a Generator that divides the domain in Voxels and loops
* through each important point in each Voxel. Almost like a uniform method but carefully
* choosing points.
*/
class BccLatticePointGenerator : public PointGenerator3{
    public:
    BccLatticePointGenerator();
    virtual void ForEach(const Bounds3f &domain, Float spacing,
                         const std::function<bool(const vec3f &)> &callback) const override;
};

class BccLatticePointGeneratorDevice{
    public:
    bb_cpu_gpu BccLatticePointGeneratorDevice();
    bb_cpu_gpu int Generate(const Bounds3f &domain, Float spacing, vec3f *points, int maxn);
};
