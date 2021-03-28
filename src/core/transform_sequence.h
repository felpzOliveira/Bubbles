#pragma once
#include <transform.h>
#include <quaternion.h>
#include <vector>

/*
* Transform Sequence provides a wrapper class for a transform sequence
* that must be computed on HOST during simulation. Device is reserved for
* solvers and SDF generation. This can be used to animate colliders between
* several points during simulation like 'key frame' animation. 
*/
struct AggregatedTransform{
    Transform immutablePre, immutablePost;
    InterpolatedTransform *interpolated;
    Float start, end;
    int owned;
};

class TransformSequence{
    public:
    std::vector<AggregatedTransform> transforms;
    Transform lastInterpolatedTransform;
    Float scopeStart, scopeEnd;

    __host__ TransformSequence();
    __host__ void AddInterpolation(Transform *t0, Transform *t1, Float s0, Float s1);
    __host__ void AddInterpolation(InterpolatedTransform *inp);
    __host__ void AddInterpolation(AggregatedTransform *kTransform);
    __host__ void AddInterpolation(AggregatedTransform *kTransform, Float s0, Float s1);
    __host__ void AddRestore(Float s0, Float s1);

    __host__ void GetLastTransform(Transform *transform);
    __host__ void Interpolate(Float t, Transform *transform, vec3f *linear=nullptr,
                              vec3f *angular=nullptr);

    __host__ void ComputeInitialTransform();
    __host__ void UpdateInterval(Float start, Float end);

    ~TransformSequence();
};
