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
    InterpolatedTransform interpolated;
    Float start, end;
    int owned;
};

struct AggregatedQuaternion{
    Float t;
    Quaternion q;
};

class TransformSequence{
    public:
    std::vector<AggregatedTransform> transforms;
    Transform lastInterpolatedTransform;
    Float scopeStart, scopeEnd;

    TransformSequence();
    void AddInterpolation(Transform *t0, Transform *t1, Float s0, Float s1);
    void AddInterpolation(InterpolatedTransform *inp);
    void AddInterpolation(AggregatedTransform *kTransform);
    void AddInterpolation(AggregatedTransform *kTransform, Float s0, Float s1);
    void AddRestore(Float s0, Float s1);

    void GetLastTransform(Transform *transform);
    void Interpolate(Float t, Transform *transform, vec3f *linear=nullptr,
                              vec3f *angular=nullptr);

    void ComputeInitialTransform();
    void UpdateInterval(Float start, Float end);

    ~TransformSequence();
};

/*
* Quaternion Sequence provides a simple wrapper for quaternion sequence transformation
* when we don't need to translate/scale anything this might be faster and easier to use
*/
class QuaternionSequence{
    public:
    std::vector<AggregatedQuaternion> quaternions;
    Float scopeStart, scopeEnd;

    QuaternionSequence();
    void AddQuaternion(const Quaternion &q1, const Float &t);
    void AddQuaternion(const Float &angle, const vec3f &axis, const Float &t);
    void Interpolate(Float t, Transform *transform, vec3f *angular=nullptr);
};

/*
* Simple interpolation if you don't want to bother with quaternions/aggregated stuff.
*/
inline void EuclideanInterpolate(Float a0, Float a1, Float t, Float s0, Float s1,
                                 vec3f axis, Transform *transform)
{
    Float f = (t - s0) / (s1 - s0);
    Float a = Lerp(a0, a1, f);
    *transform = Rotate(a, axis);
}
