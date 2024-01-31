#include <transform_sequence.h>

TransformSequence::TransformSequence() : scopeStart(0), scopeEnd(0){}

void TransformSequence::ComputeInitialTransform(){
    if(transforms.size() == 1){
        Interpolate(scopeStart, &lastInterpolatedTransform);
    }
}

void TransformSequence::UpdateInterval(Float start, Float end){
    scopeStart = Min(scopeStart, start);
    scopeEnd = Max(scopeEnd, end);
}

void TransformSequence::GetLastTransform(Transform *transform){
    Interpolate(scopeEnd, transform);
}

void TransformSequence::AddRestore(Float s0, Float s1){
    Transform firstTransform, lastTransform;
    Interpolate(scopeStart, &firstTransform);
    Interpolate(scopeEnd, &lastTransform);
    // Create a new AggregatedTransform between lastTransform -> firstTransform
    // so the animation can be looped
    AddInterpolation(&lastTransform, &firstTransform, s0, s1);
}

void TransformSequence::AddInterpolation(Transform *t0, Transform *t1, Float s0, Float s1){
    Float start = Min(s0, s1);
    Float end = Max(s0, s1);
    UpdateInterval(start, end);

    AggregatedTransform aggTransform = {
        .immutablePre  = Transform(),
        .immutablePost = Transform(),
        .interpolated = InterpolatedTransform(t0, t1, s0, s1),
        .start = start,
        .end = end,
        .owned = 0,
    };

    transforms.push_back(aggTransform);
    ComputeInitialTransform();
}

void TransformSequence::AddInterpolation(InterpolatedTransform *inp){
    AggregatedTransform aggTransform = {
        .immutablePre = Transform(),
        .immutablePost = Transform(),
        .interpolated = *inp,
        .start = inp->t0,
        .end = inp->t1,
        .owned = 0,
    };

    UpdateInterval(aggTransform.start, aggTransform.end);
    transforms.push_back(aggTransform);
    ComputeInitialTransform();
}

void TransformSequence::AddInterpolation(AggregatedTransform *kTransform,
                                         Float s0, Float s1)
{
    Float start = Min(s0, s1);
    Float end = Max(s0, s1);
    AggregatedTransform aggTransform = *kTransform;
    aggTransform.start = start;
    aggTransform.end = end;
    aggTransform.owned = 0;

    UpdateInterval(aggTransform.start, aggTransform.end);
    transforms.push_back(aggTransform);
    ComputeInitialTransform();
}

void TransformSequence::AddInterpolation(AggregatedTransform *kTransform){
    AggregatedTransform aggTransform = *kTransform;
    aggTransform.owned = 0;
    UpdateInterval(aggTransform.start, aggTransform.end);
    transforms.push_back(aggTransform);
    ComputeInitialTransform();
}

void TransformSequence::Interpolate(Float t, Transform *outTransform,
                                    vec3f *linear, vec3f *angular)
{
    AggregatedTransform *aggTransform = nullptr;
    Transform interp;

    if(t <= scopeStart){
        aggTransform = &transforms[0];
    }else if(t >= scopeEnd){
        aggTransform = &transforms[transforms.size()-1];
    }else{
        for(int i = 0; i < transforms.size(); i++){
            AggregatedTransform *agg = &transforms[i];
            if(t >= agg->start && t <= agg->end){
                aggTransform = agg;
                break;
            }
        }
    }

    AssertA(aggTransform != nullptr, "Failed to locate target intepolation");

    aggTransform->interpolated.Interpolate(t, &interp);

    *outTransform = aggTransform->immutablePost * interp * aggTransform->immutablePre;
    if(angular || linear){
        vec3f lastT, currT;
        Quaternion lastR, currR;
        Matrix4x4 lastS, currS;

        InterpolatedTransform::Decompose(outTransform->m, &currT, &currR, &currS);
        InterpolatedTransform::Decompose(lastInterpolatedTransform.m,
                                         &lastT, &lastR, &lastS);

        if(linear){
            *linear = currT - lastT;
            //TODO: We should add scaling to this velocity but it is kinda
            // hard without knowing how much we are actually scaling
        }

        if(angular){
            // Ok so we need to compute the angular velocity from two quaternions
            // going to use Lie algebra for this

            // Given 2 quaternions because this is the minimum step of a simulation
            // i'm going to assume that dq/dt = lim (q(t+h) - q(t))/h and that q(t+h) = q2
            // and q(t) = q1, with h = simulation dt.

            // but the inertial angular velocity can be used to show that:
            // dq/dt = 0.5 * q(t) * (0, Wt) where Wt is the angular velocity
            // so we can write that Wt = Im(2 * Conj(q(t)) * dq/dt) because
            // this divide is directly we don't need to receive h or care
            // about integration schemes, and can write that 
            // Wt = 2 * Im(Conj(q(t)) * q(t+h)) {/ h}, with divide outside
            Quaternion qtt = lastR.Conjugate() * currR;
            *angular = 2.0 * qtt.Image();
        }
    }

    lastInterpolatedTransform = *outTransform;
}

TransformSequence::~TransformSequence(){
    for(AggregatedTransform agg : transforms){
        if(agg.owned){
            //delete agg.interpolated;
        }
    }
}

QuaternionSequence::QuaternionSequence(){
    scopeStart = Infinity;
    scopeEnd = -Infinity;
}

void QuaternionSequence::AddQuaternion(const Quaternion &q1, const Float &t){
    scopeStart = Min(scopeStart, t);
    scopeEnd = Max(scopeEnd, t);
    quaternions.push_back({.t = t, .q = q1});
}

void QuaternionSequence::AddQuaternion(const Float &angle, const vec3f &axis,
                                       const Float &t)
{
    Transform rot = Rotate(angle, axis);
    scopeStart = Min(scopeStart, t);
    scopeEnd = Max(scopeEnd, t);
    quaternions.push_back({.t = t, .q = Quaternion(rot)});
}

void QuaternionSequence::Interpolate(Float t, Transform *transform, vec3f *angular){
    AggregatedQuaternion *aQ = nullptr, *laQ = nullptr;
    for(int i = 0; i < quaternions.size(); i++){
        AggregatedQuaternion *aggQ = &quaternions[i];
        if(aggQ->t < t){
            laQ = aggQ;
        }else{
            aQ = aggQ;
            break;
        }
    }

    if(laQ && aQ){
        Float f = (t - laQ->t) / (aQ->t - laQ->t);
        Quaternion q = Qlerp(f, laQ->q, aQ->q);
        if(angular){
            Quaternion qtt = laQ->q.Conjugate() * q;
            *angular = 2.0 * qtt.Image();
        }

        *transform = q.ToTransform();
    }else if(laQ){
        *transform = laQ->q.ToTransform();
        if(angular) *angular = vec3f(0);
    }else if(aQ){
        *transform = aQ->q.ToTransform();
        if(angular) *angular = vec3f(0);
    }
}
