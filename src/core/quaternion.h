#if !defined(QUATERNION_H)
#define QUATERNION_H
#include <geometry.h>
#include <transform.h>

class Quaternion{
    public:
    vec3f v;
    Float w;

    bb_cpu_gpu Quaternion(): v(vec3f(0)), w(1){}

    bb_cpu_gpu Quaternion &operator+=(const Quaternion &q){
        v += q.v;
        w += q.w;
        return *this;
    }

    bb_cpu_gpu friend Quaternion operator+(const Quaternion &q1, const Quaternion &q2){
        Quaternion ret = q1;
        return ret += q2;
    }

    bb_cpu_gpu Quaternion &operator-=(const Quaternion &q){
        v -= q.v;
        w -= q.w;
        return *this;
    }

    bb_cpu_gpu Quaternion operator-() const{
        Quaternion ret;
        ret.v = -v;
        ret.w = -w;
        return ret;
    }

    bb_cpu_gpu friend Quaternion operator-(const Quaternion &q1, const Quaternion &q2){
        Quaternion ret = q1;
        return ret -= q2;
    }

    bb_cpu_gpu Quaternion &operator*=(Float f){
        v *= f;
        w *= f;
        return *this;
    }

    // Hamilton product
    bb_cpu_gpu Quaternion &operator*=(Quaternion q1){
        Float w0 = w * q1.w - Dot(v, q1.v);
        vec3f v0 = w * q1.v + q1.w * v + Cross(v, q1.v);
        w = w0;
        v = v0;
        return *this;
    }

    bb_cpu_gpu Quaternion operator*(Float f) const{
        Quaternion ret = *this;
        ret.v *= f;
        ret.w *= f;
        return ret;
    }

    // Hamilton product
    bb_cpu_gpu Quaternion operator*(Quaternion q1) const{
        Quaternion ret = *this;
        Float w0 = w * q1.w - Dot(v, q1.v);
        vec3f v0 = w * q1.v + q1.w * v + Cross(v, q1.v);
        ret.w = w;
        ret.v = v;
        return ret;
    }

    bb_cpu_gpu Quaternion &operator/=(Float f){
        Float inv = 1.0 / f;
        v *= inv;
        w *= inv;
        return *this;
    }

    bb_cpu_gpu Quaternion operator/(Float f) const{
        Float inv = 1.0 / f;
        Quaternion ret = *this;
        ret.v *= inv;
        ret.w *= inv;
        return *this;
    }

    bb_cpu_gpu Quaternion Conjugate() const{
        Quaternion ret = *this;
        ret.v = -ret.v;
        return ret;
    }

    bb_cpu_gpu vec3f Image() const{
        return v;
    }

    bb_cpu_gpu Transform ToTransform() const;
    bb_cpu_gpu Quaternion(const Transform &t);
};

bb_cpu_gpu Float Qangle(const Quaternion &q1, const Quaternion &q2);
bb_cpu_gpu Quaternion Qlerp(Float t, const Quaternion &q1, const Quaternion &q2);

bb_cpu_gpu inline Quaternion operator*(Float f, const Quaternion &q){ return q * f; }
bb_cpu_gpu inline Float Dot(const Quaternion &q1, const Quaternion &q2){
    return Dot(q1.v, q2.v) + q1.w * q2.w;
}

bb_cpu_gpu inline Quaternion Normalize(const Quaternion &q){
    return q / std::sqrt(Dot(q, q));
}

class InterpolatedTransform{
    public:
    Transform tStart, tEnd;
    Float t0, t1;
    vec3f T[2];
    Quaternion R[2];
    Matrix4x4 S[2];

    bb_cpu_gpu InterpolatedTransform(Transform *e0, Transform *e1, Float s0, Float s1);
    bb_cpu_gpu static void Decompose(const Matrix4x4 &m, vec3f *T, Quaternion *R, Matrix4x4 *S);
    bb_cpu_gpu void Interpolate(Float t, Transform *transform);
};

bb_cpu_gpu void Interpolate(InterpolatedTransform *iTransform, Float t, Transform *transform);

#endif // QUATERNION_H
