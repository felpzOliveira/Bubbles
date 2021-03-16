#if !defined(QUATERNION_H)
#define QUATERNION_H
#include <transform.h>
#include <geometry.h>

class Quaternion{
    public:
    vec3f v;
    Float w;

    __bidevice__ Quaternion(): v(vec3f(0)), w(1){}

    __bidevice__ Quaternion &operator+=(const Quaternion &q){
        v += q.v;
        w += q.w;
        return *this;
    }

    __bidevice__ friend Quaternion operator+(const Quaternion &q1, const Quaternion &q2){
        Quaternion ret = q1;
        return ret += q2;
    }

    __bidevice__ Quaternion &operator-=(const Quaternion &q){
        v -= q.v;
        w -= q.w;
        return *this;
    }

    __bidevice__ Quaternion operator-() const{
        Quaternion ret;
        ret.v = -v;
        ret.w = -w;
        return ret;
    }

    __bidevice__ friend Quaternion operator-(const Quaternion &q1, const Quaternion &q2){
        Quaternion ret = q1;
        return ret -= q2;
    }

    __bidevice__ Quaternion &operator*=(Float f){
        v *= f;
        w *= f;
        return *this;
    }

    __bidevice__ Quaternion operator*(Float f) const{
        Quaternion ret = *this;
        ret.v *= f;
        ret.w *= f;
        return ret;
    }

    __bidevice__ Quaternion &operator/=(Float f){
        Float inv = 1.0 / f;
        v *= inv;
        w *= inv;
        return *this;
    }

    __bidevice__ Quaternion operator/(Float f) const{
        Float inv = 1.0 / f;
        Quaternion ret = *this;
        ret.v *= inv;
        ret.w *= inv;
        return *this;
    }

    __bidevice__ Transform ToTransform() const;
    __bidevice__ Quaternion(const Transform &t);
};

__bidevice__ Quaternion Qlerp(Float t, const Quaternion &q1, const Quaternion &q2);

__bidevice__ inline Quaternion operator*(Float f, const Quaternion &q){ return q * f; }
__bidevice__ inline Float Dot(const Quaternion &q1, const Quaternion &q2){
    return Dot(q1.v, q2.v) + q1.w * q2.w;
}

__bidevice__ inline Quaternion Normalize(const Quaternion &q){
    return q / std::sqrt(Dot(q, q));
}

#endif // QUATERNION_H
