#pragma once
#include <geometry.h>

#define Accessor2D(acc, i, j, x) acc[(i) + (x) * (j)]

template<typename T>
inline __bidevice__ void GetBarycentric(const T &x, int low, int high,
                                        int *i, T *f)
{
    T s = Floor(x);
    int offset = -low;
    *i = (int)s;

    low = 0;
    high += offset;

    if(low == high){
        *i = low;
        *f = T(0);
    }else if(*i < low){
        *i = low;
        *f = T(0);
    }else if(*i > high-1){
        *i = high-1;
        *f = 1;
    }else{
        *f = T(x-s);
    }

    *i -= offset;
}

template<typename T, typename Q> inline __bidevice__
T Bilerp(const T &f00, const T &f10, const T &f01, const T &f11, Q tx, Q ty){
    return Lerp<T, Q>(Lerp<T, Q>(f00, f10, tx), Lerp(f01, f11, tx), ty);
}

template<typename T, typename Q> inline __bidevice__
T Trilerp(const T &f000, const T &f100, const T &f010, const T &f110,
          const T &f001, const T &f101, const T &f011, const T &f111,
          Q tx, Q ty, Q tz)
{
    return Lerp<T, Q>(Bilerp<T, Q>(f000, f100, f010, f110, tx, ty),
                      Bilerp<T, Q>(f001, f101, f011, f111, tx, ty), tz);
}

template<typename T, typename Q> inline __bidevice__
T LinearGridSampler2Sample(const vec2f &pt, T *data, const vec2f &spacing,
                           const vec2f &origin, const vec2ui &res)
{
    Q fx = Q(0), fy = Q(0);
    int i = 0, j = 0;
    int ip1 = 0, jp1 = 0;

    vec2f invSpacing(1.0 / spacing.x, 1.0 / spacing.y);
    vec2f nPt = (pt - origin) * invSpacing;
    int iSize = (int)res.x;
    int jSize = (int)res.y;

    GetBarycentric<Q>(nPt.x, 0, iSize-1, &i, &fx);
    GetBarycentric<Q>(nPt.y, 0, jSize-1, &j, &fy);

    ip1 = Min(i+1, iSize-1);
    jp1 = Min(j+1, jSize-1);

    return Bilerp<T, Q>(Accessor2D(data, i, j, res.x),
                        Accessor2D(data, ip1, j, res.x),
                        Accessor2D(data, i, jp1, res.x),
                        Accessor2D(data, ip1, jp1, res.x), fx, fy);
}

template<typename T = Float> inline __bidevice__
void LinearGridSampler2Weights(const vec2f &pt, const vec2f &spacing,
                               const vec2f &origin, const vec2ui &res,
                               vec2ui *indices, T *weights)
{
    T fx(0);
    T fy(0);
    int i = 0, j = 0;
    int ip1 = 0, jp1 = 0;

    vec2f invSpacing(1.0 / spacing.x, 1.0 / spacing.y);
    vec2f nPt = (pt - origin) * invSpacing;
    int iSize = (int)res.x;
    int jSize = (int)res.y;

    GetBarycentric<T>(nPt.x, 0, iSize-1, &i, &fx);
    GetBarycentric<T>(nPt.y, 0, jSize-1, &j, &fy);

    ip1 = Min(i+1, iSize-1);
    jp1 = Min(j+1, jSize-1);

    indices[0] = vec2ui(i, j);
    indices[1] = vec2ui(ip1, j);
    indices[2] = vec2ui(i, jp1);
    indices[3] = vec2ui(ip1, jp1);

    weights[0] = (T(1) - fx) * (T(1) - fy);
    weights[1] = fx * (T(1) - fy);
    weights[2] = (T(1) - fx) * fy;
    weights[3] = fx * fy;
}

inline __bidevice__
Float CubicCatmullRom(Float a, Float b, Float c, Float d, Float x){
    Float xsq = x*x;
    Float xcu = xsq*x;

    Float minV = min(a, min(b, min(c, d)));
    Float maxV = max(a, max(b, max(c, d)));

    Float t = a * (0.0 - 0.5*x + 1.0*xsq - 0.5*xcu) +
              b * (1.0 + 0.0*x - 2.5*xsq + 1.5*xcu) +
              c * (0.0 + 0.5*x + 2.0*xsq - 1.5*xcu) +
              d * (0.0 + 0.0*x - 0.5*xsq + 0.5*xcu);

    return min(max(t, minV), maxV);
}

inline __bidevice__
Float MonotonicCatmullRom(const Float &f0, const Float &f1,
                          const Float &f2, const Float &f3, Float f)
{
    Float d1 = (f2 - f0) * 0.5;
    Float d2 = (f3 - f1) * 0.5;
    Float D1 = f2 - f1;

    if(Absf(D1) < Epsilon){
        d1 = d2 = 0;
    }

    if(Sign(D1) != Sign(d1)){
        d1 = 0;
    }

    if(Sign(D1) != Sign(d2)){
        d2 = 0;
    }

    Float a3 = d1 + d2 - 2 * D1;
    Float a2 = 3 * D1 - 2 * d1 - d2;
    Float a1 = d1;
    Float a0 = f1;

    return a3 * (f * f * f) + a2 * (f * f) + a1 * f + a0;
}

struct LinearInterpolator{
    LinearInterpolator() = default;

    template<typename Fn>
    __bidevice__ Float Interpolate(Float x, Float y, vec2ui resolution, const Fn &fn){
        int iSize = resolution.x, jSize = resolution.y;
        int ix = (int)x;
        int iy = (int)y;
        x -= ix;
        y -= iy;

        Float x00 = fn(ix, iy),
              x10 = fn(Min(ix + 1, iSize - 1), iy),
              x01 = fn(ix, Min(iy + 1, jSize - 1)),
              x11 = fn(Min(ix + 1, iSize - 1), Min(iy + 1, jSize - 1));
        return Mix(Mix(x00, x10, x), Mix(x01, x11, x), y);
    }

    __bidevice__ Float Pulse(Float x){
        return 1;
    }
};

struct MonotonicCatmull{
    MonotonicCatmull() = default;

    template<typename Fn>
    __bidevice__ Float Interpolate(Float x, Float y, vec2ui resolution, const Fn &fn){
        int iSize = resolution.x, jSize = resolution.y;
        int ix = (int)x;
        int iy = (int)y;
        x -= ix;
        y -= iy;

        int is[4] = {
            (int)Max(ix - 1, 0), ix,
            (int)Min(ix + 1, iSize - 1), (int)Min(ix + 2, iSize - 1)
        };

        int js[4] = {
            (int)Max(iy - 1, 0), iy,
            (int)Min(iy + 1, jSize - 1), (int)Min(iy + 2, jSize - 1)
        };

        Float values[4];
        for(int n = 0; n < 4; n++){
            values[n] = CubicCatmullRom(fn(is[0], js[n]),
                                        fn(is[1], js[n]),
                                        fn(is[2], js[n]),
                                        fn(is[3], js[n]), x);
        }

        return CubicCatmullRom(values[0], values[1], values[2], values[3], y);
    }

    __bidevice__ Float Pulse(Float x){
        x = Min(Absf(x), 1.f);
        return 1.f - x * x * (3.0 - 2.0 * x);
    }
};
