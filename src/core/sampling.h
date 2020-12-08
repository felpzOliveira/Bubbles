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