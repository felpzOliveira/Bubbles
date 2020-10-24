#pragma once
#include <geometry.h>

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
