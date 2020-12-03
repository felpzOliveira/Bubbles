#pragma once
#include <math.h>
#include <cutil.h>
#include <cfloat>
#include <stdio.h>
#include <stdint.h>

//#define DEBUG
//#define ASSERT_ENABLE
//#define PRINT_TIMER

#ifdef DEBUG
#define DBG_PRINT(...) printf(__VA_ARGS__)
#else
#define DBG_PRINT(...) 
#endif

#define Assure(x) __assert_check_host((x), #x, __FILE__, __LINE__, "Safe exit")
#define AssureA(x, msg) __assert_check_host((x), #x, __FILE__, __LINE__, msg)

#ifdef ASSERT_ENABLE
#define Assert(x) __assert_check((x), #x, __FILE__, __LINE__, NULL)
#define AssertA(x, msg) __assert_check((x), #x, __FILE__, __LINE__, msg)
#else
#define Assert(x)
#define AssertA(x, msg)
#endif

#define AssertAEx(x, msg) AssertA(x, msg)
#define __vec3_strfmtA(v) "%s = [%g %g %g]"
#define __vec3_strfmt(v) "[%g %g %g]"
#define __vec3_args(v) v.x, v.y, v.z
#define __vec3_argsA(v) #v, v.x, v.y, v.z
#define __vec2_strfmtA(v) "%s = [%g %g]"
#define __vec2_argsA(v) #v, v.x, v.y

#define v3fA(v) __vec3_strfmtA(v)
#define v3aA(v) __vec3_argsA(v)
#define v2fA(v) __vec2_strfmtA(v)
#define v2aA(v) __vec2_argsA(v)

#define OneMinusEpsilon 0.99999994f
#define Epsilon 0.0001f
#define Pi 3.14159265358979323846
#define InvPi 0.31830988618379067154
#define Inv2Pi 0.15915494309189533577
#define Inv4Pi 0.07957747154594766788
#define PiOver2 1.57079632679489661923
#define PiOver4 0.78539816339744830961
#define Sqrt2 1.41421356237309504880
#define MachineEpsilon 5.96046e-08
#define WaterDensity 1000.0 // density kg/m3
#define ElementaryQE 1.602176565e-19 // C
#define PermittivityEPS 8.85418782e-12 // vaccum C/V/m
#define ElectronMass 9.10938215e-31 // mass kg
#define AtomicMassUnit 1.660538921e-27 // mass kg
#define KBoltzman 1.380648e-23 // J/K Boltzman constant
#define EvToK (ElementaryQE/KBoltzman) // 1eV 

#define MIN_FLT -FLT_MAX
#define MAX_FLT  FLT_MAX
#define Infinity FLT_MAX
#define SqrtInfinity 3.1622776601683794E+18
#define IntInfinity 2147483646

typedef float Float;
//typedef double Float;

/*
* NOTE: The Inside routines for BoundsN<T> are considering Epsilons 
* to get the idea of a particle that lies on a edge. This is not correct,
* this particle should be marked as outside, however Inside is only used for
* hash sanitizing to check particle hashing is working and the hash
* must choose a bound for a edge particle so re-use Inside with care.
*/


inline __host__
void __assert_check_host(bool v, const char *name, const char *filename,
                         int line, const char *msg)
{
    if(!v){
        if(!msg)
            printf("Assert: %s (%s:%d) : (No message)\n", name, filename, line);
        else
            printf("Assert: %s (%s:%d) : (%s)\n", name, filename, line, msg);
        exit(0);
    }
}

inline __bidevice__ 
void __assert_check(bool v, const char *name, const char *filename,
                    int line, const char *msg)
{
    if(!v){
        int* ptr = nullptr;
        if(!msg)
            printf("Assert: %s (%s:%d) : (No message)\n", name, filename, line);
        else
            printf("Assert: %s (%s:%d) : (%s)\n", name, filename, line, msg);
        *ptr = 10;
    }
}

inline __bidevice__ Float Max(Float a, Float b){ return a < b ? b : a; }
inline __bidevice__ Float Min(Float a, Float b){ return a < b ? a : b; }
inline __bidevice__ Float Absf(Float v){ return v > 0 ? v : -v; }
inline __bidevice__ bool IsNaN(Float v){ return v != v; }
inline __bidevice__ Float Radians(Float deg) { return (Pi / 180) * deg; }
inline __bidevice__ Float Degrees(Float rad) { return (rad * 180 / Pi); }
inline __bidevice__ bool IsZero(Float a){ return Absf(a) < 1e-8; }
inline __bidevice__ bool IsHighpZero(Float a) { return Absf(a) < 1e-22; }
inline __bidevice__ bool IsUnsafeHit(Float a){ return Absf(a) < 1e-6; }
inline __bidevice__ bool IsUnsafeZero(Float a){ return Absf(a) < 1e-6; }
inline __bidevice__ Float Sign(Float a){
    if(IsZero(a)) return 0;
    if(a > 0) return 1;
    return -1;
}

inline __bidevice__ Float LinearRemap(Float x, Float a, Float b, Float A, Float B){
    return A + (B - A) * ((x - a) / (b - a));
}

inline __bidevice__ Float Log2(Float x){
    const Float invLog2 = 1.442695040888963387004650940071;
    return std::log(x) * invLog2;
}

inline __bidevice__ void Swap(Float *a, Float *b){
    Float c = *a; *a = *b; *b = c;
}

inline __bidevice__ bool Quadratic(Float a, Float b, Float c, Float *t0, Float *t1){
    double discr = (double)b * (double)b - 4 * (double)a * (double)c;
    if(discr < 0) return false;
    double root = std::sqrt(discr);
    double q;
    if(b < 0)
        q = -0.5 * (b - root);
    else
        q = -0.5 * (b + root);
    *t0 = q / a;
    *t1 = c / q;
    if(*t0 > *t1) Swap(t0, t1);
    return true;
}

inline __bidevice__ int Log2Int(uint64_t n){
#define S(k) if (n >= (UINT64_C(1) << k)) { i += k; n >>= k; }
    int i = -(n == 0); S(32); S(16); S(8); S(4); S(2); S(1); return i;
#undef S
}

inline __bidevice__ uint32_t FloatToBits(float f){
    uint32_t ui;
    memcpy(&ui, &f, sizeof(float));
    return ui;
}

inline __bidevice__ float BitsToFloat(uint32_t ui){
    float f;
    memcpy(&f, &ui, sizeof(uint32_t));
    return f;
}

inline __bidevice__ uint64_t FloatToBits(double f){
    uint64_t ui;
    memcpy(&ui, &f, sizeof(double));
    return ui;
}

inline __bidevice__ double BitsToFloat(uint64_t ui){
    double f;
    memcpy(&f, &ui, sizeof(uint64_t));
    return f;
}

inline __bidevice__ float NextFloatUp(float v){
    if(std::isinf(v) && v > 0.) return v;
    if(v == -0.f) v = 0.f;
    uint32_t ui = FloatToBits(v);
    if(v >= 0)
        ++ui;
    else
        --ui;
    return BitsToFloat(ui);
}

inline __bidevice__ float NextFloatDown(float v){
    if(std::isinf(v) && v < 0.) return v;
    if(v == 0.f) v = -0.f;
    uint32_t ui = FloatToBits(v);
    if(v > 0)
        --ui;
    else
        ++ui;
    return BitsToFloat(ui);
}

inline __bidevice__ double NextFloatUp(double v, int delta = 1) {
    if(std::isinf(v) && v > 0.) return v;
    if(v == -0.f) v = 0.f;
    uint64_t ui = FloatToBits(v);
    if (v >= 0.)
        ui += delta;
    else
        ui -= delta;
    return BitsToFloat(ui);
}

inline __bidevice__ double NextFloatDown(double v, int delta = 1) {
    if(std::isinf(v) && v < 0.) return v;
    if(v == 0.f) v = -0.f;
    uint64_t ui = FloatToBits(v);
    if(v > 0.)
        ui -= delta;
    else
        ui += delta;
    return BitsToFloat(ui);
}

template <typename T, typename U, typename V> 
inline __bidevice__ T Clamp(T val, U low, V high){
    if(val < low) return low;
    if(val > high) return high;
    return val;
}

template<typename T>
inline __bidevice__ Float gamma(T n){ 
    return ((Float)n * MachineEpsilon) / (1 - (Float)n * MachineEpsilon); 
}

__bidevice__ inline void swap(Float *a, Float *b){
    Float aux = *a; *a = *b; *b = aux;
}

__bidevice__ inline void swap(Float &a, Float &b){
    Float aux = a; a = b; b = aux;
}

class Ray2;
class Ray;


template<typename T> class vec1{
    public:
    T x;
    
    __bidevice__ vec1(){ x = (T)0; }
    __bidevice__ vec1(T a){ x = a; }
    __bidevice__ vec1(T a, T b): x(a){
        Assert(!HasNaN());
    }
    
    __bidevice__ bool IsZeroVector() const{
        return IsZero(x);
    }
    
    __bidevice__ bool HasNaN() const{
        return IsNaN(x);
    }
    
    __bidevice__ T operator[](int i) const{
        Assert(i == 0);
        return x;
    }
    
    __bidevice__ T &operator[](int i){
        Assert(i == 0);
        return x;
    }
    
    __bidevice__ vec1<T> operator/(T f) const{
        Assert(!IsZero(f));
        Float inv = (Float)1 / f;
        return vec1<T>(x * inv);
    }
    
    __bidevice__ vec1<T> &operator/(T f){
        Assert(!IsZero(f));
        Float inv = (Float)1 / f;
        x *= inv;
        return *this;
    }
    
    __bidevice__ vec1<T> operator-(){
        return vec1<T>(-x);
    }
    
    __bidevice__ vec1<T> operator-() const{
        return vec1<T>(-x);
    }
    
    __bidevice__ vec1<T> operator-(const vec1<T> &v) const{
        return vec1(x - v.x);
    }
    
    __bidevice__ vec1<T> operator-(const vec1<T> &v){
        return vec1(x - v.x);
    }
    
    __bidevice__ vec1<T> operator+(const vec1<T> &v) const{
        return vec1<T>(x + v.x);
    }
    
    __bidevice__ vec1<T> operator+=(const vec1<T> &v){
        x += v.x;
        return *this;
    }
    
    __bidevice__ vec1<T> operator*(T s) const{
        return vec1<T>(x * s);
    }
    
    __bidevice__ vec1<T> &operator*=(T s){
        x *= s;
        return *this;
    }
    
    __bidevice__ vec1<T> operator*(const vec1<T> &v) const{
        return vec1<T>(x * v.x);
    }
    
    __bidevice__ vec1<T> &operator*=(const vec1<T> &v){
        x *= v.x;
        return *this;
    }
    
    __bidevice__ Float LengthSquared() const{ return x * x; }
    __bidevice__ Float Length() const{ return sqrt(LengthSquared()); }
    __bidevice__ void PrintSelf() const{
        printf("P = {x : %g}\n", x);
    }
};


template<typename T> class vec2{
    public:
    T x, y;
    
    __bidevice__ vec2(){ x = y = (T)0; }
    __bidevice__ vec2(T a){ x = y = a; }
    __bidevice__ vec2(T a, T b): x(a), y(b){
        Assert(!HasNaN());
    }
    
    __bidevice__ bool IsZeroVector() const{
        return IsZero(x) && IsZero(y);
    }
    
    __bidevice__ bool HasNaN() const{
        return IsNaN(x) || IsNaN(y);
    }
    
    __bidevice__ int Dimensions() const{ return 2; }
    
    __bidevice__ T operator[](int i) const{
        Assert(i >= 0 && i < 2);
        if(i == 0) return x;
        return y;
    }
    
    __bidevice__ T &operator[](int i){
        Assert(i >= 0 && i < 2);
        if(i == 0) return x;
        return y;
    }
    
    __bidevice__ vec2<T> operator/(T f) const{
        Assert(!IsZero(f));
        Float inv = (Float)1 / f;
        return vec2<T>(x * inv, y * inv);
    }
    
    __bidevice__ vec2<T> &operator/(T f){
        Assert(!IsZero(f));
        Float inv = (Float)1 / f;
        x *= inv; y *= inv;
        return *this;
    }
    
    __bidevice__ vec2<T> operator-(){
        return vec2<T>(-x, -y);
    }
    
    __bidevice__ vec2<T> operator-() const{
        return vec2<T>(-x, -y);
    }
    
    __bidevice__ vec2<T> operator-(const vec2<T> &v) const{
        return vec2(x - v.x, y - v.y);
    }
    
    __bidevice__ vec2<T> operator-(const vec2<T> &v){
        return vec2(x - v.x, y - v.y);
    }
    
    __bidevice__ vec2<T> operator+(const vec2<T> &v) const{
        return vec2<T>(x + v.x, y + v.y);
    }
    
    __bidevice__ vec2<T> operator+=(const vec2<T> &v){
        x += v.x; y += v.y;
        return *this;
    }
    
    __bidevice__ vec2<T> operator*(T s) const{
        return vec2<T>(x * s, y * s);
    }
    
    __bidevice__ vec2<T> &operator*=(T s){
        x *= s; y *= s;
        return *this;
    }
    
    __bidevice__ vec2<T> operator*(const vec2<T> &v) const{
        return vec2<T>(x * v.x, y * v.y);
    }
    
    __bidevice__ vec2<T> &operator*=(const vec2<T> &v){
        x *= v.x; y *= v.y;
        return *this;
    }
    
    __bidevice__ vec2<T> Rotate(Float radians) const{
        Float si = std::sin(radians);
        Float co = std::cos(radians);
        return vec2<T>(x * co - y * si, x * si + y * co);
    }
    
    __bidevice__ Float LengthSquared() const{ return x * x + y * y; }
    __bidevice__ Float Length() const{ return sqrt(LengthSquared()); }
    __bidevice__ void PrintSelf() const{
        printf("P = {x : %g, y : %g}\n", x, y);
    }
};

template<typename T> class Normal3;
template<typename T> class vec3{
    public:
    T x, y, z;
    __bidevice__ vec3(){ x = y = z = (T)0; }
    __bidevice__ vec3(T a){ x = y = z = a; }
    __bidevice__ vec3(T a, T b, T c): x(a), y(b), z(c){
        Assert(!HasNaN());
    }
    
    template<typename V> __bidevice__ vec3(Normal3<V> v): x(v.x), y(v.y), z(v.z){
        Assert(!HasNaN());
    }
    
    __bidevice__ bool IsZeroVector() const{
        return IsZero(x) && IsZero(y) && IsZero(z);
    }
    
    __bidevice__ bool HasNaN(){
        return IsNaN(x) || IsNaN(y) || IsNaN(z);
    }
    
    __bidevice__ bool HasNaN() const{
        return IsNaN(x) || IsNaN(y) || IsNaN(z);
    }
    
    __bidevice__ int Dimensions() const{ return 3; }
    
    __bidevice__ T operator[](int i) const{
        Assert(i >= 0 && i < 3);
        if(i == 0) return x;
        if(i == 1) return y;
        return z;
    }
    
    __bidevice__ T &operator[](int i){
        Assert(i >= 0 && i < 3);
        if(i == 0) return x;
        if(i == 1) return y;
        return z;
    }
    
    __bidevice__ vec3<T> operator/(T f) const{
        if(IsZero(f)){
            printf("Warning: Propagating error ( division by 0 with value: %g )\n", f);
        }
        Assert(!IsZero(f));
        Float inv = (Float)1 / f;
        return vec3<T>(x * inv, y * inv, z * inv);
    }
    
    __bidevice__ vec3<T> &operator/(T f){
        Assert(!IsZero(f));
        if(IsZero(f)){
            printf("Warning: Propagating error ( division by 0 with value: %g )\n", f);
        }
        
        Float inv = (Float)1 / f;
        x *= inv; y *= inv; z *= inv;
        return *this;
    }
    
    __bidevice__ vec3<T> operator/(const vec3<T> &v) const{
        Assert(!v.HasNaN());
        Float invx = (Float)1 / v.x;
        Float invy = (Float)1 / v.y;
        Float invz = (Float)1 / v.z;
        return vec3<T>(x * invx, y * invy, z * invz);
    }
    
    __bidevice__ vec3<T> &operator/(const vec3<T> &v){
        Assert(!v.HasNaN());
        Float invx = (Float)1 / v.x;
        Float invy = (Float)1 / v.y;
        Float invz = (Float)1 / v.z;
        x = x * invx; y = y * invy; z = z * invz;
        return *this;
    }
    
    __bidevice__ vec3<T> operator-(){
        return vec3<T>(-x, -y, -z);
    }
    
    __bidevice__ vec3<T> operator-() const{
        return vec3<T>(-x, -y, -z);
    }
    
    __bidevice__ vec3<T> operator-(const vec3<T> &v) const{
        return vec3(x - v.x, y - v.y, z - v.z);
    }
    
    __bidevice__ vec3<T> &operator-=(const vec3<T> &v){
        x -= v.x; y -= v.y; z -= v.z;
        return *this;
    }
    
    __bidevice__ vec3<T> operator+(const vec3<T> &v) const{
        return vec3<T>(x + v.x, y + v.y, z + v.z);
    }
    
    __bidevice__ vec3<T> operator+=(const vec3<T> &v){
        x += v.x; y += v.y; z += v.z;
        return *this;
    }
    
    __bidevice__ vec3<T> operator*(const vec3<T> &v) const{
        return vec3<T>(x * v.x, y * v.y, z * v.z);
    }
    
    __bidevice__ vec3<T> &operator*=(const vec3<T> &v){
        x *= v.x; y *= v.y; z *= v.z;
        return *this;
    }
    
    __bidevice__ vec3<T> operator*(T s) const{
        return vec3<T>(x * s, y * s, z * s);
    }
    
    __bidevice__ vec3<T> &operator*=(T s){
        x *= s; y *= s; z *= s;
        return *this;
    }
    
    __bidevice__ Float LengthSquared() const{ return x * x + y * y + z * z; }
    __bidevice__ Float Length() const{ return sqrt(LengthSquared()); }
    __bidevice__ void PrintSelf() const{
        printf("P = {x : %g, y :  %g, z : %g}\n", x, y, z);
    }
};


template<typename T> class vec4{
    public:
    T x, y, z, w;
    __bidevice__ vec4(){ x = y = z = (T)0; }
    __bidevice__ vec4(T a){ x = y = z = w = a; }
    __bidevice__ vec4(T a, T b, T c, T d): x(a), y(b), z(c), w(d){
        Assert(!HasNaN());
    }
    
    __bidevice__ bool HasNaN(){
        return IsNaN(x) || IsNaN(y) || IsNaN(z) || IsNaN(w);
    }
    
    __bidevice__ bool HasNaN() const{
        return IsNaN(x) || IsNaN(y) || IsNaN(z) || IsNaN(w);
    }
    
    __bidevice__ bool IsZeroVector() const{
        return IsZero(x) && IsZero(y) && IsZero(z) && IsZero(w);
    }
    
    __bidevice__ T operator[](int i) const{
        Assert(i >= 0 && i < 4);
        if(i == 0) return x;
        if(i == 1) return y;
        if(i == 2) return z;
        return w;
    }
    
    __bidevice__ T &operator[](int i){
        Assert(i >= 0 && i < 4);
        if(i == 0) return x;
        if(i == 1) return y;
        if(i == 2) return z;
        return w;
    }
    
    __bidevice__ vec4<T> operator/(T f) const{
        Assert(!IsZero(f));
        if(IsZero(f)){
            printf("Warning: Propagating error ( division by 0 with value: %g )\n", f);
        }
        Float inv = (Float)1 / f;
        return vec4<T>(x * inv, y * inv, z * inv, w * inv);
    }
    
    __bidevice__ vec4<T> &operator/(T f){
        Assert(!IsZero(f));
        if(IsZero(f)){
            printf("Warning: Propagating error ( division by 0 with value: %g )\n", f);
        }
        
        Float inv = (Float)1 / f;
        x *= inv; y *= inv; z *= inv; w *= inv;
        return *this;
    }
    
    __bidevice__ vec4<T> operator/(const vec4<T> &v) const{
        Assert(!v.HasNaN());
        Float invx = (Float)1 / v.x;
        Float invy = (Float)1 / v.y;
        Float invz = (Float)1 / v.z;
        Float invw = (Float)1 / v.w;
        return vec4<T>(x * invx, y * invy, z * invz, w * invw);
    }
    
    __bidevice__ vec4<T> &operator/(const vec4<T> &v){
        Assert(!v.HasNaN());
        Float invx = (Float)1 / v.x;
        Float invy = (Float)1 / v.y;
        Float invz = (Float)1 / v.z;
        Float invw = (Float)1 / v.w;
        x = x * invx; y = y * invy; z = z * invz; w *= invw;
        return *this;
    }
    
    __bidevice__ vec4<T> operator-(){
        return vec4<T>(-x, -y, -z, -w);
    }
    
    __bidevice__ vec4<T> operator-() const{
        return vec4<T>(-x, -y, -z, -w);
    }
    
    __bidevice__ vec4<T> operator-(const vec4<T> &v) const{
        return vec4(x - v.x, y - v.y, z - v.z, w - v.w);
    }
    
    __bidevice__ vec4<T> &operator-=(const vec4<T> &v){
        x -= v.x; y -= v.y; z -= v.z; w -= v.w;
        return *this;
    }
    
    __bidevice__ vec4<T> operator+(const vec4<T> &v) const{
        return vec4<T>(x + v.x, y + v.y, z + v.z, w + v.w);
    }
    
    __bidevice__ vec4<T> operator+=(const vec4<T> &v){
        x += v.x; y += v.y; z += v.z; w += v.w;
        return *this;
    }
    
    __bidevice__ vec4<T> operator*(const vec4<T> &v) const{
        return vec4<T>(x * v.x, y * v.y, z * v.z, w * v.w);
    }
    
    __bidevice__ vec4<T> &operator*=(const vec4<T> &v){
        x *= v.x; y *= v.y; z *= v.z; w *= v.w;
        return *this;
    }
    
    __bidevice__ vec4<T> operator*(T s) const{
        return vec4<T>(x * s, y * s, z * s, w * s);
    }
    
    __bidevice__ vec4<T> &operator*=(T s){
        x *= s; y *= s; z *= s; w *= s;
        return *this;
    }
    
    __bidevice__ Float LengthSquared() const{ return x * x + y * y + z * z + w * w; }
    __bidevice__ Float Length() const{ return sqrt(LengthSquared()); }
    __bidevice__ void PrintSelf() const{
        printf("P = {x : %g, y :  %g, z : %g, w : %g}", x, y, z, w);
    }
};

template<typename T>
inline __bidevice__ bool HasZero(const vec3<T> &v){ 
    return (IsZero(v.x) || (IsZero(v.y)) || (IsZero(v.z)));
}

template<typename T>
inline __bidevice__ Float SquaredDistance(const vec2<T> &p1, const vec2<T> &p2){
    return (p1 - p2).LengthSquared();
}

template<typename T>
inline __bidevice__ Float SquaredDistance(const vec3<T> &p1, const vec3<T> &p2){
    return (p1 - p2).LengthSquared();
}

template<typename T>
inline __bidevice__ Float SquaredDistance(const vec4<T> &p1, const vec4<T> &p2){
    return (p1 - p2).LengthSquared();
}

template<typename T> 
inline __bidevice__ Float Distance(const vec2<T> &p1, const vec2<T> &p2){
    return (p1 - p2).Length();
}

template<typename T> 
inline __bidevice__ Float Distance(const vec3<T> &p1, const vec3<T> &p2){
    return (p1 - p2).Length();
}

template<typename T> 
inline __bidevice__ Float Distance(const vec4<T> &p1, const vec4<T> &p2){
    return (p1 - p2).Length();
}

template<typename T>
inline __bidevice__ vec2<T> Clamp(const vec2<T> &val, const vec2<T> &low, const vec2<T> &high){
    return vec2<T>(Clamp(val.x, low.x, high.x),
                   Clamp(val.y, low.y, high.y));
}

template<typename T>
inline __bidevice__ vec3<T> Clamp(const vec3<T> &val, const vec3<T> &low, const vec3<T> &high){
    return vec3<T>(Clamp(val.x, low.x, high.x),
                   Clamp(val.y, low.y, high.y),
                   Clamp(val.z, low.z, high.z));
}

template<typename T>
inline __bidevice__ vec4<T> Clamp(const vec4<T> &val, const vec4<T> &low, const vec4<T> &high){
    return vec4<T>(Clamp(val.x, low.x, high.x),
                   Clamp(val.y, low.y, high.y),
                   Clamp(val.z, low.z, high.z),
                   Clamp(val.w, low.w, high.w));
}

template<typename T>
inline __bidevice__ vec3<T> Clamp(const vec3<T> &val){
    return Clamp(val, vec3<T>(-1), vec3<T>(1));
}

template<typename T>
inline __bidevice__ vec2<T> Round(const vec2<T> &val){
    return vec2<T>(round(val.x), round(val.y));
}

template<typename T> inline __bidevice__ vec2<T> operator*(T s, vec2<T> &v){ return v * s; }
template<typename T> inline __bidevice__ vec3<T> operator*(T s, vec3<T> &v){ return v * s; }
template<typename T> inline __bidevice__ vec4<T> operator*(T s, vec4<T> &v){ return v * s; }
template<typename T> inline __bidevice__ vec2<T> Abs(const vec2<T> &v){
    return vec2<T>(Absf(v.x), Absf(v.y));
}

template <typename T, typename U> inline __bidevice__ 
vec2<T> operator*(U s, const vec2<T> &v){
    return v * s;
}

template <typename T, typename U> inline __bidevice__ 
vec3<T> operator*(U s, const vec3<T> &v){
    return v * s;
}

template <typename T, typename U> inline __bidevice__ 
vec4<T> operator*(U s, const vec4<T> &v){
    return v * s;
}

template<typename T> inline vec3<T> __bidevice__ Abs(const vec3<T> &v){
    return vec3<T>(Absf(v.x), Absf(v.y), Absf(v.z));
}

template<typename T> inline vec4<T> __bidevice__ Abs(const vec4<T> &v){
    return vec4<T>(Absf(v.x), Absf(v.y), Absf(v.z), Absf(v.w));
}

template<typename T> inline __bidevice__ T Dot(const vec2<T> &v1, const vec2<T> &v2){
    return v1.x * v2.x + v1.y * v2.y;
}

template<typename T> inline __bidevice__ T Dot(const vec3<T> &v1, const vec3<T> &v2){
    return v1.x * v2.x + v1.y * v2.y + v1.z * v2.z;
}

template<typename T> inline __bidevice__ T Dot(const vec4<T> &v1, const vec4<T> &v2){
    return v1.x * v2.x + v1.y * v2.y + v1.z * v2.z + v1.w * v2.w;
}

template<typename T> inline __bidevice__ T Dot2(const vec2<T> &v1){
    return Dot(v1, v1);
}

template<typename T> inline __bidevice__ T Dot2(const vec3<T> &v1){
    return Dot(v1, v1);
}

template<typename T> inline __bidevice__ T Dot2(const vec4<T> &v1){
    return Dot(v1, v1);
}

template<typename T> inline __bidevice__ T AbsDot(const vec3<T> &v1, const vec3<T> &v2){
    return Absf(Dot(v1, v2));
}

template<typename T> inline __bidevice__ T AbsDot(const vec4<T> &v1, const vec4<T> &v2){
    return Absf(Dot(v1, v2));
}

template<typename T> inline __bidevice__ vec3<T> Cross(const vec3<T> &v1, const vec3<T> &v2){
    double v1x = v1.x, v1y = v1.y, v1z = v1.z;
    double v2x = v2.x, v2y = v2.y, v2z = v2.z;
    return vec3<T>((v1y * v2z) - (v1z * v2y),
                   (v1z * v2x) - (v1x * v2z),
                   (v1x * v2y) - (v1y * v2x));
}

template<typename T> inline __bidevice__ vec2<T> Normalize(const vec2<T> &v){
    return v / v.Length();
}

template<typename T> inline __bidevice__ vec3<T> Normalize(const vec3<T> &v){
    return v / v.Length();
}

template<typename T> inline __bidevice__ vec4<T> Normalize(const vec4<T> &v){
    return v / v.Length();
}

inline __bidevice__ Float Sin(const Float &v){
    return std::sin(v);
}

template<typename T> inline __bidevice__ vec2<T> Sin(const vec2<T> &v){
    return vec2<T>(std::sin(v.x), std::sin(v.y));
}

template<typename T> inline __bidevice__ vec3<T> Sin(const vec3<T> &v){
    return vec3<T>(std::sin(v.x), std::sin(v.y), std::sin(v.z));
}

template<typename T> inline __bidevice__ vec4<T> Sin(const vec4<T> &v){
    return vec4<T>(std::sin(v.x), std::sin(v.y), std::sin(v.z), std::sin(v.w));
}

template<typename T> inline __bidevice__ T MinComponent(const vec2<T> &v){
    return Min(v.x, v.y);
}

template<typename T> inline __bidevice__ T MinComponent(const vec3<T> &v){
    return Min(v.x, Min(v.y, v.z));
}

template<typename T> inline __bidevice__ T MaxComponent(const vec2<T> &v){
    return Max(v.x, v.y);
}

template<typename T> inline __bidevice__ T MaxComponent(const vec3<T> &v){
    return Max(v.x, Max(v.y, v.z));
}

template<typename T> inline __bidevice__ int MaxDimension(const vec3<T> &v){
    return (v.x > v.y) ? ((v.x > v.z) ? 0 : 2) : ((v.y > v.z) ? 1 : 2);
}

template<typename T> inline __bidevice__ int MinDimension(const vec3<T> &v){
    return (v.x < v.y) ? ((v.x < v.z) ? 0 : 2) : ((v.y < v.z) ? 1 : 2);
}

template<typename T> inline __bidevice__ vec2<T> Min(const vec2<T> &v1, const vec2<T> &v2){
    return vec2<T>(Min(v1.x, v2.x), Min(v1.y, v2.y));
}

template<typename T> inline __bidevice__ vec3<T> Min(const vec3<T> &v1, const vec3<T> &v2){
    return vec3<T>(Min(v1.x, v2.x), Min(v1.y, v2.y), Min(v1.z, v2.z));
}

template<typename T> inline __bidevice__ vec4<T> Min(const vec4<T> &v1, const vec4<T> &v2){
    return vec4<T>(Min(v1.x, v2.x), Min(v1.y, v2.y), Min(v1.z, v2.z), Min(v1.w, v2.w));
}

template<typename T> inline __bidevice__ vec2<T> Max(const vec2<T> &v1, const vec2<T> &v2){
    return vec2<T>(Max(v1.x, v2.x), Max(v1.y, v2.y));
}

template<typename T> inline __bidevice__ vec3<T> Max(const vec3<T> &v1, const vec3<T> &v2){
    return vec3<T>(Max(v1.x, v2.x), Max(v1.y, v2.y), Max(v1.z, v2.z));
}

template<typename T> inline __bidevice__ vec4<T> Max(const vec4<T> &v1, const vec4<T> &v2){
    return vec4<T>(Max(v1.x, v2.x), Max(v1.y, v2.y), Max(v1.z, v2.z), Max(v1.w, v2.w));
}

template<typename T> inline __bidevice__ vec3<T> Permute(const vec3<T> &v, int x, int y, int z){
    return vec3<T>(v[x], v[y], v[z]);
}

template<typename T> inline __bidevice__ 
vec3<T> Flip(const vec3<T> &p){ return vec3<T>(p.z, p.y, p.x); }

template<typename T> inline __bidevice__ 
vec2<T> Flip(const vec2<T> &p){ return vec2<T>(p.y, p.x); }

template<typename T> inline __bidevice__ void 
CoordinateSystem(const vec3<T> &v1, vec3<T> *v2, vec3<T> *v3){
    if(Absf(v1.x) > Absf(v1.y)){
        Float f = sqrt(v1.x * v1.x + v1.z * v1.z);
        AssertAEx(!IsZero(f), "Zero x component coordinate system generation");
        *v2 = vec3<T>(-v1.z, 0, v1.x) / f;
    }else{
        Float f = sqrt(v1.z * v1.z + v1.y * v1.y);
        AssertAEx(!IsZero(f), "Zero y component coordinate system generation");
        *v2 = vec3<T>(0, v1.z, -v1.y) / f;
    }
    
    *v3 = Cross(v1, *v2);
}

template<typename T> inline __bidevice__
vec3<T> Sqrt(const vec3<T> &v){
    return vec3<T>(std::sqrt(v.x), std::sqrt(v.y), std::sqrt(v.z));
}

template<typename T> inline __bidevice__ 
vec3<T> Pow(const vec3<T> &v, Float val){
    return vec3<T>(std::pow(v.x, val), std::pow(v.y, val), std::pow(v.z, val));
}

template<typename T> inline __bidevice__
vec3<T> Exp(const vec3<T> &v){
    return vec3<T>(std::exp(v.x), std::exp(v.y), std::exp(v.z));
}

typedef vec1<Float> vec1f;
typedef vec1<int> vec1i;
typedef vec1<unsigned int> vec1ui;

typedef vec2<Float> vec2f;
typedef vec2<Float> Point2f;
typedef vec2<int> vec2i;
typedef vec2<int> Point2i;
typedef vec2<unsigned int> vec2ui;

typedef vec3<Float> vec3f;
typedef vec3<Float> Point3f;
typedef vec3<unsigned int> vec3ui;
typedef vec3<int> vec3i;
typedef vec3<int> Point3i;

typedef vec4<Float> vec4f;
typedef vec4<unsigned int> vec4ui;
typedef vec4<int> vec4i;

inline __bidevice__ vec3f Max(vec3f a, vec3f b){
    return vec3f(Max(a.x, b.x), Max(a.y, b.y), Max(a.z, b.z));
}

inline __bidevice__ vec3f Min(vec3f a, vec3f b){
    return vec3f(Min(a.x, b.x), Min(a.y, b.y), Min(a.z, b.z));
}

template<typename T> inline __bidevice__ T Mix(const T &p0, const T &p1, T t){
    return (1 - t) * p0 + t * p1;
}

template<typename T, typename Q> inline __bidevice__ T Lerp(const T &p0, const T &p1, Q t){
    return (1 - t) * p0 + t * p1;
}

inline __bidevice__ Float Floor(const Float &v){
    return std::floor(v);
}

template<typename T> inline __bidevice__ vec2<T> Floor(const vec2<T> &v){
    return vec2<T>(std::floor(v.x), std::floor(v.y));
}

template<typename T> inline __bidevice__ vec3<T> Floor(const vec3<T> &v){
    return vec3<T>(std::floor(v.x), std::floor(v.y), std::floor(v.z));
}

template<typename T> inline __bidevice__ T Fract(T val){
    return val - Floor(val);
}

template <typename T> inline __bidevice__ T Mod(T a, T b) {
    T result = a - (a / b) * b;
    return (T)((result < 0) ? result + b : result);
}

template<typename T> class Normal3{
    public:
    T x, y, z;
    __bidevice__ Normal3(){ x = y = z = (T)0; }
    __bidevice__ Normal3(T a){ x = y = z = a; }
    __bidevice__ Normal3(T a, T b, T c): x(a), y(b), z(c)
    {
        Assert(!HasNaN());
    }
    
    template<typename U> __bidevice__ Normal3(const vec3<U> &v):x(v.x), y(v.y), z(v.z){
        Assert(!HasNaN());
    }
    
    __bidevice__ bool HasNaN(){
        return IsNaN(x) || IsNaN(y) || IsNaN(z);
    }
    
    __bidevice__ bool HasNaN() const{
        return IsNaN(x) || IsNaN(y) || IsNaN(z);
    }
    
    __bidevice__ Normal3<T> operator-() const { return Normal3(-x, -y, -z); }
    
    __bidevice__ Normal3<T> operator+(const Normal3<T> &n) const {
        Assert(!n.HasNaN());
        return Normal3<T>(x + n.x, y + n.y, z + n.z);
    }
    
    __bidevice__ Normal3<T> &operator+=(const Normal3<T> &n) {
        Assert(!n.HasNaN());
        x += n.x; y += n.y; z += n.z;
        return *this;
    }
    __bidevice__ Normal3<T> operator-(const Normal3<T> &n) const {
        Assert(!n.HasNaN());
        return Normal3<T>(x - n.x, y - n.y, z - n.z);
    }
    
    __bidevice__ Normal3<T> &operator-=(const Normal3<T> &n) {
        Assert(!n.HasNaN());
        x -= n.x; y -= n.y; z -= n.z;
        return *this;
    }
    
    template <typename U> __bidevice__ Normal3<T> operator*(U f) const {
        return Normal3<T>(f * x, f * y, f * z);
    }
    
    template <typename U> __bidevice__ Normal3<T> &operator*=(U f) {
        x *= f; y *= f; z *= f;
        return *this;
    }
    template <typename U> __bidevice__ Normal3<T> operator/(U f) const {
        Assert(!IsZero(f));
        Float inv = (Float)1 / f;
        return Normal3<T>(x * inv, y * inv, z * inv);
    }
    
    template <typename U> __bidevice__ Normal3<T> &operator/=(U f) {
        Assert(!IsZero(f));
        Float inv = (Float)1 / f;
        x *= inv; y *= inv; z *= inv;
        return *this;
    }
    
    __bidevice__ explicit Normal3<T>(const vec3<T> &v) : x(v.x), y(v.y), z(v.z) {}
    
    __bidevice__ bool operator==(const Normal3<T> &n) const {
        return x == n.x && y == n.y && z == n.z;
    }
    
    __bidevice__ bool operator!=(const Normal3<T> &n) const {
        return x != n.x || y != n.y || z != n.z;
    }
    
    __bidevice__ T operator[](int i) const {
        Assert(i >= 0 && i < 3);
        if (i == 0) return x;
        if (i == 1) return y;
        return z;
    }
    
    __bidevice__ T &operator[](int i) {
        Assert(i >= 0 && i < 3);
        if (i == 0) return x;
        if (i == 1) return y;
        return z;
    }
    
    __bidevice__ Float LengthSquared() const { return x * x + y * y + z * z; }
    __bidevice__ Float Length() const { return sqrt(LengthSquared()); }
};

template <typename T, typename U> inline __bidevice__ 
Normal3<T> operator*(U s, const Normal3<T> &n){
    return n * s;
}

template <typename T> inline __bidevice__ Normal3<T> Normalize(const Normal3<T> &n) {
    return n / n.Length();
}

template <typename T>
inline __bidevice__ Normal3<T> Faceforward(const Normal3<T> &n, const vec3<T> &v) {
    return (Dot(n, v) < 0.f) ? -n : n;
}

template <typename T>
inline __bidevice__ Normal3<T> Faceforward(const Normal3<T> &n, const Normal3<T> &n2) {
    return (Dot(n, n2) < 0.f) ? -n : n;
}

template <typename T>
inline __bidevice__ vec3<T> Faceforward(const vec3<T> &v, const vec3<T> &v2) {
    return (Dot(v, v2) < 0.f) ? -v : v;
}

template <typename T>
inline __bidevice__ vec3<T> Faceforward(const vec3<T> &v, const Normal3<T> &n2) {
    return (Dot(v, n2) < 0.f) ? -v : v;
}

template <typename T>
inline __bidevice__ Normal3<T> Abs(const Normal3<T> &v) {
    return Normal3<T>(Absf(v.x), Absf(v.y), Absf(v.z));
}

template <typename T> inline __bidevice__ T Dot(const Normal3<T> &n1, const Normal3<T> &n2) {
    return n1.x * n2.x + n1.y * n2.y + n1.z * n2.z;
}

template <typename T> inline __bidevice__ T Dot(const Normal3<T> &n1, const vec3<T> &v) {
    return n1.x * v.x + n1.y * v.y + n1.z * v.z;
}

template <typename T> inline __bidevice__ T AbsDot(const Normal3<T> &n1, const vec3<T> &v2) {
    return Absf(n1.x * v2.x + n1.y * v2.y + n1.z * v2.z);
}

template<typename T> inline __bidevice__ vec3<T> ToVec3(const Normal3<T> &n){
    return vec3<T>(n.x, n.y, n.z);
}

template<typename T> inline __bidevice__ Normal3<T> toNormal3(const vec3<T> &v){
    return Normal3<T>(v.x, v.y, v.z);
}

typedef Normal3<Float> Normal3f;

template<typename T>
class Bounds1{
    public:
    T pMin, pMax;
    __bidevice__ Bounds1(){
        pMin = T(-FLT_MAX);
        pMax = T(FLT_MAX);
    }
    
    __bidevice__ explicit Bounds1(const T &p): pMin(p), pMax(p) {}
    __bidevice__ Bounds1(const T &p1, const T &p2)
        : pMin(Min(p1, p2)), pMax(Max(p1, p2)) {}
    
    __bidevice__ const T &operator[](int i) const{
        Assert(i == 0 || i == 1);
        return (i == 0) ? pMin : pMax;
        
    }
    __bidevice__ T &operator[](int i){
        Assert(i == 0 || i == 1);
        return (i == 0) ? pMin : pMax;
    }
    
    __bidevice__ T LengthAt(int i, int axis) const{
        Assert(axis == 0);
        return (i == 0) ? pMin : pMax;
    }
    
    __bidevice__ bool operator==(const Bounds1<T> &b) const{
        return b.pMin == pMin && b.pMax == pMax;
    }
    
    __bidevice__ T Center() const{
        return (pMin + pMax) * 0.5;
    }
    
    __bidevice__ T ExtentOn(int i) const{
        Assert(i == 0);
        return Absf(pMax - pMin);
    }
    
    __bidevice__ int MaximumExtent() const{
        return 0;
    }
    
    __bidevice__ int MinimumExtent() const{
        return 0;
    }
    
    __bidevice__ T Offset(const T &p) const{
        T o = p - pMin;
        if (pMax > pMin) o /= pMax - pMin;
        return o;
    }
    
    __bidevice__ T MinDistance(const T &p) const{
        Float x0 = Absf(pMin - p), x1 = Absf(pMax - p);
        return Min(x0, x1);
    }
    
    __bidevice__ void PrintSelf() const{
        printf("pMin = {x : %g} pMax = {x : %g}", pMin, pMax);
    }
};

template <typename T>
class Bounds2 {
    public:
    vec2<T> pMin, pMax;
    
    __bidevice__ Bounds2(){
        T minNum = FLT_MIN;
        T maxNum = FLT_MAX;
        pMin = vec2<T>(maxNum, maxNum);
        pMax = vec2<T>(minNum, minNum);
    }
    
    __bidevice__ explicit Bounds2(const vec2<T> &p) : pMin(p), pMax(p) {}
    __bidevice__ Bounds2(const vec2<T> &p1, const vec2<T> &p2)
        : pMin(Min(p1.x, p2.x), Min(p1.y, p2.y)), pMax(Max(p1.x, p2.x), Max(p1.y, p2.y)) {}
    
    __bidevice__ const vec2<T> &operator[](int i) const;
    __bidevice__ vec2<T> &operator[](int i);
    __bidevice__ bool operator==(const Bounds2<T> &b) const{
        return b.pMin == pMin && b.pMax == pMax;
    }
    
    __bidevice__ bool operator!=(const Bounds2<T> &b) const{
        return b.pMin != pMin || b.pMax != pMax;
    }
    
    __bidevice__ vec2<T> Corner(int corner) const{
        Assert(corner >= 0 && corner < 4);
        return vec2<T>((*this)[(corner & 1)].x,
                       (*this)[(corner & 2) ? 1 : 0].y);
    }
    
    __bidevice__ void Expand(Float d){
        pMin -= vec2<T>(Absf(d));
        pMax += vec2<T>(Absf(d));
    }
    
    __bidevice__ void Reduce(Float d){
        pMin += vec2<T>(Absf(d));
        pMax -= vec2<T>(Absf(d));
    }
    
    __bidevice__ T LengthAt(int i, int axis) const{
        Assert(axis == 0 || axis == 1);
        return (i == 0) ? pMin[axis] : pMax[axis];
    }
    
    __bidevice__ vec2<T> Diagonal() const { return pMax - pMin; }
    __bidevice__ T SurfaceArea() const{
        vec2<T> d = Diagonal();
        return (d.x * d.y);
    }
    
    __bidevice__ T Volume() const{
        printf("Warning: Called for volume on 2D surface\n");
        return 0;
    }
    
    __bidevice__ vec2<T> Center() const{
        return (pMin + pMax) * 0.5;
    }
    
    __bidevice__ T ExtentOn(int i) const{
        Assert(i >= 0 && i < 2);
        if(i == 0) return Absf(pMax.x - pMin.x);
        return Absf(pMax.y - pMin.y);
    }
    
    __bidevice__ int MaximumExtent() const{
        vec2<T> d = Diagonal();
        if (d.x > d.y)
            return 0;
        else
            return 1;
    }
    
    __bidevice__ int MinimumExtent() const{
        vec2<T> d = Diagonal();
        if (d.x > d.y)
            return 1;
        else
            return 0;
    }
    
    __bidevice__ vec2<T> Offset(const vec2<T> &p) const{
        vec2<T> o = p - pMin;
        if (pMax.x > pMin.x) o.x /= pMax.x - pMin.x;
        if (pMax.y > pMin.y) o.y /= pMax.y - pMin.y;
        return o;
    }
    
    __bidevice__ void BoundingSphere(vec2<T> *center, Float *radius) const{
        *center = (pMin + pMax) / 2;
        *radius = Inside(*center, *this) ? Distance(*center, pMax) : 0;
    }
    
    __bidevice__ vec2<T> MinDistance(const vec2<T> &p) const{
        Float x0 = Absf(pMin.x - p.x), x1 = Absf(pMax.x - p.x);
        Float y0 = Absf(pMin.y - p.y), y1 = Absf(pMax.y - p.y);
        return vec2<T>(Min(x0, x1), Min(y0, y1));
    }
    
    __bidevice__ bool Intersect(const Ray2 &ray, Float *tHit0=nullptr,
                                Float *tHit1=nullptr) const;
    
    template <typename U> __bidevice__ explicit operator Bounds2<U>() const{
        return Bounds2<U>((vec2<U>)pMin, (vec2<U>)pMax);
    }
    
    __bidevice__ vec2<T> Clamped(const vec2<T> &point) const{
        return Clamp(point, pMin, pMax);
    }
    
    __bidevice__ void PrintSelf() const{
        printf("pMin = {x : %g, y : %g} pMax = {x : %g, y : %g}",
               pMin.x, pMin.y, pMax.x, pMax.y);
    }
};

template <typename T>
class Bounds3 {
    public:
    vec3<T> pMin, pMax;
    
    __bidevice__ Bounds3(){
        T minNum = FLT_MIN;
        T maxNum = FLT_MAX;
        pMin = vec3<T>(maxNum, maxNum, maxNum);
        pMax = vec3<T>(minNum, minNum, minNum);
    }
    
    __bidevice__ explicit Bounds3(const vec3<T> &p) : pMin(p), pMax(p) {}
    __bidevice__ Bounds3(const vec3<T> &p1, const vec3<T> &p2)
        : pMin(Min(p1.x, p2.x), Min(p1.y, p2.y), Min(p1.z, p2.z)),
    pMax(Max(p1.x, p2.x), Max(p1.y, p2.y), Max(p1.z, p2.z)) {}
    
    __bidevice__ const vec3<T> &operator[](int i) const;
    __bidevice__ vec3<T> &operator[](int i);
    __bidevice__ bool operator==(const Bounds3<T> &b) const{
        return b.pMin == pMin && b.pMax == pMax;
    }
    
    __bidevice__ bool operator!=(const Bounds3<T> &b) const{
        return b.pMin != pMin || b.pMax != pMax;
    }
    
    __bidevice__ void Expand(Float d){
        pMin -= vec3<T>(Absf(d));
        pMax += vec3<T>(Absf(d));
    }
    
    __bidevice__ void Reduce(Float d){
        pMin += vec3<T>(Absf(d));
        pMax -= vec3<T>(Absf(d));
    }
    
    __bidevice__ vec3<T> Corner(int corner) const{
        Assert(corner >= 0 && corner < 8);
        return vec3<T>((*this)[(corner & 1)].x,
                       (*this)[(corner & 2) ? 1 : 0].y,
                       (*this)[(corner & 4) ? 1 : 0].z);
    }
    
    __bidevice__ T LengthAt(int i, int axis) const{
        Assert(axis == 0 || axis == 1 || axis == 2);
        return (i == 0) ? pMin[axis] : pMax[axis];
    }
    
    __bidevice__ vec3<T> Diagonal() const { return pMax - pMin; }
    __bidevice__ T SurfaceArea() const{
        vec3<T> d = Diagonal();
        return 2 * (d.x * d.y + d.x * d.z + d.y * d.z);
    }
    
    __bidevice__ vec3<T> Center() const{
        return (pMin + pMax) * 0.5;
    }
    
    __bidevice__ T Volume() const{
        vec3<T> d = Diagonal();
        return d.x * d.y * d.z;
    }
    
    __bidevice__ T ExtentOn(int i) const{
        Assert(i >= 0 && i < 3);
        if(i == 0) return Absf(pMax.x - pMin.x);
        if(i == 1) return Absf(pMax.y - pMin.y);
        return Absf(pMax.z - pMin.z);
    }
    
    __bidevice__ int MaximumExtent() const{
        vec3<T> d = Diagonal();
        if (d.x > d.y && d.x > d.z)
            return 0;
        else if (d.y > d.z)
            return 1;
        else
            return 2;
    }
    
    __bidevice__ int MinimumExtent() const{
        vec3<T> d = Diagonal();
        if (d.x > d.z && d.y > d.z)
            return 2;
        else if (d.x > d.y)
            return 1;
        else
            return 0;
    }
    
    __bidevice__ vec3<T> Offset(const vec3<T> &p) const{
        vec3<T> o = p - pMin;
        if (pMax.x > pMin.x) o.x /= pMax.x - pMin.x;
        if (pMax.y > pMin.y) o.y /= pMax.y - pMin.y;
        if (pMax.z > pMin.z) o.z /= pMax.z - pMin.z;
        return o;
    }
    
    __bidevice__ vec3<T> Clamped(const vec3<T> &point) const{
        return Clamp(point, pMin, pMax);
    }
    
    __bidevice__ void BoundingSphere(vec3<T> *center, Float *radius) const{
        *center = (pMin + pMax) / 2;
        *radius = Inside(*center, *this) ? Distance(*center, pMax) : 0;
    }
    
    __bidevice__ vec3<T> MinDistance(const vec3<T> &p) const{
        Float x0 = Absf(pMin.x - p.x), x1 = Absf(pMax.x - p.x);
        Float y0 = Absf(pMin.y - p.y), y1 = Absf(pMax.y - p.y);
        Float z0 = Absf(pMin.z - p.z), z1 = Absf(pMax.z - p.z);
        return vec3<T>(Min(x0, x1), Min(y0, y1), Min(z0, z1));
    }
    
    __bidevice__ bool Intersect(const Ray &ray, Float *tHit0=nullptr, 
                                Float *tHit1=nullptr) const;
    
    template <typename U> __bidevice__ explicit operator Bounds3<U>() const{
        return Bounds3<U>((vec3<U>)pMin, (vec3<U>)pMax);
    }
    
    __bidevice__ void PrintSelf() const{
        printf("pMin = {x : %g, y : %g, z : %g} pMax = {x : %g, y : %g, z : %g}",
               pMin.x, pMin.y, pMin.z, pMax.x, pMax.y, pMax.z);
    }
};

typedef Bounds1<Float> Bounds1f;
typedef Bounds1<int> Bounds1i;
typedef Bounds2<Float> Bounds2f;
typedef Bounds2<int> Bounds2i;
typedef Bounds3<Float> Bounds3f;
typedef Bounds3<int> Bounds3i;

template <typename T> inline __bidevice__ 
vec2<T> &Bounds2<T>::operator[](int i){
    Assert(i == 0 || i == 1);
    return (i == 0) ? pMin : pMax;
}

template <typename T> inline __bidevice__
Bounds2<T> Union(const Bounds2<T> &b, const vec2<T> &p){
    Bounds2<T> ret;
    ret.pMin = Min(b.pMin, p);
    ret.pMax = Max(b.pMax, p);
    return ret;
}

template <typename T> inline __bidevice__
Bounds2<T> Union(const Bounds2<T> &b1, const Bounds2<T> &b2){
    Bounds2<T> ret;
    ret.pMin = Min(b1.pMin, b2.pMin);
    ret.pMax = Max(b1.pMax, b2.pMax);
    return ret;
}

template <typename T> inline __bidevice__ 
Bounds2<T> Intersect(const Bounds2<T> &b1, const Bounds2<T> &b2){
    Bounds2<T> ret;
    ret.pMin = Max(b1.pMin, b2.pMin);
    ret.pMax = Min(b1.pMax, b2.pMax);
    return ret;
}

template <typename T> inline __bidevice__ 
bool Overlaps(const Bounds2<T> &b1, const Bounds2<T> &b2){
    bool x = (b1.pMax.x >= b2.pMin.x) && (b1.pMin.x <= b2.pMax.x);
    bool y = (b1.pMax.y >= b2.pMin.y) && (b1.pMin.y <= b2.pMax.y);
    return (x && y);
}

template <typename T> inline __bidevice__
bool Inside(const vec2<T> &p, const Bounds2<T> &b){
    bool rv = (p.x >= b.pMin.x && p.x <= b.pMax.x && 
               p.y >= b.pMin.y && p.y <= b.pMax.y);
    if(!rv){
        vec2<T> oE = b.MinDistance(p);
        rv = IsUnsafeZero(oE.x) || IsUnsafeZero(oE.y);
    }
    
    return rv;
}

template <typename T> inline __bidevice__
bool InsideExclusive(const vec2<T> &p, const Bounds2<T> &b){
    return (p.x >= b.pMin.x && p.x < b.pMax.x && 
            p.y >= b.pMin.y && p.y < b.pMax.y);
}

template <typename T, typename U> inline __bidevice__ 
Bounds2<T> Expand(const Bounds2<T> &b, U delta){
    return Bounds2<T>(b.pMin - vec2<T>(delta, delta),
                      b.pMax + vec2<T>(delta, delta));
}


template <typename T> inline __bidevice__ 
vec3<T> &Bounds3<T>::operator[](int i){
    Assert(i == 0 || i == 1);
    return (i == 0) ? pMin : pMax;
}

template <typename T> inline __bidevice__
Bounds1<T> Union(const Bounds1<T> &b, const T &p){
    Bounds1<T> ret;
    ret.pMin = Min(b.pMin, p);
    ret.pMax = Max(b.pMax, p);
    return ret;
}

template <typename T> inline __bidevice__
Bounds3<T> Union(const Bounds3<T> &b, const vec3<T> &p){
    Bounds3<T> ret;
    ret.pMin = Min(b.pMin, p);
    ret.pMax = Max(b.pMax, p);
    return ret;
}

template <typename T> inline __bidevice__
Bounds3<T> Union(const Bounds3<T> &b1, const Bounds3<T> &b2){
    Bounds3<T> ret;
    ret.pMin = Min(b1.pMin, b2.pMin);
    ret.pMax = Max(b1.pMax, b2.pMax);
    return ret;
}

template <typename T> inline __bidevice__ 
Bounds3<T> Intersect(const Bounds3<T> &b1, const Bounds3<T> &b2){
    Bounds3<T> ret;
    ret.pMin = Max(b1.pMin, b2.pMin);
    ret.pMax = Min(b1.pMax, b2.pMax);
    return ret;
}

template <typename T> inline __bidevice__ 
bool Overlaps(const Bounds3<T> &b1, const Bounds3<T> &b2){
    bool x = (b1.pMax.x >= b2.pMin.x) && (b1.pMin.x <= b2.pMax.x);
    bool y = (b1.pMax.y >= b2.pMin.y) && (b1.pMin.y <= b2.pMax.y);
    bool z = (b1.pMax.z >= b2.pMin.z) && (b1.pMin.z <= b2.pMax.z);
    return (x && y && z);
}

template <typename T> inline __bidevice__
bool Inside(const T &p, const Bounds1<T> &b){
    bool rv = (p >= b.pMin && p <= b.pMin);
    if(!rv){
        T oE = b.MinDistance(p);
        rv = IsUnsafeZero(oE);
    }
    
    return rv;
}

template <typename T> inline __bidevice__
bool Inside(const vec3<T> &p, const Bounds3<T> &b){
    bool rv = (p.x >= b.pMin.x && p.x <= b.pMax.x && p.y >= b.pMin.y &&
               p.y <= b.pMax.y && p.z >= b.pMin.z && p.z <= b.pMax.z);
    if(!rv){
        vec3<T> oE = b.MinDistance(p);
        rv = IsUnsafeZero(oE.x) || IsUnsafeZero(oE.y) || IsUnsafeZero(oE.z);
    }
    
    return rv;
}

template <typename T> inline __bidevice__
bool InsideExclusive(const vec3<T> &p, const Bounds3<T> &b){
    return (p.x >= b.pMin.x && p.x < b.pMax.x && p.y >= b.pMin.y &&
            p.y < b.pMax.y && p.z >= b.pMin.z && p.z < b.pMax.z);
}

template <typename T, typename U> inline __bidevice__ 
Bounds3<T> Expand(const Bounds3<T> &b, U delta){
    return Bounds3<T>(b.pMin - vec3<T>(delta, delta, delta),
                      b.pMax + vec3<T>(delta, delta, delta));
}

class Ray2{
    public:
    vec2f o, d;
    Float tMax;
    __bidevice__ Ray2(const vec2f &origin, const vec2f &direction, Float maxt=Infinity){
        o = origin; d = direction; tMax = maxt;
    }
    
    __bidevice__ vec2f operator()(Float t){ return o + t * d; }
    __bidevice__ vec2f operator()(Float t) const{ return o + t * d; }
};

class Ray{
    public:
    vec3f o, d;
    mutable Float tMax;
    __bidevice__ Ray(const vec3f &origin, const vec3f &direction, Float maxt=Infinity){
        o = origin; d = direction; tMax = maxt;
    }
    
    __bidevice__ vec3f operator()(Float t){ return o + t * d; }
    __bidevice__ vec3f operator()(Float t) const{ return o + t * d; }
};

inline __bidevice__ 
vec3f OffsetRayOrigin(const vec3f &p, const vec3f &pError,
                      const Normal3f &n, const vec3f &w)
{
    Float d = Dot(Abs(n), pError) + 0.0001;
    vec3f offset = d * ToVec3(n);
    if(Dot(w, ToVec3(n)) < 0) offset = -offset;
    vec3f po = p + offset;
    for(int i = 0; i < 3; ++i){
        if(offset[i] > 0)
            po[i] = NextFloatUp(po[i]);
        else if(offset[i] < 0)
            po[i] = NextFloatDown(po[i]);
    }
    
    return po;
}

template<typename T> inline  __bidevice__ 
bool Bounds2<T>::Intersect(const Ray2 &ray, Float *tHit0, Float *tHit1) const{
    Float t0 = 0, t1 = ray.tMax;
    for(int i = 0; i < 2; i++){
        Float invRayDir = 1 / ray.d[i];
        Float tNear = (pMin[i] - ray.o[i]) * invRayDir;
        Float tFar  = (pMax[i] - ray.o[i]) * invRayDir;
        if(tNear > tFar) swap(tNear, tFar);
        
        tFar *= 1 + 2 * gamma(3);
        t0 = tNear > t0 ? tNear : t0;
        t1 = tFar < t1 ? tFar : t1;
        if(t0 > t1) return false;
    }
    
    if(tHit0) *tHit0 = t0;
    if(tHit1) *tHit1 = t1;
    return true;
}

template<typename T> inline  __bidevice__ 
bool Bounds3<T>::Intersect(const Ray &ray, Float *tHit0, Float *tHit1) const{
    Float t0 = 0, t1 = ray.tMax;
    for(int i = 0; i < 3; i++){
        Float invRayDir = 1 / ray.d[i];
        Float tNear = (pMin[i] - ray.o[i]) * invRayDir;
        Float tFar  = (pMax[i] - ray.o[i]) * invRayDir;
        if(tNear > tFar) swap(tNear, tFar);
        
        tFar *= 1 + 2 * gamma(3);
        t0 = tNear > t0 ? tNear : t0;
        t1 = tFar < t1 ? tFar : t1;
        if(t0 > t1) return false;
    }
    
    if(tHit0) *tHit0 = t0;
    if(tHit1) *tHit1 = t1;
    return true;
}

inline __host__ Float rand_float(){
    return rand() / (RAND_MAX+1.f);
}

typedef struct{
    Point3f *p;
    Normal3f *n;
    vec3f *s;
    Point2f *uv;
    Point3i *indices;
    int nTriangles, nVertices;
    int nUvs, nNormals;
    AllocatorType allocator;
}ParsedMesh;