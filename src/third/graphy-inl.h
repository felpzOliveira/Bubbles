/* date = April 25th 2022 19:29 */
#pragma once
#include <iostream>
#include <vector>
#include <string>
#include <memory>
#include <stdint.h>

/////////////////////////////// X11
#include <X11/Xlib.h>
#include <X11/Xutil.h>
///////////////////////////////////

#define GMIN(a, b) ((a) < (b) ? (a) : (b))
#define GMAX(a, b) ((a) > (b) ? (a) : (b))
#define GABS(a) ((a) < 0 ? -(a) : (a))

typedef float _GraphyFPrecision;

typedef _GraphyFPrecision _Gfloat;

template<typename T> class _GraphyVec2{
    public:
    T x, y;
    _GraphyVec2(){ x = y = (T)0; }
    _GraphyVec2(T a){ x = y = a; }
    _GraphyVec2(T a, T b): x(a), y(b){}

    template<typename U>
    _GraphyVec2(_GraphyVec2<U> v): x(v.x), y(v.y){}

    T operator[](int i) const{
        if(i == 0) return x;
        return y;
    }

    T &operator[](int i){
        if(i == 0) return x;
        return y;
    }

    _GraphyVec2<T> operator/(T f) const{
        _GraphyFPrecision inv = (_GraphyFPrecision)1 / f;
        return _GraphyVec2<T>(x * inv, y * inv);
    }

    _GraphyVec2<T> &operator/(T f){
        _GraphyFPrecision inv = (_GraphyFPrecision)1 / f;
        x *= inv; y *= inv;
        return *this;
    }

    _GraphyVec2<T> operator/(const _GraphyVec2<T> &v) const{
        _GraphyFPrecision invx = (_GraphyFPrecision)1 / v.x;
        _GraphyFPrecision invy = (_GraphyFPrecision)1 / v.y;
        return _GraphyVec2<T>(x * invx, y * invy);
    }

    _GraphyVec2<T> &operator/(const _GraphyVec2<T> &v){
        _GraphyFPrecision invx = (_GraphyFPrecision)1 / v.x;
        _GraphyFPrecision invy = (_GraphyFPrecision)1 / v.y;
        x = x * invx; y = y * invy;
        return *this;
    }

    _GraphyVec2<T> operator-(){
        return _GraphyVec2<T>(-x, -y);
    }

    _GraphyVec2<T> operator-() const{
        return _GraphyVec2<T>(-x, -y);
    }

    _GraphyVec2<T> operator-(const _GraphyVec2<T> &v) const{
        return _GraphyVec2(x - v.x, y - v.y);
    }

    _GraphyVec2<T> &operator-=(const _GraphyVec2<T> &v){
        x -= v.x; y -= v.y;
        return *this;
    }

    _GraphyVec2<T> operator+(const _GraphyVec2<T> &v) const{
        return _GraphyVec2<T>(x + v.x, y + v.y);
    }

    _GraphyVec2<T> operator+=(const _GraphyVec2<T> &v){
        x += v.x; y += v.y;
        return *this;
    }

    _GraphyVec2<T> operator*(const _GraphyVec2<T> &v) const{
        return _GraphyVec2<T>(x * v.x, y * v.y);
    }

    _GraphyVec2<T> &operator*=(const _GraphyVec2<T> &v){
        x *= v.x; y *= v.y;
        return *this;
    }

    _GraphyVec2<T> operator*(T s) const{
        return _GraphyVec2<T>(x * s, y * s);
    }

    _GraphyVec2<T> &operator*=(T s){
        x *= s; y *= s;
        return *this;
    }

    _GraphyFPrecision LengthSquared() const{ return x * x + y * y; }
    _GraphyFPrecision Length() const{ return sqrt(LengthSquared()); }
    void PrintSelf() const{
        printf("P = {x : %g, y :  %g}", x, y);
    }
};

template<typename T> class _GraphyVec4{
    public:
    T x, y, z, w;
    _GraphyVec4(){ x = y = z = w = (T)0; }
    _GraphyVec4(T a){ x = y = z = w = a; }
    _GraphyVec4(T a, T b, T c, T d): x(a), y(b), z(c), w(d){}

    template<typename U>
    _GraphyVec4(_GraphyVec4<U> v): x(v.x), y(v.y), z(v.z), w(v.w){}

    T operator[](int i) const{
        if(i == 0) return x;
        if(i == 1) return y;
        if(i == 2) return z;
        return w;
    }

    T &operator[](int i){
        if(i == 0) return x;
        if(i == 1) return y;
        if(i == 2) return z;
        return w;
    }

    _GraphyVec4<T> operator/(T f) const{
        _GraphyFPrecision inv = (_GraphyFPrecision)1 / f;
        return _GraphyVec4<T>(x * inv, y * inv, z * inv, w * inv);
    }

    _GraphyVec4<T> &operator/(T f){
        _GraphyFPrecision inv = (_GraphyFPrecision)1 / f;
        x *= inv; y *= inv; z *= inv; w *= inv;
        return *this;
    }

    _GraphyVec4<T> operator/(const _GraphyVec4<T> &v) const{
        _GraphyFPrecision invx = (_GraphyFPrecision)1 / v.x;
        _GraphyFPrecision invy = (_GraphyFPrecision)1 / v.y;
        _GraphyFPrecision invz = (_GraphyFPrecision)1 / v.z;
        _GraphyFPrecision invw = (_GraphyFPrecision)1 / v.w;
        return _GraphyVec4<T>(x * invx, y * invy, z * invz, w * invw);
    }

    _GraphyVec4<T> &operator/(const _GraphyVec4<T> &v){
        _GraphyFPrecision invx = (_GraphyFPrecision)1 / v.x;
        _GraphyFPrecision invy = (_GraphyFPrecision)1 / v.y;
        _GraphyFPrecision invz = (_GraphyFPrecision)1 / v.z;
        _GraphyFPrecision invw = (_GraphyFPrecision)1 / v.w;
        x = x * invx; y = y * invy; z = z * invz; w *= invw;
        return *this;
    }

    _GraphyVec4<T> operator-(){
        return _GraphyVec4<T>(-x, -y, -z, -w);
    }

    _GraphyVec4<T> operator-() const{
        return _GraphyVec4<T>(-x, -y, -z, -w);
    }

    _GraphyVec4<T> operator-(const _GraphyVec4<T> &v) const{
        return _GraphyVec4(x - v.x, y - v.y, z - v.z, w - v.w);
    }

    _GraphyVec4<T> &operator-=(const _GraphyVec4<T> &v){
        x -= v.x; y -= v.y; z -= v.z; w -= v.w;
        return *this;
    }

    _GraphyVec4<T> operator+(const _GraphyVec4<T> &v) const{
        return _GraphyVec4<T>(x + v.x, y + v.y, z + v.z, w + v.w);
    }

    _GraphyVec4<T> operator+=(const _GraphyVec4<T> &v){
        x += v.x; y += v.y; z += v.z; w += v.w;
        return *this;
    }

    _GraphyVec4<T> operator*(const _GraphyVec4<T> &v) const{
        return _GraphyVec4<T>(x * v.x, y * v.y, z * v.z, w * v.w);
    }

    _GraphyVec4<T> &operator*=(const _GraphyVec4<T> &v){
        x *= v.x; y *= v.y; z *= v.z; w *= v.w;
        return *this;
    }

    _GraphyVec4<T> operator*(T s) const{
        return _GraphyVec4<T>(x * s, y * s, z * s, w * s);
    }

    _GraphyVec4<T> &operator*=(T s){
        x *= s; y *= s; z *= s; w *= s;
        return *this;
    }

    _GraphyFPrecision LengthSquared() const{ return x * x + y * y + z * z + w * w; }
    _GraphyFPrecision Length() const{ return sqrt(LengthSquared()); }
    void PrintSelf() const{
        printf("P = {x : %g, y :  %g, z : %g, w : %g}", x, y, z, w);
    }
};

template<typename T> inline
_GraphyVec4<T> operator*(T s, _GraphyVec4<T> &v){ return v * s; }
template<typename T> inline
_GraphyVec2<T> operator*(T s, _GraphyVec2<T> &v){ return v * s; }

template<typename T> inline _GraphyVec2<T> normalize(const _GraphyVec2<T> &v){
    return v / v.Length();
}

typedef _GraphyVec4<_Gfloat> GVec4f;
typedef _GraphyVec2<_Gfloat> GVec2f;

struct GMat3{
    _Gfloat m[3][3];

    GMat3(){
        m[0][0] = m[1][1] = m[2][2] = 1.f;
        m[0][1] = m[1][0] = m[0][2] = 0.f;
        m[1][0] = m[1][2] = 0.f;
        m[2][0] = m[2][1] = 0.f;
    }

    GMat3(_Gfloat t00, _Gfloat t11, _Gfloat t22){
        m[0][0] = t00; m[1][1] = t11; m[2][2] = t22;
        m[0][1] = m[1][0] = m[0][2] = 0.f;
        m[1][0] = m[1][2] = 0.f;
        m[2][0] = m[2][1] = 0.f;
    }

    GMat3(_Gfloat mat[3][3]){
        m[0][0] = mat[0][0]; m[1][0] = mat[1][0]; m[2][0] = mat[2][0];
        m[0][1] = mat[0][1]; m[1][1] = mat[1][1]; m[2][1] = mat[2][1];
        m[0][2] = mat[0][2]; m[1][2] = mat[1][2]; m[2][2] = mat[2][2];
    }

    GMat3(_Gfloat t00, _Gfloat t01, _Gfloat t02,
              _Gfloat t10, _Gfloat t11, _Gfloat t12,
              _Gfloat t20, _Gfloat t21, _Gfloat t22)
    {
        m[0][0] = t00; m[0][1] = t01; m[0][2] = t02;
        m[1][0] = t10; m[1][1] = t11; m[1][2] = t12;
        m[2][0] = t20; m[2][1] = t21; m[2][2] = t22;
    }

    void Set(_Gfloat c){
        for(int i = 0; i < 3; i++){
            for(int j = 0; j < 3; j++){
                m[i][j] = c;
            }
        }
    }

    GVec2f Point(GVec2f p){
        _Gfloat x = p.x, y = p.y;
        _Gfloat xp = m[0][0] * x + m[0][1] * y + m[0][2];
        _Gfloat yp = m[1][0] * x + m[1][1] * y + m[1][2];
        _Gfloat wp = m[2][0] * x + m[2][1] * y + m[2][2];
        if(wp == 1) return GVec2f(xp, yp);
        if(wp == 0){
            printf("Invalid homogeneous coordinates\n");
        }

        _Gfloat invwp = 1.f / wp;
        return GVec2f(xp, yp) * invwp;
    }

    friend GMat3 Transpose(const GMat3 &o){
        return GMat3(o.m[0][0], o.m[1][0], o.m[2][0],
                     o.m[0][1], o.m[1][1], o.m[2][1],
                     o.m[0][2], o.m[1][2], o.m[2][2]);
    }

    static GMat3 Mul(const GMat3 &m1, const GMat3 &m2){
        GMat3 r;
        for(int i = 0; i < 3; i++)
            for(int j = 0; j < 3; j++)
            r.m[i][j] = m1.m[i][0]*m2.m[0][j]+m1.m[i][1]*m2.m[1][j]+m1.m[i][2]*m2.m[2][j];
        return r;
    }

    friend _Gfloat Trace(const GMat3 &o){
        return o.m[0][0] + o.m[1][1] + o.m[2][2];
    }

    friend GMat3 Inverse(const GMat3 &o){
        _Gfloat det = (o.m[0][0] * (o.m[1][1] * o.m[2][2] - o.m[1][2] * o.m[2][1]) -
                     o.m[0][1] * (o.m[1][0] * o.m[2][2] - o.m[1][2] * o.m[2][0]) +
                     o.m[0][2] * (o.m[1][0] * o.m[2][1] - o.m[1][1] * o.m[2][0]));

        if(det == 0) return o;
        _Gfloat invDet = 1.f / det;
        _Gfloat a00 = (o.m[1][1] * o.m[2][2] - o.m[2][1] * o.m[1][2]) * invDet;
        _Gfloat a01 = (o.m[0][2] * o.m[2][1] - o.m[2][2] * o.m[0][1]) * invDet;
        _Gfloat a02 = (o.m[0][1] * o.m[1][2] - o.m[1][1] * o.m[0][2]) * invDet;
        _Gfloat a10 = (o.m[1][2] * o.m[2][0] - o.m[2][2] * o.m[1][0]) * invDet;
        _Gfloat a11 = (o.m[0][0] * o.m[2][2] - o.m[2][0] * o.m[0][2]) * invDet;
        _Gfloat a12 = (o.m[0][2] * o.m[1][0] - o.m[1][2] * o.m[0][0]) * invDet;
        _Gfloat a20 = (o.m[1][0] * o.m[2][1] - o.m[2][0] * o.m[1][1]) * invDet;
        _Gfloat a21 = (o.m[0][1] * o.m[2][0] - o.m[2][1] * o.m[0][0]) * invDet;
        _Gfloat a22 = (o.m[0][0] * o.m[1][1] - o.m[1][0] * o.m[0][1]) * invDet;
        return GMat3(a00, a01, a02, a10, a11, a12, a20, a21, a22);
    }

    friend _Gfloat Determinant(const GMat3 &o){
        return (o.m[0][0] * (o.m[1][1] * o.m[2][2] - o.m[1][2] * o.m[2][1]) -
                o.m[0][1] * (o.m[1][0] * o.m[2][2] - o.m[1][2] * o.m[2][0]) +
                o.m[0][2] * (o.m[1][0] * o.m[2][1] - o.m[1][1] * o.m[2][0]));
    }

    void PrintSelf(){
        for(int i = 0; i < 3; i++){
            printf("[ ");
            for(int j = 0; j < 3; j++){
                printf("%g ", m[i][j]);
            }

            printf("]\n");
        }
    }
};

template<typename T> inline T _GraphyClamp(T a, T edge0, T edge1){
    if(a < edge0) return edge0;
    if(a > edge1) return edge1;
    return a;
}

template<typename T> inline T _GraphyMin(T a, T b){
    if(a < b) return a;
    return b;
}

template <typename T, typename V>
inline V _GraphyLerp(T a, V x_0, V x_1){
  return (T(1) - a) * x_0 + a * x_1;
}

inline _Gfloat dot(GVec2f a, GVec2f b){
    return a.x * b.x + a.y * b.y;
}

struct GCanvasContext{
    GVec4f color;
    _Gfloat radius;
};

template<typename T>
class GArray2D{
    public:
    std::vector<T> data;
    int x;
    int y;
    int size;

    GArray2D<T>(){
        x = 0;
        y = 0;
        data.resize(0);
    }

    bool SameDimensions(const GArray2D<T> &o) const{
        return x == o.x && y == o.y;
    }

    void Init(int rx, int ry, T init=T(0)){
        x = rx > 0 ? rx : 1;
        y = ry > 0 ? ry : 1;
        size = x * y;
        data = std::vector<T>(size, init);
    }

    template <typename U>
    GArray2D<T> operator*(const U &v) const{
        GArray2D<T> o(x, y);
        for(int i = 0; i < size; i++){
            o.data[i] = v * data[i];
        }
        return o;
    }

    GArray2D<T> operator+(const GArray2D<T> &v) const{
        if(!SameDimensions(v)){
            printf("Arrays do not have same dimensions (%d %d) != (%d %d)!\n",
                    x, y, v.x, v.y);
            return GArray2D();
        }

        GArray2D<T> o(x, y);
        for(int i = 0; i < size; i++){
            o.data[i] = data[i] + v.data[i];
        }
        return o;
    }

    GArray2D<T> operator+=(const GArray2D<T> &v){
        if(!SameDimensions(v)){
            printf("Arrays do not have same dimensions (%d %d) != (%d %d)!\n",
                    x, y, v.x, v.y);
        }else{
            for(int i = 0; i < size; i++){
                data[i] = data[i] + v.data[i];
            }
        }

        return *this;
    }

    GArray2D<T> &operator=(const GArray2D<T> &arr){
        this->x = arr.x;
        this->y = arr.y;
        this->size = arr.size;
        this->data = arr.data;
        return *this;
    }

    GArray2D<T> &operator=(const T &a){
        for(int i = 0; i < size; i++){
            data[i] = a;
        }
        return *this;
    }

    T *operator[](int i){
        return &data[0] + i * y;
    }

    const T *operator[](int i) const{
        return &data[0] + i * y;
    }

    const T &At(int i, int j) const{
        i = _GraphyClamp(i, 0, x-1);
        j = _GraphyClamp(j, 0, y-1);
        return (*this)[i][j];
    }

    T &At(int i, int j){
        i = _GraphyClamp(i, 0, x-1);
        j = _GraphyClamp(j, 0, y-1);
        return (*this)[i][j];
    }

    T Get(int i, int j) const{
        i = _GraphyClamp(i, 0, x-1);
        j = _GraphyClamp(j, 0, y-1);
        return (*this)[i][j];
    }

    void Reset(T a){
        for(int i = 0; i < size; i++){
            data[i] = a;
        }
    }
};

template<typename T, typename U> inline
GArray2D<T> operator*(U s, GArray2D<T> &v){ return v * s; }

typedef enum{
    Linear, Bilinear
}GUpsampleMode;

class Graphy2DCanvas{
    public:
    Graphy2DCanvas(GArray2D<GVec4f> &_img) : img(_img){
        transform = GMat3((_Gfloat)img.x, (_Gfloat)img.y, 1.f);
    }

    Graphy2DCanvas &Color(GVec4f val){
        context.color = val;
        return *this;
    }

    Graphy2DCanvas &Color(_Gfloat r, _Gfloat g, _Gfloat b, _Gfloat a = 1){
        context.color = GVec4f(r, g, b, a);
        return *this;
    }

    Graphy2DCanvas &Color(int r, int g, int b, int a = 255){
        context.color = GVec4f(r, g, b, a) * (1.f / 255.f);
        return *this;
    }

    Graphy2DCanvas &Color(unsigned int val){
        img.Reset(GVec4f(val / 65536, val / 256 % 256, val % 256, 255) * (1.0f / 255));
        return *this;
    }

    Graphy2DCanvas &Radius(_Gfloat rad){
        context.radius = rad;
        return *this;
    }

    template<typename Fn>
    GVec4f upsample_from(int i, int j, int nx, int ny, Fn fn,
                         GUpsampleMode mode=GUpsampleMode::Linear)
    {
        if(mode == GUpsampleMode::Linear){
            float ex = (float)i / (float)width();
            float ey = (float)j / (float)height();
            int p_x = (int)(ex * nx);
            int p_y = (int)(ey * ny);
            return fn(p_x, p_y);
        }else{
            float scale_x = (float)width() / (float)nx;
            float scale_y = (float)height() / (float)ny;
            float x_ = (float)i / scale_x;
            float y_ = (float)j / scale_y;

            int x1 = GMIN((int)(std::floor(x_)), nx-1);
            int y1 = GMIN((int)(std::floor(y_)), ny-1);
            int x2 = GMIN((int)(std::ceil(x_)), nx-1);
            int y2 = GMIN((int)(std::ceil(y_)), ny-1);

            GVec4f e11 = fn(x1, y1);
            GVec4f e12 = fn(x2, y1);
            GVec4f e21 = fn(x1, y2);
            GVec4f e22 = fn(x2, y2);

            GVec4f e1 = e11 * ((_Gfloat)x2 - x_) + e12 * (x_ - (_Gfloat)x1);
            GVec4f e2 = e21 * ((_Gfloat)x2 - x_) + e22 * (x_ - (_Gfloat)x1);

            if(x1 == x2){
                e1 = e11;
                e2 = e22;
            }

            GVec4f e = e1 * ((_Gfloat)y2 - y_) + e2 * (y_ - (_Gfloat)y1);

            if(y1 == y2){
                e = e1;
            }

            return e;
        }
    }

    int width(){ return img.x; }
    int height(){ return img.y; }

    struct Line{
        Graphy2DCanvas &canvas;
        GVec4f _color;
        _Gfloat _radius;
        int n_vertices;
        GVec2f vertices[128];

        inline Line(Graphy2DCanvas &canvas):
            canvas(canvas), _color(canvas.context.color) { n_vertices = 0; }

        inline Line(Graphy2DCanvas &canvas, GVec4f col):
            canvas(canvas), _color(col) { n_vertices = 0; }

        inline Line(Graphy2DCanvas &canvas, GVec2f a, GVec2f b) :
            Line(canvas)
        {
            push(a);
            push(b);
        }

        inline Line(Graphy2DCanvas &canvas, GVec2f a, GVec2f b, GVec2f c, GVec2f d) :
            Line(canvas)
        {
            push(a);
            push(b);
            push(c);
            push(d);
        }

        inline Line &color(GVec4f col){
            _color = col;
            return *this;
        }

        inline Line &close(){
            push(vertices[0]);
            return *this;
        }

        inline Line &width(_Gfloat w){
            _radius = w * 0.5f;
            return *this;
        }

        inline void push(GVec2f vec){
            if(n_vertices < 128)
                vertices[n_vertices++] = vec;
            else
                printf("Too many vertices for line\n");
        }

        inline Line &path(GVec2f a) {
            push(a);
            return *this;
        }

        inline Line &path(GVec2f a, GVec2f b) {
            push(a);
            push(b);
            return *this;
        }

        inline Line &path(GVec2f a, GVec2f b, GVec2f c) {
            push(a);
            push(b);
            push(c);
            return *this;
        }

        inline Line &path(GVec2f a, GVec2f b, GVec2f c, GVec2f d) {
            push(a);
            push(b);
            push(c);
            push(d);
            return *this;
        }

        inline void stroke(GVec2f a, GVec2f b){
            int a_i_x = (int)(a.x + 0.5f);
            int a_i_y = (int)(a.y + 0.5f);
            int b_i_x = (int)(b.x + 0.5f);
            int b_i_y = (int)(b.y + 0.5f);

            int radius_i = (int)std::ceil(_radius + 0.5f);
            int range_lower_x  = GMIN(a_i_x, b_i_x) - radius_i;
            int range_lower_y  = GMIN(a_i_y, b_i_y) - radius_i;
            int range_higher_x = GMAX(a_i_x, b_i_x) + radius_i;
            int range_higher_y = GMAX(a_i_y, b_i_y) + radius_i;

            GVec2f direction = normalize(b - a);
            _Gfloat l = (b-a).Length();
            GVec2f tangent(-direction.y, direction.x);

            for(int i = range_lower_x; i <= range_higher_x; i++){
                for(int j = range_lower_y; j <= range_higher_y; j++){
                    GVec2f pixel = GVec2f(i + 0.5f, j + 0.5f) - a;
                    _Gfloat u = dot(tangent, pixel);
                    _Gfloat v = dot(direction, pixel);
                    if(v > 0) v = GMAX(0.f, v - l);
                    _Gfloat dist = GVec2f(u, v).Length();
                    _Gfloat alpha = _color.w * _GraphyClamp(_radius - dist, 0.f, 1.f);
                    GVec4f col = canvas.img.At(i, j);
                    canvas.img.At(i, j) = _GraphyLerp(alpha, col, _color);
                }
            }
        }

        inline ~Line(){
            for(int i = 0; i+1 < n_vertices; i++){
                stroke(canvas.transform_point(vertices[i]),
                       canvas.transform_point(vertices[i+1]));
            }
        }
    };

    struct Circle{
        Graphy2DCanvas &canvas;
        GVec2f _center;
        GVec4f _color;
        GVec4f _border_color;
        _Gfloat _radius;
        int _border;
        int _empty;

        inline Circle(Graphy2DCanvas &canvas, GVec2f center):
            canvas(canvas), _center(center), _color(canvas.context.color),
            _radius(canvas.context.radius), _border(0), _empty(0) {}

        inline Circle(Graphy2DCanvas &canvas, GVec2f center, GVec4f col):
            canvas(canvas), _center(center), _color(col),
            _radius(canvas.context.radius), _border(0), _empty(0) {}

        inline Circle &color(GVec4f color){
            _color = color;
            return *this;
        }

        inline Circle &color(_Gfloat r, _Gfloat g, _Gfloat b, _Gfloat a=1){
            _color = GVec4f(r, g, b, a);
            return *this;
        }

        inline Circle &radius(_Gfloat r){
            _radius = r;
            return *this;
        }

        inline Circle &border(int b=1, GVec4f bcolor=GVec4f(1,1,1,1)){
            _border = b;
            _border_color = bcolor;
            return *this;
        }

        inline Circle &empty(){
            _empty = 1;
            return *this;
        }

        ~Circle(){
            GVec2f center = canvas.transform_point(_center);
            int c_i = (int)(center.x + 0.5f);
            int c_j = (int)(center.y + 0.5f);
            int radius_i = (int)std::ceil(_radius + 0.5f);
            _Gfloat off = GMAX(0.98 * (_Gfloat)radius_i, 3.f);
            for(int i = -radius_i; i <= radius_i; i++){
                for(int j = -radius_i; j <= radius_i; j++){
                    int cir = 1;
                    if(_border){
                        if(i == -radius_i || i == radius_i ||
                           j == -radius_i || j == radius_i)
                        {
                            canvas.img.At(c_i + i, c_j + j) = _border_color;
                            cir = 0;
                        }
                    }

                    if(cir){
                        GVec2f v = center - GVec2f(c_i, c_j) - GVec2f(i, j);
                        _Gfloat dist = v.Length();
                        if(dist < off && _empty) continue;

                        _Gfloat alpha = _color.w * _GraphyClamp(_radius - dist, 0.f, 1.f);
                        GVec4f col = canvas.img.At(c_i + i, c_j + j);
                        canvas.img.At(c_i + i, c_j + j) = _GraphyLerp(alpha, col, _color);
                    }
                }
            }
        }
    };

    Circle circle(GVec2f center){
        return Circle(*this, center);
    }

    Circle circle(_Gfloat x, _Gfloat y){
        return Circle(*this, GVec2f(x, y));
    }

    GVec2f transform_point(GVec2f p){
        return transform.Point(p);
    }

    Line path(_Gfloat xa, _Gfloat ya, _Gfloat xb, _Gfloat yb){
        return path(GVec2f(xa, ya), GVec2f(xb, yb));
    }

    Line path(){
        return Line(*this);
    }

    Line path(GVec2f a, GVec2f b){
        return Line(*this).path(a, b);
    }

    Line path(GVec2f a, GVec2f b, GVec2f c, GVec2f d){
        return Line(*this, a, b, c, d);
    }

    Line rect(GVec2f a, GVec2f b){
        return Line(*this, a, GVec2f(a.x, b.y), b, GVec2f(b.x, a.y)).close();
    }

    // direct image set
    template<typename Fn>
    void for_each_pixel(Fn fn){
        for(int i = 0; i < img.x; i++){
            for(int j = 0; j < img.y; j++){
                img.At(i, j) = fn(i, j);
            }
        }
    }

    GCanvasContext context;
    GArray2D<GVec4f> &img;
    GMat3 transform;
};

///////////////////// X11
#include <dlfcn.h>

typedef Display *(*XOpenDisplayPtr)(char *);
typedef GC(*XDefaultGCPtr)(Display *, int);
typedef void(*XPutImagePtr)(Display *, Drawable, GC, XImage *,int, int, int,
                            int, unsigned int, unsigned int);
typedef void(*XStoreNamePtr)(Display *, Window, char *);
typedef int(*XPendingPtr)(Display *);
typedef void(*XNextEventPtr)(Display *, XEvent *);
typedef Bool(*XFilterEventPtr)(XEvent *event, Window w);
typedef Visual*(*XDefaultVisualPtr)(Display *, int);
typedef Window(*XCreateSimpleWindowPtr)(Display *, Window, int, int, unsigned int,
                                 unsigned int, unsigned int, unsigned long, unsigned long);
typedef Window(*XRootWindowPtr)(Display *, int);
typedef void(*XSelectInputPtr)(Display *, Window, long);
typedef void(*XMapWindowPtr)(Display *, Window);
typedef Atom(*XInternAtomPtr)(Display *, char *, Bool);
typedef Status (*XSetWMProtocolsPtr)(Display *, Window, Atom *, int);
typedef void(*XFlushPtr)(Display *);
typedef void(*XUnmapWindowPtr)(Display *, Window);
typedef void(*XDestroyWindowPtr)(Display *, Window);
typedef void(*XCloseDisplayPtr)(Display *);

typedef XImage *(*XCreateImagePtr)(Display *, Visual *, unsigned int, int, int, char *,
                           unsigned int, unsigned int, int, int);

class ImageX11;

class BaseX11{
    public:
    void *display;
    void *visual;
    unsigned long window;
    ImageX11 *img;
    Atom wmDelete;
    void *libX11;

    XOpenDisplayPtr xOpenDisplay;
    XDefaultGCPtr xDefaultGC;
    XPutImagePtr xPutImage;
    XStoreNamePtr xStoreName;
    XPendingPtr xPending;
    XNextEventPtr xNextEvent;
    XFilterEventPtr xFilterEvent;
    XDefaultVisualPtr xDefaultVisual;
    XCreateSimpleWindowPtr xCreateSimpleWindow;
    XRootWindowPtr xRootWindow;
    XSelectInputPtr xSelectInput;
    XMapWindowPtr xMapWindow;
    XUnmapWindowPtr xUnmapWindow;
    XInternAtomPtr xInternAtom;
    XSetWMProtocolsPtr xSetWMProtocols;
    XFlushPtr xFlush;
    XDestroyWindowPtr xDestroyWindow;
    XCloseDisplayPtr xCloseDisplay;
    XCreateImagePtr xCreateImage;
};

using GUIBase = BaseX11;
/////////////////////////

class GWindow : public GUIBase{
    public:
    std::string window_name;
    int width, height;
    GArray2D<GVec4f> buffer;
    std::unique_ptr<Graphy2DCanvas> canvas;
    bool is_alive;

    explicit GWindow(const std::string title, int width =800, int height=600) :
        window_name(title), width(width), height(height)
    {
        int rv = init_lib();
        if(!rv) return;

        create_window();
        set_title(window_name);
        buffer.Init(width, height);
        canvas = std::unique_ptr<Graphy2DCanvas>(new Graphy2DCanvas(buffer));
    }

    ~GWindow(){
        close_lib();
    }

    Graphy2DCanvas get_canvas(){
        return *canvas;
    }

    inline int init_lib();
    inline void close_lib();
    inline void create_window();
    inline void set_title(std::string title);
    inline void redraw();
    inline void process_event();
    inline void close_window();
    inline bool is_opened(){ return is_alive; }

    void update(){
        if(is_alive){
            redraw();
            process_event();
        }
    }
};


/////////////////////// X11
class ImageX11{
    public:
    XImage *image;
    std::vector<uint8_t> image_data;
    int width, height;
    ImageX11(Display *display, Visual *visual, int width, int height, GWindow *winPtr)
     : width(width), height(height){
        image_data.resize(width * height * 4);
        image = winPtr->xCreateImage(display, visual, 24, ZPixmap, 0,
                                (char *)image_data.data(), width, height, 32, 0);
    }

    void set_data(const GArray2D<GVec4f> &color){
        auto p = image_data.data();
        for(int j = 0; j < height; j++){
            for(int i = 0; i < width; i++){
                auto c = color[i][height - j - 1];
                *p++ = uint8_t(_GraphyClamp(int(c[2] * 255.0f), 0, 255));
                *p++ = uint8_t(_GraphyClamp(int(c[1] * 255.0f), 0, 255));
                *p++ = uint8_t(_GraphyClamp(int(c[0] * 255.0f), 0, 255));
                *p++ = uint8_t(_GraphyClamp(int(c[3] * 255.0f), 0, 255));
            }
        }
    }

    ~ImageX11() {
        delete image;
    }
};

static void *load_symbol(void *handle, const char *name){
    void *ptr = dlsym(handle, name);
    if(!ptr){
        std::cout << "Failed to load symbol " << name << " [ " <<
            dlerror() << " ]" << std::endl;
    }
    return ptr;
}

inline int GWindow::init_lib(){
    int rv = 0;
    libX11 = dlopen("libX11.so", RTLD_LAZY);
    if(libX11){
        xOpenDisplay = (XOpenDisplayPtr)load_symbol(libX11, "XOpenDisplay");
        xDefaultGC   = (XDefaultGCPtr)load_symbol(libX11, "XDefaultGC");
        xPutImage    = (XPutImagePtr)load_symbol(libX11, "XPutImage");
        xStoreName   = (XStoreNamePtr)load_symbol(libX11, "XStoreName");
        xPending     = (XPendingPtr)load_symbol(libX11, "XPending");
        xNextEvent   = (XNextEventPtr)load_symbol(libX11, "XNextEvent");
        xFilterEvent = (XFilterEventPtr)load_symbol(libX11, "XFilterEvent");
        xFlush       = (XFlushPtr)load_symbol(libX11, "XFlush");
        xDefaultVisual = (XDefaultVisualPtr)load_symbol(libX11, "XDefaultVisual");
        xRootWindow    = (XRootWindowPtr)load_symbol(libX11, "XRootWindow");
        xSelectInput   = (XSelectInputPtr)load_symbol(libX11, "XSelectInput");
        xMapWindow     = (XMapWindowPtr)load_symbol(libX11, "XMapWindow");
        xInternAtom    = (XInternAtomPtr)load_symbol(libX11, "XInternAtom");
        xUnmapWindow   = (XUnmapWindowPtr)load_symbol(libX11, "XUnmapWindow");
        xDestroyWindow = (XDestroyWindowPtr)load_symbol(libX11, "XDestroyWindow");
        xCreateImage   = (XCreateImagePtr)load_symbol(libX11, "XCreateImage");
        xCloseDisplay  = (XCloseDisplayPtr)load_symbol(libX11, "XCloseDisplay");
        xSetWMProtocols = (XSetWMProtocolsPtr)load_symbol(libX11, "XSetWMProtocols");
        xCreateSimpleWindow =
                (XCreateSimpleWindowPtr)load_symbol(libX11, "XCreateSimpleWindow");

        rv = xOpenDisplay && xDefaultGC && xPutImage && xStoreName &&
             xPending && xNextEvent && xFilterEvent && xDefaultVisual &&
             xCreateSimpleWindow && xRootWindow && xSelectInput &&
             xFlush && xMapWindow && xInternAtom && xSetWMProtocols &&
             xUnmapWindow && xDestroyWindow && xCloseDisplay ? 1 : 0;
    }

    if(!rv){
        printf("Failed to get symbols\n");
    }

    return rv;
}

inline void GWindow::close_lib(){
    if(libX11){
        dlclose(libX11);
    }
}

inline void GWindow::redraw(){
    img->set_data(buffer);
    xPutImage((Display *)display, window, xDefaultGC((Display *)display, 0),
               img->image, 0, 0, 0, 0, width, height);
}

inline void GWindow::set_title(std::string title) {
  xStoreName((Display *)display, window, (char *)title.c_str());
}

inline void GWindow::process_event(){
    while(xPending((Display *)display)){
        XEvent ev;
        xNextEvent((Display *)display, &ev);
        int filtered = xFilterEvent(&ev, None);
        switch (ev.type) {
            case Expose:
            break;
            case ButtonPress:
            break;
            case KeyPress:
            //key_pressed = true;
            break;
            case ClientMessage:{
                if(filtered) return;
                if(ev.xclient.data.l[0] == wmDelete){
                    close_window();
                    return;
                }
            } break;
            default:{}
        }
    }
}

inline void GWindow::create_window(){
    display = xOpenDisplay(nullptr);
    visual = xDefaultVisual((Display *)display, 0);
    window = xCreateSimpleWindow((Display *)display, xRootWindow((Display *)display, 0),
                                 0, 0, width, height, 1, 0, 0);
    xSelectInput((Display *)display, window,
                 ButtonPressMask | ExposureMask | KeyPressMask | KeyReleaseMask);
    xMapWindow((Display *)display, window);
    wmDelete = xInternAtom((Display *)display, (char *)"WM_DELETE_WINDOW", False);

    Atom protocols[] = { wmDelete };
    xSetWMProtocols((Display *)display, window,
                    protocols, sizeof(protocols) / sizeof(Atom));

    xFlush((Display *)display);

    img = new ImageX11((Display *)display, (Visual *)visual, width, height, this);
    is_alive = true;
}

inline void GWindow::close_window(){
    if(is_alive){
        xUnmapWindow((Display *)display, window);
        xDestroyWindow((Display *)display, window);
        xFlush((Display *)display);
        xCloseDisplay((Display *)display);
        is_alive = false;
    }
}

///////////////////////////
