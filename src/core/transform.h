#pragma once

#include <geometry.h>
#include <cutil.h>
#include <interaction.h>

struct ParsedMesh;

struct Matrix2x2{
    Float m[2][2];
    __bidevice__ Matrix2x2(){
        m[0][0] = m[1][1] = 1.f;
        m[0][1] = m[1][0] = 0.f;
    }

    __bidevice__ Matrix2x2(Float mat[2][2]){
        m[0][0] = mat[0][0]; m[0][1] = mat[0][1];
        m[1][0] = mat[1][0]; m[1][1] = mat[1][1];
    }

    __bidevice__ Matrix2x2 (Float t00, Float t01, Float t10, Float t11){
        m[0][0] = t00; m[0][1] = t01;
        m[1][0] = t10; m[1][1] = t11;
    }

    __bidevice__ void Set(Float c){
        m[0][0] = m[1][1] = c;
        m[0][1] = m[1][0] = c;
    }

    __bidevice__ friend Matrix2x2 Transpose(const Matrix2x2 &o){
        return Matrix2x2(o.m[0][0], o.m[1][0], o.m[0][1], o.m[1][1]);
    }

    __bidevice__ static Matrix2x2 Mul(const Matrix2x2 &m1, const Matrix2x2 &m2){
        Matrix2x2 r;
        for(int i = 0; i < 2; i++){
            for(int j = 0; j < 2; j++){
                r.m[i][j] = m1.m[i][0]*m2.m[0][j]+m1.m[i][1]*m2.m[1][j];
            }
        }
        return r;
    }

    __bidevice__ friend Float Trace(const Matrix2x2 &o){
        return o.m[0][0] + o.m[1][1];
    }

    __bidevice__ friend Float Determinant(const Matrix2x2 &o){
        return o.m[0][0]*o.m[1][1] - o.m[0][1]*o.m[1][0];
    }

    __bidevice__ friend Matrix2x2 Inverse(const Matrix2x2 &o){
        Float det = o.m[0][0]*o.m[1][1] - o.m[0][1]*o.m[1][0];
        AssertA(!IsZero(det), "Zero determinant on matrix inverse");
        if(IsZero(det)) return o;

        Float invDet = 1.0f / det;
        Float a00 =  o.m[1][1] * invDet;
        Float a01 = -o.m[0][1] * invDet;
        Float a10 = -o.m[1][0] * invDet;
        Float a11 =  o.m[0][0] * invDet;
        return Matrix2x2(a00, a01, a10, a11);
    }

    __bidevice__ friend Matrix2x2 HighpInverse(const Matrix2x2 &o){
        Float det = o.m[0][0]*o.m[1][1] - o.m[0][1]*o.m[1][0];
        AssertA(!IsHighpZero(det), "Zero determinant on matrix inverse");
        if(IsHighpZero(det)) return o;

        Float invDet = 1.0f / det;
        Float a00 =  o.m[1][1] * invDet;
        Float a01 = -o.m[0][1] * invDet;
        Float a10 = -o.m[1][0] * invDet;
        Float a11 =  o.m[0][0] * invDet;
        return Matrix2x2(a00, a01, a10, a11);
    }

    __bidevice__ void TensorAdd(const vec2f &v){
        Float x2 = v.x * v.x, y2 = v.y * v.y;
        Float xy = v.x * v.y;
        m[0][0] += x2; m[0][1] += xy;
        m[1][0] += xy; m[1][1] += y2;
    }

    __bidevice__ void PrintSelf(){
        for(int i = 0; i < 2; i++){
            printf("[ ");
            for(int j = 0; j < 2; j++){
                printf("%g ", m[i][j]);
            }

            printf("]\n");
        }
    }
};

struct Matrix3x3{
    Float m[3][3];

    __bidevice__ Matrix3x3(){
        m[0][0] = m[1][1] = m[2][2] = 1.f;
        m[0][1] = m[1][0] = m[0][2] = 0.f;
        m[1][0] = m[1][2] = 0.f;
        m[2][0] = m[2][1] = 0.f;
    }

    __bidevice__ Matrix3x3(Float mat[3][3]){
        m[0][0] = mat[0][0]; m[1][0] = mat[1][0]; m[2][0] = mat[2][0];
        m[0][1] = mat[0][1]; m[1][1] = mat[1][1]; m[2][1] = mat[2][1];
        m[0][2] = mat[0][2]; m[1][2] = mat[1][2]; m[2][2] = mat[2][2];
    }

    __bidevice__ Matrix3x3(Float t00, Float t01, Float t02,
                           Float t10, Float t11, Float t12,
                           Float t20, Float t21, Float t22)
    {
        m[0][0] = t00; m[0][1] = t01; m[0][2] = t02;
        m[1][0] = t10; m[1][1] = t11; m[1][2] = t12;
        m[2][0] = t20; m[2][1] = t21; m[2][2] = t22;
    }

    __bidevice__ void Set(Float c){
        for(int i = 0; i < 3; i++){
            for(int j = 0; j < 3; j++){
                m[i][j] = c;
            }
        }
    }

    __bidevice__ Matrix3x3 operator*(Float s) const{
        Matrix3x3 r;
        for(int i = 0; i < 3; i++)
            for(int j = 0; j < 3; j++)
                r.m[i][j] = m[i][j] * s;
        return r;
    }

    __bidevice__ vec3f Vec(vec3f v){
        return vec3f(m[0][0] * v.x + m[0][1] * v.y + m[0][2] * v.z,
                     m[1][0] * v.x + m[1][1] * v.y + m[1][2] * v.z,
                     m[2][0] * v.x + m[2][1] * v.y + m[2][2] * v.z);
    }

    __bidevice__ Matrix3x3 operator+(const Matrix3x3 &o) const{
        Matrix3x3 r;
        for(int i = 0; i < 3; i++)
            for(int j = 0; j < 3; j++)
                r.m[i][j] = m[i][j] + o.m[i][j];
        return r;
    }

    __bidevice__ friend Matrix3x3 Transpose(const Matrix3x3 &o){
        return Matrix3x3(o.m[0][0], o.m[1][0], o.m[2][0],
                         o.m[0][1], o.m[1][1], o.m[2][1],
                         o.m[0][2], o.m[1][2], o.m[2][2]);
    }

    __bidevice__ static Matrix3x3 Mul(const Matrix3x3 &m1, const Matrix3x3 &m2){
        Matrix3x3 r;
        for(int i = 0; i < 3; i++)
            for(int j = 0; j < 3; j++)
            r.m[i][j] = m1.m[i][0]*m2.m[0][j]+m1.m[i][1]*m2.m[1][j]+m1.m[i][2]*m2.m[2][j];
        return r;
    }

    __bidevice__ friend vec2f Translation(const Matrix3x3 &m1){
        return vec2f(m1.m[0][2], m1.m[1][2]);
    }

    __bidevice__ friend Float Trace(const Matrix3x3 &o){
        return o.m[0][0] + o.m[1][1] + o.m[2][2];
    }

    __bidevice__ friend Matrix3x3 Inverse(const Matrix3x3 &o){
        Float det = (o.m[0][0] * (o.m[1][1] * o.m[2][2] - o.m[1][2] * o.m[2][1]) -
                     o.m[0][1] * (o.m[1][0] * o.m[2][2] - o.m[1][2] * o.m[2][0]) +
                     o.m[0][2] * (o.m[1][0] * o.m[2][1] - o.m[1][1] * o.m[2][0]));

        AssertA(!IsZero(det), "Zero determinant on matrix inverse");
        if(IsZero(det)) return o;
        Float invDet = 1.f / det;
        Float a00 = (o.m[1][1] * o.m[2][2] - o.m[2][1] * o.m[1][2]) * invDet;
        Float a01 = (o.m[0][2] * o.m[2][1] - o.m[2][2] * o.m[0][1]) * invDet;
        Float a02 = (o.m[0][1] * o.m[1][2] - o.m[1][1] * o.m[0][2]) * invDet;
        Float a10 = (o.m[1][2] * o.m[2][0] - o.m[2][2] * o.m[1][0]) * invDet;
        Float a11 = (o.m[0][0] * o.m[2][2] - o.m[2][0] * o.m[0][2]) * invDet;
        Float a12 = (o.m[0][2] * o.m[1][0] - o.m[1][2] * o.m[0][0]) * invDet;
        Float a20 = (o.m[1][0] * o.m[2][1] - o.m[2][0] * o.m[1][1]) * invDet;
        Float a21 = (o.m[0][1] * o.m[2][0] - o.m[2][1] * o.m[0][0]) * invDet;
        Float a22 = (o.m[0][0] * o.m[1][1] - o.m[1][0] * o.m[0][1]) * invDet;
        return Matrix3x3(a00, a01, a02, a10, a11, a12, a20, a21, a22);
    }

    __bidevice__ friend Matrix3x3 HighpInverse(const Matrix3x3 &o){
        Float det = (o.m[0][0] * (o.m[1][1] * o.m[2][2] - o.m[1][2] * o.m[2][1]) -
                     o.m[0][1] * (o.m[1][0] * o.m[2][2] - o.m[1][2] * o.m[2][0]) +
                     o.m[0][2] * (o.m[1][0] * o.m[2][1] - o.m[1][1] * o.m[2][0]));

        AssertA(!IsHighpZero(det), "Zero determinant on matrix inverse");
        if(IsHighpZero(det)) return o;
        Float invDet = 1.f / det;
        Float a00 = (o.m[1][1] * o.m[2][2] - o.m[2][1] * o.m[1][2]) * invDet;
        Float a01 = (o.m[0][2] * o.m[2][1] - o.m[2][2] * o.m[0][1]) * invDet;
        Float a02 = (o.m[0][1] * o.m[1][2] - o.m[1][1] * o.m[0][2]) * invDet;
        Float a10 = (o.m[1][2] * o.m[2][0] - o.m[2][2] * o.m[1][0]) * invDet;
        Float a11 = (o.m[0][0] * o.m[2][2] - o.m[2][0] * o.m[0][2]) * invDet;
        Float a12 = (o.m[0][2] * o.m[1][0] - o.m[1][2] * o.m[0][0]) * invDet;
        Float a20 = (o.m[1][0] * o.m[2][1] - o.m[2][0] * o.m[1][1]) * invDet;
        Float a21 = (o.m[0][1] * o.m[2][0] - o.m[2][1] * o.m[0][0]) * invDet;
        Float a22 = (o.m[0][0] * o.m[1][1] - o.m[1][0] * o.m[0][1]) * invDet;
        return Matrix3x3(a00, a01, a02, a10, a11, a12, a20, a21, a22);
    }

    __bidevice__ friend Float Determinant(const Matrix3x3 &o){
        return (o.m[0][0] * (o.m[1][1] * o.m[2][2] - o.m[1][2] * o.m[2][1]) -
                o.m[0][1] * (o.m[1][0] * o.m[2][2] - o.m[1][2] * o.m[2][0]) +
                o.m[0][2] * (o.m[1][0] * o.m[2][1] - o.m[1][1] * o.m[2][0]));
    }

    __bidevice__ void TensorAdd(const vec3f &v){
        Float x2 = v.x * v.x, y2 = v.y * v.y, z2 = v.z * v.z;
        Float xy = v.x * v.y, xz = v.x * v.z, yz = v.y * v.z;
        m[0][0] += x2; m[0][1] += xy; m[0][2] += xz;
        m[1][0] += xy; m[1][1] += y2; m[1][2] += yz;
        m[2][0] += xz; m[2][1] += yz; m[2][2] += z2;
    }

    __bidevice__ void PrintSelf(){
        for(int i = 0; i < 3; i++){
            printf("[ ");
            for(int j = 0; j < 3; j++){
                printf("%g ", m[i][j]);
            }

            printf("]\n");
        }
    }
};

struct Matrix4x4{
    Float m[4][4];

    __bidevice__ Matrix4x4(){
        m[0][0] = m[1][1] = m[2][2] = m[3][3] = 1.f;
        m[0][1] = m[0][2] = m[0][3] = m[1][0] = m[1][2] = m[1][3] = m[2][0] =
            m[2][1] = m[2][3] = m[3][0] = m[3][1] = m[3][2] = 0.f;
    }

    __bidevice__ Matrix4x4(const Matrix3x3 &mat){
        for(int i = 0; i < 3; i++){
            for(int j = 0; j < 3; j++){
                m[i][j] = mat.m[i][j];
            }
        }

        m[0][3] = 0; m[1][3] = 0; m[2][3] = 0; m[3][3] = 1;
    }

    __bidevice__ Matrix4x4(Float mat[4][4]);
    __bidevice__ Matrix4x4(Float t00, Float t01, Float t02, Float t03, Float t10, Float t11,
                           Float t12, Float t13, Float t20, Float t21, Float t22, Float t23,
                           Float t30, Float t31, Float t32, Float t33);

    __bidevice__ friend Matrix4x4 Transpose(const Matrix4x4 &);

    __bidevice__ static Matrix4x4 Mul(const Matrix4x4 &m1, const Matrix4x4 &m2) {
        Matrix4x4 r;
        for (int i = 0; i < 4; ++i)
            for (int j = 0; j < 4; ++j)
            r.m[i][j] = m1.m[i][0] * m2.m[0][j] + m1.m[i][1] * m2.m[1][j] +
            m1.m[i][2] * m2.m[2][j] + m1.m[i][3] * m2.m[3][j];
        return r;
    }

    __bidevice__ friend vec3f Translation(const Matrix4x4 &m1){
        return vec3f(m1.m[0][3], m1.m[1][3], m1.m[2][3]);
    }

    __bidevice__ friend Matrix4x4 Inverse(const Matrix4x4 &);

    __bidevice__ void PrintSelf(){
        for(int i = 0; i < 4; ++i){
            printf("[ ");
            for(int j = 0; j < 4; ++j){
                printf("%g  ", m[i][j]);
            }

            printf("]\n");
        }
    }
};

class Transform2{
    public:
    Matrix3x3 m, mInv;
    __bidevice__ Transform2(){}
    __bidevice__ Transform2(const Matrix3x3 &m) : m(m), mInv(Inverse(m)){}
    __bidevice__ Transform2(const Matrix3x3 &mat, const Matrix3x3 &inv): m(mat), mInv(inv){}
    __bidevice__ friend Transform2 Inverse(const Transform2 &t){
        return Transform2(t.mInv, t.m);
    }

    __bidevice__ vec2f Vector(const vec2f &p) const{
        Float x = p.x, y = p.y;
        Float xp = m.m[0][0] * x + m.m[0][1] * y;
        Float yp = m.m[1][0] * x + m.m[1][1] * y;
        return vec2f(xp, yp);
    }

    __bidevice__ vec2f Vector(const vec2f &p, vec2f *pError) const{
        Float x = p.x, y = p.y;
        Float xp = m.m[0][0] * x + m.m[0][1] * y;
        Float yp = m.m[1][0] * x + m.m[1][1] * y;
        Float xAbsSum = Absf(m.m[0][0] * x) + Absf(m.m[0][1] * y);
        Float yAbsSum = Absf(m.m[1][0] * x) + Absf(m.m[1][1] * y);
        *pError = gamma(3) * vec2f(xAbsSum, yAbsSum);
        return vec2f(xp, yp);
    }

    __bidevice__ vec2f Point(const vec2f &p) const{
        Float x = p.x, y = p.y;
        Float xp = m.m[0][0] * x + m.m[0][1] * y + m.m[0][2];
        Float yp = m.m[1][0] * x + m.m[1][1] * y + m.m[1][2];
        Float wp = m.m[2][0] * x + m.m[2][1] * y + m.m[2][2];

        if(IsZero(wp - 1.)){
            return vec2f(xp, yp);
        }else{
            AssertA(!IsZero(wp), "Invalid homogeneous coordinate normalization");
            return vec2f(xp, yp) / wp;
        }
    }

    __bidevice__ vec2f Point(const vec2f &p, vec2f *pError) const{
        Float x = p.x, y = p.y;
        Float xp = m.m[0][0] * x + m.m[0][1] * y + m.m[0][2];
        Float yp = m.m[1][0] * x + m.m[1][1] * y + m.m[1][2];
        Float wp = m.m[2][0] * x + m.m[2][1] * y + m.m[2][2];

        Float xAbsSum = Absf(m.m[0][0] * x) + Absf(m.m[0][1] * y) + Absf(m.m[0][2]);
        Float yAbsSum = Absf(m.m[1][0] * x) + Absf(m.m[1][1] * y) + Absf(m.m[1][2]);
        *pError = gamma(3) * vec2f(xAbsSum, yAbsSum);

        if(IsZero(wp - 1.)){
            return vec2f(xp, yp);
        }else{
            AssertA(!IsZero(wp), "Invalid homogeneous coordinate normalization");
            return vec2f(xp, yp) / wp;
        }
    }

    __bidevice__ Ray2 operator()(const Ray2 &r) const{
        vec2f oError, dError;
        vec2f o = Point(r.o, &oError);
        vec2f d = Vector(r.d, &dError);
        Float len2 = d.LengthSquared();
        if(len2 > 0 && !IsZero(len2)){
            Float dt = Dot(Abs(d), oError) / len2;
            o += d * dt;
        }

        return Ray2(o, d, r.tMax);
    }

    __bidevice__ Bounds2f operator()(const Bounds2f &b) const{
        Bounds2f ret(Point(vec2f(b.pMin.x, b.pMin.y)));
        ret = Union(ret, Point(vec2f(b.pMin.x, b.pMax.y)));
        ret = Union(ret, Point(vec2f(b.pMin.x, b.pMin.y)));
        ret = Union(ret, Point(vec2f(b.pMax.x, b.pMax.y)));
        ret = Union(ret, Point(vec2f(b.pMax.x, b.pMin.y)));
        return ret;
    }

    __bidevice__ SurfaceInteraction2 operator()(const SurfaceInteraction2 &si) const{
        SurfaceInteraction2 ret;
        ret.p = Point(si.p);
        ret.n = Vector(si.n);
        ret.shape = si.shape;
        return ret;
    }

    __bidevice__ Transform2 operator*(const Transform2 &t2) const;
};

__bidevice__ Transform2 Scale2(Float u);
__bidevice__ Transform2 Scale2(Float x, Float y);
__bidevice__ Transform2 Translate2(Float u);
__bidevice__ Transform2 Translate2(Float x, Float y);
__bidevice__ Transform2 Rotate2(Float alpha);

class Transform{
    public:
    Matrix4x4 m, mInv;
    // Transform Public Methods
    __bidevice__ Transform() {}
    __bidevice__ Transform(const Float mat[4][4]){
        m = Matrix4x4(mat[0][0], mat[0][1], mat[0][2], mat[0][3], mat[1][0],
                      mat[1][1], mat[1][2], mat[1][3], mat[2][0], mat[2][1],
                      mat[2][2], mat[2][3], mat[3][0], mat[3][1], mat[3][2],
                      mat[3][3]);
        mInv = Inverse(m);
    }

    __bidevice__ Transform(const Matrix4x4 &m) : m(m), mInv(Inverse(m)) {}
    __bidevice__ Transform(const Matrix4x4 &m, const Matrix4x4 &mInv) : m(m), mInv(mInv) {}

    __bidevice__ friend Transform Inverse(const Transform &t) {
        return Transform(t.mInv, t.m);
    }

    __bidevice__ friend Transform Transpose(const Transform &t) {
        return Transform(Transpose(t.m), Transpose(t.mInv));
    }

    template<typename T> __bidevice__ vec3<T> Vector(const vec3<T> &v) const{
        T x = v.x, y = v.y, z = v.z;
        return vec3<T>(m.m[0][0] * x + m.m[0][1] * y + m.m[0][2] * z,
                       m.m[1][0] * x + m.m[1][1] * y + m.m[1][2] * z,
                       m.m[2][0] * x + m.m[2][1] * y + m.m[2][2] * z);
    }

    template<typename T> __bidevice__ Normal3<T> Normal(const Normal3<T> &n) const{
        T x = n.x, y = n.y, z = n.z;
        return Normal3<T>(mInv.m[0][0] * x + mInv.m[1][0] * y + mInv.m[2][0] * z,
                          mInv.m[0][1] * x + mInv.m[1][1] * y + mInv.m[2][1] * z,
                          mInv.m[0][2] * x + mInv.m[1][2] * y + mInv.m[2][2] * z);
    }

    template<typename T> __bidevice__ vec3<T> Vector(const vec3<T> &v, vec3<T> *absError) const{
        T x = v.x, y = v.y, z = v.z;
        absError->x =
            gamma(3) * (Absf(m.m[0][0] * v.x) + Absf(m.m[0][1] * v.y) +
                        Absf(m.m[0][2] * v.z));
        absError->y =
            gamma(3) * (Absf(m.m[1][0] * v.x) + Absf(m.m[1][1] * v.y) +
                        Absf(m.m[1][2] * v.z));
        absError->z =
            gamma(3) * (Absf(m.m[2][0] * v.x) + Absf(m.m[2][1] * v.y) +
                        Absf(m.m[2][2] * v.z));
        return vec3<T>(m.m[0][0] * x + m.m[0][1] * y + m.m[0][2] * z,
                       m.m[1][0] * x + m.m[1][1] * y + m.m[1][2] * z,
                       m.m[2][0] * x + m.m[2][1] * y + m.m[2][2] * z);
    }

    template<typename T> __bidevice__ vec3<T> Point(const vec3<T> &p) const{
        T x = p.x, y = p.y, z = p.z;
        T xp = m.m[0][0] * x + m.m[0][1] * y + m.m[0][2] * z + m.m[0][3];
        T yp = m.m[1][0] * x + m.m[1][1] * y + m.m[1][2] * z + m.m[1][3];
        T zp = m.m[2][0] * x + m.m[2][1] * y + m.m[2][2] * z + m.m[2][3];
        T wp = m.m[3][0] * x + m.m[3][1] * y + m.m[3][2] * z + m.m[3][3];
        if (wp == 1)
            return vec3<T>(xp, yp, zp);
        else{
            AssertAEx(!IsZero(wp), "Zero transform wp [Point3f]");
            return vec3<T>(xp, yp, zp) / wp;
        }
    }

    template<typename T> __bidevice__ vec3<T> Point(const vec3<T> &p, vec3<T> *pError) const{
        T x = p.x, y = p.y, z = p.z;
        T xp = (m.m[0][0] * x + m.m[0][1] * y) + (m.m[0][2] * z + m.m[0][3]);
        T yp = (m.m[1][0] * x + m.m[1][1] * y) + (m.m[1][2] * z + m.m[1][3]);
        T zp = (m.m[2][0] * x + m.m[2][1] * y) + (m.m[2][2] * z + m.m[2][3]);
        T wp = (m.m[3][0] * x + m.m[3][1] * y) + (m.m[3][2] * z + m.m[3][3]);

        T xAbsSum = (Absf(m.m[0][0] * x) + Absf(m.m[0][1] * y) +
                     Absf(m.m[0][2] * z) + Absf(m.m[0][3]));
        T yAbsSum = (Absf(m.m[1][0] * x) + Absf(m.m[1][1] * y) +
                     Absf(m.m[1][2] * z) + Absf(m.m[1][3]));
        T zAbsSum = (Absf(m.m[2][0] * x) + Absf(m.m[2][1] * y) +
                     Absf(m.m[2][2] * z) + Absf(m.m[2][3]));
        *pError = gamma(3) * vec3<T>(xAbsSum, yAbsSum, zAbsSum);
        if(wp == 1)
            return vec3<T>(xp, yp, zp);
        else{
            AssertAEx(!IsZero(wp), "Zero transform wp [Point3f][2]");
            return vec3<T>(xp, yp, zp) / wp;
        }
    }

    __bidevice__ Ray operator()(const Ray &r) const{
        vec3f oError, dError;
        vec3f o = Point(r.o, &oError);
        vec3f d = Vector(r.d, &dError);
        Float len2 = d.LengthSquared();
        if(len2 > 0 && !IsZero(len2)){
            Float dt = Dot(Abs(d), oError) / len2;
            o += d * dt;
        }

        return Ray(o, d, r.tMax);
    }

    __bidevice__ SurfaceInteraction operator()(const SurfaceInteraction &si) const{
        SurfaceInteraction ret;
        ret.p = Point(si.p);
        ret.n = Normal(si.n);
        ret.shape = si.shape;
        return ret;
    }

    __bidevice__ void Mesh(ParsedMesh *mesh) const;

    __bidevice__ bool IsIdentity() const {
        return (m.m[0][0] == 1.f && m.m[0][1] == 0.f && m.m[0][2] == 0.f &&
                m.m[0][3] == 0.f && m.m[1][0] == 0.f && m.m[1][1] == 1.f &&
                m.m[1][2] == 0.f && m.m[1][3] == 0.f && m.m[2][0] == 0.f &&
                m.m[2][1] == 0.f && m.m[2][2] == 1.f && m.m[2][3] == 0.f &&
                m.m[3][0] == 0.f && m.m[3][1] == 0.f && m.m[3][2] == 0.f &&
                m.m[3][3] == 1.f);
    }

    __bidevice__ const Matrix4x4 &GetMatrix() const { return m; }
    __bidevice__ const Matrix4x4 &GetInverseMatrix() const { return mInv; }

    __bidevice__ Transform operator*(const Transform &t2) const;
    __bidevice__ bool SwapsHandedness() const;

    __bidevice__ Bounds3f operator()(const Bounds3f &b) const;
};

__bidevice__ Transform Translate(const vec3f &delta);
__bidevice__ Transform Translate(Float x, Float y, Float z);
__bidevice__ Transform Translate(Float u);
__bidevice__ Transform Scale(const vec3f &delta);
__bidevice__ Transform Scale(Float x, Float y, Float z);
__bidevice__ Transform Scale(Float u);
__bidevice__ Transform RotateX(Float theta, bool radians=false);
__bidevice__ Transform RotateY(Float theta, bool radians=false);
__bidevice__ Transform RotateZ(Float theta, bool radians=false);
__bidevice__ Transform Rotate(Float theta, const vec3f &axis, bool radians=false);
__bidevice__ Transform RotateAround(Float theta, const vec3f &axis, const vec3f &point);
__bidevice__ bool SolveLinearSystem2x2(const Float A[2][2], const Float B[2], Float *x0,
                                       Float *x1);
