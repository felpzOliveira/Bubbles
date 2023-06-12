#include <iostream>
#include <cmath>
#include <stdio.h>
#include <cstddef>
#include <cutil.h>


template<typename T>
struct SVDVec3{
    T m[3];

    bb_cpu_gpu SVDVec3(T a){
        m[0] = a;
        m[1] = a;
        m[2] = a;
    }

    bb_cpu_gpu SVDVec3(T x, T y, T z){
        m[0] = x;
        m[1] = y;
        m[2] = z;
    }

    bb_cpu_gpu SVDVec3(){
        m[0] = 0;
        m[1] = 0;
        m[2] = 0;
    }

    bb_cpu_gpu SVDVec3 operator=(const SVDVec3 &v){
        m[0] = v.m[0]; m[1] = v.m[1]; m[2] = v.m[2];
        return *this;
    }

    bb_cpu_gpu SVDVec3 operator*(const SVDVec3 &v) const{
        return SVDVec3(m[0] * v.m[0], m[1] * v.m[1], m[2] * v.m[2]);
    }

    template<typename Q> bb_cpu_gpu
    SVDVec3 operator*(Q s) const{
        return SVDVec3(m[0] * s, m[1] * s, m[2] * s);
    }

    bb_cpu_gpu T operator[](int i) const{ return m[i]; }
    bb_cpu_gpu T &operator[](int i){ return m[i]; }
};


template<typename T>
struct SVDMat3{
    T m[3][3];

    bb_cpu_gpu
    SVDMat3(T a00, T a01, T a02,
            T a10, T a11, T a12,
            T a20, T a21, T a22)
    {
        m[0][0] = a00; m[0][1] = a01; m[0][2] = a02;
        m[1][0] = a10; m[1][1] = a11; m[1][2] = a12;
        m[2][0] = a20; m[2][1] = a21; m[2][2] = a22;
    }

    bb_cpu_gpu SVDMat3(){
        for(int i = 0; i < 3; i++)
            for(int j = 0; j < 3; j++)
                m[i][j] = i == j ? 1 : 0;
    }

    bb_cpu_gpu
    SVDMat3(size_t M, size_t N, const T& s = T(0)){
        if(M != 3 || N != 3)
            printf("{SVDMat3} Unsupported dimensions: %d x %d\n", (int)M, (int)N);

        for(int i = 0; i < 3; i++)
            for(int j = 0; j < 3; j++)
                m[i][j] = s;
    }

    bb_cpu_gpu
    SVDMat3 operator+(const SVDMat3 &v) const{
        SVDMat3 mat;
        for(int i = 0; i < 3; i++)
            for(int j = 0; j < 3; j++)
                mat.m[i][j] = m[i][j] + v.m[i][j];
        return mat;
    }

    template<typename Q>
    bb_cpu_gpu SVDMat3 operator*(const Q &s) const{
        SVDMat3 mat;
        for(int i = 0; i < 3; i++)
            for(int j = 0; j < 3; j++)
                mat.m[i][j] = m[i][j] * s;
        return mat;
    }

    bb_cpu_gpu SVDMat3 Transpose(){
        SVDMat3 mat;
        for(int i = 0; i < 3; i++)
            for(int j = 0; j < 3; j++)
                mat.m[j][i] = m[i][j];
        return mat;
    }

    bb_cpu_gpu SVDMat3 operator=(const SVDMat3 &v){
        for(int i = 0; i < 3; i++)
            for(int j = 0; j < 3; j++)
                m[i][j] = v.m[i][j];
        return *this;
    }

    bb_cpu_gpu static SVDMat3 ScaleMatrix(T sx, T sy, T sz){
        return SVDMat3(sx, 0, 0,
                        0,sy, 0,
                        0, 0,sz);
    }

    template<typename Q>
    bb_cpu_gpu static SVDMat3 Vvt(Q qx, Q qy, Q qz){
        T x = T(qx); T y = T(qy); T z = T(qz);
        return SVDMat3(x * x, x * y, x * z,
                       y * x, y * y, y * z,
                       z * x, z * y, z * z);
    }

    bb_cpu_gpu T Determinant(){
        return (m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) -
                m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0]) +
                m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]));
    }

    bb_cpu_gpu SVDVec3<T> PointMul(T x, T y, T z){
        T xp = m[0][0] * x + m[0][1] * y + m[0][2] * z;
        T yp = m[1][0] * x + m[1][1] * y + m[1][2] * z;
        T zp = m[2][0] * x + m[2][1] * y + m[2][2] * z;
        return SVDVec3<T>(xp, yp, zp);
    }

    bb_cpu_gpu
    SVDMat3 operator*(const SVDMat3 &v) const{
        SVDMat3 mat;
        for(int i = 0; i < 3; i++){
            for(int j = 0; j < 3; j++){
                T val = T(0);
                for(int s = 0; s < 3; s++)
                    val += m[i][s] * v.m[s][j];
                mat.m[i][j] = val;
            }
        }

        return mat;
    }

    bb_cpu_gpu T operator()(int i, int j) const{ return m[i][j]; }
    bb_cpu_gpu T &operator()(int i, int j){ return m[i][j]; }

    bb_cpu_gpu size_t rows(){ return 3; }
    bb_cpu_gpu size_t cols(){ return 3; }
};

template<typename T>
inline std::ostream &operator<<(std::ostream &out, const SVDMat3<T> &v){
    for(int i = 0; i < 3; i++){
        out << "[ ";
        for(int j = 0; j < 3; j++){
            out << v.m[i][j] << " ";
        }
        if(i < 2)
            out << "],";
        else
            out << "]";
    }
    return out;
}

template<typename T>
inline std::ostream &operator<<(std::ostream &out, const SVDVec3<T> &v){
    out << "[ " << v[0] << " " << v[1] << " " << v[2] << " ]";
    return out;
}

template<typename T, typename Q> inline bb_cpu_gpu SVDVec3<T> operator*(Q s, SVDVec3<T> &v){ return v * s; }

template <typename T> inline bb_cpu_gpu
T sign(T a, T b) {
    return b >= 0.0 ? std::fabs(a) : -std::fabs(a);
}

template <typename T> inline bb_cpu_gpu
T svd_max(T a, T b){
    return a > b ? a : b;
}

template <typename T> inline bb_cpu_gpu
T pythag(T a, T b) {
    T at = std::fabs(a);
    T bt = std::fabs(b);
    T ct;
    T result;

    if (at > bt) {
        ct = bt / at;
        result = at * std::sqrt(1 + ct * ct);
    } else if (bt > 0) {
        ct = at / bt;
        result = bt * std::sqrt(1 + ct * ct);
    } else {
        result = 0;
    }

    return result;
}

template <typename T> bb_cpu_gpu
void SVD3(const SVDMat3<T>& a, SVDMat3<T>& u, SVDVec3<T>& w, SVDMat3<T>& v){
    int M = 3;
    int N = 3;
    const int m = (int)M;
    const int n = (int)N;

    int flag, i = 0, its = 0, j = 0, jj = 0, k = 0, l = 0, nm = 0;
    T c = 0, f = 0, h = 0, s = 0, x = 0, y = 0, z = 0;
    T anorm = 0, g = 0, scale = 0;

    // Prepare workspace
    SVDVec3<T> rv1;
    u = a;
    w = SVDVec3<T>();
    v = SVDMat3<T>();

    // Householder reduction to bidiagonal form
    for (i = 0; i < n; i++) {
        // left-hand reduction
        l = i + 1;
        rv1[i] = scale * g;
        g = s = scale = 0;
        if (i < m) {
            for (k = i; k < m; k++) {
                scale += std::fabs(u(k, i));
            }
            if (scale) {
                for (k = i; k < m; k++) {
                    u(k, i) /= scale;
                    s += u(k, i) * u(k, i);
                }
                f = u(i, i);
                g = -sign(std::sqrt(s), f);
                h = f * g - s;
                u(i, i) = f - g;
                if (i != n - 1) {
                    for (j = l; j < n; j++) {
                        for (s = 0, k = i; k < m; k++) {
                            s += u(k, i) * u(k, j);
                        }
                        f = s / h;
                        for (k = i; k < m; k++) {
                            u(k, j) += f * u(k, i);
                        }
                    }
                }
                for (k = i; k < m; k++) {
                    u(k, i) *= scale;
                }
            }
        }
        w[i] = scale * g;

        // right-hand reduction
        g = s = scale = 0;
        if (i < m && i != n - 1) {
            for (k = l; k < n; k++) {
                scale += std::fabs(u(i, k));
            }
            if (scale) {
                for (k = l; k < n; k++) {
                    u(i, k) /= scale;
                    s += u(i, k) * u(i, k);
                }
                f = u(i, l);
                g = -sign(std::sqrt(s), f);
                h = f * g - s;
                u(i, l) = f - g;
                for (k = l; k < n; k++) {
                    rv1[k] = (T)u(i, k) / h;
                }
                if (i != m - 1) {
                    for (j = l; j < m; j++) {
                        for (s = 0, k = l; k < n; k++) {
                            s += u(j, k) * u(i, k);
                        }
                        for (k = l; k < n; k++) {
                            u(j, k) += s * rv1[k];
                        }
                    }
                }
                for (k = l; k < n; k++) {
                    u(i, k) *= scale;
                }
            }
        }
        anorm = svd_max(anorm, (std::fabs((T)w[i]) + std::fabs(rv1[i])));
    }

    // accumulate the right-hand transformation
    for (i = n - 1; i >= 0; i--) {
        if (i < n - 1) {
            if (g) {
                for (j = l; j < n; j++) {
                    v(j, i) = ((u(i, j) / u(i, l)) / g);
                }
                // T division to avoid underflow
                for (j = l; j < n; j++) {
                    for (s = 0, k = l; k < n; k++) {
                        s += u(i, k) * v(k, j);
                    }
                    for (k = l; k < n; k++) {
                        v(k, j) += s * v(k, i);
                    }
                }
            }
            for (j = l; j < n; j++) {
                v(i, j) = v(j, i) = 0;
            }
        }
        v(i, i) = 1;
        g = rv1[i];
        l = i;
    }

    // accumulate the left-hand transformation
    for (i = n - 1; i >= 0; i--) {
        l = i + 1;
        g = w[i];
        if (i < n - 1) {
            for (j = l; j < n; j++) {
                u(i, j) = 0;
            }
        }
        if (g) {
            g = 1 / g;
            if (i != n - 1) {
                for (j = l; j < n; j++) {
                    for (s = 0, k = l; k < m; k++) {
                        s += u(k, i) * u(k, j);
                    }
                    f = (s / u(i, i)) * g;
                    for (k = i; k < m; k++) {
                        u(k, j) += f * u(k, i);
                    }
                }
            }
            for (j = i; j < m; j++) {
                u(j, i) = u(j, i) * g;
            }
        } else {
            for (j = i; j < m; j++) {
                u(j, i) = 0;
            }
        }
        ++u(i, i);
    }

    // diagonalize the bidiagonal form
    for (k = n - 1; k >= 0; k--) {
        // loop over singular values
        for (its = 0; its < 30; its++) {
            // loop over allowed iterations
            flag = 1;
            for (l = k; l >= 0; l--) {
                // test for splitting
                nm = l - 1;
                if (std::fabs(rv1[l]) + anorm == anorm) {
                    flag = 0;
                    break;
                }
                if (std::fabs((T)w[nm]) + anorm == anorm) {
                    break;
                }
            }
            if (flag) {
                c = 0;
                s = 1;
                for (i = l; i <= k; i++) {
                    f = s * rv1[i];
                    if (std::fabs(f) + anorm != anorm) {
                        g = w[i];
                        h = pythag(f, g);
                        w[i] = (T)h;
                        h = 1 / h;
                        c = g * h;
                        s = -f * h;
                        for (j = 0; j < m; j++) {
                            y = u(j, nm);
                            z = u(j, i);
                            u(j, nm) = y * c + z * s;
                            u(j, i) = z * c - y * s;
                        }
                    }
                }
            }
            z = w[k];
            if (l == k) {
                // convergence
                if (z < 0) {
                    // make singular value nonnegative
                    w[k] = -z;
                    for (j = 0; j < n; j++) {
                        v(j, k) = -v(j, k);
                    }
                }
                break;
            }
            if (its >= 30) {
                printf("No convergence after 30 iterations");
                return;
            }

            // shift from bottom 2 x 2 minor
            x = w[l];
            nm = k - 1;
            y = w[nm];
            g = rv1[nm];
            h = rv1[k];
            f = ((y - z) * (y + z) + (g - h) * (g + h)) / (2 * h * y);
            g = pythag(f, (T)1);
            f = ((x - z) * (x + z) +
                 h * ((y / (f + sign(g, f))) - h)) /
                x;

            // next QR transformation
            c = s = 1;
            for (j = l; j <= nm; j++) {
                i = j + 1;
                g = rv1[i];
                y = w[i];
                h = s * g;
                g = c * g;
                z = pythag(f, h);
                rv1[j] = z;
                c = f / z;
                s = h / z;
                f = x * c + g * s;
                g = g * c - x * s;
                h = y * s;
                y = y * c;
                for (jj = 0; jj < n; jj++) {
                    x = v(jj, j);
                    z = v(jj, i);
                    v(jj, j) = x * c + z * s;
                    v(jj, i) = z * c - x * s;
                }
                z = pythag(f, h);
                w[j] = z;
                if (z) {
                    z = 1 / z;
                    c = f * z;
                    s = h * z;
                }
                f = (c * g) + (s * y);
                x = (c * y) - (s * g);
                for (jj = 0; jj < m; jj++) {
                    y = u(jj, j);
                    z = u(jj, i);
                    u(jj, j) = y * c + z * s;
                    u(jj, i) = z * c - y * s;
                }
            }
            rv1[l] = 0;
            rv1[k] = f;
            w[k] = x;
        }
    }
}

typedef SVDMat3<double> SVDMat3d;
typedef SVDVec3<double> SVDVec3d;
typedef SVDMat3<float> SVDMat3f;
typedef SVDVec3<float> SVDVec3f;

