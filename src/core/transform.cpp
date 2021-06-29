#include <transform.h>
#include <string.h>
#include <quaternion.h>
#include <shape.h>

//PBRT is an freaking excellent reference!
__bidevice__ void Interpolate(InterpolatedTransform *iTransform, Float t, Transform *transform){
    iTransform->Interpolate(t, transform);
}

__bidevice__ void InterpolatedTransform::Interpolate(Float t, Transform *transform){
    if(t <= t0){
        *transform = tStart;
        return;
    }

    if(t >= t1){
        *transform = tEnd;
        return;
    }

    // After decomposition the interpolated result can be computed by literally
    // interpolating each property (translation, rotation, scale) and combining
    // them in a single matrix given by TRS.

    // Compute interpolation value (0, 1)
    Float dt = (t - t0) / (t1 - t0);

    // For translation perform Linear interpolation on the decomposed translation
    vec3f translation = Lerp(T[0], T[1], dt);

    // For rotation apply Quaternion interpolation
    Quaternion quat = Qlerp(dt, R[0], R[1]);

    // For scale perform Linear interpolation on each element
    Matrix4x4 scale;
    for(int i = 0; i < 3; i++){
        for(int j = 0; j < 3; j++){
            scale.m[i][j] = Lerp(S[0].m[i][j], S[1].m[i][j], dt);
        }
    }

    // combine
    *transform = Translate(translation) * quat.ToTransform() * Transform(scale);
}

__bidevice__ void InterpolatedTransform::Decompose(const Matrix4x4 &m, vec3f *T,
                                                   Quaternion *Rquat, Matrix4x4 *S)
{
    // The decomposition works by attempting to solve M = TRS where M is affine
    // and compute values for T (translation) R (rotation) S (scale) matrix.
    // The translation is pretty easy to get by inspecting the last column of M.
    // We than remove this component and obtain a matrix M* that contains only RS.
    // Rotation can be computed by applying polar decomposition, i.e.: iterate:
    //       Mi+1 = 0.5 * (Mi + (Mi^t)^-1)     with M0 = M*
    // whenever (Mi+1 - Mi) becomes too small the matrix Mi contains the best
    // approximation of the rotation in the original M*.
    // Finally the scale matrix can be computed by reversing M* = RS => S = R^-1 * M*

    T->x = m.m[0][3];
    T->y = m.m[1][3];
    T->z = m.m[2][3];

    Matrix4x4 M = m;
    for (int i = 0; i < 3; ++i) M.m[i][3] = M.m[3][i] = 0.f;
    M.m[3][3] = 1.f;

    Float norm;
    int count = 0;
    Matrix4x4 R = M;
    do{
        Matrix4x4 Rnext;
        Matrix4x4 Rit = Inverse(Transpose(R));
        for(int i = 0; i < 4; ++i)
            for(int j = 0; j < 4; ++j)
                Rnext.m[i][j] = 0.5f * (R.m[i][j] + Rit.m[i][j]);

        norm = 0;
        for(int i = 0; i < 3; ++i){
            Float n = Absf(R.m[i][0] - Rnext.m[i][0]) +
                      Absf(R.m[i][1] - Rnext.m[i][1]) +
                      Absf(R.m[i][2] - Rnext.m[i][2]);

            norm = Max(norm, n);
        }
        R = Rnext;
    }while(++count < 100 && norm > .0001);

    *Rquat = Quaternion(R);

    *S = Matrix4x4::Mul(Inverse(R), M);
}

__bidevice__ InterpolatedTransform::InterpolatedTransform(Transform *e0, Transform *e1,
                                                          Float s0, Float s1)
: tStart(*e0), tEnd(*e1), t0(s0), t1(s1)
{
    Decompose(tStart.m, &T[0], &R[0], &S[0]);
    Decompose(tEnd.m, &T[1], &R[1], &S[1]);
    // Flip for choosing faster path on rotations
    if(Dot(R[0], R[1]) < 0){
        R[1] = -R[1];
    }
}

__bidevice__ bool SolveLinearSystem2x2(const Float A[2][2], const Float B[2], 
                                       Float *x0, Float *x1)
{
    Float det = A[0][0] * A[1][1] - A[0][1] * A[1][0];
    if (Absf(det) < 1e-10f) return false;
    *x0 = (A[1][1] * B[0] - A[0][1] * B[1]) / det;
    *x1 = (A[0][0] * B[1] - A[1][0] * B[0]) / det;
    if (IsNaN(*x0) || IsNaN(*x1)) return false;
    return true;
}

__bidevice__ Matrix4x4::Matrix4x4(Float mat[4][4]) { memcpy(m, mat, 16 * sizeof(Float)); }

__bidevice__ Matrix4x4::Matrix4x4(Float t00, Float t01, Float t02, Float t03, Float t10,
                                  Float t11, Float t12, Float t13, Float t20, Float t21,
                                  Float t22, Float t23, Float t30, Float t31, Float t32,
                                  Float t33) 
{
    m[0][0] = t00; m[0][1] = t01; m[0][2] = t02; m[0][3] = t03;
    m[1][0] = t10; m[1][1] = t11; m[1][2] = t12; m[1][3] = t13;
    m[2][0] = t20; m[2][1] = t21; m[2][2] = t22; m[2][3] = t23;
    m[3][0] = t30; m[3][1] = t31; m[3][2] = t32; m[3][3] = t33;
}

__bidevice__ Matrix4x4 Transpose(const Matrix4x4 &m){
    return Matrix4x4(m.m[0][0], m.m[1][0], m.m[2][0], m.m[3][0],
                     m.m[0][1], m.m[1][1], m.m[2][1], m.m[3][1],
                     m.m[0][2], m.m[1][2], m.m[2][2], m.m[3][2],
                     m.m[0][3], m.m[1][3], m.m[2][3], m.m[3][3]);
}

__bidevice__ Matrix4x4 Inverse(const Matrix4x4 &m){
    int indxc[4], indxr[4];
    int ipiv[4] = {0, 0, 0, 0};
    Float minv[4][4];
    memcpy(minv, m.m, 4 * 4 * sizeof(Float));
    for (int i = 0; i < 4; i++) {
        int irow = 0, icol = 0;
        Float big = 0.f;
        // Choose pivot
        for (int j = 0; j < 4; j++) {
            if (ipiv[j] != 1) {
                for (int k = 0; k < 4; k++) {
                    if (ipiv[k] == 0) {
                        if (std::abs(minv[j][k]) >= big) {
                            big = Float(Absf(minv[j][k]));
                            irow = j;
                            icol = k;
                        }
                    } else if (ipiv[k] > 1)
                        printf("Singular matrix in MatrixInvert\n");
                }
            }
        }
        ++ipiv[icol];
        // Swap rows _irow_ and _icol_ for pivot
        if (irow != icol) {
            for (int k = 0; k < 4; ++k) swap(minv[irow][k], minv[icol][k]);
        }
        indxr[i] = irow;
        indxc[i] = icol;
        if (minv[icol][icol] == 0.f) printf("Singular matrix in MatrixInvert\n");
        
        // Set $m[icol][icol]$ to one by scaling row _icol_ appropriately
        Float pivinv = 1. / minv[icol][icol];
        minv[icol][icol] = 1.;
        for (int j = 0; j < 4; j++) minv[icol][j] *= pivinv;
        
        // Subtract this row from others to zero out their columns
        for (int j = 0; j < 4; j++) {
            if (j != icol) {
                Float save = minv[j][icol];
                minv[j][icol] = 0;
                for (int k = 0; k < 4; k++) minv[j][k] -= minv[icol][k] * save;
            }
        }
    }
    // Swap columns to reflect permutation
    for (int j = 3; j >= 0; j--) {
        if (indxr[j] != indxc[j]) {
            for (int k = 0; k < 4; k++)
                swap(minv[k][indxr[j]], minv[k][indxc[j]]);
        }
    }
    return Matrix4x4(minv);
}

__bidevice__ Transform2 Scale2(Float u){
    return Scale2(u, u);
}

__bidevice__ Transform2 Scale2(Float x, Float y){
    AssertA(!IsZero(x) && !IsZero(y), "Zero scale Matrix3x3");
    Float ix = 1.0 / x;
    Float iy = 1.0 / y;
    Matrix3x3 m(x, 0, 0, 0, y, 0, 0, 0, 1);
    Matrix3x3 inv(ix, 0, 0, 0, iy, 0, 0, 0, 1);
    return Transform2(m, inv);
}

__bidevice__ Transform2 Translate2(Float x, Float y){
    Matrix3x3 m(1, 0, x, 0, 1, y, 0, 0, 1);
    Matrix3x3 inv(1, 0, -x, 0, 1, -y, 0, 0, 1);
    return Transform2(m, inv);
}

__bidevice__ Transform2 Translate2(Float u){
    return Translate2(u, u);
}

__bidevice__ Transform2 Rotate2(Float alpha){
    Float co = std::cos(alpha);
    Float si = std::sin(alpha);
    Float coi = std::cos(-alpha);
    Float sii = std::sin(-alpha);
    Matrix3x3 m(co, -si, 0, si, co, 0, 0, 0, 1);
    Matrix3x3 minv(coi, -sii, 0, sii, coi, 0, 0, 0, 1);
    return Transform2(m, minv);
}

__bidevice__ Transform2 Transform2::operator*(const Transform2 &t2) const{
    return Transform2(Matrix3x3::Mul(m, t2.m), Matrix3x3::Mul(t2.mInv, mInv));
}

__bidevice__ Transform Translate(const vec3f &delta) {
    Matrix4x4 m(1, 0, 0, delta.x, 0, 1, 0, delta.y, 0, 0, 1, delta.z, 0, 0, 0,
                1);
    Matrix4x4 minv(1, 0, 0, -delta.x, 0, 1, 0, -delta.y, 0, 0, 1, -delta.z, 0,
                   0, 0, 1);
    return Transform(m, minv);
}

__bidevice__ Transform Translate(Float x, Float y, Float z){
    return Translate(vec3f(x, y, z));
}

__bidevice__ Transform Translate(Float u){
    return Translate(vec3f(u, u, u));
}

__bidevice__ Transform Scale(Float x, Float y, Float z) {
    Matrix4x4 m(x, 0, 0, 0, 0, y, 0, 0, 0, 0, z, 0, 0, 0, 0, 1);
    Matrix4x4 minv(1 / x, 0, 0, 0, 0, 1 / y, 0, 0, 0, 0, 1 / z, 0, 0, 0, 0, 1);
    return Transform(m, minv);
}

__bidevice__ Transform Scale(Float u){
    return Scale(u, u, u);
}

__bidevice__ Transform Scale(const vec3f &delta){
    return Scale(delta.x, delta.y, delta.z);
}

__bidevice__ Transform RotateX(Float theta) {
    Float sinTheta = std::sin(Radians(theta));
    Float cosTheta = std::cos(Radians(theta));
    Matrix4x4 m(1, 0, 0, 0, 0, cosTheta, -sinTheta, 0, 0, sinTheta, cosTheta, 0,
                0, 0, 0, 1);
    return Transform(m, Transpose(m));
}

__bidevice__ Transform RotateY(Float theta) {
    Float sinTheta = std::sin(Radians(theta));
    Float cosTheta = std::cos(Radians(theta));
    Matrix4x4 m(cosTheta, 0, sinTheta, 0, 0, 1, 0, 0, -sinTheta, 0, cosTheta, 0,
                0, 0, 0, 1);
    return Transform(m, Transpose(m));
}

__bidevice__ Transform RotateZ(Float theta) {
    Float sinTheta = std::sin(Radians(theta));
    Float cosTheta = std::cos(Radians(theta));
    Matrix4x4 m(cosTheta, -sinTheta, 0, 0, sinTheta, cosTheta, 0, 0, 0, 0, 1, 0,
                0, 0, 0, 1);
    return Transform(m, Transpose(m));
}

__bidevice__ Transform Rotate(Float theta, const vec3f &axis) {
    vec3f a = Normalize(axis);
    Float sinTheta = std::sin(Radians(theta));
    Float cosTheta = std::cos(Radians(theta));
    Matrix4x4 m;
    // Compute rotation of first basis vector
    m.m[0][0] = a.x * a.x + (1 - a.x * a.x) * cosTheta;
    m.m[0][1] = a.x * a.y * (1 - cosTheta) - a.z * sinTheta;
    m.m[0][2] = a.x * a.z * (1 - cosTheta) + a.y * sinTheta;
    m.m[0][3] = 0;
    
    // Compute rotations of second and third basis vectors
    m.m[1][0] = a.x * a.y * (1 - cosTheta) + a.z * sinTheta;
    m.m[1][1] = a.y * a.y + (1 - a.y * a.y) * cosTheta;
    m.m[1][2] = a.y * a.z * (1 - cosTheta) - a.x * sinTheta;
    m.m[1][3] = 0;
    
    m.m[2][0] = a.x * a.z * (1 - cosTheta) - a.y * sinTheta;
    m.m[2][1] = a.y * a.z * (1 - cosTheta) + a.x * sinTheta;
    m.m[2][2] = a.z * a.z + (1 - a.z * a.z) * cosTheta;
    m.m[2][3] = 0;
    return Transform(m, Transpose(m));
}

// Rotate following the path around the sphere centered at point instead of origin
__bidevice__ Transform RotateAround(Float theta, const vec3f &axis, const vec3f &point){
    return Translate(point) * Rotate(theta, axis) * Translate(-point);
}

__bidevice__ Transform Transform::operator*(const Transform &t2) const{
    return Transform(Matrix4x4::Mul(m, t2.m), Matrix4x4::Mul(t2.mInv, mInv));
}

__bidevice__ bool Transform::SwapsHandedness() const{
    Float det = m.m[0][0] * (m.m[1][1] * m.m[2][2] - m.m[1][2] * m.m[2][1]) -
        m.m[0][1] * (m.m[1][0] * m.m[2][2] - m.m[1][2] * m.m[2][0]) +
        m.m[0][2] * (m.m[1][0] * m.m[2][1] - m.m[1][1] * m.m[2][0]);
    return det < 0;
}

__bidevice__ Bounds3f Transform::operator()(const Bounds3f &b) const{
    const Transform &M = *this;
    Bounds3f ret(M.Point(vec3f(b.pMin.x, b.pMin.y, b.pMin.z)));
    ret = Union(ret, M.Point(vec3f(b.pMax.x, b.pMin.y, b.pMin.z)));
    ret = Union(ret, M.Point(vec3f(b.pMin.x, b.pMax.y, b.pMin.z)));
    ret = Union(ret, M.Point(vec3f(b.pMin.x, b.pMin.y, b.pMax.z)));
    ret = Union(ret, M.Point(vec3f(b.pMin.x, b.pMax.y, b.pMax.z)));
    ret = Union(ret, M.Point(vec3f(b.pMax.x, b.pMax.y, b.pMin.z)));
    ret = Union(ret, M.Point(vec3f(b.pMax.x, b.pMin.y, b.pMax.z)));
    ret = Union(ret, M.Point(vec3f(b.pMax.x, b.pMax.y, b.pMax.z)));
    return ret;
}

__bidevice__ void Transform::Mesh(ParsedMesh *mesh) const{
    int it = Max(mesh->nVertices, mesh->nNormals);
    for(int i = 0; i < it; i++){
        if(i < mesh->nVertices){
            Point3f p = mesh->p[i];
            mesh->p[i] = Point(p);
        }
        
        if(i < mesh->nNormals){
            Normal3f n = mesh->n[i];
            mesh->n[i] = Normal(n);
        }
    }

    mesh->transform = *this;
}
