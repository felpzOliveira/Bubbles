/* date = May 5th 2022 21:25 */
#pragma once
#include <kernel.h>
#include <grid.h>
#include <cutil.h>
#include <bound_util.h>

/**************************************************************/
//               M A R R O N E   M E T H O D                  //
//                     Sphere Scan Test                       //
/**************************************************************/

/*
* Implementation of the sphere scan test performed by Marrone in:
*  Fast free-surface detection and level-set function definition in SPH solvers
*
* and also in:
*  A fast algorithm for free-surface particles detection in 2D and 3D SPH methods
*
* Here we also implement the method present in:
*  Simple free-surface detection in two and three-dimensional SPH solver
*
* as an adaptation to Marrone however we use the same formulation as Marrone but
* with the cover vector presented and the small radius considerations.
*
* The implementation here does not compute the eigenvalues described in the paper,
* because of the issues when computing the invertible tensor B, see 'doring.h'.
* Here we perform a complete geometric test based on the scan test described in the paper.
*/


// Frisvald method with the fast fix presented by Pixar
inline __bidevice__ vec3f GetNormalTangent(vec3f n){
    // apply modified frisvald method
    Float sign = copysignf(1.f, n.z);
    Float a = -1.f / (sign + n.z);
    Float b = n.x * n.y * a;
    return vec3f(1.f + sign * n.x * n.x * a, sign * b, -sign * n.x);
}

inline __bidevice__ vec2f GetNormalTangent(vec2f n){
    // for 2D a simple clokwise rotation gets it
    return vec2f(n.y, -n.x);
}

// Reference point 'T' and τ in the paper
template<typename T> inline __bidevice__
void MarroneAuxiliarVectors(T p, T normal, T &t, T &tau, Float h){
    tau = GetNormalTangent(normal);
    t = p + normal * h;
}

// Check for the 2D case the conditions under equation 4, page 7.
inline __bidevice__
bool MarroneConeArcCheck(vec2f pji, vec2f pjt, vec2f tau, vec2f ni, Float pji_len, Float h){
    Float arc = Absf(Dot(ni, pjt)) + Absf(Dot(tau, pjt));
    return arc <= h;
}

// Check for the 3D case the conditions under equation 5, page 10.
inline __bidevice__
bool MarroneConeArcCheck(vec3f pji, vec3f pjt, vec3f tau, vec3f ni, Float pji_len, Float h){
    Float arc = acos(Dot(ni, pji) / pji_len);
    return arc <= PiOver4;
}

// The method: simply apply geometric test under all particles that have normal.
// Note that Bubbles can have normal vector as (0,0,0) since normal vectors
// are computed using SPH.
template<typename T, typename U, typename Q> inline __bidevice__
int MarroneBoundaryConeCheck(ParticleSet<T> *pSet, Grid<T, U, Q> *domain,
                             Float h, T pi, T ni, int pId)
{
    T t, tau;
    int inside = 0;
    Float h_sqrt2 = 1.414213562 * h;

    // Compute τ and T
    MarroneAuxiliarVectors(pi, ni, t, tau, h);

    // For each particle in the neighborhood check conditions 4 and 5.
    ForAllNeighbors(domain, pSet, pi, [&](int j) -> int{
        if(pId == j) return 0;

        T pj = pSet->GetParticlePosition(j);
        T pjt = pj - t;
        T pji = pj - pi;

        Float pji_len = pji.Length();
        Float pjt_len = pjt.Length();

        // [NaN Handling] In case particle are too close the sphere
        //                scan test presented by Marrone is impossible, give up.
        if(IsZero(pji_len)){
            return 0;
        }

        if((pji_len >= h_sqrt2) && (pjt_len < h)){
            inside = 1;
            return 1;
        }

        bool arc_check = MarroneConeArcCheck(pji, pjt, tau, ni, pji_len, h);
        if((pji_len < h_sqrt2) && arc_check){
            inside = 1;
            return 1;
        }

        return 0;
    });

    return inside;
}

template<typename T, typename U, typename Q> inline __bidevice__
int MarroneBoundaryIsInteriorParticle(ParticleSet<T> *pSet, Grid<T, U, Q> *domain,
                                      Float h, int pId)
{
    T t, tau;
    T pi = pSet->GetParticlePosition(pId);
    T ni = pSet->GetParticleNormal(pId);

    // [NaN Handling] In case the particle is really interior or has a symmetric
    //                neighborhood, sph cannot compute the normal vector, give up.
    if(ni.Length() < 0.92) return 1;

    return MarroneBoundaryConeCheck(pSet, domain, h, pi, ni, pId);
}

template<typename T, typename U, typename Q> inline __host__
void MarroneBoundary(ParticleSet<T> *pSet, Grid<T, U, Q> *domain, Float h){
    int N = pSet->GetParticleCount();

    AutoParallelFor("MarroneBoundary", N, AutoLambda(int i){
        int inside = MarroneBoundaryIsInteriorParticle(pSet, domain, h, i);
        pSet->SetParticleV0(i, 1-inside);
    });
}

/************************************************************************/
//                A D A P T E D   V E R S I O N                         //
/************************************************************************/
/*
* Adaptation from Marrone using the cover vector and small radius considerations from:
*    Simple free-surface detection in two and three-dimensional SPH solver
*/
template<typename T, typename U, typename Q> inline __bidevice__
T MarroneAdaptComputeCoverVector(ParticleSet<T> *pSet, Grid<T, U, Q> *domain,
                                 Float h, T pi, int pId, int &n_counter)
{
    T bi(0);
    Float bilen = 0;
    Float h2 = 2.0 * h;
    n_counter = 0;

    // compute the cover vector, equation 6, page 5.
    ForAllNeighbors(domain, pSet, pi, [&](int j) -> int{
        if(j == pId) return 0;
        T pj = pSet->GetParticlePosition(j);

        T xij = pi - pj;
        Float len = xij.Length();
        if(IsZero(len)){
            return 0;
        }

        if(len < h2){
            n_counter += 1;
            bi = bi + xij / len;
        }

        return 0;
    });

    if(n_counter > 0){
        bilen = bi.Length();
        if(!IsZero(bilen)){
            bi = bi / bilen;
        }
    }

    return bi;
}

template<typename T, typename U, typename Q> inline __bidevice__
int MarroneAdaptIsInteriorParticle(ParticleSet<T> *pSet, Grid<T, U, Q> *domain,
                                   Float h, int pId)
{
    T pi = pSet->GetParticlePosition(pId);

    // nth values from equation 8, page 6.
    int threshold = domain->GetDimensions() == 2 ? 4 : 15;

    int n_counter = 0;
    T bi = MarroneAdaptComputeCoverVector(pSet, domain, h, pi, pId, n_counter);

    if(n_counter <= threshold) return 0;

    return MarroneBoundaryConeCheck(pSet, domain, h, pi, bi, pId);
}

template<typename T, typename U, typename Q> inline __bidevice__
void MarroneAdaptComputeWorkQueue(ParticleSet<T> *pSet, Grid<T, U, Q> *domain,
                                  WorkQueue<vec4f> *workQ, Float h, int pId)
{
    T pi = pSet->GetParticlePosition(pId);
    pSet->SetParticleV0(pId, 0);

    // nth values from equation 8, page 6.
    int threshold = domain->GetDimensions() == 2 ? 4 : 15;

    int n_counter = 0;
    T bi = MarroneAdaptComputeCoverVector(pSet, domain, h, pi, pId, n_counter);

    if(n_counter <= threshold){
        pSet->SetParticleV0(pId, 1);
    }else{
        vec4f val(0);
        for(int i = 0; i < domain->GetDimensions(); i++){
            val[i] = bi[i];
        }
        val[3] = pId;
        workQ->Push(val);
    }
}

template<typename T, typename U, typename Q> inline __bidevice__
void MarroneAdaptProcessWorkQueue(ParticleSet<T> *pSet, Grid<T, U, Q> *domain,
                                  WorkQueue<vec4f> *workQ, Float h)
{
    T bi;
    vec4f val = workQ->Fetch();
    int pId = (int)val[3];
    for(int i = 0; i < domain->GetDimensions(); i++){
        bi[i] = val[i];
    }

    T pi = pSet->GetParticlePosition(pId);
    int inside = MarroneBoundaryConeCheck(pSet, domain, h, pi, bi, pId);
    pSet->SetParticleV0(pId, 1-inside);
}

template<typename T, typename U, typename Q> inline __host__
void MarroneAdaptBoundary(ParticleSet<T> *pSet, Grid<T, U, Q> *domain, Float h,
                          WorkQueue<vec4f> *workQ)
{
    int N = pSet->GetParticleCount();
    AutoParallelFor("MarroneAdaptWorkQ", N, AutoLambda(int i){
        MarroneAdaptComputeWorkQueue(pSet, domain, workQ, h, i);
    });

    AutoParallelFor("MarroneAdaptWorkQBoundary", workQ->size, AutoLambda(int i){
        MarroneAdaptProcessWorkQueue(pSet, domain, workQ, h);
    });
}

template<typename T, typename U, typename Q> inline __host__
void MarroneAdaptBoundary(ParticleSet<T> *pSet, Grid<T, U, Q> *domain, Float h){
    int N = pSet->GetParticleCount();
    AutoParallelFor("MarroneAdaptBoundary", N, AutoLambda(int i){
        int inside = MarroneAdaptIsInteriorParticle(pSet, domain, h, i);
        pSet->SetParticleV0(i, 1-inside);
    });
}
