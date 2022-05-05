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
* The method is partially implemented here. The implementation here does not compute
* the eigenvalues described in the paper, because of the issues when computing the
* invertible tensor B, see 'doring.h'. Here we perform a complete geometric test
* based on the scan test described in the paper.
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
int MarroneBoundaryIsInteriorParticle(ParticleSet<T> *pSet, Grid<T, U, Q> *domain,
                                      Float h, int pId)
{
    T t, tau;
    int inside = 0;
    Float h_sqrt2 = 1.414213562 * h;
    T pi = pSet->GetParticlePosition(pId);
    T ni = pSet->GetParticleNormal(pId);

    // [NaN Handling] 1- In case the particle is really interior
    //                   sph cannot compute the normal vector, give up.
    if(ni.Length() < 0.98) return 1;

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

        // [NaN Handling] 2- In case particle are too close the sphere
        //                   scan test presented by Marrone is impossible, give up.
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

template<typename T, typename U, typename Q> __global__
void MarroneBoundaryKernel(ParticleSet<T> *pSet, Grid<T, U, Q> *domain, Float h){
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if(i < pSet->GetParticleCount()){
        int inside = MarroneBoundaryIsInteriorParticle(pSet, domain, h, i);
        pSet->SetParticleV0(i, 1-inside);
    }
}

template<typename T, typename U, typename Q> inline __host__
void MarroneBoundary(ParticleSet<T> *pSet, Grid<T, U, Q> *domain, Float h){
    int N = pSet->GetParticleCount();
    if(!GetSystemUseCPU()){
        GPULaunch(N, GPUKernel(MarroneBoundaryKernel<T, U, Q>), pSet, domain, h);
    }else{
        ParallelFor(0, N, [&](int i) -> void{
            int inside = MarroneBoundaryIsInteriorParticle(pSet, domain, h, i);
            pSet->SetParticleV0(i, 1-inside);
        });
    }
}
