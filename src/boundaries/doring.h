/* date = November 1th 2021 21:50 */
#pragma once
#include <kernel.h>
#include <grid.h>
#include <cutil.h>
#include <bound_util.h>
#include <transform.h>

/**************************************************************/
//        R A N D L E S - D O R I N G   M E T H O D           //
//                     Randles-Doring                         //
/**************************************************************/

/*
* This method is quite interesting, it is a result of several different
* works. Unfortunatelly I was unnable to reliably generate the boundary
* for 3D because picking μ values is quite difficult, but it kinda works
* for 2D. Anyways it is here for completeness reasons, but it shouldn't be used.
* This comes from the paper:
*    A fast algorithm for free-surface particles detection
*                in 2D and 3D SPH methods
* However in here I only implement the first stage of classification based
* on the renormalization matrix B.
*/

/*
* Disclaimer: The μ value Marrone uses is μ = 0.75, I couldn't make it
* work under this value of μ, but μ = 0.45 seems to hold for 2D during
* simulation. I'm however unnable to consistently generate boundaries for
* 3D. I don't see anything in their paper that indicates that we need to change
* anything for 3D, but the paper itself doesn't show any images in 3D of the
* actual boundary so maybe there is something else that needs to be done.
*/

// default kernel size expansion size for the Random-Doring/Marrone
#define RDM_KERNEL_EXPANSION 3.0

// this μ value seems to work decently for 2D
#define RDM_MU_2D 0.45

// this μ is from the paper, I could not find a correct value
#define RDM_MU_3D 0.75

inline bb_cpu_gpu vec2f RandlesDoringGradient(Float rho, Float distance, vec2f dir){
    SphStdKernel2 kernel(rho);
    return kernel.gradW(distance, dir);
}

inline bb_cpu_gpu vec3f RandlesDoringGradient(Float rho, Float distance, vec3f dir){
    SphStdKernel3 kernel(rho);
    return kernel.gradW(distance, dir);
}

/*
* Solve the 2D characteristic equation of a given tensor:
* Given the renormalization A solves the relation λ² - trace(A)λ + det(A) = 0
* and return the minimum value. Solutions are:
*     λ1 = trace(A)/2 + sqrt(T²/4 - det(A))
*     λ2 = trace(A)/2 - sqrt(T²/4 - det(A))
*/
inline bb_cpu_gpu Float ComputeMinEigenvalue(const Matrix2x2 &m){
    Float trace  = Trace(m);
    Float det    = Determinant(m);
    Float trace2 = trace * trace;
    Float sdelta = sqrt(Max(0.f, trace2 - 4.0 * det));
    Float L1 = 0.5 * (trace + sdelta);
    Float L2 = 0.5 * (trace - sdelta);
    return Min(L1, L2);
}

/*
* Solves the 2D characteristic equation of a given tensor when
* the inversion is zero, i.e.: λ² - trace(A)λ = 0, solutions are:
*     λ1 = 0
*     λ2 = trace(A)
* In the original paper there is nothing talking about λ = 0,
* it makes sense since this implies det(A) = 0 and A not invertible
* but since the formulation is based on the fact that it is
* possible to invert the renormalization matrix I won't return
* 0 here.
*/
inline bb_cpu_gpu Float ComputeMinEigenvalueZero(const Matrix2x2 &m){
    return Trace(m);
}

/*
* Use SVD for computing the 3D eigenvalues. Should be more stable than
* using cardano. I dont however expects this to fix the 3D issues.
*/
inline bb_cpu_gpu Float ComputeMinEigenvalue(const Matrix3x3 &m){
    vec3f S;
    m.Eigenvalues(S);
    return Min(S[0], Min(S[1], S[2]));
}

inline bb_cpu_gpu Float ComputeMinEigenvalueZero(const Matrix3x3 &m){
    return ComputeMinEigenvalue(m);
}

/*
* Main routine, computes the minimum eigenvalue for each particle
* based on the renormalization proposed by Marrone. There are some
* things it does not make explicit, such as:
*      - What to do with λ = 0,
*      - What exactly to do if particles don't have enough neighbors within h,
*      - How to refine μ
* So this algorithm is kind of a guess based on the paper, since I couldn't
* find any base implementation and was unable to get any response from the author.
*/
template<typename T, typename U, typename Q, typename M> inline bb_cpu_gpu
void RandlesDoringEigenvalue(ParticleSet<T> *pSet, Grid<T, U, Q> *domain,
                             Float h, Float *L, int pId)
{
    Float rho = RDM_KERNEL_EXPANSION * h;
    T pi = pSet->GetParticlePosition(pId);
    Float mass = pSet->GetMass();
    unsigned int cellId = domain->GetLinearHashedPosition(pi);
    int any_parts = 0;

    M Ti;
    Ti.Set(0);
    /* Make this query in N3x3 just so we get a big particle neighborhood */
    domain->ForAllNeighborsOf(cellId, 3, [&](Cell<Q> *cell, U cid, int lid) -> int{
        int n = cell->GetChainLength();
        ParticleChain *pChain = cell->GetChain();
        for(int i = 0; i < n; i++){
            T pj = pSet->GetParticlePosition(pChain->pId);
            Float dj = pSet->GetParticleDensity(pChain->pId);

            Float distance = Distance(pj, pi);
            T dir(0);
            if(distance > 0){
                dir = (pj - pi) / distance;
            }

            T v = RandlesDoringGradient(rho, distance, dir);
            if(v.LengthSquared() > 0){
                v = Normalize(v);
            }

            Ti.TensorAdd(v * (mass / dj));

            pChain = pChain->next;
            any_parts += 1;
        }

        return 0;
    });

    if(any_parts == 0){
        printf("[RDM] Error: No particles in tensor\n");
        L[pId] = 0;
    }else{
        Float pL = 0;
        Float det = Determinant(Ti);
        if(IsZero(det)){
            /*
            * This is not strictly correct as the renormalization Ti is not
            * invertible, but we can simplify computation. This is a terrible
            * situation because if we set the λ for this particle to be
            * zero, it could be added to boundary however setting to +-Infinity
            * will stretch the λ range of the domain and miss-compute other
            * particles. I can't find any references to what to do here
            * so I'll attempt to compute the λ values for the singular Ti,
            * I have no idea if this is valid or close to correct, but it is
            * what it is.
            */
            pL = ComputeMinEigenvalueZero(Ti);
        }else{
            pL = ComputeMinEigenvalue(HighpInverse(Ti));
        }

        L[pId] = pL;
    }
}

inline bb_cpu_gpu
int RandlesDoringIsBoundaryParticle(Float *L, Float mu, Float Lmax,
                                    Float Lmin, int pId)
{
    Float pL = L[pId];
    Float ref = (pL - Lmin) / (Lmax - Lmin);
    return ref < mu ? 1 : 0;
}

inline vec2f RandlesDoringGetEigenvalueLimits(Float *L, int N){
    Float Lmin = Infinity;
    Float Lmax = -Infinity;
    for(int i = 0; i < N; i++){
        Float Li = L[i];
        Lmax = Max(Li, Lmax);
        Lmin = Min(Li, Lmin);
    }

    return vec2f(Lmin, Lmax);
}

inline bb_cpu_gpu void
RandlesDoringEigenvalueDecl(ParticleSet3 *pSet, Grid3 *domain, Float h, Float *L, int i)
{
    RandlesDoringEigenvalue<vec3f, vec3ui, Bounds3f, Matrix3x3>(pSet, domain, h, L, i);
}

inline bb_cpu_gpu void
RandlesDoringEigenvalueDecl(ParticleSet2 *pSet, Grid2 *domain, Float h, Float *L, int i)
{
    RandlesDoringEigenvalue<vec2f, vec2ui, Bounds2f, Matrix2x2>(pSet, domain, h, L, i);
}

template<typename T, typename U, typename Q> inline
void RandlesDoringEigenvalueImpl(ParticleSet<T> *pSet, Grid<T, U, Q> *domain,
                                 Float h, Float *L)
{
    int N = pSet->GetParticleCount();
    AutoParallelFor("RandlesDoringEigenvalue", N, AutoLambda(int i){
        RandlesDoringEigenvalueDecl(pSet, domain, h, L, i);
    });
}

template<typename T, typename U, typename Q> inline
void RandlesDoringBoundary(ParticleSet<T> *pSet, Grid<T, U, Q> *domain,
                           Float h, Float mu=-1)
{
    static int warned = 0;
    if(warned == 0){
        printf("[RDM] Warning: Randles-Doring method implementation is quite unstable.\n"
               "               The μ value should be manually adjusted. For 2D μ = 0.45 is decent.\n"
               "               Could not find a suitable μ for 3D.\n");
        warned = 1;
    }

    if(mu < 0){
        mu = domain->GetDimensions() == 2 ? RDM_MU_2D : RDM_MU_3D;
    }

    int N = pSet->GetParticleCount();
    Float *L = cudaAllocateUnregisterVx(Float, N);
    for(int i = 0; i < N; i++) L[i] = 0;

    RandlesDoringEigenvalueImpl(pSet, domain, h, L);

    vec2f Ls = RandlesDoringGetEigenvalueLimits(L, N);
    AutoParallelFor("RandlesDoringIsBoundaryParticle", N, AutoLambda(int i){
        int b = RandlesDoringIsBoundaryParticle(L, mu, Ls.y, Ls.x, i);
        pSet->SetParticleV0(i, b);
    });
    //printf("[RDM] Range [λmin, λmax] : [%g, %g]\n", Ls.x, Ls.y);

    cudaFree(L);
}
