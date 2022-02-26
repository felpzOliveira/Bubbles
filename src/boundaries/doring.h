/* date = November 1th 2021 21:50 */
#pragma once
#include <kernel.h>
#include <grid.h>
#include <cutil.h>
#include <bound_util.h>
#include <transform.h>

/**************************************************************/
//        R A N D L E S - D O R I N G   M E T H O D           //
//                Randles-Doring/Marrone Method               //
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

inline __bidevice__ vec2f RandlesDoringGradient(Float rho, Float distance, vec2f dir){
    SphStdKernel2 kernel(rho);
    return kernel.gradW(distance, dir);
}

inline __bidevice__ vec3f RandlesDoringGradient(Float rho, Float distance, vec3f dir){
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
inline __bidevice__ Float ComputeMinEigenvalue(const Matrix2x2 &m){
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
* possible to invert the renormalization matrix I'll won't return
* 0 here.
*/
inline __bidevice__ Float ComputeMinEigenvalueZero(const Matrix2x2 &m){
    return Trace(m);
}

/*
* Solves the 3D characterstic equation of a given tensor:
* Given the renormalization A solves the relation:
*    -λ³ + trace(A)λ² - 0.5(trace(A)² - trace(A²))λ + det(A) = 0
* Not going to do Newton, this is going to simply be Cardano solutions
* I'll use double to increase precision but there are a lot of sqrts
* and cbrt so precision will struggle.
*/
inline __bidevice__ Float ComputeMinEigenvalue(const Matrix3x3 &m){
    Matrix3x3 m2 = Matrix3x3::Mul(m, m);
    /* division by A = -1 to bring this to: x³ + bx² + cx + d = 0 */
    double B = -Trace(m);
    double C = 0.5 * (B * B - Trace(m2));
    double D = -Determinant(m);
    double B3 = B / 3.0;

    /* compute c/3a - b²/9a with a = 1 */
    double q = C / 3.0 - B * B / 9.0;
    /* compute -b³/27a³ + bc/6a² -d/2a with a = 1*/
    double r = (-B3 * B3 * B3) + (B*C/6.0) - (D/2.0);
    /* compute term inside square root: r*r + q*q*q */
    double r1 = r*r + q*q*q;

    // if r1 > 0 than there are 3 distinct solutions 1 real and 2 complex
    if(r1 > 0){
        /* compute positive part */
        double s = r + sqrt(r1);
        /* compute cubic */
        s = s < 0 ? -cbrt(-s) : cbrt(s);
        // now get the other side
        double t = r - sqrt(r1);
        t = t < 0 ? -cbrt(-t) : cbrt(t);
        double L1 = s + t - B3;
        return (Float)L1;
    }else if(IsHighpZero(r1)){
        // All roots are real, with two equals
        double r13 = r < 0 ? -cbrt(-r) : cbrt(r);
        double L1 = 2.0 * r13 - B3;
        double L2 = -r13 -B3;
        return (Float)Min(L1, L2);
    }else{
        // all different roots
        q = -q;
        double q3 = q * q * q;
        q3 = acos(r / sqrt(q3));
        double r13 = 2.0 * sqrt(q);
        /* rotate solutions */
        double L1 = -B3 + r13 * cos((q3) / 3.0);
        double L2 = -B3 + r13 * cos((q3 + TwoPi) / 3.0);
        double L3 = -B3 + r13 * cos((q3 + 2.0 * TwoPi) / 3.0);
        return (Float)Min(Min(L1, L2), L3);
    }
}

/*
* Solves the 3D characteristic equation of a given tensor when
* the inversion is zero, i.e.:
*    λ(-λ² + trace(A)λ - 0.5(trace(A)² - trace(A²)) = 0
* Solutions are λ = 0 and the second degree poly.
* Again, I'm not sure what to do with λ = 0 for det(A) = 0,
* for 3D it is possible that the characteristic has no solutions
* besides 0, so it will need to return here, but I'm not sure
* about consequences of doing so.
*/
inline __bidevice__ Float ComputeMinEigenvalueZero(const Matrix3x3 &m){
    Matrix3x3 m2 = Matrix3x3::Mul(m, m);
    /* Divide by A = -1 to get x² + bx + c = 0 */
    double B = -Trace(m);
    double C = 0.5 * (B * B - Trace(m2));
    /* Solve the 2d equation */
    double delta =  B * B - 4.0 * C;
    if(delta > 0 || IsZero(delta)){ // the original equation has 3 solutions
        double sdelta = sqrt(Max(0, delta));
        double L1 = 0.5 * (-B + sdelta);
        double L2 = 0.5 * (-B - sdelta);
        return (Float)Min(L1, L2);
    }else{ // There is no solution
        return 0;
    }
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
template<typename T, typename U, typename Q, typename M> inline __bidevice__
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

inline __bidevice__
int RandlesDoringIsBoundaryParticle(Float *L, Float mu, Float Lmax,
                                    Float Lmin, int pId)
{
    Float pL = L[pId];
    Float ref = (pL - Lmin) / (Lmax - Lmin);
    return ref < mu ? 1 : 0;
}

template<typename T, typename U, typename Q, typename M> __global__
void RandlesDoringEigenvalueKernel(ParticleSet<T> *pSet,
                                   Grid<T, U, Q> *domain, Float h, Float *L)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if(i < pSet->GetParticleCount()){
        RandlesDoringEigenvalue<T, U, Q, M>(pSet, domain, h, L, i);
    }
}

template<typename T>
__global__ void RandlesDoringGetBoundaryKernel(ParticleSet<T> *pSet, Float mu,
                                               Float *L, vec2f Lrange)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if(i < pSet->GetParticleCount()){
        int b = RandlesDoringIsBoundaryParticle(L, mu, Lrange.y, Lrange.x, i);
        pSet->SetParticleV0(i, b);
    }
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

inline __host__
void RandlesDoringEigenvalueImpl(ParticleSet2 *pSet, Grid2 *domain,
                                 Float h, Float *L)
{
    int N = pSet->GetParticleCount();
    if(!GetSystemUseCPU()){
        GPULaunch(N,
        GPUKernel(RandlesDoringEigenvalueKernel<vec2f, vec2ui, Bounds2f, Matrix2x2>),
                  pSet, domain, h, L);
    }else{
        ParallelFor(0, N, [&](int i) -> void{
            RandlesDoringEigenvalue<vec2f, vec2ui, Bounds2f, Matrix2x2>(pSet, domain, h, L, i);
        });
    }
}

inline __host__
void RandlesDoringEigenvalueImpl(ParticleSet3 *pSet, Grid3 *domain,
                                 Float h, Float *L)
{
    int N = pSet->GetParticleCount();
    if(!GetSystemUseCPU()){
        GPULaunch(N,
        GPUKernel(RandlesDoringEigenvalueKernel<vec3f, vec3ui, Bounds3f, Matrix3x3>),
                  pSet, domain, h, L);
    }else{
        ParallelFor(0, N, [&](int i) -> void{
            RandlesDoringEigenvalue<vec3f, vec3ui, Bounds3f, Matrix3x3>(pSet, domain, h, L, i);
        });
    }
}

template<typename T, typename U, typename Q> inline __host__
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

    if(!GetSystemUseCPU()){
        GPULaunch(N, GPUKernel(RandlesDoringGetBoundaryKernel<T>),
                  pSet, mu, L, Ls);
    }else{
        ParallelFor(0, N, [&](int i) -> void{
            int b = RandlesDoringIsBoundaryParticle(L, mu, Ls.y, Ls.x, i);
            pSet->SetParticleV0(i, b);
        });
    }
    //printf("[RDM] Range [λmin, λmax] : [%g, %g]\n", Ls.x, Ls.y);

    cudaFree(L);
}
