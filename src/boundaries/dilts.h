/* date = August 17th 2021 21:16 */
#pragma once
#include <grid.h>
#include <cutil.h>

/**************************************************************/
//                  D I L T S   M E T H O D                   //
//                      Spoke version                         //
/**************************************************************/

/*
* The original Dilts method for 3D is quite troublesome
* to implement in GPU. However the Spoke version is quite
* precise and considering we accept a few false positives
* in the interior of the fluid I'll implement this version.
*/

template<typename T> inline __bidevice__
bool DoesSpheresIntersect(T pi, T pj, Float ri, Float rj){
    Float d = Distance(pi, pj);
    return !(d > (ri + rj) - Epsilon);
}

/*
* Uniformly samples the circle around pi in 2D.
*/
inline __bidevice__
vec2f DiltsSpokeTakeUniformPoint(vec2f pi, Float r, int &done,
                                 int &sp, int &unused, int spoke_samples)
{
    Float samples = (Float)spoke_samples;
    Float angle = (Float)sp * TwoPi / samples;
    Float co, so;
#if defined(__CUDA_ARCH__)
    __sincosf(angle, &so, &co);
#else
    so = sin(angle);
    co = cos(angle);
#endif
    vec2f of = r * vec2f(co, so);
    sp += 1;
    done = (sp >= spoke_samples) ? 1 : 0;
    return pi + of;
}

/*
* Uniformly samples the sphere around pi in 3D.
* TODO: It would be best to sample this with the sphere density method.
*/
inline __bidevice__
vec3f DiltsSpokeTakeUniformPoint(vec3f pi, Float r, int &done,
                                 int &i, int &j, int spoke_samples)
{
    Float samples = (Float)spoke_samples;
    Float u = i * Pi / samples;
    Float v = j * TwoPi / samples;
    Float sinu, cosu, sinv, cosv;
#if defined(__CUDA_ARCH__)
    __sincosf(u, &sinu, &cosu);
    __sincosf(v, &sinv, &cosv);
#else
    sinu = sin(u), cosu = cos(u);
    sinv = sin(v), cosv = cos(v);
#endif

    j += 1;
    if(j > spoke_samples){
        i += 1;
        j = 0;
    }

    done = (i > spoke_samples) ? 1 : 0;
    return r * vec3f(sinu * cosv, sinu * sinv, cosu) + pi;

}

template<typename T, typename U, typename Q> inline __bidevice__
int DiltsSpokeParticleIsBoundary(Grid<T, U, Q> *domain, ParticleSet<T> *pSet, int pId){
    int *neighbors = nullptr;
    T candidates[256];
    int max_size = 256;
    int n = 0;
    T pi = pSet->GetParticlePosition(pId);
    unsigned int cellId = domain->GetLinearHashedPosition(pi);
    int count = domain->GetNeighborsOf(cellId, &neighbors);

    // TODO: grab compensation factor from spacing multiplier and pass it here,
    //       most simulations implemented in bubbles use 1.8/2.0 scaling
    //       so I'll just hardcode 0.9 here.
    Float rad = pSet->GetRadius() * 0.9;

    for(int i = 0; i < count; i++){
        Cell<Q> *cell = domain->GetCell(neighbors[i]);
        ParticleChain *pChain = cell->GetChain();
        int size = cell->GetChainLength();
        for(int j = 0; j < size; j++){
            if(pChain->pId != pId){
                T pj = pSet->GetParticlePosition(pChain->pId);
                if(DoesSpheresIntersect(pi, pj, rad, rad)){
                    if(!(n < max_size)){
                        printf("Too many hits?\n");
                    }else{
                        candidates[n++] = pj;
                    }
                }
            }

            pChain = pChain->next;
        }
    }

    int done = 0;
    int iti = 0, itj = 0;
    // hardcode sample count, I do believe 32 is waaay more than enough
    int samples = 32;
    while(!done){
        int is_inside = 0;
        T point = DiltsSpokeTakeUniformPoint(pi, rad, done, iti, itj, samples);
        for(int s = 0; s < n; s++){
            T pj = candidates[s];
            Float d = Distance(pj, point);
            if(d <= rad){
                is_inside = 1;
                break;
            }
        }

        if(!is_inside){
            return 1;
        }
    }
    return 0;
}

template<typename T, typename U, typename Q>
__global__ void DiltsComputeKernel(Grid<T, U, Q> *domain, ParticleSet<T> *pSet){
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if(i < pSet->GetParticleCount()){
        int b = DiltsSpokeParticleIsBoundary(domain, pSet, i);
        pSet->SetParticleV0(i, b);
    }
}

template<typename T, typename U, typename Q> __host__
void DiltsSpokeBoundary(Grid<T, U, Q> *domain, ParticleSet<T> *pSet){
    int N = pSet->GetParticleCount();
    GPULaunch(N, GPUKernel(DiltsComputeKernel<T, U, Q>), domain, pSet);
}
