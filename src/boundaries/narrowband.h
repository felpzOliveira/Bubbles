/* date = November 16th 2021 21:38 */
#pragma once
#include <geometry.h>
#include <particle.h>
#include <grid.h>

/**************************************************************/
//                 N A R R O W  -  B A N D                    //
/**************************************************************/

/*
* Computes the narrow-band from a mathematical perspective, i.e.:
* the tubular regions over the boundary. Boundary should be extracted
* over some of the methods implemented in bubbles, here we simply expand it
* to a ball of radius ρ, i.e.: Given a particle pi that belongs to the boundary
* a particle pj is said to be on the narrow-band if the distance between pi and pj
* is within the ball of radius ρ.
*
* For the extension routine to work it needs that the classification was performed
* on particles v0 buffer (all methods in bubbles do that).
*/

template<typename T, typename U, typename Q> __bidevice__
void NarrowBandCompute(ParticleSet<T> *pSet, Grid<T, U, Q> *domain, int queryDepth,
                       Float rho, int pId)
{
    T pi = pSet->GetParticlePosition(pId);
    unsigned int cellId = domain->GetLinearHashedPosition(pi);
    domain->ForAllNeighborsOf(cellId, queryDepth,
    [&](Cell<Q> *cell, U cid, int lid) -> int{
        int n = cell->GetChainLength();
        ParticleChain *pChain = cell->GetChain();
        for(int i = 0; i < n; i++){
            T pj = pSet->GetParticlePosition(pChain->pId);
            Float dist = Distance(pi, pj);
            if(dist <= rho && pSet->GetParticleV0(pChain->pId) == 1){
                pSet->SetParticleV0(pId, 2);
                return 1;
            }
            pChain = pChain->next;
        }
        return 0;
    });
}

template<typename T, typename U, typename Q> __global__
void NarrowBandComputeKernel(ParticleSet<T> *pSet, Grid<T, U, Q> *domain,
                             int queryDepth, Float rho)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if(i < pSet->GetParticleCount()){
        if(pSet->GetParticleV0(i) == 0){
            NarrowBandCompute(pSet, domain, queryDepth, rho, i);
        }
    }
}

template<typename T, typename U, typename Q> __host__
void GeometricalNarrowBand(ParticleSet<T> *pSet, Grid<T, U, Q> *domain, Float rho){
    int use_cpu = GetSystemUseCPU();
    int N = pSet->GetParticleCount();
    T cellDims = domain->GetCellSize();
    Float maxd = MinComponent(cellDims);
    int queryDepth = (int)(std::ceil(rho / maxd)) + 1;

    if(!use_cpu){
        GPULaunch(N, GPUKernel(NarrowBandComputeKernel<T, U, Q>),
                  pSet, domain, queryDepth, rho);
    }else{
        ParallelFor(0, N, [&](int i) -> void{
            NarrowBandCompute(pSet, domain, queryDepth, rho, i);
        });
    }
}
