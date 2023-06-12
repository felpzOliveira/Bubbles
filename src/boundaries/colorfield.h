/* date = August 18th 2021 19:13 */
#pragma once
#include <grid.h>
#include <cutil.h>
#include <kernel.h>

/**************************************************************/
//             C O L O R   F I E L D   M E T H O D            //
//                Classic Color Field method                  //
/**************************************************************/

/*
* This is the classic Color Field method. It is based on computing the
* density of each particles and detect regions where the gradient is != 0.
* It is quite accurate but it can have some strong failures on edge cases.
*/

#define CF_KERNEL_EXPANSION 3.0

inline bb_cpu_gpu vec3f KernelGradient(Float rho, Float distance, vec3f dir){
    SphStdKernel3 kernel3d(rho);
    return kernel3d.gradW(distance, dir);
}

inline bb_cpu_gpu vec2f KernelGradient(Float rho, Float distance, vec2f dir){
    SphStdKernel2 kernel2d(rho);
    return kernel2d.gradW(distance, dir);
}

template<typename T, typename U, typename Q> bb_cpu_gpu
int CFParticleIsBoundary(ParticleSet<T> *pSet, Grid<T, U, Q> *domain,
                             Float h, int pId)
{
    Float rho = CF_KERNEL_EXPANSION * h;

    int *neighbors = nullptr;
    Float mass = pSet->GetMass();
    T pi = pSet->GetParticlePosition(pId);
    unsigned int cellId = domain->GetLinearHashedPosition(pi);
    int count = domain->GetNeighborsOf(cellId, &neighbors);

    T s(0.f);

    for(int i = 0; i < count; i++){
        Cell<Q> *cell = domain->GetCell(neighbors[i]);
        ParticleChain *pChain = cell->GetChain();
        int size = cell->GetChainLength();
        for(int j = 0; j < size; j++){
            T pj = pSet->GetParticlePosition(pChain->pId);
            Float dj = pSet->GetParticleDensity(pChain->pId);
            Float distance = Distance(pi, pj);

            if(!IsZero(dj) && !IsZero(distance)){
                T dir = (pj - pi) / distance;
                s += (mass / dj) * KernelGradient(rho, distance, dir);
            }

            pChain = pChain->next;
        }
    }

    Float l2 = s.Length();
    if(l2 > 7.0){
        return 1;
    }

    return 0;
}

template<typename T, typename U, typename Q>
void CFBoundary(ParticleSet<T> *pSet, Grid<T, U, Q> *domain, Float h){
    int N = pSet->GetParticleCount();
    AutoParallelFor("CFParticleIsBoundary", N, AutoLambda(int i){
        int b = CFParticleIsBoundary(pSet, domain, h, i);
        pSet->SetParticleV0(i, b);
    });
}
