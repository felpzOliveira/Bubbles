/* date = August 18th 2021 17:51 */
#pragma once
#include <grid.h>
#include <cutil.h>
#include <kernel.h>

/**************************************************************/
//                X I A O W E I   M E T H O D                 //
//                    Asymmetry Value method                  //
/**************************************************************/

/*
* Xiaowei methods is the method mentioned in the paper:
*      Staggered meshless solid-fluid coupling
* I refer to this method as 'Asymmetry Value' method because
* it attempts to compute a particle property referred as 'asymmetry'.
* It is an extension of the classic color field method.
*/
#define XIAOWEI_KERNEL_EXPANSION 6.0

template<typename T, typename U, typename Q> __bidevice__
int XiaoweiParticleIsBoundary(ParticleSet<T> *pSet, Grid<T, U, Q> *domain,
                              Float h, int pId)
{
    /*
    * All simulations in Bubbles (so far) use the same target density,
    * the Water density, so I'm gonna hardcode it here. If ever we change
    * this, this routine needs to get the target density given by the solver.
    */
    Float rhot = 0.7 * WaterDensity;
    Float rho = XIAOWEI_KERNEL_EXPANSION * h;

    Float di = pSet->GetParticleDensity(pId);
    if(di < rhot){
        return 1;
    }

    int *neighbors = nullptr;
    Float mass = pSet->GetMass();
    T pi = pSet->GetParticlePosition(pId);
    unsigned int cellId = domain->GetLinearHashedPosition(pi);
    int count = domain->GetNeighborsOf(cellId, &neighbors);

    SphStdKernel3 kernel3d(rho);
    SphStdKernel2 kernel2d(rho);
    Float Wsum = 0;
    T sum(0.0f);

    for(int i = 0; i < count; i++){
        Cell<Q> *cell = domain->GetCell(neighbors[i]);
        ParticleChain *pChain = cell->GetChain();
        int size = cell->GetChainLength();
        for(int j = 0; j < size; j++){
            Float W = 0;
            T pj = pSet->GetParticlePosition(pChain->pId);
            Float distance = Distance(pi, pj);
            if(domain->GetDimensions() == 3){ // 3d
                W = kernel3d.W(distance);
            }else{ // 2d
                W = kernel2d.W(distance);
            }

            sum += pj * W;
            Wsum += W;

            pChain = pChain->next;
        }
    }

    sum = sum / Wsum;
    Float asymmetry = Distance(pi, sum);

    // I don't quite understand how this const is picked
    if(asymmetry > 0.1 * rho){
        return 1;
    }

    return 0;
}

template<typename T, typename U, typename Q> __global__
void XiaoweiBoundaryKernel(ParticleSet<T> *pSet, Grid<T, U, Q> *domain, Float h){
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if(i < pSet->GetParticleCount()){
        int b = XiaoweiParticleIsBoundary(pSet, domain, h, i);
        pSet->SetParticleV0(i, b);
    }
}

template<typename T, typename U, typename Q> __host__
void XiaoweiBoundaryCPU(ParticleSet<T> *pSet, Grid<T, U, Q> *domain, Float h, int i){
    if(i < pSet->GetParticleCount()){
        int b = XiaoweiParticleIsBoundary(pSet, domain, h, i);
        pSet->SetParticleV0(i, b);
    }
}

template<typename T, typename U, typename Q> __host__
void XiaoweiBoundary(ParticleSet<T> *pSet, Grid<T, U, Q> *domain, Float h){
    int use_cpu = GetSystemUseCPU();
    int N = pSet->GetParticleCount();
    if(!use_cpu){
        GPULaunch(N, GPUKernel(XiaoweiBoundaryKernel<T, U, Q>), pSet, domain, h);
    }else{
        ParallelFor(0, N, [&](int i) -> void{
            XiaoweiBoundaryCPU(pSet, domain, h, i);
        });
    }
}
