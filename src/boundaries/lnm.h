#pragma once
#include <grid.h>
#include <cutil.h>

/**************************************************************/
//      L A Y E R E D   N E I G H B O R   M E T H O D         //
//                      Trivial version                       //
/**************************************************************/

/*
* The following is the direct imlementation for the version used in thesis.
* It strictly implements L1 and L2 searches only.
* TODO: Implement generic Fn search for Li-boundaries for GPU.
*/

// going to classify on v0 buffer as simulation step should already be resolved
template<typename T, typename Q>
__bidevice__ void LNMBoundaryLNSet(ParticleSet<T> *pSet, Cell<Q> *self, int L){
    int count = self->GetChainLength();
    ParticleChain *pChain = self->GetChain();
    for(int i = 0; i < count; i++){
        pSet->SetParticleV0(pChain->pId, L);
        pChain = pChain->next;
    }
}

template<typename T, typename U, typename Q>
__bidevice__ void LNMBoundaryL2Asymmetry(ParticleSet<T> *pSet, Cell<Q> *self, Float h,
                                         Grid<T, U, Q> *domain, int *neighbors, int nCount)
{
    SphStdKernel2 kernel2(h);
    SphStdKernel3 kernel3(h);
    int count = self->GetChainLength();
    ParticleChain *pSelfChain = self->GetChain();
    const int dim = domain->dimensions;
    
    for(int i = 0; i < count; i++){
        T sum(0);
        Float Wsum = 0;
        T pi = pSet->GetParticlePosition(pSelfChain->pId);
        for(int v = 0; v < nCount; v++){
            Cell<Q> *cell = domain->GetCell(neighbors[v]);
            int level = cell->GetLevel(); // promote
            if(level == -1 || level == 2){
                int size = cell->GetChainLength();
                ParticleChain *pChain = cell->GetChain();
                for(int j = 0; j < size; j++){ // asymmetry acc
                    T pj = pSet->GetParticlePosition(pChain->pId);
                    Float distance = Distance(pi, pj);
                    Float W = 0;
                    if(dim == 3){
                        W = kernel3.W(distance);
                    }else{ // dim == 2
                        W = kernel2.W(distance);
                    }
                    
                    Wsum += W;
                    sum += pj * W;
                    
                    pChain = pChain->next;
                }
            }
        }
        
        sum = sum / Wsum;
        Float asymmetry = Distance(pi, sum);
        if(asymmetry > 1e-8){
            pSet->SetParticleV0(pSelfChain->pId, 2);
        }
        
        pSelfChain = pSelfChain->next;
    }
}

template<typename T, typename U, typename Q>
__global__ void LNMBoundaryL2Kernel(ParticleSet<T> *pSet, Grid<T, U, Q> *domain, 
                                    Float preDelta, Float h, int algorithm)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if(id < domain->GetActiveCellCount()){
        int i = domain->GetActiveCellId(id);
        Cell<Q> *self = domain->GetCell(i);
        if(self->GetChainLength() > 0 && self->GetLevel() != 1){
            int *neighbors = nullptr;
            int count = domain->GetNeighborsOf(i, &neighbors);
            for(int i = 0; i < count; i++){
                Cell<Q> *cell = domain->GetCell(neighbors[i]);
                if(cell->GetLevel() == 1 && cell->GetChainLength() < preDelta){
                    self->SetLevel(2);
                    if(algorithm == 0){ // fast
                        LNMBoundaryLNSet(pSet, self, 2);
                    }else if(algorithm == 1){ // asymmetry
                        LNMBoundaryL2Asymmetry(pSet, self, h, domain, 
                                               neighbors, count);
                    }
                    
                    break;
                }
            }
        }
    }
}


template<typename T, typename U, typename Q>
__global__ void LNMBoundaryL1Kernel(ParticleSet<T> *pSet, Grid<T, U, Q> *domain){
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if(id < domain->GetActiveCellCount()){
        const int dim = domain->dimensions;
        int threshold = (dim == 3) ? 27 : 9;
        int i = domain->GetActiveCellId(id);
        Cell<Q> *self = domain->GetCell(i);
        self->SetLevel(-1);
        int *neighbors = nullptr;
        int count = domain->GetNeighborsOf(i, &neighbors);
        if(count != threshold){
            self->SetLevel(1);
            LNMBoundaryLNSet(pSet, self, 1);
        }else{
            for(int i = 0; i < count; i++){
                Cell<Q> *cell = domain->GetCell(neighbors[i]);
                if(cell->GetChainLength() == 0){
                    self->SetLevel(1);
                    LNMBoundaryLNSet(pSet, self, 1);
                    break;
                }
            }
        }
    }
}

/*
* Actual boundary routine.
*/
template<typename T, typename U, typename Q>
__host__ void LNMBoundary(ParticleSet<T> *pSet, Grid<T, U, Q> *domain, 
                          Float h, int algorithm=0)
{
    T len = domain->GetCellSize();
    Float maxd = MinComponent(len);
    Float delta = std::pow((maxd / h), (Float)domain->dimensions); // eq 3.14
    
    /* Classify L1 */
    GPULaunch(domain->GetActiveCellCount(), 
              GPUKernel(LNMBoundaryL1Kernel<T, U, Q>), pSet, domain);
    
    /* Filter L2 */
    GPULaunch(domain->GetActiveCellCount(), GPUKernel(LNMBoundaryL2Kernel<T, U, Q>), 
              pSet, domain, delta, h, algorithm);
}


/**************************************************************/
//                 V O X E L   C L A S S I F I E R            */
/**************************************************************/
template<typename T, typename U, typename Q>
__bidevice__ bool LNMComputeOnce(Grid<T, U, Q> *domain, int refLevel, unsigned int cellId){
    int *neighbors = nullptr;
    int count = domain->GetNeighborsOf(cellId, &neighbors);
    Cell<Q> *self = domain->GetCell(cellId);
    int level = self->GetLevel();
    if(level > -1) return false;
    
    if(self->GetChainLength() == 0){
        self->SetLevel(0);
        return true;
    }
    
    if(refLevel > 0){
        for(int i = 0; i < count; i++){
            if(cellId != neighbors[i]){
                Cell<Q> *cell = domain->GetCell(neighbors[i]);
                
                if(cell->GetLevel() == refLevel - 1){
                    self->SetLevel(refLevel);
                    return true;
                }
            }
        }
    }
    
    return false;
}

template<typename T, typename U, typename Q>
__global__ void LNMOnceKernel(Grid<T, U, Q> *domain, int level){
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if(i < domain->GetCellCount()){
        if(LNMComputeOnce(domain, level, i)){
            domain->indicator = 1; // NOTE: Concurrency
        }
    }
}


template<typename T, typename U, typename Q>
__host__ void LNMInvalidateCells(Grid<T, U, Q> *domain){
    int total = domain->GetCellCount();
    for(int i = 0; i < total; i++){
        Cell<Q> *cell = domain->GetCell(i);
        cell->SetLevel(-1);
    }
}

// Lazy implementation
template<typename T, typename U, typename Q>
__host__ int LNMClassifyLazyGPU(Grid<T, U, Q> *domain, int levels=-1){
    bool done = false;
    int level = 0;
    int N = domain->GetCellCount();
    
    while(!done){
        domain->indicator = 0;
        GPULaunch(N, GPUKernel(LNMOnceKernel<T, U, Q>), domain, level);
        
        level++;
        done = (domain->indicator == 0 || (levels > 0 && level > levels));
    }
    
    domain->SetLNMMaxLevel(level);
    
    return level;
}
