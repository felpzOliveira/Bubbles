#pragma once
#include <grid.h>
#include <cutil.h>

// TODO: Implement GPU version and non iterative version

template<typename T, typename U, typename Q>
__bidevice__ bool CNMComputeOnce(Grid<T, U, Q> *domain, int refLevel, unsigned int cellId){
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

/* Set grid levels by Closest Neighbor Method, return the maximum level found */
template<typename T, typename U, typename Q>
__host__ int CNMComputeBoundary(Grid<T, U, Q> *domain, int levels=-1){
    int total = domain->GetCellCount();
    for(int i = 0; i < total; i++){
        Cell<Q> *cell = domain->GetCell(i);
        cell->SetLevel(-1);
    }
    
    bool done = false;
    int level = 0;
    while(!done){
        int changed = 0;
        for(unsigned int i = 0; i < total; i++){
            if(CNMComputeOnce(domain, level, i)){
                changed = 1;
            }
        }
        
        level ++;
        done = (changed == 0);
        if(levels > 0) done |= level > levels;
    }
    
    domain->SetCNMMaxLevel(level);
    
    return level;
}

template<typename T, typename U, typename Q>
__global__ void CNMOnceKernel(Grid<T, U, Q> *domain, int level){
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if(i < domain->GetCellCount()){
        if(CNMComputeOnce(domain, level, i)){
            domain->indicator = 1; // NOTE: Concurrency
        }
    }
}

template<typename T, typename U, typename Q>
__host__ void CNMInvalidateCells(Grid<T, U, Q> *domain){
    int total = domain->GetCellCount();
    for(int i = 0; i < total; i++){
        Cell<Q> *cell = domain->GetCell(i);
        cell->SetLevel(-1);
    }
}

// Lazy implementation
template<typename T, typename U, typename Q>
__host__ int CNMClassifyLazyGPU(Grid<T, U, Q> *domain, int levels=-1){
    bool done = false;
    int level = 0;
    int nThreads = CUDA_THREADS_PER_BLOCK;
    int N = domain->GetCellCount();
    
    while(!done){
        domain->indicator = 0;
        CNMOnceKernel<<<(N + nThreads - 1) / nThreads, nThreads>>>(domain, level);
        cudaDeviceAssert();
        
        level++;
        done = (domain->indicator == 0 || (levels > 0 && level > levels));
    }
    
    domain->SetCNMMaxLevel(level);
    
    return level;
}

template<typename T, typename U, typename Q, typename F>
__host__ void CNMProcessDeep(Grid<T, U, Q> *domain, F callback){
    unsigned int total = domain->GetCellCount();
    for(unsigned int i = 0; i < total; i++){
        Cell<Q> *cell = domain->GetCell(i);
        cell->SetLevel(-1);
    }
    
    bool done = false;
    int level = 0;
    while(!done){
        int changed = 0;
        for(unsigned int i = 0; i < total; i++){
            if(CNMComputeOnce(domain, level, i)){
                callback(domain->GetCell(i), level);
                changed = 1;
            }
        }
        
        level++;
        done = (changed == 0);
    }
    
    domain->SetCNMMaxLevel(level);
}