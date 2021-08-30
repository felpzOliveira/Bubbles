#pragma once
#include <grid.h>
#include <cutil.h>
#include <dilts.h>
#include <bound_util.h>

/**************************************************************/
//      L A Y E R E D   N E I G H B O R   M E T H O D         //
//                      Trivial version                       //
/**************************************************************/

/*
* The following is the direct imlementation for the version used in thesis.
* It strictly implements L1 and L2 searches only.
* TODO: Implement generic Fn search for Li-boundaries for GPU, i.e.: port from CPU.
*/

__bidevice__ inline int LNMDomainQuerySize(int len1D, int dim){
    return (int)pow(len1D, dim);
}

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

template<typename T, typename Q>
__bidevice__ void LNMBoundaryPushWork(ParticleSet<T> *pSet, Cell<Q> *self,
                                      LNMWorkQueue *workQ)
{
    int count = self->GetChainLength();
    ParticleChain *pChain = self->GetChain();
    for(int i = 0; i < count; i++){
        workQ->Push(pChain->pId);
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

template<typename T, typename U, typename Q> __global__
void LNMBoundaryL2GeomKernel(ParticleSet<T> *pSet, Grid<T, U, Q> *domain,
                             Float preDelta, Float h, LNMWorkQueue *workQ)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if(i < workQ->size){
        int id = workQ->Fetch();
        T pi = pSet->GetParticlePosition(id);
        unsigned int cellId = domain->GetLinearHashedPosition(pi);
        Cell<Q> *cell = domain->GetCell(cellId);
        // TODO: This needs to be a queue in order to perform decently
        if(cell->GetLevel() == 2){
            int b = DiltsSpokeParticleIsBoundary(domain, pSet, id);
            if(b > 0){
                pSet->SetParticleV0(id, 2);
            }

            workQ->ids[id] = b;
        }
    }
}

template<typename T, typename U, typename Q>
__global__ void LNMBoundaryL2Kernel(ParticleSet<T> *pSet, Grid<T, U, Q> *domain, 
                                    Float preDelta, Float h, int algorithm,
                                    LNMWorkQueue *workQ)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if(id < domain->GetActiveCellCount()){
        int i = domain->GetActiveCellId(id);
        Cell<Q> *self = domain->GetCell(i);
        if(self->GetChainLength() > 0 && self->GetLevel() != 1){
            int *neighbors = nullptr;
            int count = domain->GetNeighborsOf(i, &neighbors);
            for(int s = 0; s < count; s++){
                Cell<Q> *cell = domain->GetCell(neighbors[s]);
                if(cell->GetLevel() == 1 && (cell->GetChainLength() <= preDelta)){
                    self->SetLevel(2);
                    if(algorithm == 0){ // fast
                        LNMBoundaryLNSet(pSet, self, 2);
                    }else if(algorithm == 1){ // asymmetry
                        LNMBoundaryL2Asymmetry(pSet, self, h, domain, 
                                               neighbors, count);
                    }else if(algorithm >= 2 && workQ){
                        LNMBoundaryPushWork(pSet, self, workQ);
                    }
                    
                    break;
                }
            }
        }
    }
}

template<typename T, typename U, typename Q>
__global__ void LNMBoundaryExtendL2Kernel(ParticleSet<T> *pSet, Grid<T, U, Q> *domain){
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if(id < domain->GetActiveCellCount()){
        int i = domain->GetActiveCellId(id);
        Cell<Q> *self = domain->GetCell(i);
        // promote unbounded L2 levels to L3
        if(self->GetLevel() == 2){
            LNMBoundaryLNSet(pSet, self, 3);
        }
    }
}

template<typename T, typename U, typename Q>
__global__ void LNMBoundaryLKKernel(ParticleSet<T> *pSet, Grid<T, U, Q> *domain,
                                    Float preDelta, Float h)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if(id < domain->GetActiveCellCount()){
        int i = domain->GetActiveCellId(id);
        Cell<Q> *self = domain->GetCell(i);
        if(self->GetChainLength() > 0 && self->GetLevel() != 1){
            int *neighbors = nullptr;
            int count = domain->GetNeighborsOf(i, &neighbors);
            for(int s = 0; s < count; s++){
                Cell<Q> *cell = domain->GetCell(neighbors[s]);
                if(cell->GetLevel() == 1 && cell->GetChainLength() <= preDelta){
                    self->SetLevel(-2);
                    LNMBoundaryLNSet(pSet, self, 2);
                    break;
                }
            }
        }else if(self->GetLevel() == 1){
            LNMBoundaryLNSet(pSet, self, 1);
        }else if(self->GetLevel() > 2){
            LNMBoundaryLNSet(pSet, self, self->GetLevel());
        }
    }
}

template<typename T, typename U, typename Q>
__global__ void LNMBoundaryL1Kernel(ParticleSet<T> *pSet, Grid<T, U, Q> *domain){
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if(id < domain->GetActiveCellCount()){
        int threshold = LNMDomainQuerySize(3, domain->dimensions);
        int i = domain->GetActiveCellId(id);
        Cell<Q> *self = domain->GetCell(i);
        int p = self->GetChainLength();
        self->SetLevel(-1);
        int *neighbors = nullptr;
        int count = domain->GetNeighborsOf(i, &neighbors);
        if(p < 1) return;
        if(count != threshold){
            self->SetLevel(1);
            LNMBoundaryLNSet(pSet, self, 1);
        }else{
            for(int s = 0; s < count; s++){
                Cell<Q> *cell = domain->GetCell(neighbors[s]);
                if(cell->GetChainLength() == 0){
                    self->SetLevel(1);
                    LNMBoundaryLNSet(pSet, self, 1);
                    break;
                }
            }
        }
    }
}

template<typename U> __bidevice__ inline
bool LNMBoundaryAreCoords1x1(const U u0, const U u1, int dim){
    int eval = 0;
    for(int s = 0; s < dim; s++){
        int ef = Absf(int(u0[s]) - int(u1[s]));
        if(ef < 2){
            eval += ef;
        }else{
            eval = 0;
            break;
        }
    }

    return eval > 0 && eval <= dim;
}

/*
 * This call perform full L1 and L2 classification in one step for the case
 * in the paper i.e.: F2(p) = 1. This is mostly symbolic, just to show
 * that it is possible to write a function to perform full classification
 * over N2x2.
 */
template<typename T, typename U, typename Q>
__global__ void LNMBoundarySingleKernel(ParticleSet<T> *pSet, Grid<T, U, Q> *domain,
                                        int preDelta)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if(id < domain->GetActiveCellCount()){
        int p = 0, b = 0;
        bool is_l1 = false, is_l2 = false;
        U cand[125];
        U sources[125];
        int *nei = nullptr;
        const int dim = domain->dimensions;
        int threshold = LNMDomainQuerySize(3, dim);
        int i = domain->GetActiveCellId(id);
        Cell<Q> *self = domain->GetCell(i);
        U u = domain->GetCellIndex(i);

        self->SetLevel(-1);
        if(self->GetChainLength() == 0) return;

        int count = domain->GetNeighborsOf(i, &nei);

        if(count != threshold){
            self->SetLevel(1);
            LNMBoundaryLNSet(pSet, self, 1);
            return;
        }

        domain->ForAllNeighborsOf(i, 2, [&](Cell<Q> *cell, U cid, int lid) -> int{
            if(i == lid) return 0;

            int n = cell->GetChainLength();
            bool is_n1x1 = LNMBoundaryAreCoords1x1(cid, u, dim);
            if(n == 0){
                if(is_n1x1){
                    is_l1 = true;
                    return 1;
                }else{
                    sources[b++] = cid;
                }
            }else if(is_n1x1 && n <= preDelta){
                count = domain->GetNeighborsOf(lid, &nei);
                if(count != threshold){
                    is_l2 = true;
                    p = 0, b = 0;
                    return 1;
                }else if(!is_l2){
                    cand[p++] = cid;
                }
            }
            return 0;
        });

        if(is_l1){
            self->SetLevel(1);
            LNMBoundaryLNSet(pSet, self, 1);
        }else if(b > 0 && p > 0){
            for(int s = 0; s < b && !is_l2; s++){
                U src = sources[s];
                for(int t = 0; t < p && !is_l2; t++){
                    U cnd = cand[t];
                    is_l2 = LNMBoundaryAreCoords1x1(src, cnd, dim);
                }
            }
        }

        if(is_l2){
            self->SetLevel(2);
            LNMBoundaryLNSet(pSet, self, 2);
        }
    }
}

template<typename T, typename U, typename Q>
__host__ void LNMBoundarySingle(ParticleSet<T> *pSet, Grid<T, U, Q> *domain, Float h){
    /* Get the minimum in case its not a regular grid */
    T len = domain->GetCellSize();
    Float maxd = MinComponent(len);

    Float delta = std::pow((maxd / h), (Float)domain->dimensions); // eq 3.14
    delta -= domain->dimensions > 2 ? 0 : 1; // offset compensation for 2D

    /* N2x2 classification */
    GPULaunch(domain->GetActiveCellCount(),
              GPUKernel(LNMBoundarySingleKernel<T, U, Q>), pSet, domain, delta);
}

/*
* Actual boundary routine.
*/
template<typename T, typename U, typename Q>
__host__ void LNMBoundary(ParticleSet<T> *pSet, Grid<T, U, Q> *domain,
                          Float h, int algorithm=0, LNMWorkQueue *workQ=nullptr)
{
    /* Get the minimum in case its not a regular grid */
    T len = domain->GetCellSize();
    Float maxd = MinComponent(len);
    
    Float delta = std::pow((maxd / h), (Float)domain->dimensions); // eq 3.14
    delta -= domain->dimensions > 2 ? 0 : 1; // offset compensation for 2D

    if(workQ){
        workQ->Reset();
    }

    /* Classify L1 */
    GPULaunch(domain->GetActiveCellCount(),
              GPUKernel(LNMBoundaryL1Kernel<T, U, Q>), pSet, domain);

    /* Filter L2 by Voxel */
    GPULaunch(domain->GetActiveCellCount(), GPUKernel(LNMBoundaryL2Kernel<T, U, Q>),
              pSet, domain, delta, h, algorithm, workQ);

    if(algorithm >= 2 && workQ){
        /* Apply Geometric intersection */
        int N = workQ->size;
        GPULaunch(N, GPUKernel(LNMBoundaryL2GeomKernel<T, U, Q>),
                   pSet, domain, delta, h, workQ);
    }

}

/**************************************************************/
//                 V O X E L   C L A S S I F I E R            */
/**************************************************************/
template<typename T, typename U, typename Q>
__bidevice__ bool LNMComputeOnce(Grid<T, U, Q> *domain, int refLevel, unsigned int cellId){
    int *neighbors = nullptr;
    int count = domain->GetNeighborsOf(cellId, &neighbors);
    const int dim = domain->dimensions;
    int threshold = (dim == 3) ? 27 : 9;

    Cell<Q> *self = domain->GetCell(cellId);
    int level = self->GetLevel();
    if(level > -1) return false;

    if(self->GetChainLength() == 0){
        self->SetLevel(0);
        return true;
    }

    if(count != threshold){
        self->SetLevel(1);
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
__host__ int LNMClassifyLazyGPU(Grid<T, U, Q> *domain, int levels=-1, int startLevel=0){
    bool done = false;
    int level = startLevel;
    int N = domain->GetCellCount();
    
    while(!done){
        domain->indicator = 0;
        GPULaunch(N, GPUKernel(LNMOnceKernel<T, U, Q>), domain, level);
        
        if(domain->indicator == 0){
            done = true;
        }

        if(levels > -1){
            done = (level >= levels);
        }

        if(!done){
            level++;
        }
    }
    
    domain->SetLNMMaxLevel(level);
    
    return level;
}

template<typename T, typename U, typename Q>
__global__ void LNMParticleAttributesKernel(Grid<T, U, Q> *domain, ParticleSet<T> *pSet){
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if(i < pSet->GetParticleCount()){
        T pi = pSet->GetParticlePosition(i);
        unsigned int cellId = domain->GetLinearHashedPosition(pi);
        Cell<Q> *cell = domain->GetCell(cellId);
        int level = cell->GetLevel();
        if(level > 0){
            pSet->SetParticleV0(i, level);
        }else{
            pSet->SetParticleV0(i, 0);
        }
    }
}

template<typename T, typename U, typename Q> __host__ 
void LNMAssignParticlesAttributesGPU(Grid<T, U, Q> *domain, ParticleSet<T> *pSet){
    int N = pSet->GetParticleCount();
    GPULaunch(N, GPUKernel(LNMParticleAttributesKernel<T, U, Q>), domain, pSet);
}

template<typename T, typename U, typename Q>
__host__ void LNMBoundaryExtended(ParticleSet<T> *pSet, Grid<T, U, Q> *domain,
                                  Float h, int max_level, int algorithm=0)
{
    // classify voxels
    LNMClassifyLazyGPU<T, U, Q>(domain, max_level);
    LNMAssignParticlesAttributesGPU<T, U, Q>(domain, pSet);

    /* Get the minimum in case its not a regular grid */
    T len = domain->GetCellSize();
    Float maxd = MinComponent(len);

    Float delta = std::pow((maxd / h), (Float)domain->dimensions); // eq 3.14
    delta -= domain->dimensions > 2 ? 0 : 1; // offset compensation for 2D

    /* Filter and Classify LK */
    GPULaunch(domain->GetActiveCellCount(), GPUKernel(LNMBoundaryLKKernel<T, U, Q>),
              pSet, domain, delta, h);

    /* Promote L2 to L3 as the interface is compromised */
    GPULaunch(domain->GetActiveCellCount(), GPUKernel(LNMBoundaryExtendL2Kernel<T, U, Q>),
              pSet, domain);
}

