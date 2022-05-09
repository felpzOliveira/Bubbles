/* date = August 21st 2021 22:12 */
#pragma once
#include <grid.h>
#include <cutil.h>

template<typename T> inline __bidevice__ int QueuePushEntry(T *Q){
#if defined(__CUDA_ARCH__)
    return atomicAdd(&Q->size, 1);
#else
    return __atomic_fetch_add(&Q->size, 1, __ATOMIC_SEQ_CST);
#endif
}

template<typename T> inline __bidevice__ int QueueGetEntry(T *Q){
#if defined(__CUDA_ARCH__)
    return atomicAdd(&Q->entry, 1);
#else
    return __atomic_fetch_add(&Q->entry, 1, __ATOMIC_SEQ_CST);
#endif
}

template<typename T>
class WorkQueue{
    public:
    int size;
    int entry;
    int capacity;
    int counter;
    T *ids;
    __bidevice__ WorkQueue(){}
    __host__ void SetSlots(int n){
        ids = cudaAllocateVx(T, n);
        for(int i = 0; i < n; i++){
            ids[i] = T(0);
        }
        size = 0;
        entry = 0;
        counter = 0;
        capacity = n;
    }

    __bidevice__ int Push(T id){
        int at = QueuePushEntry(this);
        if(at >= capacity){
            printf("Too many entries\n");
        }
        ids[at] = id;
        return at;
    }

    __bidevice__ T Fetch(int *where=nullptr){
        int at = QueueGetEntry(this);
        if(where) *where = at;
        return ids[at];
    }

    __bidevice__ void IncreaseCounter(){
    #if defined(__CUDA_ARCH__)
        atomicAdd(&counter, 1);
    #else
        __atomic_fetch_add(&counter, 1, __ATOMIC_SEQ_CST);
    #endif
    }

    __host__ void ResetEntry(){
        entry = 0;
    }

    __host__ void Reset(){
        entry = 0;
        size = 0;
        counter = 0;
    }
};

typedef WorkQueue<int> LNMWorkQueue;
typedef WorkQueue<int> IntWorkQueue;
typedef WorkQueue<vec3f> SandimWorkQueue3;
typedef WorkQueue<vec2f> SandimWorkQueue2;

template<typename T, typename U, typename Q, typename Func> __bidevice__
void ForAllNeighbors(Grid<T, U, Q> *domain, ParticleSet<T> *pSet, T pi, Func fn,
                     int depth=1)
{
    int neighbors[343]; // at most 7³
    if(depth > 3){
        printf("Warning: Too large domain search, max is 3 ( 7³ )\n");
        depth = 3;
    }

    unsigned int cellId = domain->GetLinearHashedPosition(pi);
    int count = domain->GetNeighborListFor(cellId, depth, neighbors);
    int terminate = 0;

    for(int i = 0; i < count && !terminate; i++){
        Cell<Q> *cell = domain->GetCell(neighbors[i]);
        ParticleChain *pChain = cell->GetChain();
        int size = cell->GetChainLength();
        for(int j = 0; j < size && !terminate; j++){
            terminate = fn(pChain->pId);
            pChain = pChain->next;
        }
    }
}

template<typename T, typename U, typename Q, typename Func> __bidevice__
void ForAllNeighbors(Grid<T, U, Q> *domain, ParticleSet<T> *pSet, int pId, Func fn){
    int *neighbors = nullptr;
    T pi = pSet->GetParticlePosition(pId);
    unsigned int cellId = domain->GetLinearHashedPosition(pi);
    int count = domain->GetNeighborsOf(cellId, &neighbors);
    int terminate = 0;

    for(int i = 0; i < count && !terminate; i++){
        Cell<Q> *cell = domain->GetCell(neighbors[i]);
        ParticleChain *pChain = cell->GetChain();
        int size = cell->GetChainLength();
        for(int j = 0; j < size && !terminate; j++){
            if(pChain->pId != pId){
                terminate = fn(pChain->pId);
            }

            pChain = pChain->next;
        }
    }
}
