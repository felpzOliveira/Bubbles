/* date = August 21st 2021 22:12 */
#pragma once
#include <grid.h>
#include <cutil.h>

template<typename T> inline bb_cpu_gpu int QueuePushEntry(T *Q){
#if defined(__CUDA_ARCH__)
    return atomicAdd(&Q->size, 1);
#else
    return __atomic_fetch_add(&Q->size, 1, __ATOMIC_SEQ_CST);
#endif
}

template<typename T> inline bb_cpu_gpu int QueueGetEntry(T *Q){
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
    bb_cpu_gpu WorkQueue(){}

    void SetSlots(int n, bool registered=true){
        if(registered)
            ids = cudaAllocateVx(T, n);
        else
            ids = cudaAllocateUnregisterVx(T, n);

        for(int i = 0; i < n; i++){
            ids[i] = T(0);
        }
        size = 0;
        entry = 0;
        counter = 0;
        capacity = n;
    }

    bb_cpu_gpu int Push(T id){
        int at = QueuePushEntry(this);
        if(at >= capacity){
            printf("Too many entries\n");
        }
        ids[at] = id;
        return at;
    }

    bb_cpu_gpu T At(int index){
        return ids[index];
    }

    bb_cpu_gpu T *Ref(int index){
        return &ids[index];
    }

    bb_cpu_gpu T Fetch(int *where=nullptr){
        int at = QueueGetEntry(this);
        if(where) *where = at;
        return ids[at];
    }

    bb_cpu_gpu void IncreaseCounter(){
    #if defined(__CUDA_ARCH__)
        atomicAdd(&counter, 1);
    #else
        __atomic_fetch_add(&counter, 1, __ATOMIC_SEQ_CST);
    #endif
    }

    void ResetEntry(){
        entry = 0;
    }

    void Reset(){
        entry = 0;
        size = 0;
        counter = 0;
    }
};

template<typename BaseQueue, unsigned int N>
class MultiWorkQueue{
    public:
    BaseQueue *qs[N];
    size_t count = 0;

    bb_cpu_gpu MultiWorkQueue(){
        for(size_t i = 0; i < N; i++){
            qs[i] = nullptr;
        }
    }

    void SetSlots(int maxItems, int id){
        if(qs[id] == nullptr){
            qs[id] = cudaAllocateVx(BaseQueue, 1);
            qs[id]->SetSlots(maxItems);
        }
    }

    bb_cpu_gpu BaseQueue *FetchQueueFor(int id){
        return qs[id];
    }

    void Reset(int id=-1){
        if(id >= 0) qs[id]->Reset();
        else{
            for(size_t i = 0; i < count; i++){
                qs[i]->Reset();
            }
        }
    }
};

template<typename T>
class DualWorkQueue{
    public:
    MultiWorkQueue<WorkQueue<T>, 2> mQueue;
    int active = 0;

    DualWorkQueue() = default;

    void SetSlots(int maxItems){
        mQueue.SetSlots(maxItems, 0);
        mQueue.SetSlots(maxItems, 1);
        active = 0;
    }

    bb_cpu_gpu WorkQueue<T> *ActiveQueue(){
        return mQueue.FetchQueueFor(active);
    }

    bb_cpu_gpu WorkQueue<T> *NextQueue(){
        return mQueue.FetchQueueFor(1-active);
    }

    void Flip(int reset=1){
        WorkQueue<T> *Q = ActiveQueue();
        Q->Reset();
        active = 1 - active;
    }

    bb_cpu_gpu void Push(T id){
        WorkQueue<T> *Q = NextQueue();
        Q->Push(id);
    }

    bb_cpu_gpu T Fetch(int *where=nullptr){
        WorkQueue<T> *Q = ActiveQueue();
        return Q->Fetch(where);
    }
};

typedef WorkQueue<int> LNMWorkQueue;
typedef WorkQueue<int> IntWorkQueue;
typedef WorkQueue<vec3f> SandimWorkQueue3;
typedef WorkQueue<vec2f> SandimWorkQueue2;

template<typename T, typename U, typename Q, typename Func> bb_cpu_gpu
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

template<typename T, typename U, typename Q, typename Func> bb_cpu_gpu
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
