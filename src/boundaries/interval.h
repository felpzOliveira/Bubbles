/* date = May 19th 2021 15:25 */
#pragma once
#include <grid.h>
#include <cutil.h>
#include <bound_util.h>

/**************************************************************/
//               I N T E R V A L   M E T H O D                //
/**************************************************************/

/*
* Implementation of Sandim's new interval method for boundary detection in:
* Simple and reliable boundary detection for meshfree particle methods
*                       using interval analysis
*/

#define INTERVAL_LABEL_COVERED 0
#define INTERVAL_LABEL_UNCOVERED 1
#define INTERVAL_LABEL_PARTIALLY_COVERED 2

template<typename T>
class Stack{
    public:
    T data[256];
    unsigned int _top;
    unsigned int capacity;
    unsigned int size;
    __bidevice__ Stack(){
        _top = 0;
        capacity = 256;
        size = 0;
    }

    __bidevice__ void push(T item){
        if(!(_top < capacity)){
            printf("Not enough space\n");
            return;
        }

        memcpy(&data[_top++], &item, sizeof(T));
        size += 1;
    }

    __bidevice__ bool empty(){ return size < 1; }

    __bidevice__ void pop(){
        if(size > 0 && _top > 0){
            _top--;
            size--;
        }
    }

    __bidevice__ void top(T *item){
        if(size > 0 && _top > 0){
            memcpy(item, &data[_top-1], sizeof(T));
        }
    }

    __bidevice__ ~Stack(){}
};

template<typename T, typename Q> inline __bidevice__
void IntervalBoundaryGetBoundsFor(ParticleSet<T> *pSet, int i,
                                  Q *bounds, Float h)
{
    T p = pSet->GetParticlePosition(i);
    *bounds = Q(p - T(h), p + T(h));
}

template<typename T, typename U, typename Q> __bidevice__
bool IntervalParticleIsInterior(Grid<T, U, Q> *domain, ParticleSet<T> *pSet,
                                int max_depth, Float h, int pId)
{
    struct IntervalData{
        Q bq;
        int depth;
    };

    Q Bi;
    Stack<IntervalData> stack;
    IntervalBoundaryGetBoundsFor<T, Q>(pSet, pId, &Bi, h);

    int q = INTERVAL_LABEL_UNCOVERED;
    Q Bq = Bi;
    auto fn = [&](int j) -> int{
        Q Bj;
        IntervalBoundaryGetBoundsFor<T, Q>(pSet, j, &Bj, h);
        if(Inside(Bq, Bj)){
            q = INTERVAL_LABEL_COVERED;
            return 1;
        }

        if(Overlaps(Bq, Bj)){
            q = INTERVAL_LABEL_PARTIALLY_COVERED;
        }

        return 0;
    };

    stack.push({Bi, 0});
    while(!stack.empty()){
        IntervalData iBq;
        stack.top(&iBq);
        stack.pop();

        Bq = iBq.bq;
        q = INTERVAL_LABEL_UNCOVERED;
        ForAllNeighbors(domain, pSet, pId, fn);

        if(q == INTERVAL_LABEL_COVERED){
            // continue looping
        }else if(q == INTERVAL_LABEL_UNCOVERED){
            return false;
        }else{
            if(iBq.depth < max_depth){
                Q inner[8];
                int s = SplitBounds(Bq, &inner[0]);
                for(int k = 0; k < s; k++){
                    stack.push({inner[k], iBq.depth+1});
                }
            }else{
                return false;
            }
        }
    }

    return true;
}

template<typename T, typename U, typename Q> __global__
void IntervalBoundaryKernel(Grid<T, U, Q> *domain, ParticleSet<T> *pSet,
                            Float h, int max_depth)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if(i < pSet->GetParticleCount()){
        int is_interior = IntervalParticleIsInterior<T, U, Q>(domain, pSet,
                                                              max_depth, h, i);
        if(!is_interior){
            pSet->SetParticleV0(i, 1);
        }
    }
}

template<typename T, typename U, typename Q>
__host__ void IntervalBoundary(ParticleSet<T> *pSet, Grid<T, U, Q> *domain,
                               Float h, int max_depth = 4)
{
    int N = pSet->GetParticleCount();
    // 2/3 , 2/2
    Float scale = domain->GetDimensions() == 3 ? 0.666 : 1.0;
    GPULaunch(N, GPUKernel(IntervalBoundaryKernel<T, U, Q>),
                            domain, pSet, h * scale, max_depth);
}
