/* date = May 19th 2021 15:25 */
#pragma once
#include <grid.h>
#include <cutil.h>
#include <bound_util.h>

/**************************************************************/
//               I N T E R V A L   M E T H O D                //
//                Geometrical AABB intersection               //
/**************************************************************/

/*
* Implementation of Sandim's new interval method for boundary detection in:
* Simple and reliable boundary detection for meshfree particle methods
*                       using interval analysis
*
* Sandim uses recursive methods for classification, since we are running
* on GPU I'll implement a simple GPU friendly stack (not very efficient tho)
* and unroll the recursion to be stack-based.
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

template<typename Q> struct IntervalData{
    Q bq;
    int depth;
};

inline __bidevice__ int IntervalIsBBInsideQuery(Bounds2f box, vec2f p,
                                                Float h, int *max_query)
{
    int inside = 0;
    vec2f points[] = {
        box.pMin, box.pMax,
        vec2f(box.pMin.x, box.pMax.y),
        vec2f(box.pMax.x, box.pMin.y),
    };

    for(int i = 0; i < 4; i++){
        Float dist = Distance(p, points[i]);
        inside += (dist < h ? 1 : 0);
    }

    *max_query = 4;
    return inside;
}

inline __bidevice__ int IntervalIsBBInsideQuery(Bounds3f box, vec3f p,
                                                Float h, int *max_query)
{
    int inside = 0;
    vec3f p0 = box.pMin;
    vec3f p1 = box.pMax;
    vec3f points[] = {
        p0,
        vec3f(p1.x, p0.y, p0.z),
        vec3f(p1.x, p0.y, p1.z),
        vec3f(p0.x, p0.y, p1.z),
        vec3f(p0.x, p1.y, p0.z),
        vec3f(p1.x, p1.y, p0.z),
        p1,
        vec3f(p0.x, p1.y, p1.z)
    };

    for(int i = 0; i < 8; i++){
        Float dist = Distance(p, points[i]);
        inside += (dist < h ? 1 : 0);
    }

    *max_query = 8;
    return inside;
}

template<typename T, typename U, typename Q> __bidevice__
bool IntervalParticleIsInterior(Grid<T, U, Q> *domain, ParticleSet<T> *pSet,
                                int max_depth, Float h, int pId)
{
    Q Bi;
    Stack<IntervalData<Q>> stack;
    T pi = pSet->GetParticlePosition(pId);
    IntervalBoundaryGetBoundsFor<T, Q>(pSet, pId, &Bi, h);
    int q = INTERVAL_LABEL_UNCOVERED;

    Q Bq = Bi;

    stack.push({Bi, 0});
    bool reached_max_depth = false;
    while(!stack.empty()){
        IntervalData<Q> iBq;
        stack.top(&iBq);
        stack.pop();

        // start query on the given depth, assume it is uncovered, i.e.: boundary
        Bq = iBq.bq;

        // check if the query is being made from a 'useless' bounding box, i.e.:
        // completely inside the original particle

        int m_query = 0;
        int inside = IntervalIsBBInsideQuery(Bq, pi, h, &m_query);
        if(inside == m_query){
            // this bounding box will not give any usefull information
            continue;
        }

        q = INTERVAL_LABEL_UNCOVERED;

        // check neighboors to see if this assumption is correct
        ForAllNeighbors(domain, pSet, pId, [&](int j) -> int{
            int max_query = 0;
            T pj = pSet->GetParticlePosition(j);
            int inside = IntervalIsBBInsideQuery(Bq, pj, h, &max_query);
            // Possible cases:
            // 1- inside = max_query => the BB is entirely inside the ball
            //                          this means we don't need to continue
            if(inside == max_query){
                q = INTERVAL_LABEL_COVERED;
                return 1;
            }

            // 2- inside > 0 => the BB is partially covered we need to continue
            if(inside > 0){
                q = INTERVAL_LABEL_PARTIALLY_COVERED;
            }

            return 0;
        });

        // if the label changed, adjust accordingly
        if(q == INTERVAL_LABEL_COVERED){
            // if the current bounding box is covered we need to inspect
            // others but not subdivide this one
        }else if(q == INTERVAL_LABEL_UNCOVERED){
            // if we found a bounding box that is completely uncovered
            // than it means we are done, because there is a portion of the
            // particle that is not being touched by its neighboors
            return false;
        }else{
            // we need to subdivide and further query the BBs, if depth allows
            if(iBq.depth < max_depth){
                Q inner[8];
                int s = SplitBounds(Bq, &inner[0]);
                // push the new bounding box into the stack
                for(int k = 0; k < s; k++){
                    stack.push({inner[k], iBq.depth+1});
                }
            }else{
                // if we are unsure about the result and depth is already too large
                // assume it is a boundary particle. Since the query is always
                // exclusive I think this means we can terminate the query
                // as there will always be one portion of the BB that will be
                // unresolved
                reached_max_depth = true;
                break;
            }
        }
    }

    // i'm not really sure what reaching here means. One idea might be
    // that if too many bounding boxes are completely covered it loops
    // with label 'INTERVAL_LABEL_COVERED' and the exit is the exhaust
    // of all BBs. In this situation I guess this particle is interior.
    // However we need to account for the order in which the stack is
    // processed. If we reach the max depth it means we were subdividing
    // and we are unsure about a segment of the BB, for this case it is
    // best to mark the particle as boundary.
    return !reached_max_depth;
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
    GPULaunch(N, GPUKernel(IntervalBoundaryKernel<T, U, Q>),
                           domain, pSet, h, max_depth);
}
