/* date = August 21st 2021 21:56 */
#pragma once
#include <grid.h>
#include <cutil.h>
#include <kernel.h>
#include <util.h>
#include <bound_util.h>
#include <convhull3d.h>

/**************************************************************/
//                S A N D I M   M E T H O D                   //
//                 Convex Hull based method                   //
/**************************************************************/

/*
* In here we implement Sandim's method for boundary extraction
* based on the HPR operator presented in the paper:
*     Boundary detection in particle-based fluids
*
* This was originally a full CUDA implementation and it can
* still be used to perform timing. However I see now that
* this method needs a VERY good Convex Hull computation and the
* one I made for CUDA was simply not robust enough. So I'm adding
* qhull library to perform this task. Unfortunatelly qhull is not
* runnable on kernel code, so i'll do everything in CUDA but Convex Hulls
* are handled through multi-threaded CPU computation on 3D, for 2D i'll
* use a simple Jarvis March on CUDA.
*/

#define SANDIM_GAMMA 1.3 // this is also a guess
/*
* This is the distance considered for detecting out-of-domain particles
* this realy depends on the quality of the convex hull implentation.
* For us this gives decent results.
*/
#define SANDIM_SMALL_DISTANCE 1e-5

/*
 * Computes, for every particle, if this particle can be resolved through
 * Sandim's method. Because this method relies on Convex Hull and Bubbles
 * is (for now) a particle-based simulator it is possible that a given
 * solver state has particles overlapping. Convex Hull cannot be resolved
 * in these cases so we mark those particles as out-of-domain and keep the
 * particle with lowest id. Once computation is finished we simply copy the
 * result back into the particles that were disconsidered. Every out-of-domain
 * particle stores the id of the lowest particle sister that will be resolved +1.
 * This avoids the 0 particle. I'm also negating this value for simpler tests.
 */
template<typename T, typename U, typename Q> bb_cpu_gpu
void SandimComputeWorkQueueFor(ParticleSet<T> *pSet, Grid<T, U, Q> *domain,
                               int *workQ, int pId)
{
    // do this on bucket level as it is faster and yields the same result
    T pi = pSet->GetParticlePosition(pId);
    Bucket *bucket = pSet->GetParticleBucket(pId);
    int minCandidate = pSet->GetParticleCount() + 2;
    for(int i = 0; i < bucket->Count(); i++){
        int j = bucket->Get(i);
        if(j != pId){
            T pj = pSet->GetParticlePosition(j);
            Float d = Distance(pi, pj);
            if(d < SANDIM_SMALL_DISTANCE){
                if(pId > j){
                    minCandidate = Min(j+1, minCandidate);
                }
            }
        }
    }

    if(!(minCandidate > pSet->GetParticleCount())){
        minCandidate = -minCandidate;
    }

    workQ[pId] = minCandidate;
}

template<typename T, typename U, typename Q> inline
void SandimComputeWorkQueue(ParticleSet<T> *pSet, Grid<T, U, Q> *domain, int *workQ){
    int N = pSet->GetParticleCount();
    AutoParallelFor("SandimComputeWorkQueue", N, AutoLambda(int i){
        SandimComputeWorkQueueFor(pSet, domain, workQ, i);
    });
}

/*
 * This method is very memory intensive, so instead of generating viewpoints
 * based on the location they lie on the grid (empty cell) we also only add
 * them if the offending particle is within the Sandim suggested range 4 * rho.
 * This reduces *a lot* the memory required to store the inverted points for
 * some voxels.
 */
template<typename T, typename U, typename Q, typename SandimWorkQueue> bb_cpu_gpu
void SandimComputeVP(ParticleSet<T> *pSet, Grid<T, U, Q> *domain,
                     SandimWorkQueue *vpWorkQ, unsigned int id)
{
    int *neighbors = nullptr;
    int count = domain->GetNeighborsOf(id, &neighbors);
    Cell<Q> *cell = domain->GetCell(id);
    T vp = cell->bounds.Center();
    Float radius = 2.0 * domain->GetCellSize()[0];
    if(cell->GetChainLength() == 0){
        for(int i = 0; i < count; i++){
            Cell<Q> *nei = domain->GetCell(neighbors[i]);
            int count = nei->GetChainLength();
            int add = 0;
            ParticleChain *pChain = nei->GetChain();
            for(int i = 0; i < count; i++){
                T pi = pSet->GetParticlePosition(pChain->pId);
                Float d = Distance(vp, pi);
                if(d <= radius){
                    add = 1;
                    break;
                }

                pChain = pChain->next;
            }

            if(add){
                vpWorkQ->Push(cell->bounds.Center());
                break;
            }
        }
    }
}

template<typename T, typename U, typename Q, typename SandimWorkQueue> inline
void SandimComputeViewPointsImpl(ParticleSet<T> *pSet, Grid<T, U, Q> *domain,
                                 SandimWorkQueue *vpWorkQ)
{
    unsigned int N = domain->GetCellCount();
    AutoParallelFor("SandimComputeVP", N, AutoLambda(int i){
        SandimComputeVP(pSet, domain, vpWorkQ, i);
    });
}

inline void SandimComputeViewPoints(ParticleSet3 *pSet, Grid3 *domain,
                                    SandimWorkQueue3 *vpWorkQ)
{
    SandimComputeViewPointsImpl<vec3f, vec3ui, Bounds3f, SandimWorkQueue3>(
                                                        pSet, domain, vpWorkQ);
}

inline void SandimComputeViewPoints(ParticleSet2 *pSet, Grid2 *domain,
                                    SandimWorkQueue2 *vpWorkQ)
{
    SandimComputeViewPointsImpl<vec2f, vec2ui, Bounds2f, SandimWorkQueue2>(
                                                        pSet, domain, vpWorkQ);
}

/*
 * This is the grid implementation Sandim suggests to use for the domain.
 * I particularly don't like it, I feel the solver grid is better but
 * it is here if you wish to use it, 'bbtool boundary ... -method sandim' uses this
 * implementation instead of the default one. This grid is not completely symmetric,
 * use with care.
 */
template<typename T, typename U, typename Q> inline
Grid<T, U, Q> *SandimComputeCompatibleGridImpl(ParticleSet<T> *pSet, Float h){
    Float rho = h;
    Float cellSize = 2.0 * rho;
    Q bounds = UtilComputeParticleSetBounds(pSet);
    T dims = (bounds.pMax - bounds.pMin) / cellSize;
    U idim = U(dims) + U(3);

    T grid_min = bounds.pMin - cellSize;
    T grid_max = grid_min + T(idim) * cellSize;
    return MakeGrid(idim, grid_min, grid_max);
}

inline Grid3 *SandimComputeCompatibleGrid(ParticleSet3 *pSet, Float h){
    return SandimComputeCompatibleGridImpl<vec3f, vec3ui, Bounds3f>(pSet, h);
}

inline Grid2 *SandimComputeCompatibleGrid(ParticleSet2 *pSet, Float h){
    return SandimComputeCompatibleGridImpl<vec2f, vec2ui, Bounds2f>(pSet, h);
}

/*
 * Performs exponential flip.
 */
template<typename T, typename U, typename Q, typename SandimWorkQ> bb_cpu_gpu
int SandimExponentialFlip(T vp, Float gamma, IndexedParticle<T> *invPoints,
                          int max, ParticleSet<T> *pSet,
                          Grid<T, U, Q> *domain, int *partQ)
{
    Float maxnorm = 0;
    int n = 0;
    Float cellSize = domain->GetCellSize()[0];
    Float searchRadius = 2.0 * cellSize;
    int depth = 2; // depth = 2 makes sure the ball 4 * rho is reached
    int would_need = 0;
    // Compute max norm
    ForAllNeighbors(domain, pSet, vp, [&](int pId) -> int{
        if(partQ[pId] >= 0){
            T pj = pSet->GetParticlePosition(pId);
            T pi = pj - vp;
            Float norm = pi.Length();
            if(norm <= searchRadius){
                maxnorm = Max(pi.Length(), maxnorm);
                would_need += 1;
            }
        }
        return 0;
    }, depth);

    if(would_need == 0){ // BUG, viewpoint without particles
        printf("[BUG] Zero norm ( %d )\n", would_need);
        return 0;
    }

    would_need = 0;

    maxnorm *= 1.1;

    // flip
    ForAllNeighbors(domain, pSet, vp, [&](int pId) -> int{
        if(n < max && partQ[pId] >= 0){
            T pj = pSet->GetParticlePosition(pId);
            T pi = (pj - vp);
            Float norm = pi.Length();
            pi = pi  / maxnorm;
            if(norm <= searchRadius){
                norm = pi.Length();
                Float invGamma = 1.0 / pow(norm, gamma);
                invPoints[n++] = {pId, pi * invGamma};
                would_need = n;
            }
        }

        return 0;
    }, depth);

    if(n < max){ // negate viewpoint indice so we can easily remove without sort
        invPoints[n++] = {-1, vp};
    }else{
        printf("Could not add viewpoint to flipped list ( %d )\n", would_need + 1);
    }

    return n;
}

template<typename T, typename U, typename Q, typename SandimWorkQ> bb_cpu_gpu
void SandimComputeVPFlip(ParticleSet<T> *pSet, Grid<T, U, Q> *domain, int *partQ,
                         SandimWorkQ *vpWorkQ, IndexedParticle<T> *points,
                         int *vp_count, int maxp)
{
    int where = 0;
    T vp = vpWorkQ->Fetch(&where);
    IndexedParticle<T> *invPoints = &points[maxp * where];
    // Compute exponential flip for VP
    unsigned int cellId = domain->GetLinearHashedPosition(vp);
    int n = SandimExponentialFlip<T, U, Q, SandimWorkQ>(vp, SANDIM_GAMMA,
                                invPoints, maxp, pSet, domain, partQ);
    vp_count[where] = n;
}

/*
 * Simple JarvisMarch for the 2D case where solving in the GPU is easy.
 */
bb_cpu_gpu inline
void ConvexHullJarvisMarch2D(IndexedParticle<vec2f> *points, int length,
                             ParticleSet2 *pSet)
{
    Float yMin = Infinity;
    int chosen = 0;
    int source = 0;

    auto LeftOn = [&](vec2f a, vec2f b, vec2f x) -> bool{
        Float area = ((b.x - a.x) * (x.y - a.y) - (x.x - a.x) * (b.y - a.y));
        return area > 0 || IsZero(area);
    };

    for(int i = 0; i < length; i++){
        vec2f p = points[i].p;
        if(yMin > p.y){
            chosen = i;
            yMin = p.y;
        }
    }

    source = chosen;
    do{
        int next = (chosen + 1) % length;
        int pId = points[chosen].pId;
        if(pId >= 0){
            pSet->SetParticleV0(pId, 1);
        }

        for(int j = 0; j < length; j++){
            if(j != chosen){
                vec2f pp = points[chosen].p;
                vec2f pq = points[next].p;
                vec2f pi = points[j].p;
                if(!LeftOn(pp, pi, pq)){
                    next = j;
                }
            }
        }

        chosen = next;
    }while(chosen != source);
}

static bb_cpu_gpu
void SandimComputeConvexHull2D(IndexedParticle<vec2f> *points, int *vp_count,
                               SandimWorkQueue2 *vpWorkQ, ParticleSet2 *pSet, int maxp)
{
    int where = 0;
    vec2f vp = vpWorkQ->Fetch(&where);
    IndexedParticle<vec2f> *invPoints = &points[where * maxp];
    int len = vp_count[where];
    ConvexHullJarvisMarch2D(invPoints, len, pSet);
}

/*
 * Apply QuickHull to 3D case. Solves on CPU using the multi-threading framework
 * from cutil.h.
 */
inline
void SandimConvexHull3(IndexedParticle<vec3f> *points, int *vp_count,
                       SandimWorkQueue3 *vpWorkQ, ParticleSet3 *pSet,
                       int maxp)
{
    ConvexHullPrepare();

    ParallelFor(0, vpWorkQ->size, [&](int i) -> void{
        int len = vp_count[i];
        IndexedParticle<vec3f> *ips = &points[maxp * i];

        if(len < 4){
            for(int s = 0; s < len; s++){
                int id = ips[s].pId;
                if(id >= 0) pSet->SetParticleV0(id, 1);
            }

            return;
        }

        ConvexHull3D(ips, maxp, len, [&](int id){
            int pId = ips[id].pId;
            if(pId >= 0){
                pSet->SetParticleV0(pId, 1);
            }
        });
    });

    ConvexHullFinish();
}

template<typename T, typename U, typename Q, typename SandimWorkQ> inline
void SandimComputeHPRImpl(ParticleSet<T> *pSet, Grid<T, U, Q> *domain, int *partQ,
                          SandimWorkQ *vpWorkQ, IndexedParticle<T> **pts,
                          int **vps, const int maxN)
{
    int N = vpWorkQ->size;
    int len = maxN * N;
    double totalLen = len * sizeof(IndexedParticle<T>) + N * sizeof(int);
    double mb = totalLen / 1e+6;
    //printf("[SANDIM] GPU Extra memory required: %g (%g Mb) ( VP = %d x N = %d )\n",
           //totalLen, mb, N, maxN);
    IndexedParticle<T> *points = cudaAllocateUnregisterVx(IndexedParticle<T>, len);
    int *vp_count = cudaAllocateUnregisterVx(int, N);
    AutoParallelFor("SandimComputeFlip", N, AutoLambda(int i){
        SandimComputeVPFlip(pSet, domain, partQ, vpWorkQ, points, vp_count, maxN);
    });

    vpWorkQ->ResetEntry();
    *pts = points;
    *vps = vp_count;
}

template<typename T> inline bb_cpu_gpu
void SandimRemapUnboundedParticles(ParticleSet<T> *pSet, int *partQ, int i){
    if(i < pSet->GetParticleCount()){
        int id = partQ[i];
        if(id < 0){
            int curr = pSet->GetParticleV0(-id-1);
            pSet->SetParticleV0(i, curr);
        }
    }
}

inline
void SandimComputeHPR(ParticleSet3 *pSet, Grid3 *domain, int *partQ,
                      SandimWorkQueue3 *vpWorkQ, int maxFlipSlots)
{
    IndexedParticle<vec3f> *points;
    int *vp_count;
    const int maxN = maxFlipSlots;
    SandimComputeHPRImpl<vec3f, vec3ui, Bounds3f, SandimWorkQueue3>(pSet, domain,
                partQ, vpWorkQ, &points, &vp_count, maxN);

    SandimConvexHull3(points, vp_count, vpWorkQ, pSet, maxN);

    int N = pSet->GetParticleCount();
    AutoParallelFor("SandimRemapUnboundedParticles", N, AutoLambda(int i){
        SandimRemapUnboundedParticles<vec3f>(pSet, partQ, i);
    });

    cudaFree(points);
    cudaFree(vp_count);
}

inline
void SandimComputeHPR(ParticleSet2 *pSet, Grid2 *domain, int *partQ,
                      SandimWorkQueue2 *vpWorkQ, int maxFlipSlots)
{
    IndexedParticle<vec2f> *points;
    int *vp_count;
    const int maxN = maxFlipSlots;
    int N = pSet->GetParticleCount();
    SandimComputeHPRImpl<vec2f, vec2ui, Bounds2f, SandimWorkQueue2>(pSet, domain,
                partQ, vpWorkQ, &points, &vp_count, maxN);

    AutoParallelFor("SandimComputeConvexHull2D", vpWorkQ->size, AutoLambda(int i){
        SandimComputeConvexHull2D(points, vp_count, vpWorkQ, pSet, maxN);
    });

    AutoParallelFor("SandimRemapUnboundedParticles", N, AutoLambda(int i){
        SandimRemapUnboundedParticles<vec2f>(pSet, partQ, i);
    });

    cudaFree(points);
    cudaFree(vp_count);
}

template<typename T, typename U, typename Q, typename SandimWorkQ> inline
void SandimBoundary(ParticleSet<T> *pSet, Grid<T, U, Q> *domain,
                    SandimWorkQ *vpWorkQ, int maxFlipSlots=512)
{
    if(vpWorkQ->capacity < domain->GetCellCount()){
        printf("Warning: WorkQueue does not garantee viewpoint capacity, dangerous run\n");
    }

    int nparts = pSet->GetParticleCount();
    int *partWorkQ = cudaAllocateUnregisterVx(int, nparts);
    SandimComputeWorkQueue(pSet, domain, partWorkQ);
    SandimComputeViewPoints(pSet, domain, vpWorkQ);
    SandimComputeHPR(pSet, domain, partWorkQ, vpWorkQ, maxFlipSlots);
    cudaFree(partWorkQ);
}

