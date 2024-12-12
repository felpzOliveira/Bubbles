/* date = August 23rd 2024 10:29 */
#pragma once
#include <grid.h>
#include <cutil.h>
#include <kernel.h>
#include <util.h>
#include <transform.h>
#include <svd3_cuda.h>

/**************************************************************/
//                N I C O L A S   M E T H O D                 //
//                     Resampling PCA                         //
/**************************************************************/

/*
* In here we implement the classification scheme presented in:
*   Boundary particle resampling for surface reconstruction in liquid animation
*
* I'm not sure I'm using correct radius, not really trusting in the rejection method.
* I also did not implement the normal adjustment the paper presents. It claims we need
* move the particles based on that, however the x it provides does not apply to li = 3,
* the equation xn = xi +- 0.5ρuk, k = 1,...,3-li does not hold for li = 3, so I dont
* quite understand how it expects to shift points on li = 3 layer.
* TODO: This only supports 3D and GPU. We should probably do something about 2D
*       and CPU.
*/

inline
bb_cpu_gpu int NicolasSVDLevel(float a, float b, float c){
    float alpha = 0.2f;
    float t[3] = {a, b, c};
    if(a > c) Swap(a, c);
    if(a > b) Swap(a, b);
    if(b > c) Swap(b, c);
    t[0] = a;
    t[1] = b;
    t[2] = c;

    if(t[1] <= alpha * t[2]){ // σ2 <= α σ3
        return 1;
    }else if(t[0] <= alpha * t[2]){ // σ1 <= α σ3
        return 2;
    }
    return 3;
}

inline
void NicolasClassifier(ParticleSet3 *pSet, Grid3 *domain, float kernelRadius,
                       int fixed=0, vec3f *us=nullptr, vec4ui *buckets=nullptr)
{
    int N = pSet->GetParticleCount();
    vec3ui res = domain->GetIndexCount();
    vec3f cell = domain->GetCellSize();
    printf("{%g %g %g} {%u %u %u} {%g}\n", cell.x, cell.y, cell.z, res.x, res.y, res.z, kernelRadius);
    AutoParallelFor("Nicolas", N, AutoLambda(int pId){
        int *neighbors = nullptr;
        vec3f pi = pSet->GetParticlePosition(pId);
        unsigned int cellId = domain->GetLinearHashedPosition(pi);
        int count = domain->GetNeighborsOf(cellId, &neighbors);

        float A[3][3];
        float U[3][3];
        float S[3];
        float V[3][3];
        vec3f centroid(0.f);
        float counter = 0.f;

        for(int i = 0; i < 3; i++){
            for(int j = 0; j < 3; j++){
                A[i][j] = 0;
                U[i][j] = 0;
                V[i][j] = 0;
            }

            S[i] = 0;
        }

        for(int i = 0; i < count; i++){
            Cell3 *cell = domain->GetCell(neighbors[i]);
            ParticleChain *pChain = cell->GetChain();
            int size = cell->GetChainLength();
            for(int j = 0; j < size; j++){
                vec3f pj = pSet->GetParticlePosition(pChain->pId);

                Float dij = Distance(pi, pj);
                if(IsWithinStd(dij, kernelRadius)){
                    centroid += pj;
                    counter += 1.f;
                }

                pChain = pChain->next;
            }
        }

        if(counter < 4){
            pSet->SetParticleV0(pId, fixed > 0 ? 1 : 0);
            if(us){
                us[3 * pId + 0] = vec3f(1,0,0);
                us[3 * pId + 1] = vec3f(0,1,0);
                us[3 * pId + 2] = vec3f(0,0,1);
            }
            return;
        }

        Float invCounter = 1.f / counter;
        centroid *= invCounter;

        for(int i = 0; i < count; i++){
            Cell3 *cell = domain->GetCell(neighbors[i]);
            ParticleChain *pChain = cell->GetChain();
            int size = cell->GetChainLength();
            for(int j = 0; j < size; j++){
                vec3f pj = pSet->GetParticlePosition(pChain->pId);
                Float dij = Distance(pi, pj);
                if(IsWithinStd(dij, kernelRadius)){
                    vec3f v = pj - centroid;
                    float x2 = v.x * v.x, y2 = v.y * v.y, z2 = v.z * v.z;
                    float xy = v.x * v.y, xz = v.x * v.z, yz = v.y * v.z;
                    A[0][0] += x2; A[0][1] += xy; A[0][2] += xz;
                    A[1][0] += xy; A[1][1] += y2; A[1][2] += yz;
                    A[2][0] += xz; A[2][1] += yz; A[2][2] += z2;
                }

                pChain = pChain->next;
            }
        }

        for(int i = 0; i < 3; i++){
            for(int j = 0; j <3; j++){
                A[i][j] *= invCounter;
            }
        }

        svd(A[0][0], A[0][1], A[0][2], A[1][0], A[1][1], A[1][2], A[2][0], A[2][1], A[2][2],
            U[0][0], U[0][1], U[0][2], U[1][0], U[1][1], U[1][2], U[2][0], U[2][1], U[2][2],
            S[0], S[1], S[2],
            V[0][0], V[0][1], V[0][2], V[1][0], V[1][1], V[1][2], V[2][0], V[2][1], V[2][2]);

        int level = NicolasSVDLevel(S[0], S[1], S[2]);

        if(fixed > 0)
            pSet->SetParticleV0(pId, level != 3 ? 1 : 0);
        else
            pSet->SetParticleV0(pId, level);

        if(us){
            vec3f u1, u2, u3;
            u1.x = U[0][0]; u1.y = U[1][0]; u1.z = U[2][0];
            u2.x = U[0][1]; u2.y = U[1][1]; u2.z = U[2][1];
            u3.x = U[0][2]; u3.y = U[1][2]; u3.z = U[2][2];

            us[3 * pId + 0] = u1;
            us[3 * pId + 1] = u2;
            us[3 * pId + 2] = u3;
        }
    });

    if(buckets){
        uint32_t _buckets[4] = {0, 0, 0, 0};
        for(int i = 0; i < N; i++){
            int level = pSet->GetParticleV0(i);
            _buckets[level] += 1;
        }

        buckets->x = _buckets[0];
        buckets->y = _buckets[1];
        buckets->z = _buckets[2];
        buckets->w = _buckets[3];
    }
}

struct NicolasRefinedSystem{
    vec3f *newParticles;
    int *newLabels;
    int size;
};

inline bb_cpu_gpu
bool NicolasCheckSafeRegionFor(ParticleSet3 *pSet, Grid3 *domain,
                               Float rho, int refId, vec3f pi)
{
    bool accept = true;
    unsigned int cellId = domain->GetLinearHashedPosition(pi);

    domain->ForAllNeighborsOf(cellId, 3, [&](Cell3 *cell, vec3ui cid, int lid) -> int{
        int n = cell->GetChainLength();
        ParticleChain *pChain = cell->GetChain();
        for(int i = 0; i < n; i++){
            if(pChain->pId != refId){
                vec3f pj = pSet->GetParticlePosition(pChain->pId);
                Float dij = Distance(pj, pi);
                if(dij < rho){
                    accept = false;
                    return 1;
                }
            }

            pChain = pChain->next;
        }

        return 0;
    });

    return accept;
}

inline
void NicolasCheckSafeRegion(ParticleSet3 *pSet, Grid3 *domain, Float rho,
                            vec3f *newParticles, int *newLabels, int *id,
                            int *sourceId, int offset)
{
    int size = *id - offset;
    AutoParallelFor("Nicolas_SafeRegion", size, AutoLambda(int i){
        int pId = i + offset;
        vec3f pi = newParticles[pId];
        int refId = sourceId[pId];

        if(!NicolasCheckSafeRegionFor(pSet, domain, rho * 0.5f, refId, pi)){
            newLabels[pId] = -1;
        }
    });
}

inline
void NicolasSpawnNewParticles_Q0(ParticleSet3 *pSet, Grid3 *domain,
                                 Float rho, vec3f *us, NicolasWorkQueue3 *q0,
                                 vec3f *newParticles, int *newLabels, int *id,
                                 int *sourceId, int offset)
{
    if(q0->size == 0)
        return;

    AutoParallelFor("Nicolas_Q0", q0->size, AutoLambda(int i){
        int where = 0;
        int pId = q0->Fetch(&where);

        vec3f pi = pSet->GetParticlePosition(pId);

        vec3f u1 = us[3 * pId + 0];
        vec3f u2 = us[3 * pId + 1];
        vec3f u3 = us[3 * pId + 2];

        // We are going to replace pi with 6 particles
        // given by xn = xi +- 0.5ρ uk, k = 1, 2, 3 (page 5)
        vec3f p1 = pi + rho * 0.5f * u1;
        vec3f p2 = pi - rho * 0.5f * u1;

        vec3f p3 = pi + rho * 0.5f * u2;
        vec3f p4 = pi - rho * 0.5f * u2;

        vec3f p5 = pi + rho * 0.5f * u3;
        vec3f p6 = pi - rho * 0.5f * u3;

        vec3f ps[6] = {p1, p2, p3, p4, p5, p6};

        for(int i = 0; i < 6; i++){
            int target = atomic_increase_get(id);
            newParticles[target] = ps[i];
            newLabels[target] = 0;
            sourceId[target] = pId;
        }
    });
}

inline
void NicolasSpawnNewParticles_Q1(ParticleSet3 *pSet, Grid3 *domain,
                                 Float rho, vec3f *us, NicolasWorkQueue3 *q1,
                                 vec3f *newParticles, int *newLabels, int *id,
                                 int *sourceId, int offset)
{
    if(q1->size == 0)
        return;

    AutoParallelFor("Nicolas_Q1", q1->size, AutoLambda(int i){
        int where = 0;
        int pId = q1->Fetch(&where);

        vec3f pi = pSet->GetParticlePosition(pId);

        vec3f u1 = us[3 * pId + 0];
        vec3f u2 = us[3 * pId + 1];

        // We are going to replace pi with 4 particles
        // given by xn = xi +- 0.5ρ uk, k = 1, 2 (page 5)
        vec3f p1 = pi + rho * 0.5f * u1;
        vec3f p2 = pi - rho * 0.5f * u1;

        vec3f p3 = pi + rho * 0.5f * u2;
        vec3f p4 = pi - rho * 0.5f * u2;

        vec3f ps[4] = {p1, p2, p3, p4};


        for(int i = 0; i < 4; i++){
            int target = atomic_increase_get(id);
            newParticles[target] = ps[i];
            newLabels[target] = 1;
            sourceId[target] = pId;
        }
    });
}

inline
void NicolasSpawnNewParticles_Q2(ParticleSet3 *pSet, Grid3 *domain,
                                 Float rho, vec3f *us, NicolasWorkQueue3 *q2,
                                 vec3f *newParticles, int *newLabels, int *id,
                                 int *sourceId, int offset)
{
    if(q2->size == 0)
        return;

    AutoParallelFor("Nicolas_Q2", q2->size, AutoLambda(int i){
        int where = 0;
        int pId = q2->Fetch(&where);

        vec3f pi = pSet->GetParticlePosition(pId);

        vec3f u1 = us[3 * pId + 0];

        // We are going to replace pi with 2 particles
        // given by xn = xi +- 0.5ρ uk, k = 1 (page 5)
        vec3f p1 = pi + rho * 0.5f * u1;
        vec3f p2 = pi - rho * 0.5f * u1;

        vec3f ps[2] = {p1, p2};

        for(int i = 0; i < 2; i++){
            int target = atomic_increase_get(id);
            newParticles[target] = ps[i];
            newLabels[target] = 2;
            sourceId[target] = pId;
        }
    });
}

inline
NicolasRefinedSystem NicolasRefinement(ParticleSet3 *pSet, Grid3 *domain,
                                       float kernelRadius, int fixed=0)
{
    vec4ui buckets;
    vec3f *newParticles = nullptr;
    int *newLabels = nullptr;
    int *id = nullptr;
    int *sourceId = nullptr;
    uint32_t queueLen[3] = {0, 0, 0};
    NicolasWorkQueue3 *queues[3] = {nullptr, nullptr, nullptr};
    uint32_t totalParticles = pSet->GetParticleCount();
    vec3f *us = cudaAllocateUnregisterVx(vec3f, 3 * pSet->GetParticleCount());

    NicolasClassifier(pSet, domain, kernelRadius, fixed, us, &buckets);

    queueLen[0] = buckets.x * 6; // level 0 requires 5 new particles + original
    queueLen[1] = buckets.y * 4; // level 1 requires 3 new particles + original
    queueLen[2] = buckets.z * 2; // level 2 requires 1 new particle + original

    totalParticles += queueLen[0] - buckets.x;
    totalParticles += queueLen[1] - buckets.y;
    totalParticles += queueLen[2] - buckets.z;
    // NOTE: level 3 is not refined

    // NOTE: Since bubbles data structures cant be resized we need to get new storage
    newParticles = cudaAllocateUnregisterVx(vec3f, totalParticles);
    newLabels = cudaAllocateUnregisterVx(int, totalParticles);
    sourceId = cudaAllocateUnregisterVx(int, totalParticles);
    id = cudaAllocateUnregisterVx(int, 1);

    for(int i = 0; i < 3; i++){
        queues[i] = cudaAllocateVx(NicolasWorkQueue3, 1);
        queues[i]->SetSlots(buckets[i]);
    }

    int counter = 0;
    for(int i = 0; i < pSet->GetParticleCount(); i++){
        int level = pSet->GetParticleV0(i);
        if(level == 3){
            newParticles[counter] = pSet->GetParticlePosition(i);
            newLabels[counter] = level;
            sourceId[counter] = i;
            counter++;
            continue;
        }

        int at = queues[level]->size;
        queues[level]->ids[at] = i;
        queues[level]->size += 1;
    }

    *id = counter;
    // Spawn particles located at level = 0
    NicolasSpawnNewParticles_Q0(pSet, domain, kernelRadius * 0.5, us, queues[0],
                                newParticles, newLabels, id, sourceId, counter);

    // Spawn particles located at level = 1
    NicolasSpawnNewParticles_Q1(pSet, domain, kernelRadius * 0.5, us, queues[1],
                                newParticles, newLabels, id, sourceId, counter);

    // Spawn particles located at level = 2
    NicolasSpawnNewParticles_Q2(pSet, domain, kernelRadius * 0.5, us, queues[2],
                                newParticles, newLabels, id, sourceId, counter);

    // Check safe region for generated particles
    NicolasCheckSafeRegion(pSet, domain, kernelRadius * 0.5, newParticles, newLabels, id,
                           sourceId, counter);

    counter = *id;
    cudaFree(us);
    cudaFree(id);
    cudaFree(sourceId);
    return {newParticles, newLabels, counter};
}
