#include <delaunay.h>
#include <gDel3D/GpuDelaunay.h>
#include <gDel3D/GPU/HostToKernel.h>
#include <gDel3D/GPU/KerCommon.h>
#include <gDel3D/CommonTypes.h>
#include <kernel.h>
#include <util.h>
#include <vector>
#include <interval.h>
#include <fstream>
#include <cutil.h>
#include <unordered_map>

struct DelaunaySet{
    int nPos = 0;
    int offset = 0;
    int totalSize = 0;
    Point3 *positions;

    bb_cpu_gpu void SetParticlePosition(int id, vec3f v){
        if(nPos > 0){
            int trueId = id - offset;
            positions[trueId] = {v.x, v.y, v.z};
        }
    }

    bb_cpu_gpu vec3f GetParticlePosition(int id){
        if(nPos > 0){
            int trueId = id - offset;
            Point3 p = positions[trueId];
            return vec3f(p._p[0], p._p[1], p._p[2]);
        }
        return vec3f();
    }

    bb_cpu_gpu bool InSet(int id){
        return id >= offset;
    }

    void Cleanup(){
        cudaFree(positions);
    }
};

inline
bb_cpu_gpu vec3f DelaunayPosition(ParticleSet3 *pSet, DelaunaySet *dSet, int id){
    int count = pSet->GetParticleCount();
    if(id < count){
        printf("[ERROR] Invalid position query\n");
        return pSet->GetParticlePosition(id);
    }
    else
        return dSet->GetParticlePosition(id);
}

template<unsigned int N>
bb_cpu_gpu void ParticleFlag(ParticleSet3 *pSet, Grid3 *grid, int pId, Float radius,
                             int threshold)
{
    int *neighbors = nullptr;
    vec3f pi = pSet->GetParticlePosition(pId);
    unsigned int cellId = grid->GetLinearHashedPosition(pi);
    Cell3 *cell = grid->GetCell(cellId);
    int count = grid->GetNeighborsOf(cellId, &neighbors);

    vec3f pnei[N];
    int pjs[N];
    int iterator = 0;

    Float twoRadius = 2.0 * radius;
    auto check_add = [&](vec3f pj, int j) -> bool{
        if(j == pId)
            return iterator < N;

        if(iterator == N)
            return false;

        Float dist = Distance(pj, pi);
        if(dist < 1e-4)
            return true;

        if(dist < twoRadius){
            bool accept = true;
            for(int k = 0; k < iterator && accept; k++){
                Float d = Distance(pj, pnei[k]);
                if(d < 1e-4 || j == pjs[k])
                    accept = false;
            }

            if(accept){
                pnei[iterator] = pj;
                pjs[iterator] = j;
                iterator += 1;
            }
        }
        return iterator < N;
    };

    bool search = true;
    for(int i = 0; i < count && search; i++){
        Cell3 *cell = grid->GetCell(neighbors[i]);
        ParticleChain *pChain = cell->GetChain();
        int size = cell->GetChainLength();
        for(int j = 0; j < size && search; j++){
            vec3 pj = pSet->GetParticlePosition(pChain->pId);
            search = check_add(pj, pChain->pId);
        }
    }

    if(iterator == N)
        pSet->SetParticleV0(pId, 0);
    else
        pSet->SetParticleV0(pId, 1);
}

void DelaunayClassifyNeighbors(ParticleSet3 *pSet, Grid3 *domain, int threshold,
                               Float spacing, Float mu)
{
    int maxSize = 0;
    unsigned int totalCells = domain->GetCellCount();
    for(unsigned int i = 0; i < totalCells; i++){
        Cell3 *cell = domain->GetCell(i);
        maxSize = Max(maxSize, cell->GetChainLength());
    }

    Float radius = mu * spacing;
    AutoParallelFor("Delaunay_GetDomain", pSet->GetParticleCount(), AutoLambda(int i){
        ParticleFlag<3>(pSet, domain, i, radius, threshold);
    });
}

bb_cpu_gpu bool DelaunayIsTetValid(Tet tet, TetOpp opp, char v, uint32_t pLen){
    bool valid = true;
    if(!isTetAlive(v)) valid = false;

    for(int s = 0; s < 4; s++){
        if(opp._t[s] == -1) valid = false;
        if(tet._v[s] == pLen) valid = false; // inf point
    }

    return valid;
}

bb_cpu_gpu int matching_faces(i3 *tFaces, i3 face){
    for(int i = 0; i < 4; i++){
        if(tFaces[i] == face)
            return i;
    }

    return -1;
}

bb_cpu_gpu vec3ui matching_orientation(i3 face, Tet &tet){
    for(int i = 0; i < 4; i++){
        const int *orderVi = TetViAsSeenFrom[i];
        i3 _face(tet._v[orderVi[0]], tet._v[orderVi[1]], tet._v[orderVi[2]]);
        if(face == _face){
            return vec3ui(tet._v[orderVi[0]], tet._v[orderVi[1]], tet._v[orderVi[2]]);
        }
    }

    printf("Did not find face, bug?\n");
    return vec3ui();
}

bb_cpu_gpu bool
check_triangle(i3 tri, uint32_t *ids, ParticleSet3 *pSet, DelaunaySet *dSet,
               Float radius)
{
    int idA = ids[tri.t[0]], idB = ids[tri.t[1]], idC = ids[tri.t[2]];
    Float r = 2.0 * radius;
    vec3f pA = DelaunayPosition(pSet, dSet, idA);
    vec3f pB = DelaunayPosition(pSet, dSet, idB);
    vec3f pC = DelaunayPosition(pSet, dSet, idC);
    Float aB = Distance(pA, pB);
    Float aC = Distance(pA, pC);
    Float bD = Distance(pB, pC);

    bool valid = aB < r && aC < r && bD < r;
    if(!valid){
        //printf("AB= %g, AC= %g, BD= %g, R= %g, [%d](%s)\n", aB, aC, bD, r, s, bin);
    }
    return valid;
}

static void
DelaunayWorkQueueAndFilter(GpuDel *triangulator, DelaunayTriangulation &triangulation,
                           ParticleSet3 *pSet, DelaunaySet *dSet, uint32_t *ids,
                           Float mu, Float spacing, DelaunayWorkQueue *delQ,
                           TimerList &timer)
{
    Tet *kernTet = toKernelPtr(triangulator->_tetVec);
    TetOpp *kernTetOpp = toKernelPtr(triangulator->_oppVec);
    char *kernChar = toKernelPtr(triangulator->_tetInfoVec);

    uint32_t size = triangulator->_tetVec.size();
    uint32_t pLen = triangulation.pLen;
    delQ->SetSlots(size * 4, false);

    Float radius = mu * spacing;
    char *tetFlags = cudaAllocateUnregisterVx(char, size);
    int *extraCounter = cudaAllocateUnregisterVx(int, 1);

    *extraCounter = 0;
    timer.Start("Tetrahedra Filtering");
    /*
    * Classify the tetras in 3 possibilites:
    * 1 - completely valid -> flag= 0xf
    * 2 - at least one edge is invalid -> flag= 0x0
    * 3 - virtual/invalid tetras -> flag= -1
    */
    AutoParallelFor("Delaunay_Filter", size, AutoLambda(int i){
        Tet tet = kernTet[i];
        const TetOpp botOpp = loadOpp(kernTetOpp, i);
        int A = tet._v[0], B = tet._v[1],
            C = tet._v[2], D = tet._v[3];
        tetFlags[i] = -1;

        if(DelaunayIsTetValid(tet, botOpp, kernChar[i], pLen)){
            int idA = ids[A], idB = ids[B],
                idC = ids[C], idD = ids[D];

            vec3f pA = DelaunayPosition(pSet, dSet, idA);
            vec3f pB = DelaunayPosition(pSet, dSet, idB);
            vec3f pC = DelaunayPosition(pSet, dSet, idC);
            vec3f pD = DelaunayPosition(pSet, dSet, idD);

            Float rA = radius;
            Float rB = radius;
            Float rC = radius;
            Float rD = radius;

            Float rAB = rA+rB;
            Float rAC = rA+rC;
            Float rAD = rA+rD;
            Float rBC = rB+rC;
            Float rBD = rB+rD;
            Float rCD = rC+rD;

            Float aB = Distance(pA, pB);
            Float aC = Distance(pA, pC);
            Float aD = Distance(pA, pD);
            Float bC = Distance(pB, pC);
            Float bD = Distance(pB, pD);
            Float cD = Distance(pC, pD);

            bool is_tetra_valid =  aB < rAB && aC < rAC && aD < rAD &&
                                   bC < rBC && bD < rBD &&
                                   cD < rCD;
            if(!is_tetra_valid){
                tetFlags[i] = 0;
            }else
                tetFlags[i] = 0xf;
        }
    });

    /*
    * TODO: The following two kernels can be merged into a single one
    * it is just splitted so we can compute the actual contribution of each one
    */
    /*
    * Process only the completely valid tetras (flag == 0xf), these provide valid
    * triangles no matter what and can be pushed into the unique tri list directly.
    */
    timer.StopAndNext("Find Unique Triangles - Part 1");
    AutoParallelFor("Delaunay_UniqueTriangles_NoTetras", size, AutoLambda(int tetIndex){
        Tet tet = kernTet[tetIndex];
        int A = tet._v[0], B = tet._v[1],
            C = tet._v[2], D = tet._v[3];
        const TetOpp botOpp = loadOpp(kernTetOpp, tetIndex);

        int triStates = tetFlags[tetIndex];
        i3 faces[] = { i3(A, B, C), i3(A, B, D), i3(A, D, C), i3(B, D, C) };
        int info[] = { 0, 0, 0, 0 };
        int opp[] = { D, C, B, A };

        if(triStates <= 0)
            return;

        for(int botVi = 0; botVi < 4; botVi++){
            const int topTi = botOpp.getOppTet(botVi);
            const Tet topTet  = kernTet[topTi];
            int triStatesTi = tetFlags[topTi];

            if(triStatesTi <= 0)
                continue;

            int a = topTet._v[0], b = topTet._v[1],
                c = topTet._v[2], d = topTet._v[3];
            i3 topFaces[] = {i3(a, b, c), i3(a, b, d), i3(a, d, c), i3(b, d, c)};

            for(int k = 0; k < 4; k++){
                int where = matching_faces(faces, topFaces[k]);
                if(where >= 0)
                    info[where] += 1;
            }
        }

        for(int s = 0; s < 4; s++){
            if(info[s] == 0){
                vec3ui tri = matching_orientation(faces[s], tet);
                delQ->Push({tri, opp[s]});
            }
        }
    });

    timer.StopAndNext("Find Unique Triangles - Part 2");
    /*
    * Process the possibly invalid tetras (flag == 0). Checks each face and
    * validate against neighbors and previously added faces. Generate missing
    * faces and possibly helps fill-in possible holes in the mesh.
    */
    AutoParallelFor("Delaunay_UniqueTriangles_Tetras", size, AutoLambda(int tetIndex){
        Tet tet = kernTet[tetIndex];
        const TetOpp botOpp = loadOpp(kernTetOpp, tetIndex);
        int A = tet._v[0], B = tet._v[1],
            C = tet._v[2], D = tet._v[3];

        int tetState = tetFlags[tetIndex];
        i3 faces[] = { i3(A, B, C), i3(A, B, D), i3(A, D, C), i3(B, D, C) };
        int info[] = { 0, 0, 0, 0 };
        int opp[] = { D, C, B, A };

        if(tetState != 0)
            return;

        for(int botVi = 0; botVi < 4; botVi++){
            const int topTi = botOpp.getOppTet(botVi);
            const Tet topTet  = kernTet[topTi];
            int triStatesTi = tetFlags[topTi];

            int a = topTet._v[0], b = topTet._v[1],
                c = topTet._v[2], d = topTet._v[3];
            i3 topFaces[] = {i3(a, b, c), i3(a, b, d), i3(a, d, c), i3(b, d, c)};

            for(int k = 0; k < 4; k++){
                int where = matching_faces(faces, topFaces[k]);
                if(where >= 0){
                    // found a matching face, check if neighboor is valid
                    if(triStatesTi == -1){
                        // this tetra is invalid (virtual)
                    }else if(triStatesTi == 0 && info[where] == 0){
                        // this tetra is broken (a face was removed)
                        // if the index of this tetra is bigger we process it
                        if(tetIndex < topTi)
                            info[where] = 2;
                        else // solved by neighbor
                            info[where] = 1;
                    }else{
                        // this tetra is valid, face was already processed
                        info[where] = 1;
                    }
                }
            }
        }

        for(int s = 0; s < 4; s++){
            bool valid_face = check_triangle(faces[s], ids, pSet, dSet, radius);
            if(!valid_face)
                continue;

            if(info[s] == 0 || info[s] == 2){
                vec3ui tri = matching_orientation(faces[s], tet);
                delQ->Push({tri, opp[s]});
                atomic_increase_get(extraCounter);
            }
        }
    });

    timer.Stop();

    cudaFree(extraCounter);
    cudaFree(tetFlags);
}

void
DelaunaySurface(DelaunayTriangulation &triangulation, SphParticleSet3 *sphSet,
                Float spacing, Float mu, Grid3 *domain, SphSolver3 *solver,
                TimerList &timer)
{
    GpuDel triangulator;
    vec3i *u_tri = nullptr;
    uint32_t *u_ids = nullptr;
    DelaunayWorkQueue *delQ = nullptr;
    int pointNum = 0;

    Float kernel = sphSet->GetKernelRadius();
    vec3f cellSize = domain->GetCellSize();
    Float dist = mu * spacing;

    DelaunaySet *dSet = nullptr;
    ParticleSet3 *pSet = sphSet->GetParticleSet();

    printf(" * Delaunay: [μ = %g] [h = %g]\n", mu, spacing);

    std::cout << " - Adjusting domain..." << std::flush;

    int totalCount = 0;
    int supports = 0;
    for(int i = 0; i < pSet->GetParticleCount(); i++){
        if(pSet->GetParticleV0(i) > 0){
            supports += 3;
            totalCount += 4;
        }else{
        #if defined(DELAUNAY_WITH_INTERIOR)
            totalCount += 1;
        #endif
        }
    }

    dSet = cudaAllocateUnregisterVx(DelaunaySet, 1);
    u_ids = cudaAllocateUnregisterVx(uint32_t, totalCount);
    dSet->positions = cudaAllocateUnregisterVx(Point3, totalCount);
    dSet->totalSize = totalCount;
    dSet->offset = pSet->GetParticleCount();

    int *iIndex = cudaAllocateUnregisterVx(int, 1);
    int *refIndex = cudaAllocateUnregisterVx(int, 1);
    *iIndex = 0;
    *refIndex = 0;

    timer.Start("Particle Filter and Spawn");

#if defined(DELAUNAY_WITH_INTERIOR)
    AutoParallelFor("Delaunay_Particle-Filter",pSet->GetParticleCount(),AutoLambda(int pId)
    {
        int bi = pSet->GetParticleV0(pId);
        if(bi > 0)
            return;

        int acceptpart = 1;
        Bucket *bucket = pSet->GetParticleBucket(pId);
        vec3f pi = pSet->GetParticlePosition(pId);
        for(int i = 0; i < bucket->Count(); i++){
            int j = bucket->Get(i);
            if(j != pId){
                vec3f pj = pSet->GetParticlePosition(j);
                Float d = Distance(pi, pj);
                if(d < 1e-4){
                    if(pId > j){
                        acceptpart = 0;
                        break;
                    }
                }
            }
        }

        if(acceptpart == 1){
            int key = atomic_increase_get(iIndex);
            int id = pSet->GetParticleCount() + key;
            dSet->positions[key] = {pi.x, pi.y, pi.z};
            u_ids[key] = id;
        }
    });
#endif

    *refIndex = *iIndex;
    *iIndex = 0;

    AutoParallelFor("Delaunay_Tet-Spawn", pSet->GetParticleCount(), AutoLambda(int pId){
        int bi = pSet->GetParticleV0(pId);

        if(bi == 0)
            return;

        int acceptpart = 1;
        Bucket *bucket = pSet->GetParticleBucket(pId);
        vec3f pi = pSet->GetParticlePosition(pId);

        for(int i = 0; i < bucket->Count(); i++){
            int j = bucket->Get(i);
            if(j != pId){
                vec3f pj = pSet->GetParticlePosition(j);
                Float d = Distance(pi, pj);
                if(d < 1e-4){
                    if(pId > j){
                        acceptpart = 0;
                        break;
                    }
                }
            }
        }

        if(acceptpart == 1){
            int key = atomic_increase_get(iIndex);
            int id = pSet->GetParticleCount() + *refIndex + 4 * key;
            Float edgeLen = mu * spacing;
            const Float one_over_sqrt2 = 0.7071067811865475;

            Float a = edgeLen * one_over_sqrt2;
            Float ha = a * 0.5f;

            vec3f u0 = vec3f(+ha, +ha, +ha);
            vec3f u1 = vec3f(-ha, +ha, -ha);
            vec3f u2 = vec3f(-ha, -ha, +ha);
            vec3f u3 = vec3f(+ha, -ha, -ha);

            vec3f p0 = pi + u0;//
            vec3f p1 = pi + u1;// - u0;
            vec3f p2 = pi + u2;// - u0;
            vec3f p3 = pi + u3;// - u0;

            dSet->positions[*refIndex + 4 * key + 0] = {p0.x, p0.y, p0.z};
            dSet->positions[*refIndex + 4 * key + 1] = {p1.x, p1.y, p1.z};
            dSet->positions[*refIndex + 4 * key + 2] = {p2.x, p2.y, p2.z};
            dSet->positions[*refIndex + 4 * key + 3] = {p3.x, p3.y, p3.z};
            u_ids[*refIndex + 4 * key + 0] = id + 0;
            u_ids[*refIndex + 4 * key + 1] = id + 1;
            u_ids[*refIndex + 4 * key + 2] = id + 2;
            u_ids[*refIndex + 4 * key + 3] = id + 3;
        }
    });

    timer.Stop();

    dSet->nPos = *refIndex + 4 * (*iIndex);
    triangulation.pointVec.resize(dSet->nPos);

    for(int i = 0, j = 0; i < dSet->nPos; i++){
        triangulation.pointVec[j++] = dSet->positions[i];
    }

    pointNum = triangulation.pointVec.size();
    Float frac = 100.0 * supports / pSet->GetParticleCount();
    std::cout << "done\n - Support points " << supports <<
                 " ( " << frac << "% )" << std::endl;

    std::cout << " - Running delaunay triangulation..." << std::flush;
    timer.Start("Delaunay Triangulation");
    triangulator.compute(triangulation.pointVec, &triangulation.output);
    timer.Stop();
    triangulation.pLen = pointNum;
    std::cout << "done" << std::endl;

    delQ = cudaAllocateUnregisterVx(DelaunayWorkQueue, 1);
    DelaunayWorkQueueAndFilter(&triangulator, triangulation, pSet, dSet,
                                u_ids, mu, spacing, delQ, timer);

    if(delQ->size == 0){
        std::cout << " - Filter gave no triangles\n";
        return;
    }

    std::cout << " - Aggregating triangles ( " << delQ->size << " )..." << std::flush;

    u_tri = cudaAllocateUnregisterVx(vec3i, delQ->size);
    triangulation.shrinked.resize(delQ->size);

    timer.Start("(Extra) Mark Boundary");
    AutoParallelFor("Delaunay_MarkTriangles", delQ->size, AutoLambda(int i){
        DelaunayTriangleInfo *uniqueTri = delQ->Ref(i);
        vec3ui tri = uniqueTri->tri;

        int A = tri.x;
        int B = tri.y;
        int C = tri.z;

        vec3i index = vec3i(A, C, B);
        u_tri[i] = index;
    });
    timer.Stop();

    std::cout << "done" << std::endl;

    std::cout << " - Copying to host..." << std::flush;
    int n_it = pSet->GetParticleCount();
    int max_it = Max(n_it, delQ->size);

    // NOTE: This remapping does not really do anything usefull, it just
    //       recompute vertices indices so that we can output a smaller
    //       file without unusefull points
    // TODO: We can probably remove this and do some changes in kernel
    //       but it is not actually important
    std::unordered_map<uint32_t, uint32_t> remap;
    std::map<i2, uint32_t, i2Comp> edgeMap;
    uint32_t runningId = 0;

    timer.Start("(Extra) Reduce Indices");
    for(int i = 0; i < max_it; i++){
        if(i < delQ->size){
            int A = u_tri[i].x;
            int B = u_tri[i].y;
            int C = u_tri[i].z;
            int idA = u_ids[A];
            int idB = u_ids[B];
            int idC = u_ids[C];

            i2 edges[3] = {i2(A, B), i2(A, C), i2(B, C)};
            for(int s = 0; s < 3; s++){
                if(edgeMap.find(edges[s]) == edgeMap.end()){
                    edgeMap[edges[s]] = 1;
                }else{
                    edgeMap[edges[s]] += 1;
                }
            }

            vec3f pA = DelaunayPosition(pSet, dSet, idA);
            vec3f pB = DelaunayPosition(pSet, dSet, idB);
            vec3f pC = DelaunayPosition(pSet, dSet, idC);

            uint32_t aId, bId, cId;
            if(remap.find(idA) == remap.end()){
                aId = runningId++;
                remap[idA] = aId;
                triangulation.shrinkedPs.push_back(pA);
            }else
                aId = remap[idA];

            if(remap.find(idB) == remap.end()){
                bId = runningId++;
                remap[idB] = bId;
                triangulation.shrinkedPs.push_back(pB);
            }else
                bId = remap[idB];

            if(remap.find(idC) == remap.end()){
                cId = runningId++;
                remap[idC] = cId;
                triangulation.shrinkedPs.push_back(pC);
            }else
                cId = remap[idC];

            triangulation.shrinked[i] = vec3i(aId, bId, cId);
        }
    }
    timer.Stop();

    int counter = 0;
    for(auto it : edgeMap){
        if(it.second > 2)
            counter += 1;
    }

    std::cout << "done" << std::endl;
    std::cout << " - Edges with > 2 : " << counter << std::endl;
    std::cout << " - Finished building mesh" << " ( " << triangulation.shrinkedPs.size() << " vertices | "
              << triangulation.shrinked.size() << " triangles )" << std::endl;

    triangulator.cleanup();
    dSet->Cleanup();
    cudaFree(dSet);
    cudaFree(delQ->ids);
    cudaFree(delQ);
    cudaFree(u_tri);
    cudaFree(u_ids);
    cudaFree(iIndex);
    cudaFree(refIndex);
}

inline bb_cpu_gpu bool shiftface(vec3i &face, size_t i){
    if(face[0] == i){
        face = vec3i(face[1], face[2], i);
    }else if(face[1] == i){
        face = vec3i(face[0], face[2], i);
    }else if(face[2] == i){
        face = vec3i(face[0], face[1], i);
    }
    return face[2] == i;
}

// Trivial forward/backwards taubin
void TaubinSmoothOnce(vec3f *inOut, vec3f *tmp, size_t verticeCount,
                      vec3i *faces, size_t faceCount, Float lambda, Float mu)
{
    vec3f *inOutBuf = inOut;
    vec3f *tmpBuf = tmp;
    vec3i *faceBuf = faces;
    AutoParallelFor("Taubin Smooth - Forward", verticeCount, AutoLambda(int index){
        vec3f laplacian(0.f, 0.f, 0.f);
        int count = 0;
        for(size_t fi = 0; fi < faceCount; fi++){
            vec3i face = faceBuf[fi];
            if(shiftface(face, index)){
                vec3f vi = inOutBuf[face[2]];
                vec3f vj = inOutBuf[face[0]];
                vec3f vl = inOutBuf[face[1]];
                vec3f edge1 = vj - vi;
                vec3f edge2 = vl - vi;
                laplacian += edge1 + edge2;
                count += 2;
            }
        }

        if(count > 0){
            Float inv = 1.f / (Float)count;
            tmpBuf[index] = inOutBuf[index] + lambda * (laplacian * inv - inOutBuf[index]);
        }else{
            tmpBuf[index] = inOutBuf[index];
        }
    });

    AutoParallelFor("Taubin Smooth - Backwards", verticeCount, AutoLambda(int index){
        vec3f laplacian(0.f, 0.f, 0.f);
        int count = 0;
        for(size_t fi = 0; fi < faceCount; fi++){
            vec3i face = faceBuf[fi];
            if(shiftface(face, index)){
                vec3f vi = tmpBuf[face[2]];
                vec3f vj = tmpBuf[face[0]];
                vec3f vl = tmpBuf[face[1]];
                vec3f edge1 = vj - vi;
                vec3f edge2 = vl - vi;
                laplacian += edge1 + edge2;
                count += 2;
            }
        }

        if(count > 0){
            Float inv = 1.f / (Float)count;
            inOutBuf[index] = tmpBuf[index] + mu * (laplacian * inv - tmpBuf[index]);
        }else{
            inOutBuf[index] = tmpBuf[index];
        }
    });
}

// Trivial laplacian
void LaplacianSmoothOnce(vec3f *readBuf, vec3f *writeBuf, size_t verticeCount,
                         vec3i *faces, size_t faceCount)
{
    vec3f *sourceBuf = readBuf;
    vec3f *destBuf = writeBuf;
    vec3i *faceBuf = faces;
    AutoParallelFor("Laplacian Smooth", verticeCount, AutoLambda(int index){
        vec3f laplacianSum(0.f, 0.f, 0.f);
        int count = 0;

        for(size_t fi = 0; fi < faceCount; fi++){
            vec3i face = faceBuf[fi];
            if(shiftface(face, index)){
                vec3f vi = sourceBuf[face[2]];
                vec3f vj = sourceBuf[face[0]];
                vec3f vl = sourceBuf[face[1]];
                vec3f edge1 = vj - vi;
                vec3f edge2 = vl - vi;
                laplacianSum += edge1 + edge2;
                count += 2;
            }
        }

        if(count > 0){
            Float inv = 1.f / (Float)count;
            destBuf[index] = sourceBuf[index] + laplacianSum * inv;
        }else{
            destBuf[index] = sourceBuf[index];
        }
    });
}

void DelaunaySmooth(DelaunayTriangulation &triangulation, MeshSmoothOpts opts){
    size_t vsize = triangulation.shrinkedPs.size();
    size_t fsize = triangulation.shrinked.size();
    vec3f *bufferA = cudaAllocateUnregisterVx(vec3f, vsize);
    vec3f *bufferB = cudaAllocateUnregisterVx(vec3f, vsize);
    vec3i *faceBuf = cudaAllocateUnregisterVx(vec3i, fsize);

    memcpy(bufferA, triangulation.shrinkedPs.data(), vsize * sizeof(vec3f));
    memcpy(faceBuf, triangulation.shrinked.data(), fsize * sizeof(vec3i));

    vec3f *buffers[2] = {bufferA, bufferB};
    vec3f *outputBuf = bufferA;
    int active = 1;
    for(int i = 0; i < opts.iterations; i++){
        if(opts.method == LaplacianSmooth){
            vec3f *writeBuf = buffers[active];
            vec3f *readBuf  = buffers[1-active];

            LaplacianSmoothOnce(readBuf, writeBuf, vsize, faceBuf, fsize);
            active = 1 - active;
            outputBuf = writeBuf;
        }else if(opts.method == TaubinSmooth){
            TaubinSmoothOnce(bufferA, bufferB, vsize, faceBuf, fsize, opts.lambda, opts.mu);
            outputBuf = bufferA;
        }
    }

    for(int i = 0; i < vsize; i++){
        triangulation.shrinkedPs[i] = outputBuf[i];
    }

    cudaFree(bufferA);
    cudaFree(bufferB);
    cudaFree(faceBuf);
}

void
DelaunayGetTriangleMesh(DelaunayTriangulation &triangulation, HostTriangleMesh3 *mesh){
    if(mesh){
        std::vector<vec3f> *points = &triangulation.shrinkedPs;
        for(int i = 0; i < (int)points->size(); i++){
            vec3f p = points->at(i);
            mesh->addPoint(p);
        }

        for(vec3i index : triangulation.shrinked){
            vec3ui face(index[0], index[1], index[2]);
            mesh->addPointUvNormalTriangle(face, face, face);
        }
    }
}

