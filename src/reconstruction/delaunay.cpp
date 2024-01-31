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

#define is_bit_set(n, b) (((n) >> (b)) & 1)
#define set_bit(n, b) ((n) |= (1 << (b)))

struct DelaunaySet{
    int nPos = 0;
    int offset = 0;
    Point3 *positions;

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

bb_cpu_gpu
void printBinary(int num, char bin[9]){
    for(int i = 7; i >= 0; i--){
        if(is_bit_set(num, i))
            bin[7-i] = '1';
        else
            bin[7-i] = '0';
    }

    bin[8] = 0;
}

inline
bb_cpu_gpu vec3f DelaunayPosition(ParticleSet3 *pSet, DelaunaySet *dSet, int id){
    int count = pSet->GetParticleCount();
    if(id < count)
        return pSet->GetParticlePosition(id);
    else
        return dSet->GetParticlePosition(id);
}

inline bb_cpu_gpu void DelaunaySetBoundary(DelaunaySet *dSet, int id, int *u_bound){
    if(!dSet->InSet(id))
        u_bound[id] = 1;
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


void DumpPoints(ParticleSet3 *pSet){
    FILE *fp = fopen("test_points.txt", "wb");
    if(fp){
        int nCount = pSet->GetParticleCount();
        int total = 0;
        std::stringstream ss;
        for(int i = 0; i < nCount; i++){
            vec3f pi = pSet->GetParticlePosition(i);
            int vi = pSet->GetParticleV0(i);
            if(vi >= 0){
                total += 1;
                ss << pi.x << " " << pi.y << " " << pi.z << " " << vi << std::endl;
            }
        }

        fprintf(fp, "%d\n", total);
        fprintf(fp, "%s", ss.str().c_str());
        fclose(fp);
    }
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
__debug_check_triangle(i3 tri, uint32_t *ids, ParticleSet3 *pSet, DelaunaySet *dSet,
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
                           int *bound, Float mu, Float spacing, DelaunayWorkQueue *delQ,
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
    AutoParallelFor("Delaunay_Filter", size, AutoLambda(int i){
        Tet tet = kernTet[i];
        int A = tet._v[0], B = tet._v[1],
            C = tet._v[2], D = tet._v[3];
        const TetOpp botOpp = loadOpp(kernTetOpp, i);

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

#if 1
            bool is_tetra_valid =  aB < rAB && aC < rAC && aD < rAD &&
                                   bC < rBC && bD < rBD &&
                                   cD < rCD;
            if(!is_tetra_valid){
                DelaunaySetBoundary(dSet, idA, bound);
                DelaunaySetBoundary(dSet, idB, bound);
                DelaunaySetBoundary(dSet, idC, bound);
                DelaunaySetBoundary(dSet, idD, bound);
                tetFlags[i] = 0;
            }else
                tetFlags[i] = 0x0f;
#else
            int ABCvalid = aB < rAB && aC < rAC && bC < rBC;
            int ABDvalid = aB < rAB && bD < rBD && aD < rAD;
            int BCDvalid = bC < rBC && bD < rBD && cD < rCD;
            int ACDvalid = aC < rAC && aD < rAD && cD < rCD;
            // ABC ABD ACD BCD
            //  1   1   1   1
            if(ABCvalid) { set_bit(tetFlags[i], 0); }
            if(ABDvalid) { set_bit(tetFlags[i], 1); }
            if(ACDvalid) { set_bit(tetFlags[i], 2); }
            if(BCDvalid) { set_bit(tetFlags[i], 3); }


            //tetFlags[i] |= ABCvalid ? (1 << 0) : 0;
            //tetFlags[i] |= ABDvalid ? (1 << 1) : 0;
            //tetFlags[i] |= ACDvalid ? (1 << 2) : 0;
            //tetFlags[i] |= BCDvalid ? (1 << 3) : 0;
#endif
        }
    });

    /*
    * TODO: The following two kernels can be merged into a single one
    * it is just splitted so we can compute the actual contribution of each one
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
            const int topVi = botOpp.getOppVi(botVi);
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
            const int topVi = botOpp.getOppVi(botVi);
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
            bool valid_face = __debug_check_triangle(faces[s], ids, pSet, dSet, radius);
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

    printf("Extra info = %d\n", *extraCounter);
    cudaFree(extraCounter);
    cudaFree(tetFlags);
}

void
DelaunaySurface(DelaunayTriangulation &triangulation, SphParticleSet3 *sphSet,
                Float spacing, Float mu, Grid3 *domain, SphSolver3 *solver,
                TimerList &timer)
{
    GpuDel triangulator;
    int *u_bound = nullptr;
    vec3i *u_tri = nullptr;
    uint32_t *u_ids = nullptr;
    DelaunayWorkQueue *delQ = nullptr;
    int pointNum = 0;

    Float kernel = sphSet->GetKernelRadius();
    vec3f cellSize = domain->GetCellSize();
    Float dist = mu * spacing;

    DelaunaySet *dSet = nullptr;
    ParticleSet3 *pSet = sphSet->GetParticleSet();
    std::vector<int> *boundary = &triangulation.boundary;

    printf(" * Delaunay: [μ = %g] [h = %g]\n", mu, spacing);

    std::cout << " - Adjusting domain..." << std::flush;
    // TODO: Adjust based on extended domain
    u_bound = cudaAllocateUnregisterVx(int, pSet->GetParticleCount());
    boundary->reserve(pSet->GetParticleCount());

    int totalCount = 0;
    for(int i = 0; i < pSet->GetParticleCount(); i++){
        //totalCount += 1;
        totalCount += pSet->GetParticleV0(i) > 0 ? 4 : 1;
    }

    dSet = cudaAllocateUnregisterVx(DelaunaySet, 1);
    u_ids = cudaAllocateUnregisterVx(uint32_t, totalCount);
    dSet->positions = cudaAllocateUnregisterVx(Point3, totalCount);
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

        u_bound[pId] = 0;
    });
#endif

    *refIndex = *iIndex;
    *iIndex = 0;
#if 1
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
            Float edgeLen = mu * spacing * 0.9;
            const Float one_over_sqrt2 = 0.7071067811865475;

            Float a = edgeLen * one_over_sqrt2;
            Float ha = 0.5 * a;

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

        u_bound[pId] = 0;
    });
#endif
    timer.Stop();

    dSet->nPos = *refIndex + 4 * (*iIndex);
    triangulation.pointVec.resize(dSet->nPos);

    for(int i = 0, j = 0; i < dSet->nPos; i++){
        triangulation.pointVec[j++] = dSet->positions[i];
    }

    pointNum = triangulation.pointVec.size();
    std::cout << "done\n - Support points " << dSet->nPos << std::endl;

    std::cout << " - Running delaunay triangulation..." << std::flush;
    timer.Start("Delaunay Triangulation");
    triangulator.compute(triangulation.pointVec, &triangulation.output);
    timer.Stop();
    triangulation.pLen = pointNum;
    std::cout << "done" << std::endl;

    delQ = cudaAllocateUnregisterVx(DelaunayWorkQueue, 1);
    DelaunayWorkQueueAndFilter(&triangulator, triangulation, pSet, dSet,
                                u_ids, u_bound, mu, spacing, delQ, timer);

    if(delQ->size == 0){
        std::cout << " - Filter gave no triangles\n";
        return;
    }

    std::cout << " - Aggregating triangles ( " << delQ->size << " )..." << std::flush;

    u_tri = cudaAllocateUnregisterVx(vec3i, delQ->size);
    triangulation.shrinked.resize(delQ->size);

    timer.Start("(Extra) Mark Boundary");
    AutoParallelFor("Delaunay_MarkTrisAndBoundary", delQ->size, AutoLambda(int i){
        DelaunayTriangleInfo *uniqueTri = delQ->Ref(i);
        vec3ui tri = uniqueTri->tri;
        int opp = uniqueTri->opp;

        int A = tri.x;
        int B = tri.y;
        int C = tri.z;

        int idA = u_ids[A];
        int idB = u_ids[B];
        int idC = u_ids[C];
        int idD = u_ids[opp];

        vec3i index = vec3i(A, C, B);

        DelaunaySetBoundary(dSet, idA, u_bound);
        DelaunaySetBoundary(dSet, idB, u_bound);
        DelaunaySetBoundary(dSet, idC, u_bound);
        DelaunaySetBoundary(dSet, idD, u_bound);
        u_tri[i] = index;
    });
    timer.Stop();

    std::cout << "done" << std::endl;

    std::cout << " - Copying to host..." << std::flush;
    int n_it = pSet->GetParticleCount();
    int b_len = 0;
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

        if(i < pSet->GetParticleCount()){
            b_len += u_bound[i] > 0;
            boundary->push_back(u_bound[i]);
        }
    }
    timer.Stop();

    int counter = 0;
    for(auto it : edgeMap){
        if(it.second > 2)
            counter += 1;
    }

    std::cout << "done ( boundary: " << b_len << " / " << n_it << " ) " << std::endl;
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
    cudaFree(u_bound);
    cudaFree(iIndex);
    cudaFree(refIndex);
}

void
DelaunayWriteBoundary(DelaunayTriangulation &triangulation, SphParticleSet3 *sphSet,
                      const char *path)
{
    FILE *fp = fopen(path, "w");
    ParticleSet3 *pSet = sphSet->GetParticleSet();
    if(fp){
        std::vector<int> &boundary = triangulation.boundary;
        //std::vector<int> boundary;
        //int n = UtilGetBoundaryState(pSet, &boundary);
        //printf(" Filter ( %d / %d )\n", n, pSet->GetParticleCount());
        fprintf(fp, "%lu\n", boundary.size());
        for(int i = 0; i < boundary.size(); i++){
            vec3f p = pSet->GetParticlePosition(i);
            fprintf(fp, "%g %g %g %d\n", p.x, p.y, p.z, boundary[i]);
        }
        fclose(fp);
    }
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

