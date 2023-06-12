#include <delaunay.h>
#include <gDel3D/GpuDelaunay.h>
#include <gDel3D/GPU/HostToKernel.h>
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
    int nNor = 0;
    int offset = 0;
    vec3f *positions;
    vec3f *normals;

    __bidevice__ vec3f GetParticlePosition(int id){
        if(nPos > 0){
            int trueId = id - offset;
            return positions[trueId];
        }
        return vec3f();
    }

    __bidevice__ void SetParticlePosition(int id, vec3f p){
        if(nPos > 0){
            int trueId = id - offset;
            positions[trueId] = p;
        }
    }

    __bidevice__ vec3f GetParticleNormal(int id){
        if(nNor > 0){
            int trueId = id - offset;
            return normals[trueId];
        }
        return vec3f();
    }

    __bidevice__ bool InSet(int id){
        return id >= offset;
    }

    void Build(std::vector<vec3f> &pos, std::vector<vec3f> &nor, int off){
        nPos = pos.size();
        nNor = nor.size();
        offset = off;

        if(nPos > 0 && nNor > 0){
            positions = cudaAllocateUnregisterVx(vec3f, nPos);
            normals = cudaAllocateUnregisterVx(vec3f, nNor);
            memcpy(positions, pos.data(), pos.size() * sizeof(vec3f));
            memcpy(normals, nor.data(), nor.size() * sizeof(vec3f));
        }
    }

    void Cleanup(){
        cudaFree(positions);
        cudaFree(normals);
    }
};


struct DelaunaySmallBucket{
    int id;
    int neighbors[3];
    vec3f pneighbors[3];
    int size;
    int density;
};

struct DelaunaySetBuilder{
    std::vector<vec3f> pos;
    std::vector<vec3f> nor;

    void PushParticle(vec3f p, vec3f n){
        pos.push_back(p);
        nor.push_back(n);
    }

    size_t Size(){
        return pos.size();
    }

    DelaunaySet *Build(int total_p){
        DelaunaySet *dSet = cudaAllocateUnregisterVx(DelaunaySet, 1);
        dSet->Build(pos, nor, total_p);
        return dSet;
    }
};


inline
__bidevice__ vec3f DelaunayPosition(ParticleSet3 *pSet, DelaunaySet *dSet, int id){
    int count = pSet->GetParticleCount();
    if(id < count)
        return pSet->GetParticlePosition(id);
    else
        return dSet->GetParticlePosition(id);
}

inline
__bidevice__ void DelaunaySetPosition(ParticleSet3 *pSet, DelaunaySet *dSet,
                                      int id, vec3f p)
{
    int count = pSet->GetParticleCount();
    if(id < count)
        pSet->SetParticlePosition(id, p);
    else
        dSet->SetParticlePosition(id, p);
}

inline __bidevice__ void DelaunaySetBoundary(DelaunaySet *dSet, int id, int *u_bound){
    if(!dSet->InSet(id))
        u_bound[id] = 1;
}

__bidevice__ vec3f
GeometricalNormal(vec3f a, vec3f b, vec3f c, bool normalized=true){
    vec3f n;
    vec3f n1 = a - b;
    vec3f n2 = b - c;
    n.x = n1.y * n2.z - n1.z * n2.y;
    n.y = n1.z * n2.x - n1.x * n2.z;
    n.z = n1.x * n2.y - n1.y * n2.x;
    return normalized ? SafeNormalize(n) : n;
}

__bidevice__ void ParticleFlag(ParticleSet3 *pSet, Grid3 *grid, int pId, Float radius,
                               int threshold)
{
    int *neighbors = nullptr;
    vec3f pi = pSet->GetParticlePosition(pId);
    unsigned int cellId = grid->GetLinearHashedPosition(pi);
    Cell3 *cell = grid->GetCell(cellId);
    int count = grid->GetNeighborsOf(cellId, &neighbors);

    vec3f pnei[3] = {vec3f(), vec3f(), vec3f()};
    int iterator = 0;

    Float twoRadius = 2.0 * radius * 0.5;
    auto check_add = [&](vec3f pj, int j) -> bool{
        if(j == pId)
            return iterator < 3;

        if(iterator == 3)
            return false;

        Float dist = Distance(pj, pi);
        if(dist < 1e-4)
            return true;

        if(dist < twoRadius){
            bool accept = true;
            for(int k = 0; k < iterator && accept; k++){
                Float d = Distance(pj, pnei[k]);
                if(d < 1e-4)
                    accept = false;
            }

            if(accept){
                pnei[iterator] = pj;
                iterator += 1;
            }
        }
        return iterator < 3;
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

    if(iterator == 3)
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
        ParticleFlag(pSet, domain, i, radius, threshold);
    });

    //DumpPoints(pSet);
    //exit(0);
}

void SpawnTetra1(vec3f pi, int pId, vec3f n, int id, DelaunaySetBuilder &dBuilder,
                 std::vector<uint32_t> &ids, Point3HVec &pointVec, Float edgeLen)
{
    const vec3f u0 = vec3f(0.57735, 0.57735, 0.57735);
    const vec3f u1 = vec3f(0.57735, -0.57735, -0.57735);
    const vec3f u2 = vec3f(-0.57735, 0.57735, -0.57735);
    const vec3f u3 = vec3f(-0.57735, -0.57735, 0.57735);
    const vec3f d0 = Normalize(u1 - u0);
    const vec3f d1 = Normalize(u2 - u0);
    const vec3f d2 = Normalize(u3 - u0);

    vec3f p1 = pi + d0 * edgeLen;
    vec3f p2 = pi + d1 * edgeLen;
    vec3f p3 = pi + d2 * edgeLen;

    ids.push_back(pId);
    pointVec.push_back({pi.x, pi.y, pi.z});

    ids.push_back(id + 0);
    ids.push_back(id + 1);
    ids.push_back(id + 2);

    pointVec.push_back({p1.x, p1.y, p1.z});
    pointVec.push_back({p2.x, p2.y, p2.z});
    pointVec.push_back({p3.x, p3.y, p3.z});

    dBuilder.PushParticle(p1, n);
    dBuilder.PushParticle(p2, n);
    dBuilder.PushParticle(p3, n);
}


std::unordered_map<uint32_t, int> verticesFlag;
void CheckPart(ParticleSet3 *pSet, Grid3 *grid, int pId, DelaunaySetBuilder &dBuilder,
               std::vector<uint32_t> &ids, Point3HVec &pointVec, Float dist)
{
    int acceptpart = 1;
    vec3f pi = pSet->GetParticlePosition(pId);
#if 1
    Bucket *bucket = pSet->GetParticleBucket(pId);
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
#else
    int *neighbors = nullptr;
    unsigned int cellId = grid->GetLinearHashedPosition(pi);
    int count = grid->GetNeighborsOf(cellId, &neighbors);
    for(int i = 0; i < count; i++){
        Cell3 *cell = grid->GetCell(neighbors[i]);
        ParticleChain *pChain = cell->GetChain();
        int size = cell->GetChainLength();

        for(int j = 0; j < size; j++){
            if(pChain->pId != pId){
                vec3f pj = pSet->GetParticlePosition(pChain->pId);
                Float d = Distance(pi, pj);
                if(d < 1e-4){
                    if(pId > j){
                        acceptpart = 0;
                        break;
                    }
                }
            }
        }
    }
#endif
    if(acceptpart == 1){
        int totalP = pSet->GetParticleCount() + dBuilder.Size();
        vec3f n = pSet->GetParticleNormal(pId);
        int bi = pSet->GetParticleV0(pId);
        if(bi > 0){
            //SpawnTetra2(pi, pId, n, totalP, dBuilder, ids, pointVec, 0.9 * dist);
            SpawnTetra1(pi, pId, n, totalP, dBuilder, ids, pointVec, 0.9 * dist);
            //SpawnParticles(pi, pId, n, totalP, dBuilder, ids, pointVec, cellId, grid, pSet);
        }else{
            ids.push_back(pId);
            pointVec.push_back({pi.x, pi.y, pi.z});
        }
    }else{
        verticesFlag[pId] = -1;
        pSet->SetParticleV0(pId, -1);
    }
}

__bidevice__ bool DelaunayIsTetValid(Tet tet, TetOpp opp, char v, uint32_t pLen){
    bool valid = true;
    if(!isTetAlive(v)) valid = false;

    for(int s = 0; s < 4; s++){
        if(opp._t[s] == -1) valid = false;
        if(tet._v[s] == pLen) valid = false; // inf point
    }

    return valid;
}

__bidevice__ int matching_faces(i3 *tFaces, i3 face){
    for(int i = 0; i < 4; i++){
        if(tFaces[i] == face)
            return i;
    }

    return -1;
}

__bidevice__ vec3ui matching_orientation(i3 face, Tet &tet){
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
    delQ->SetSlots(size, false);

    Float radius = mu * spacing;
    char *tetFlags = cudaAllocateUnregisterVx(char, size);

    timer.Start("Tetrahedra Filtering");
    AutoParallelFor("Delaunay_Filter", size, AutoLambda(int i){
        Tet tet = kernTet[i];
        int A = tet._v[0], B = tet._v[1],
            C = tet._v[2], D = tet._v[3];
        const TetOpp botOpp = kernTetOpp[i];

        tetFlags[i] = 0;

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
                DelaunaySetBoundary(dSet, idA, bound);
                DelaunaySetBoundary(dSet, idB, bound);
                DelaunaySetBoundary(dSet, idC, bound);
                DelaunaySetBoundary(dSet, idD, bound);
            }else
                tetFlags[i] = 1;
        }
    });

    timer.StopAndNext("Find Unique Triangles");
    AutoParallelFor("Delaunay_UniqueTriangles", size, AutoLambda(int i){
        Tet tet = kernTet[i];
        int A = tet._v[0], B = tet._v[1],
            C = tet._v[2], D = tet._v[3];
        const TetOpp botOpp = kernTetOpp[i];

        i3 faces[] = {i3(A, B, C), i3(A, B, D), i3(A, D, C), i3(B, D, C)};
        int info[] = {0, 0, 0, 0};
        int opp[] = {D, C, B, A};

        if(tetFlags[i] == 0)
            return;

        for(int botVi = 0; botVi < 4; botVi++){
            const int topTi = botOpp.getOppTet(botVi);
            const Tet topTet  = kernTet[topTi];
            if(tetFlags[topTi] == 0)
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
                delQ->Push(vec4ui(tri.x, tri.y, tri.z, opp[s]));
            }
        }
    });
    timer.Stop();

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
    std::vector<uint32_t> *ids = nullptr;
    DelaunayWorkQueue *delQ = nullptr;
    DelaunaySetBuilder dBuilder;
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

    timer.Start("Particle Filter and Spawn");
    for(int i = 0; i < pSet->GetParticleCount(); i++){
        CheckPart(pSet, domain, i, dBuilder, triangulation.ids,
                  triangulation.pointVec, dist);
        u_bound[i] = 0;
    }
    timer.Stop();

    pointNum = triangulation.pointVec.size();
    std::cout << "done\n - Support points " << dBuilder.Size() << std::endl;

    dSet = dBuilder.Build(pSet->GetParticleCount());

    std::cout << " - Running delaunay triangulation..." << std::flush;
    timer.Start("Delaunay Triangulation");
    triangulator.compute(triangulation.pointVec, &triangulation.output);
    timer.Stop();
    triangulation.pLen = pointNum;
    std::cout << "done" << std::endl;

    ids = &triangulation.ids;
    u_ids = cudaAllocateUnregisterVx(uint32_t, ids->size());
    memcpy(u_ids, ids->data(), sizeof(uint32_t) * ids->size());

    delQ = cudaAllocateUnregisterVx(DelaunayWorkQueue, 1);
    DelaunayWorkQueueAndFilter(&triangulator, triangulation, pSet, dSet,
                                u_ids, u_bound, mu, spacing, delQ, timer);

    std::cout << " - Aggregating triangles..." << std::flush;

    u_tri = cudaAllocateUnregisterVx(vec3i, delQ->size);
    triangulation.shrinked.resize(delQ->size);

    timer.Start("(Extra) Mark Boundary");
    AutoParallelFor("Delaunay_MarkTrisAndBoundary", delQ->size, AutoLambda(int i){
        vec4ui uniqueTri = delQ->At(i);
        vec3ui tri = vec3ui(uniqueTri.x, uniqueTri.y, uniqueTri.z);
        int opp = uniqueTri.w;

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
    //       but it is not actually affecting performance
    std::unordered_map<uint32_t, uint32_t> remap;
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

            vec3f pA = DelaunayPosition(pSet, dSet, idA);
            vec3f pB = DelaunayPosition(pSet, dSet, idB);
            vec3f pC = DelaunayPosition(pSet, dSet, idC);

            uint32_t aId, bId, cId;
            if(remap.find(idA) == remap.end()){
                aId = runningId++;
                remap[idA] = aId;
                verticesFlag[idA] = 1;
                triangulation.shrinkedPs.push_back(pA);
            }else
                aId = remap[idA];

            if(remap.find(idB) == remap.end()){
                bId = runningId++;
                remap[idB] = bId;
                verticesFlag[idB] = 1;
                triangulation.shrinkedPs.push_back(pB);
            }else
                bId = remap[idB];

            if(remap.find(idC) == remap.end()){
                cId = runningId++;
                remap[idC] = cId;
                verticesFlag[idC] = 1;
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

    std::vector<uint32_t> missing;
    for(int i = 0; i < pSet->GetParticleCount(); i++){
        int bi = pSet->GetParticleV0(i);
        if(bi > 0){
            if(verticesFlag.find(i) == verticesFlag.end()){
                missing.push_back(i);
                pSet->SetParticleV0(i, 2);
            }
        }
    }

    std::cout << "done ( boundary: " << b_len << " / " << n_it << " ) " << std::endl;
    std::cout << " - Finished building mesh" << " ( " << triangulation.shrinkedPs.size() << " vertices | "
              << triangulation.shrinked.size() << " triangles )" << std::endl;
    std::cout << " - Unreferenced: " << missing.size() << std::endl;

    DumpPoints(pSet);

    triangulator.cleanup();
    dSet->Cleanup();
    cudaFree(dSet);
    cudaFree(delQ->ids);
    cudaFree(delQ);
    cudaFree(u_tri);
    cudaFree(u_ids);
    cudaFree(u_bound);
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

