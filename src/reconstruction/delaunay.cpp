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

    __host__ void Build(std::vector<vec3f> &pos, std::vector<vec3f> &nor, int off){
        nPos = pos.size();
        nNor = nor.size();
        offset = off;

        if(nPos > 0 && nNor > 0){
            positions = cudaAllocateVx(vec3f, nPos);
            normals = cudaAllocateVx(vec3f, nNor);
            memcpy(positions, pos.data(), pos.size() * sizeof(vec3f));
            memcpy(normals, nor.data(), nor.size() * sizeof(vec3f));
        }
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

    __host__ void PushParticle(vec3f p, vec3f n){
        pos.push_back(p);
        nor.push_back(n);
    }

    __host__ size_t Size(){
        return pos.size();
    }

    __host__ DelaunaySet *Build(int total_p){
        DelaunaySet *dSet = cudaAllocateVx(DelaunaySet, 1);
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
__bidevice__ vec3f DelaunayNormal(ParticleSet3 *pSet, DelaunaySet *dSet, int id){
    int count = pSet->GetParticleCount();
    if(id < count)
        return pSet->GetParticleNormal(id);
    else
        return dSet->GetParticleNormal(id);
}

inline __bidevice__ void DelaunaySetBoundary(DelaunaySet *dSet, int id, int *u_bound){
    if(!dSet->InSet(id))
        u_bound[id] = 1;
}

__host__ void
DelaunayIndexedMapHandleTriangle(int a, int b, int c, int d,
                                 DelaunayTriangleIndexedMap &indexedMap)
{
    i3 key(a, b, c);
    DelaunayIndexInfo indexInfo = { vec3ui(a, b, c), d, 0 };
    if(indexedMap.find(key) != indexedMap.end()){
        indexInfo = indexedMap[key];
    }

    indexInfo.counter += 1;
    indexedMap[key] = indexInfo;
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
                               DelaunaySmallBucket *buckets)
{
    int *neighbors = nullptr;
    vec3f pi = pSet->GetParticlePosition(pId);
    DelaunaySmallBucket *bucket = &buckets[pId];
    unsigned int cellId = grid->GetLinearHashedPosition(pi);
    Cell3 *cell = grid->GetCell(cellId);
    int count = grid->GetNeighborsOf(cellId, &neighbors);

    int within_radius = 0;
    int density = 0;
    for(int i = 0; i < 3; i++){
        bucket->neighbors[i] = -1;
        bucket->pneighbors[i] = vec3f();
    }
    bucket->id = pId;
    bucket->size = 0;
    bucket->density = 0;

    radius *= 0.9;
    auto fn_accept = [&](vec3f _pj, int _j){
        Float dist = Distance(pi, _pj);
        if(dist < 1e-4 || pId == _j || dist >= radius)
            return;

        int accept = 1;
        for(int i = 0; i < 3; i++){
            int nj = bucket->neighbors[i];
            if(nj == -1)
                break;

            vec3f np = bucket->pneighbors[i];
            Float d = Distance(np, _pj);
            if(d < 1e-4 || d >= radius){
                accept = 0;
                break;
            }
        }

        if(accept){
            if(within_radius < 3){
                bucket->neighbors[within_radius] = _j;
                bucket->pneighbors[within_radius] = _pj;
            }
            within_radius += 1;
        }
    };

    for(int i = 0; i < count; i++){
        Cell3 *cell = grid->GetCell(neighbors[i]);
        ParticleChain *pChain = cell->GetChain();
        int size = cell->GetChainLength();
        for(int j = 0; j < size; j++){
            vec3f pj = pSet->GetParticlePosition(pChain->pId);
            fn_accept(pj, pChain->pId);
            if(pChain->pId != pId){
                density += 1;
            }
            pChain = pChain->next;
        }
    }

    bucket->size = 1+within_radius;
    bucket->density = density;
}

__host__ void SpawnParticles(vec3f pi, int pId, vec3f n, int id, DelaunaySetBuilder &dBuilder,
                          std::vector<uint32_t> &ids, Point3HVec &pointVec, Float radius)
{
    ids.push_back(pId);
    pointVec.push_back({pi.x, pi.y, pi.z});

    for(int i = 0; i < 10; i++){
        Float u0 = rand_float();
        Float u1 = rand_float();
        Float r = radius;

        Float theta = TwoPi * u0;
        Float phi = acos(1.f - 2.f * u1);
        Float x = r * sin(phi) * cos(theta);
        Float y = r * sin(phi) * sin(theta);
        Float z = r * cos(phi);

        vec3f p = pi + vec3f(x, y, z);
        ids.push_back(id + i);
        pointVec.push_back({p.x, p.y, p.z});
        dBuilder.PushParticle(p, n);
    }
}

__host__ void SpawnTetra1(vec3f pi, int pId, vec3f n, int id, DelaunaySetBuilder &dBuilder,
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

__host__ void SpawnTetra2(vec3f p0, vec3f p1, vec3f n, int id, DelaunaySetBuilder &dBuilder,
                          std::vector<uint32_t> &ids, Point3HVec &pointVec, Float edgeLen)
{
    Matrix3x3 I;
    const vec3f u0 = vec3f(0.57735, 0.57735, 0.57735);
    const vec3f u1 = vec3f(0.57735, -0.57735, -0.57735);
    const vec3f u2 = vec3f(-0.57735, 0.57735, -0.57735);
    const vec3f u3 = vec3f(-0.57735, -0.57735, 0.57735);
    vec3f dir = Normalize(p1 - p0);
    vec3f edir = Normalize(u1 - u0);
    vec3f v = Cross(edir, dir);

    Float c = Dot(edir, dir);
    if(c == -1){
        // TODO: direction dir is the negative of edir, all we need to do is rotate
        //       p1 by 180 or simply swap p1 and p0 and scale the tetra
        printf("Invalid direction for build\n");
        return;
    }

    // Efficiently Building a Matrix to Rotate One Vector to Another
    //    Tomas Moller and John F. Hughes
    Matrix3x3 vx( 0,   -v.z,  v.y,
                  v.z,  0,   -v.x,
                 -v.y,  v.x,  0 );
    Matrix3x3 vx2 = Matrix3x3::Mul(vx, vx);
    Matrix3x3 R = I + vx + vx2 * (1.f / (1.f + c));

    vec3f ru2 = R.Vec(u2);
    vec3f ru3 = R.Vec(u3);

    vec3f dir_u2 = Normalize(ru2 - p1);
    vec3f dir_u3 = Normalize(ru3 - p1);

    vec3f np1 = p1 + dir_u2 * edgeLen;
    vec3f np2 = p1 + dir_u3 * edgeLen;

    ids.push_back(id + 0);
    ids.push_back(id + 1);

    pointVec.push_back({np1.x, np1.y, np1.z});
    pointVec.push_back({np2.x, np2.y, np2.z});

    dBuilder.PushParticle(np1, n);
    dBuilder.PushParticle(np2, n);
}

__host__ void SpawnTetra3(vec3f p0, vec3f p1, vec3f p2, vec3f n, int id,
                          DelaunaySetBuilder &dBuilder, std::vector<uint32_t> &ids,
                          Point3HVec &pointVec, Float edgeLen)
{
    vec3f nb = GeometricalNormal(p0, p1, p2);
    vec3f bcenter = (p0 + p1 + p2) * 0.3333f;
    Float e = edgeLen;
    Float d0 = Distance(p0, bcenter);
    Float d0_2 = d0 * d0;
    Float ed2 = e * e;

    Float d2 = d0_2;
    if(ed2 < d2){
        printf("Warning: large triangle?\n");
    }else{
        Float h = Max(0.1 * edgeLen, sqrt(ed2 - d2) * 0.8);
        vec3f np = bcenter + h * nb;

        //Float e0 = Distance(p0, np);
        //Float e1 = Distance(p1, np);
        //Float e2 = Distance(p2, np);
        //printf("Ratio {%g %g %g}\n", e0/edgeLen, e1/edgeLen, e2/edgeLen);
        ids.push_back(id + 0);
        pointVec.push_back({np.x, np.y, np.z});
        dBuilder.PushParticle(np, n);
    }
}

__host__ void CheckPart(ParticleSet3 *pSet, Grid3 *grid, int pId, int threshold,
                        DelaunaySetBuilder &dBuilder, std::vector<uint32_t> &ids,
                        Point3HVec &pointVec, Float dist, DelaunaySmallBucket *sbucket)
{
    vec3f pi = pSet->GetParticlePosition(pId);
    unsigned int cellId = grid->GetLinearHashedPosition(pi);
    Cell3 *cell = grid->GetCell(cellId);
    Bucket *bucket = pSet->GetParticleBucket(pId);
    int acceptpart = 1;
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
        int totalP = pSet->GetParticleCount() + dBuilder.Size();
        vec3f n = pSet->GetParticleNormal(pId);
        if(sbucket->density < threshold){
            SpawnTetra1(pi, pId, n, totalP, dBuilder, ids, pointVec, 0.9 * dist);
            //SpawnParticles(pi, pId, n, totalP, dBuilder, ids, pointVec, dist);
        }else{
            ids.push_back(pId);
            pointVec.push_back({pi.x, pi.y, pi.z});
        }
    #if 0
        else if(sbucket->size == 2){
            vec3f p1 = sbucket->pneighbors[0];
            SpawnTetra2(pi, p1, n, totalP, dBuilder, ids, pointVec, 0.99 * dist);
        }else if(sbucket->size == 3){
            vec3f p1 = sbucket->pneighbors[0];
            vec3f p2 = sbucket->pneighbors[1];
            SpawnTetra3(pi, p1, p2, n, totalP, dBuilder, ids, pointVec, 0.99 * dist);
        }
    #endif
    }
}

__bidevice__ bool
TriangleMatchesDirection(vec3f a, vec3f b, vec3f c, vec3f ng){
    vec3f n = GeometricalNormal(a, b, c, false);
    Float d = Dot(n, ng);
    return (d >= 0);
}

__bidevice__ vec3i
DelaunayOrientation(vec3f a, vec3f b, vec3f c, vec3f op, int A, int B, int C,
                    vec3f na, vec3f nb, vec3f nc)
{
    vec3f ng = SafeNormalize(op - a);
    if(!TriangleMatchesDirection(a, b, c, ng))
        return vec3i(A, B, C);
    if(TriangleMatchesDirection(a, c, b, ng)){
        vec3f navg = SafeNormalize(na + nb + nc);
        Float dot = Dot(GeometricalNormal(a, b, c), navg);
        if(dot < 0)
            return vec3i(A, C, B);
        else
            return vec3i(A, B, C);
    }
    return vec3i(A, C, B);
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

__bidevice__ int face_inside(i3 *tFaces, i3 face){
    for(int i = 0; i < 4; i++){
        if(tFaces[i] == face)
            return i;
    }

    return -1;
}

__host__ static void
DelaunayWorkQueueAndFilter(GpuDel *triangulator, DelaunayTriangulation &triangulation,
                           ParticleSet3 *pSet, DelaunaySet *dSet, uint32_t *ids,
                           int *bound, Float mu, Float spacing, DelaunayWorkQueue *delQ)
{
    // TODO: Cpu stuff
    Tet *kernTet = toKernelPtr(triangulator->_tetVec);
    TetOpp *kernTetOpp = toKernelPtr(triangulator->_oppVec);
    char *kernChar = toKernelPtr(triangulator->_tetInfoVec);

    uint32_t size = triangulator->_tetVec.size();
    uint32_t pLen = triangulation.pLen;
    delQ->SetSlots(size, false);

    Float radius = mu * spacing;
    int *tetFlags = cudaAllocateUnregisterVx(int, size);

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
            }else{
                tetFlags[i] = 1;
            }
        }
    });

    AutoParallelFor("Delaunay_UniqueTriangles", size, AutoLambda(int i){
        Tet tet = kernTet[i];
        int A = tet._v[0], B = tet._v[1],
            C = tet._v[2], D = tet._v[3];
        const TetOpp botOpp = kernTetOpp[i];

        i3 faces[] = {i3(A, B, C), i3(A, B, D), i3(A, D, C), i3(B, D, C)};
        int info[] = {0, 0, 0, 0};
        vec4ui tris[] = { vec4ui(A, B, C, D), vec4ui(A, B, D, C),
                          vec4ui(A, D, C, B), vec4ui(B, D, C, A) };

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
                int where = face_inside(faces, topFaces[k]);
                if(where >= 0){
                    info[where] += 1;
                }
            }
        }

        for(int s = 0; s < 4; s++){
            if(info[s] == 0){
                delQ->Push(tris[s]);
            }
        }
    });

    cudaFree(tetFlags);
}

__host__ void
DelaunaySurface(DelaunayTriangulation &triangulation, SphParticleSet3 *sphSet,
                Float spacing, Float mu, Grid3 *domain)
{
    GpuDel triangulator;
    int *u_bound = nullptr;
    vec3i *u_tri = nullptr;
    uint32_t *u_ids = nullptr;
    std::vector<uint32_t> *ids = nullptr;
    DelaunayWorkQueue *delQ = nullptr;
    DelaunaySmallBucket *buckets = nullptr;
    DelaunaySetBuilder dBuilder;
    int pointNum = 0;
    const int threshold = 30;

    DelaunaySet *dSet = nullptr;
    ParticleSet3 *pSet = sphSet->GetParticleSet();
    std::vector<int> *boundary = &triangulation.boundary;

    printf(" * Delaunay: [μ = %g] [h = %g]\n", mu, spacing);

    std::cout << " - Adjusting domain..." << std::flush;
    // TODO: Adjust based on extended domain
    u_bound = cudaAllocateUnregisterVx(int, pSet->GetParticleCount());
    boundary->reserve(pSet->GetParticleCount());

    buckets = cudaAllocateUnregisterVx(DelaunaySmallBucket, pSet->GetParticleCount());

    Float dist = mu * spacing;

    AutoParallelFor("Delaunay_GetDomain", pSet->GetParticleCount(), AutoLambda(int i){
        ParticleFlag(pSet, domain, i, dist, buckets);
    });

    for(int i = 0; i < pSet->GetParticleCount(); i++){
        CheckPart(pSet, domain, i, threshold, dBuilder, triangulation.ids,
                  triangulation.pointVec, dist, &buckets[i]);
        u_bound[i] = 0;
    }

    pointNum = triangulation.pointVec.size();
    std::cout << "done\n - Support points " << dBuilder.Size() << std::endl;

    dSet = dBuilder.Build(pSet->GetParticleCount());

    std::cout << " - Running delaunay triangulation..." << std::flush;
    triangulator.compute(triangulation.pointVec, &triangulation.output);
    triangulation.pLen = pointNum;
    std::cout << "done" << std::endl;

    ids = &triangulation.ids;
    u_ids = cudaAllocateUnregisterVx(uint32_t, ids->size());
    memcpy(u_ids, ids->data(), sizeof(uint32_t) * ids->size());

    delQ = cudaAllocateUnregisterVx(DelaunayWorkQueue, 1);
    DelaunayWorkQueueAndFilter(&triangulator, triangulation, pSet, dSet,
                                u_ids, u_bound, mu, spacing, delQ);

    std::cout << " - Aggregating triangles..." << std::flush;

    u_tri = cudaAllocateUnregisterVx(vec3i, delQ->size);
    triangulation.shrinked.resize(delQ->size);

    AutoParallelFor("Delaunay_OrderTriangles", delQ->size, AutoLambda(int i){
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
        vec3f pA = DelaunayPosition(pSet, dSet, idA);
        vec3f pB = DelaunayPosition(pSet, dSet, idB);
        vec3f pC = DelaunayPosition(pSet, dSet, idC);
        vec3f pD = DelaunayPosition(pSet, dSet, idD);

        vec3f _nA = DelaunayNormal(pSet, dSet, idA);
        vec3f _nB = DelaunayNormal(pSet, dSet, idB);
        vec3f _nC = DelaunayNormal(pSet, dSet, idC);

        vec3i index = DelaunayOrientation(pA, pB, pC, pD, A, B, C,
                                          _nA, _nB, _nC);

        DelaunaySetBoundary(dSet, idA, u_bound);
        DelaunaySetBoundary(dSet, idB, u_bound);
        DelaunaySetBoundary(dSet, idC, u_bound);
        DelaunaySetBoundary(dSet, idD, u_bound);
        u_tri[i] = index;
    });

    std::cout << "done" << std::endl;

    std::cout << " - Copying to host..." << std::flush;
    int n_it = pSet->GetParticleCount();
    int b_len = 0;
    int max_it = Max(n_it, delQ->size);
    for(int i = 0; i < max_it; i++){
        if(i < pSet->GetParticleCount()){
            b_len += u_bound[i] > 0;
            boundary->push_back(u_bound[i]);
        }
        if(i < delQ->size)
            triangulation.shrinked[i] = u_tri[i];
    }
    std::cout << "done ( boundary: " << b_len << " / " << n_it << " ) " << std::endl;

    cudaFree(delQ->ids);
    cudaFree(delQ);
    cudaFree(u_tri);
    cudaFree(u_ids);
    cudaFree(u_bound);
    cudaFree(buckets);
    triangulator.cleanup();
}

__host__ void
DelaunayWriteBoundary(DelaunayTriangulation &triangulation, SphParticleSet3 *sphSet,
                      const char *path)
{
    FILE *fp = fopen(path, "w");
    ParticleSet3 *pSet = sphSet->GetParticleSet();
    if(fp){
        std::vector<int> &boundary = triangulation.boundary;
        fprintf(fp, "%lu\n", boundary.size());
        for(int i = 0; i < boundary.size(); i++){
            vec3f p = pSet->GetParticlePosition(i);
            fprintf(fp, "%g %g %g %d\n", p.x, p.y, p.z, boundary[i]);
        }
        fclose(fp);
    }
}

__host__ void
DelaunayGetTriangleMesh(DelaunayTriangulation &triangulation, HostTriangleMesh3 *mesh){
    if(mesh){
        PredWrapper predWrapper;
        predWrapper.init(triangulation.pointVec, triangulation.output.ptInfty);
        for(int i = 0; i < (int)predWrapper.pointNum(); i++){
            const Point3 pt = predWrapper.getPoint(i);
            mesh->addPoint(vec3f(pt._p[0], pt._p[1], pt._p[2]));
        }

        for(vec3i index : triangulation.shrinked){
            vec3ui face(index[0], index[1], index[2]);
            mesh->addPointUvNormalTriangle(face, face, face);
        }
    }
}

__host__ void
DelaunayWritePly(DelaunayTriangulation &triangulation, const char *path){
    UtilGDel3DWritePly(&triangulation.shrinked, &triangulation.pointVec,
                       &triangulation.output, path);
}
