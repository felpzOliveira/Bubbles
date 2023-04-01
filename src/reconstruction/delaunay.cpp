#include <delaunay.h>
#include <gDel3D/GpuDelaunay.h>
#include <kernel.h>
#include <util.h>
#include <vector>
#include <interval.h>
#include <fstream>

template<typename DelaunayMap> __host__ bool
DelaunayCanPushTriangle(int a, int b, int c, DelaunayMap &triMap){
    if(!(a < 0 || b < 0 || c < 0)){
        i3 i_val(a, b, c);
        return triMap.find(i_val) == triMap.end();
    }
    return false;
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

#define PUSH_TRIANGLE(a, b, c, triMap)do{\
    triMap[i3(a, b, c)] = vec3ui(a, b, c);\
}while(0)

#define MARK_TRIANGLE(a, b, c, vertexMap, triMap)do{\
    vertexMap[a] += 1;\
    vertexMap[b] += 1;\
    vertexMap[c] += 1;\
    triMap[i3(a, b, c)] = 1;\
}while(0)

#define PUSH_INDEX_INTO_LIST(a, b, c, indexList)do{\
    indexList.push_back(vec3i(a, b, c));\
}while(0)

__host__ int CheckPart(ParticleSet3 *pSet, Grid3 *grid, int pId, Float &pR){
    vec3f pi = pSet->GetParticlePosition(pId);
    Bucket *bucket = pSet->GetParticleBucket(pId);
    int acceptpart = bucket->Count() < 2 ? 0 : 1;
    pR = Infinity;
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
            }else{
                pR = Min(pR, d);
            }
        }
    }
    return acceptpart;
}

__host__ vec3f
GeometricalNormal(vec3f a, vec3f b, vec3f c, bool normalized=true){
    vec3f n;
    vec3f n1 = a - b;
    vec3f n2 = b - c;
    n.x = n1.y * n2.z - n1.z * n2.y;
    n.y = n1.z * n2.x - n1.x * n2.z;
    n.z = n1.x * n2.y - n1.y * n2.x;
    return normalized ? SafeNormalize(n) : n;
}

__host__ bool
TriangleMatchesDirection(vec3f a, vec3f b, vec3f c, vec3f ng){
    vec3f n = GeometricalNormal(a, b, c, false);
    return !(Dot(n, ng) < 0);
}

__host__ vec3i
DelaunayOrientation(vec3f a, vec3f b, vec3f c, vec3f ng, int A, int B, int C){
    if(TriangleMatchesDirection(a, b, c, ng))
        return vec3i(A, B, C);
    else{
        if(!TriangleMatchesDirection(a, c, b, ng)){
            printf("Inverted triangle does not match normal ( %g %g %g ) {%d %d %d}\n",
                    ng.x, ng.y, ng.z, A, B, C);
            vec3f n0 = GeometricalNormal(a, b, c);
            vec3f n1 = GeometricalNormal(a, c, b);
            Float d0 = Dot(n0, ng);
            Float d1 = Dot(n1, ng);
            printf("Forward  ( %g %g %g ) --> %g\n", n0.x, n0.y, n0.z, d0);
            printf("Backward ( %g %g %g ) --> %g\n", n1.x, n1.y, n1.z, d1);
        }
        return vec3i(A, C, B);
    }
}

__host__ void
DelaunayCacheStore(std::vector<int> &boundary, const char *path){
    FILE *fp = fopen(path, "wb");
    int *mem = boundary.data();
    if(fp){
        int c_mem = boundary.size();
        fwrite(&c_mem, sizeof(int), 1, fp);
        fwrite(mem, sizeof(int) * c_mem, 1, fp);
        fclose(fp);
    }
}

__host__ int
DelaunayCacheLoad(std::vector<int> &boundary, const char *path){
    FILE *fp = fopen(path, "rb");
    int rv = -1;
    if(fp){
        int c_mem = 0;
        fread(&c_mem, sizeof(int), 1, fp);
        if(c_mem > 0){
            int *mem = new int[c_mem];
            fread(mem, sizeof(int) * c_mem, 1, fp);
            for(int i = 0; i < c_mem; i++){
                boundary.push_back(mem[i]);
            }
            rv = c_mem;
            delete[] mem;
        }
        fclose(fp);
    }
    return rv;
}

#define DELAUNAY_CACHE_BOUNDARY 0

__host__ void
DelaunayPushParticleCellBoundary(std::vector<int> &boundary, vec3f p, Grid3 *domain){
    int id = domain->GetLinearHashedPosition(p);
    Cell3 *cell = domain->GetCell(id);
    ParticleChain *pChain = cell->GetChain();
    int size = cell->GetChainLength();

    for(int i = 0; i < size; i++){
        boundary[pChain->pId] = 1;
        pChain = pChain->next;
    }
}

__host__ void
DelaunaySurface(DelaunayTriangulation &triangulation, SphParticleSet3 *sphSet,
                Grid3 *domain)
{
    ParticleSet3 *pSet = sphSet->GetParticleSet();
    DelaunayTriangleIndexedMap keyMap;
    DelaunayTriangleMap triMap;
    std::vector<int> boundary;
    GpuDel triangulator;

    Float spacing = 0.02;
    Float mu = 2.0;

    int pointNum = 0;
    boundary.reserve(pSet->GetParticleCount());
    for(int i = 0; i < pSet->GetParticleCount(); i++){
        Float pR = 0;
        boundary.push_back(0);
        if(CheckPart(pSet, domain, i, pR)){
            vec3f vi = pSet->GetParticlePosition(i);
            triangulation.ids.push_back(i);
            triangulation.partRMap[i] = mu * spacing;
            triangulation.pointVec.push_back({vi.x, vi.y, vi.z});
            pointNum += 1;
        }
    }

    printf("Point Count %d / %d\n", pointNum, pSet->GetParticleCount());
    triangulator.compute(triangulation.pointVec, &triangulation.output);
    triangulation.pLen = pointNum;

    std::vector<uint32_t> *ids = &triangulation.ids;
    DelaunayFloatTriangleMap &partRMap = triangulation.partRMap;
    DelaunayVertexMap &vertexMap = triangulation.vertexMap;

    std::vector<vec3i> indexList;
    long int totalTris = 0, totalInserted = 0;

    GDel3D_ForEachRealTetra(&triangulation.output, triangulation.pLen,
    [&](Tet tet, TetOpp botOpp, int i) -> void
    {
        int A = tet._v[0], B = tet._v[1],
            C = tet._v[2], D = tet._v[3];

        int idA = ids->at(A), idB = ids->at(B),
            idC = ids->at(C), idD = ids->at(D);
        vec3f pA = pSet->GetParticlePosition(idA);
        vec3f pB = pSet->GetParticlePosition(idB);
        vec3f pC = pSet->GetParticlePosition(idC);
        vec3f pD = pSet->GetParticlePosition(idD);

        Float rA = partRMap[idA];
        Float rB = partRMap[idB];
        Float rC = partRMap[idC];
        Float rD = partRMap[idD];

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
            boundary[idA] = 1;
            boundary[idB] = 1;
            boundary[idC] = 1;
            boundary[idD] = 1;
            return;
        }

        DelaunayIndexedMapHandleTriangle(A, B, C, D, keyMap);
        DelaunayIndexedMapHandleTriangle(A, B, D, C, keyMap);
        DelaunayIndexedMapHandleTriangle(A, D, C, B, keyMap);
        DelaunayIndexedMapHandleTriangle(B, D, C, A, keyMap);
    });

    for(auto it : keyMap){
        totalTris += 1;
        DelaunayIndexInfo indexInfo = it.second;
        vec3ui tri = indexInfo.baseTriangle;

        if(indexInfo.counter > 1)
            continue;

        int A = tri.x;
        int B = tri.y;
        int C = tri.z;

        int idA = ids->at(A);
        int idB = ids->at(B);
        int idC = ids->at(C);
        int idD = ids->at(indexInfo.oposite);
        vec3f pA = pSet->GetParticlePosition(idA);
        vec3f pB = pSet->GetParticlePosition(idB);
        vec3f pC = pSet->GetParticlePosition(idC);
        vec3f pD = pSet->GetParticlePosition(idD);

        vec3f nA = pSet->GetParticleNormal(idA);
        vec3f nB = pSet->GetParticleNormal(idB);
        vec3f nC = pSet->GetParticleNormal(idC);

        vec3f n = SafeNormalize((nA + nC + nB));
        vec3i index = DelaunayOrientation(pA, pB, pC, n, A, B, C);
        MARK_TRIANGLE(A, B, C, vertexMap, triMap);
        PUSH_INDEX_INTO_LIST(index[0], index[1], index[2], triangulation.shrinked);

        boundary[idA] = 1;
        boundary[idB] = 1;
        boundary[idC] = 1;
        boundary[idD] = 1;

        //DelaunayPushParticleCellBoundary(boundary, pA, domain);
        //DelaunayPushParticleCellBoundary(boundary, pB, domain);
        //DelaunayPushParticleCellBoundary(boundary, pC, domain);
        //DelaunayPushParticleCellBoundary(boundary, pD, domain);

        totalInserted += 1;
    }

    printf("Total: %ld, Inserted: %ld\n", totalTris, totalInserted);
    ////////////////////////////////////
    FILE *fp = fopen("bound.txt", "w");
    if(fp){
        fprintf(fp, "%lu\n", boundary.size());
        for(int i = 0; i < boundary.size(); i++){
            vec3f p = pSet->GetParticlePosition(i);
            fprintf(fp, "%g %g %g %d\n", p.x, p.y, p.z, boundary[i]);
        }
        fclose(fp);
    }
    ////////////////////////////////////
    UtilGDel3DWritePly(&triangulation.shrinked, &triangulation.pointVec,
                       &triangulation.output, "delaunay2.ply");
}


__host__ void
DelaunaySurface2(DelaunayTriangulation &triangulation, SphParticleSet3 *sphSet,
                Grid3 *domain)
{
    ParticleSet3 *pSet = sphSet->GetParticleSet();
    DelaunayTriangleMap triMap;
    std::vector<int> boundary;
    GpuDel triangulator;

    Float spacing = 0.02;
    Float mu = 2.0;

    int n = 0;
    if(DELAUNAY_CACHE_BOUNDARY){
        std::cout << " * Extracting boundary ... " << std::flush;
        IntervalBoundary(pSet, domain, spacing, PolygonSubdivision);
        n = UtilGetBoundaryState(pSet, &boundary);
        DelaunayCacheStore(boundary, "cache");
    }else{
        std::cout << " * Loading boundary cache ... " << std::flush;
        n = DelaunayCacheLoad(boundary, "cache");
    }

    if(n < 0){
        printf("Failed to get compute boundary\n");
        exit(0);
    }
    printf("%d / %d\n", n, pSet->GetParticleCount());

    int pointNum = 0;
    for(int i = 0; i < pSet->GetParticleCount(); i++){
        Float pR = 0;
        if(boundary[i] && CheckPart(pSet, domain, i, pR)){
            vec3f vi = pSet->GetParticlePosition(i);
            triangulation.ids.push_back(i);
            triangulation.partRMap[i] = mu * spacing;
            triangulation.pointVec.push_back({vi.x, vi.y, vi.z});
            pointNum += 1;
        }
    }
    printf("Point Count %d / %d\n", pointNum, n);
    triangulator.compute(triangulation.pointVec, &triangulation.output);
    triangulation.pLen = pointNum;

    std::vector<uint32_t> *ids = &triangulation.ids;
    DelaunayFloatTriangleMap &partRMap = triangulation.partRMap;
    DelaunayVertexMap &vertexMap = triangulation.vertexMap;

    std::vector<vec3i> indexList;
    GDel3D_ForEachRealTetra(&triangulation.output, triangulation.pLen,
    [&](Tet tet, TetOpp botOpp, int i) -> void
    {
        int A = tet._v[0], B = tet._v[1],
            C = tet._v[2], D = tet._v[3];
        int idA = ids->at(A), idB = ids->at(B),
            idC = ids->at(C), idD = ids->at(D);
        vec3f pA = pSet->GetParticlePosition(idA);
        vec3f pB = pSet->GetParticlePosition(idB);
        vec3f pC = pSet->GetParticlePosition(idC);
        vec3f pD = pSet->GetParticlePosition(idD);

        vec3f nA = pSet->GetParticleNormal(idA);
        vec3f nB = pSet->GetParticleNormal(idB);
        vec3f nC = pSet->GetParticleNormal(idC);
        vec3f nD = pSet->GetParticleNormal(idD);

        Float rA = partRMap[idA];
        Float rB = partRMap[idB];
        Float rC = partRMap[idC];
        Float rD = partRMap[idD];

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

#if 0
        bool is_tetra_valid =  aB < rAB && aC < rAC && aD < rAD &&
                               bC < rBC && bD < rBD &&
                               cD < rCD;

        if(!is_tetra_valid)
            return;
#endif
        // ABC
        if(aC < rAC && aB < rAB && bC < rBC){
            if(DelaunayCanPushTriangle(A, B, C, triMap)){
                vec3f n = SafeNormalize((nA + nC + nB));
                vec3i index = DelaunayOrientation(pA, pB, pC, n, A, B, C);
                MARK_TRIANGLE(A, B, C, vertexMap, triMap);
                PUSH_INDEX_INTO_LIST(index[0], index[1], index[2],
                                    triangulation.shrinked);
            }
        }
        // ABD
        if(aB < rAB && aD < rAD && bD < rBD){
            if(DelaunayCanPushTriangle(A, B, D, triMap)){
                vec3f n = SafeNormalize((nA + nB + nD));
                vec3i index = DelaunayOrientation(pA, pB, pD, n, A, B, D);
                MARK_TRIANGLE(A, B, D, vertexMap, triMap);
                PUSH_INDEX_INTO_LIST(index[0], index[1], index[2],
                                    triangulation.shrinked);
            }
        }
        // ADC
        if(aD < rAD && aC < rAC && cD < rCD){
            if(DelaunayCanPushTriangle(A, D, C, triMap)){
                vec3f n = SafeNormalize((nA + nC + nD));
                vec3i index = DelaunayOrientation(pA, pD, pC, n, A, D, C);
                MARK_TRIANGLE(A, D, C, vertexMap, triMap);
                PUSH_INDEX_INTO_LIST(index[0], index[1], index[2],
                                    triangulation.shrinked);
            }
        }
        // BDC
        if(bD < rBD && bC < rBC && cD < rCD){
            if(DelaunayCanPushTriangle(B, D, C, triMap)){
                vec3f n = SafeNormalize((nB + nC + nD));
                vec3i index = DelaunayOrientation(pB, pD, pC, n, B, D, C);
                MARK_TRIANGLE(B, D, C, vertexMap, triMap);
                PUSH_INDEX_INTO_LIST(index[0], index[1], index[2],
                                    triangulation.shrinked);
            }
        }
    });

    UtilGDel3DWritePly(&triangulation.shrinked, &triangulation.pointVec,
                       &triangulation.output, "delaunay2.ply");
}

__host__ void
DelaunayWritePly(DelaunayTriangulation &triangulation, const char *path){
    UtilGDel3DWritePly(&triangulation.shrinked, &triangulation.pointVec,
                       &triangulation.output, path);
}
