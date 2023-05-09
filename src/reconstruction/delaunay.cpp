#include <delaunay.h>
#include <gDel3D/GpuDelaunay.h>
#include <kernel.h>
#include <util.h>
#include <vector>
#include <interval.h>
#include <fstream>

#define PUSH_TRIANGLE(a, b, c, triMap)do{\
    triMap[i3(a, b, c)] = vec3ui(a, b, c);\
}while(0)

#define MARK_TRIANGLE(a, b, c, vertexMap, triMap)do{\
    vertexMap[a] += 1;\
    vertexMap[b] += 1;\
    vertexMap[c] += 1;\
    triMap[i3(a, b, c)] = 1;\
}while(0)

#define MARK_CONNECTIONS(a, b, c, conMap)do{\
    if(conMap.find(a) == conMap.end()){\
        std::unordered_set<uint32_t> set;\
        set.insert(b);\
        set.insert(c);\
        conMap[a] = set;\
    }else{\
        conMap[a].insert(b);\
        conMap[a].insert(c);\
    }\
}while(0)

#define PUSH_INDEX_INTO_LIST(a, b, c, indexList)do{\
    indexList.push_back(vec3i(a, b, c));\
}while(0)


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
    Float d = Dot(n, ng);
    return (d >= 0);
}

__host__ vec3i
DelaunayOrientation(vec3f a, vec3f b, vec3f c, vec3f op, int A, int B, int C,
                    vec3f na, vec3f nb, vec3f nc, bool &remove, Bounds3f &bounds)
{
#if 1
    remove = false;
    vec3f ng = Normalize(op - a);
    if(!TriangleMatchesDirection(a, b, c, ng))
        return vec3i(A, B, C);
    if(TriangleMatchesDirection(a, c, b, ng)){
        vec3f navg = Normalize(na + nb + nc);
        Float dot = Dot(GeometricalNormal(a, b, c), navg);
        if(dot == 0 || dot == -0){
            printf("Dot = %g\n", dot);
            //remove = true;
        }

        if(dot < 0)
            return vec3i(A, C, B);
        else
            return vec3i(A, B, C);
        static int tri_inv_counter = 0;
        tri_inv_counter++;
        printf("Inverted triangle does not match normal ( %g %g %g ) {%d %d %d}\n",
                ng.x, ng.y, ng.z, A, B, C);
        vec3f n0 = GeometricalNormal(a, b, c);
        vec3f n1 = GeometricalNormal(a, c, b);
        Float d0 = Dot(n0, ng);
        Float d1 = Dot(n1, ng);
        printf("Forward  ( %g %g %g ) --> %g\n", n0.x, n0.y, n0.z, d0);
        printf("Backward ( %g %g %g ) --> %g\n", n1.x, n1.y, n1.z, d1);
        printf("inv count = %d\n", tri_inv_counter);
    }
    return vec3i(A, C, B);
#endif
}

__host__ bool
DelaunayFilterTetrahedron(vec3f a, vec3f b, vec3f c, vec3f d, Float threshold){
    vec3f f0 = a - d;
    vec3f f1 = b - d;
    vec3f f2 = c - d;
    Float vol = (1.f / 6.f) * Absf(Dot(f0, Cross(f1, f2)));
    return vol < threshold;
}

__host__ bool
DelaunayFilterTriangle(vec3f a, vec3f b, vec3f c, Float threshold){
    Float area = 0.5f * Cross(b - a, c - a).Length();
    return area < threshold;
}

__host__ void
DelaunaySurface(DelaunayTriangulation &triangulation, SphParticleSet3 *sphSet,
                Float spacing, Float mu, Grid3 *domain)
{
    ParticleSet3 *pSet = sphSet->GetParticleSet();
    DelaunayTriangleIndexedMap keyMap;
    DelaunayTriangleMap triMap;
    std::vector<int> &boundary = triangulation.boundary;
    GpuDel triangulator;
    Bounds3f bounds = domain->GetBounds();

    printf(" * Delaunay: [μ = %g] [h = %g]\n", mu, spacing);
    int pointNum = 0;
    boundary.reserve(pSet->GetParticleCount());

    std::cout << " - Removing dangerous particles... " << std::flush;
    for(int i = 0; i < pSet->GetParticleCount(); i++){
        Float pR = 0;
        boundary.push_back(0);
        if(CheckPart(pSet, domain, i, pR)){
            vec3f vi = pSet->GetParticlePosition(i);
            triangulation.ids.push_back(i);
            triangulation.partRMap[i] = mu * spacing;
            //triangulation.partRMap[i] = mu * pR;
            triangulation.pointVec.push_back({vi.x, vi.y, vi.z});
            pointNum += 1;
        }
    }

    std::cout << (pSet->GetParticleCount() - pointNum) << " removed" << std::endl;
    triangulator.compute(triangulation.pointVec, &triangulation.output);
    triangulation.pLen = pointNum;

    printf(" - Processing tetras...\n");
    std::vector<uint32_t> *ids = &triangulation.ids;
    DelaunayFloatTriangleMap &partRMap = triangulation.partRMap;
    DelaunayVertexMap &vertexMap = triangulation.vertexMap;

    std::vector<vec3i> indexList;
    long int totalTris = 0, totalInserted = 0;

    //Float maxRatio = 0.f;
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

#if 0
        Float edge_long = Max(Max(Max(aB, aC), Max(aD, bC)), Max(bD, cD));
        Float edge_smal = Min(Min(Min(aB, aC), Min(aD, bC)), Min(bD, cD));
        Float ratio = (edge_long / edge_smal);
        //is_tetra_valid &= !DelaunayFilterTetrahedron(pA, pB, pC, pD, 0.04);
        if(is_tetra_valid){
            maxRatio = Max(maxRatio, ratio);
            is_tetra_valid &= ratio < 5;
        }
#endif

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

    //printf("Ratio = %g\n", maxRatio);
#if 0
    DD_edgeMap edgeMap;

    auto add_edge = [](DD_edgeMap &map, i2 edge){
        if(map.find(edge) == map.end())
            map[edge] = 1;
        else
            map[edge] += 1;
    };
#endif
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

        vec3f _nA = pSet->GetParticleNormal(idA);
        vec3f _nB = pSet->GetParticleNormal(idB);
        vec3f _nC = pSet->GetParticleNormal(idC);

        boundary[idA] = 1;
        boundary[idB] = 1;
        boundary[idC] = 1;
        boundary[idD] = 1;

        bool remove = false;
        vec3i index = DelaunayOrientation(pA, pB, pC, pD, A, B, C,
                                          _nA, _nB, _nC, remove, bounds);
        if(remove)
            continue;
        MARK_TRIANGLE(A, B, C, vertexMap, triMap);
        PUSH_INDEX_INTO_LIST(index[0], index[1], index[2], triangulation.shrinked);
        //MARK_CONNECTIONS(A, B, C, conMap);
        //MARK_CONNECTIONS(B, A, C, conMap);
        //MARK_CONNECTIONS(C, B, A, conMap);

        //i2 eAB({A, B});
        //i2 eAC({A, C});
        //i2 eBC({B, C});

        //add_edge(edgeMap, eAB);
        //add_edge(edgeMap, eAC);
        //add_edge(edgeMap, eBC);

        totalInserted += 1;
    }
#if 0
    for(auto it = edgeMap.begin(); it != edgeMap.end(); it++){
        if(it->second > 2)
            printf("Edge appears %d times\n", it->second);
    }
#endif
    printf(" - Total triangles: %ld, Inserted: %ld\n", totalTris, totalInserted);
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
