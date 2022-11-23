#include <delaunay.h>
#include <gDel3D/GpuDelaunay.h>
#include <kernel.h>
#include <util.h>
#include <vector>

#define TRI_MAP_ADD(a, b, c, u_map)do{\
    i3 i_val(a, b, c);\
    if(u_map.find(i_val) == u_map.end()){\
        u_map[i_val] = 1;\
    }\
}while(0)

#define TRI_SET_ADD_CHECK(a, b, c, u_map, u_set, op_map, op, pset, ids)do{\
    i3 i_val(a, b, c);\
    if(u_set.find(i_val) == u_set.end()){\
        if(u_map.find(i_val) == u_map.end()){\
            u_map[i_val] = 1;\
            op_map[i_val] = op;\
        }else{\
            uint32_t other_op = op_map[i_val];\
            if(!SameSide(a, b, c, op, other_op, pset, ids)) u_map[i_val] += 1;\
        }\
    }\
}while(0)

__host__ bool SameSide(uint32_t tA, uint32_t tB, uint32_t tC, uint32_t tOp,
                       uint32_t tOp2, ParticleSet3 *pSet, std::vector<uint32_t> *ids)
{
    vec3f pA = pSet->GetParticlePosition(ids->at(tA));
    vec3f pB = pSet->GetParticlePosition(ids->at(tB));
    vec3f pC = pSet->GetParticlePosition(ids->at(tC));
    vec3f pOp = pSet->GetParticlePosition(ids->at(tOp));
    vec3f pOp2 = pSet->GetParticlePosition(ids->at(tOp2));

    vec3f M = (pA + pB + pC) * 0.33333;
    vec3f MOP = pOp - M;
    vec3f MOP2 = pOp2 - M;

    return Dot(MOP, MOP2) > -0.05;
}

__host__ int CheckPart(ParticleSet3 *pSet, Grid3 *grid, int pId){
    vec3f pi = pSet->GetParticlePosition(pId);
    Bucket *bucket = pSet->GetParticleBucket(pId);
    int minCandidate = pSet->GetParticleCount() + 2;
    for(int i = 0; i < bucket->Count(); i++){
        int j = bucket->Get(i);
        if(j != pId){
            vec3f pj = pSet->GetParticlePosition(j);
            Float d = Distance(pi, pj);
            if(d < 1e-5){
                if(pId > j){
                    minCandidate = Min(j+1, minCandidate);
                }
            }
        }
    }

    if(!(minCandidate > pSet->GetParticleCount())){
        minCandidate = -minCandidate;
    }
    return minCandidate;
}

__host__ void
DelaunayTriangulate(DelaunayTriangulation &triangulation, SphParticleSet3 *sphSet,
                    Grid3 *domain)
{
    GpuDel triangulator;
    ParticleSet3 *pSet = sphSet->GetParticleSet();
    int pointNum = 0;
    for(int i = 0; i < pSet->GetParticleCount(); i++){
        if(CheckPart(pSet, domain, i) >= 0){
            vec3f vi = pSet->GetParticlePosition(i);
            triangulation.ids.push_back(i);
            triangulation.pointVec.push_back({vi.x, vi.y, vi.z});
            pointNum++;
        }
    }

    triangulator.compute(triangulation.pointVec, &triangulation.output);
    triangulation.pLen = pointNum;
}

__host__ void
DelaunayShrink(DelaunayTriangulation &triangulation, SphParticleSet3 *sphSet,
               std::vector<int> &boundary)
{
    DelaunayTriangleMap triMap, triOpMap;
    DelaunayTriangleSet removedSet;
    ParticleSet3 *pSet = sphSet->GetParticleSet();
    std::vector<uint32_t> *ids = &triangulation.ids;
    GDel3D_ForEachRealTetra(&triangulation.output, triangulation.pLen,
    [&](Tet tet, TetOpp botOpp, int i) -> void{
        int A = tet._v[0], B = tet._v[1],
            C = tet._v[2], D = tet._v[3];
        TRI_MAP_ADD(A, C, B, triMap);
        TRI_MAP_ADD(A, B, D, triMap);
        TRI_MAP_ADD(A, D, C, triMap);
        TRI_MAP_ADD(B, D, C, triMap);
    });

    Float radius = 2.0 * sphSet->GetKernelRadius();
    for(auto it = triMap.begin(); it != triMap.end(); it++){
        i3 indices = it->first;
        uint32_t A = indices.t[0];
        uint32_t B = indices.t[1];
        uint32_t C = indices.t[2];
        vec3f p0 = pSet->GetParticlePosition(ids->at(A));
        vec3f p1 = pSet->GetParticlePosition(ids->at(B));
        vec3f p2 = pSet->GetParticlePosition(ids->at(C));

        Float d0 = Distance(p0, p1),
              d1 = Distance(p1, p2),
              d2 = Distance(p0, p2);

        if(!(d0 < radius &&
            d1 < radius &&
            d2 < radius))
        {
            removedSet.insert(indices);
        }
    }

    triMap.clear();

    GDel3D_ForEachRealTetra(&triangulation.output, triangulation.pLen,
    [&](Tet tet, TetOpp botOpp, int i) -> void{
        uint32_t A = tet._v[0], B = tet._v[1],
                 C = tet._v[2], D = tet._v[3];
        bool lA = boundary[ids->at(A)];
        bool lB = boundary[ids->at(B)];
        bool lC = boundary[ids->at(C)];
        bool lD = boundary[ids->at(D)];
        if(lA || lB || lC)
            TRI_SET_ADD_CHECK(A, C, B, triMap, removedSet, triOpMap, D, pSet, ids);
        if(lA || lB || lD)
            TRI_SET_ADD_CHECK(A, B, D, triMap, removedSet, triOpMap, C, pSet, ids);
        if(lA || lD || lC)
            TRI_SET_ADD_CHECK(A, D, C, triMap, removedSet, triOpMap, B, pSet, ids);
        if(lB || lC || lD)
            TRI_SET_ADD_CHECK(B, D, C, triMap, removedSet, triOpMap, A, pSet, ids);
    });

    for(auto it = triMap.begin(); it != triMap.end(); it++){
        if(it->second == 1){
            i3 index = it->first;
            int A = index.t[0];
            int B = index.t[1];
            int C = index.t[2];
            vec3f p0 = pSet->GetParticlePosition(ids->at(A));
            vec3f p1 = pSet->GetParticlePosition(ids->at(B));
            vec3f p2 = pSet->GetParticlePosition(ids->at(C));
            vec3f n0 = pSet->GetParticleNormal(ids->at(A));
            vec3f n1 = pSet->GetParticleNormal(ids->at(B));
            vec3f n2 = pSet->GetParticleNormal(ids->at(C));
            vec3f n = Normalize((n0 + n1 + n2));

            vec3f e0 = p1 - p0;
            vec3f e1 = p2 - p0;
            vec3f ng = Normalize(Cross(e0, e1));
            Float d = Dot(ng, n);
            if(d > 0){
                Swap(A, C);
            }

            index.t[0] = C;
            index.t[1] = B;
            index.t[2] = A;
            triangulation.shrinked.push_back(index);
        }
    }
}

__host__ void
DelaunayWritePly(DelaunayTriangulation &triangulation, const char *path){
    UtilGDel3DWritePly(&triangulation.shrinked, &triangulation.pointVec,
                       &triangulation.output, path);
}
