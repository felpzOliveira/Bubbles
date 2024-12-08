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

/*
* NOTE: Extremely slow, unoptimized code
* for computing the 3D SIG, delaunay-based
* surface reconstruction of a fluid. I'm missing
* the vertice smoothing within triangle picking
* but it shouldnt be too hard to add.
*/

struct DelaunayTriangleInfo{
    vec3ui tri;
    int opp;

    bb_cpu_gpu
    DelaunayTriangleInfo(int){}

    bb_cpu_gpu
    DelaunayTriangleInfo(vec3ui _tri, int op){
        tri = _tri;
        opp = op;
    }
};

typedef WorkQueue<DelaunayTriangleInfo> DelaunayWorkQueue;

struct DelaunaySet{
    int nPos = 0;
    int offset = 0;
    int totalSize = 0;
    Point3 *positions;
    int internalRefId;

    bb_cpu_gpu void SetParticlePosition(int id, vec3f v){
        if(nPos > 0){
            int trueId = id - offset;
            positions[trueId] = {v.x, v.y, v.z};
        }
    }

    bb_cpu_gpu int GetTrueId(int id){
        return id - offset;
    }

    bb_cpu_gpu vec3f GetParticlePosition(int id){
        if(nPos > 0){
            int trueId = id - offset;
            Point3 p = positions[trueId];
            return vec3f(p._p[0], p._p[1], p._p[2]);
        }
        return vec3f();
    }

    bb_cpu_gpu bool IsInternal(int id){
        return (id - offset) < internalRefId;
    }

    bb_cpu_gpu bool InSet(int id){
        return id >= offset;
    }

    void Cleanup(){
        cudaFree(positions);
    }
};

DelaunayOptions DelaunayDefaultOptions(){
    return DelaunayOptions{
        .extendBoundary = true,
        .withInteriorParticles = true,
        .use_1nn_radius = false,
        .use_alpha_shapes = false,
        .outType = GatherSurface,
        .mu = 1.1f,
        .alpha = 50.f,
        .spacing = 0.02f,
    };
}

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

bb_cpu_gpu
Float triangle_area(const vec3f &pA, const vec3f &pB, const vec3f &pC){
    Float ab = (pA-pB).Length();
    Float ac = (pA-pC).Length();
    Float bc = (pB-pC).Length();
    Float s = (ab + bc + ac) * 0.5;
    return std::sqrt(s * (s - ab) * (s - ac) * (s - bc));
}

bb_cpu_gpu bool
check_triangle(i3 tri, uint32_t *ids, ParticleSet3 *pSet, DelaunaySet *dSet,
               Float *radius)
{
    int idA = ids[tri.t[0]], idB = ids[tri.t[1]], idC = ids[tri.t[2]];
    vec3f pA = DelaunayPosition(pSet, dSet, idA);
    vec3f pB = DelaunayPosition(pSet, dSet, idB);
    vec3f pC = DelaunayPosition(pSet, dSet, idC);
    Float aB = Distance(pA, pB);
    Float aC = Distance(pA, pC);
    Float bC = Distance(pB, pC);

    Float rA = radius[0];
    Float rB = radius[1];
    Float rC = radius[2];

    Float area = triangle_area(pA, pB, pC);
    bool by_area = !IsZero(area) && !std::isnan(area);
    bool by_edges = aB < (rA + rB) && aC < (rA + rC) && bC < (rB + rC);
    return by_area && by_edges;
}

void
dump_particles(vec3f *ptr, int size, const char *path){
    std::ofstream ofs(path);
    ofs << size << std::endl;
    for(int i = 0; i < size; i++){
        vec3f p = ptr[i];
        ofs << p.x << " " << p.y << " " << p.z << std::endl;
    }
    ofs.close();
}

static void
DelaunayWorkQueueAll(GpuDel *triangulator, DelaunayTriangulation &triangulation,
                     DelaunayWorkQueue *delQ, TimerList &timer)
{
    Tet *kernTet = toKernelPtr(triangulator->_tetVec);
    TetOpp *kernTetOpp = toKernelPtr(triangulator->_oppVec);
    char *kernChar = toKernelPtr(triangulator->_tetInfoVec);

    uint32_t size = triangulator->_tetVec.size();
    uint32_t pLen = triangulation.pLen;
    delQ->SetSlots(size * 4, false);

    char *tetFlags = cudaAllocateUnregisterVx(char, size);
    timer.Start("Tetrahedra Filtering");

    AutoParallelFor("Delaunay_Filter", size, AutoLambda(int i){
        Tet tet = kernTet[i];
        const TetOpp botOpp = loadOpp(kernTetOpp, i);
        tetFlags[i] = -1;
        if(DelaunayIsTetValid(tet, botOpp, kernChar[i], pLen)){
            tetFlags[i] = 0xf;
        }
    });

    timer.StopAndNext("Gather");

    AutoParallelFor("Delaunay_Gather", size, AutoLambda(int tetIndex){
        Tet tet = kernTet[tetIndex];
        const TetOpp botOpp = loadOpp(kernTetOpp, tetIndex);
        int triStates = tetFlags[tetIndex];
        if(triStates <= 0)
            return;

        int A = tet._v[0], B = tet._v[1],
            C = tet._v[2], D = tet._v[3];
        i3 faces[] = { i3(A, B, C), i3(A, B, D), i3(A, D, C), i3(B, D, C) };
        int info[] = { 0, 0, 0, 0 };
        int opp[] = { D, C, B, A };

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
                if(where >= 0){
                    if(tetIndex > topTi){
                        info[where] = 1;
                    }
                }
            }
        }

        for(int s = 0; s < 4; s++){
            if(info[s] == 0){
                vec3ui tri = matching_orientation(faces[s], tet);
                delQ->Push({tri, opp[s]});
            }
        }
    });

    timer.Stop();
    cudaFree(tetFlags);
}

static void
DelaunayWorkQueueConvexHull(GpuDel *triangulator, DelaunayTriangulation &triangulation,
                            DelaunayWorkQueue *delQ, TimerList &timer)
{
    Tet *kernTet = toKernelPtr(triangulator->_tetVec);
    TetOpp *kernTetOpp = toKernelPtr(triangulator->_oppVec);
    char *kernChar = toKernelPtr(triangulator->_tetInfoVec);

    uint32_t size = triangulator->_tetVec.size();
    uint32_t pLen = triangulation.pLen;
    delQ->SetSlots(size * 4, false);

    char *tetFlags = cudaAllocateUnregisterVx(char, size);
    timer.Start("Tetrahedra Filtering");

    AutoParallelFor("Delaunay_Filter", size, AutoLambda(int i){
        Tet tet = kernTet[i];
        const TetOpp botOpp = loadOpp(kernTetOpp, i);
        tetFlags[i] = -1;
        if(DelaunayIsTetValid(tet, botOpp, kernChar[i], pLen)){
            tetFlags[i] = 0xf;
        }
    });

    timer.StopAndNext("Find Unique Triangles");

    AutoParallelFor("Delaunay_Unique", size, AutoLambda(int tetIndex){
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
                if(where >= 0){
                    info[where] = 1;
                }
            }
        }

        for(int s = 0; s < 4; s++){
            if(info[s] == 0){
                vec3ui tri = matching_orientation(faces[s], tet);
                delQ->Push({tri, opp[s]});
            }
        }
    });

    timer.Stop();
    cudaFree(tetFlags);
}

template<typename TriangleAcceptorFn>
static void DelaunayForEachFinalTriangle(GpuDel *triangulator, DelaunayTriangulation &triangulation,
                                         ParticleSet3 *pSet, DelaunaySet *dSet,
                                         uint32_t *ids, char *tetFlags, Float *fRadius,
                                         bool extended, bool checkRadius,
                                         TriangleAcceptorFn &acceptor)
{
    Tet *kernTet = toKernelPtr(triangulator->_tetVec);
    TetOpp *kernTetOpp = toKernelPtr(triangulator->_oppVec);
    char *kernChar = toKernelPtr(triangulator->_tetInfoVec);
    uint32_t size = triangulator->_tetVec.size();

    /*
    * TODO: The following two kernels can be merged into a single one
    * it is just splitted so we can compute the actual contribution of each one
    */
    /*
    * Process only the completely valid tetras (flag == 0xf), these provide valid
    * triangles no matter what and can be pushed into the unique tri list directly.
    */
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
            #if 0
                bool accept = true;
                if(extended){
                    accept = !(dSet->IsInternal(ids[tri.x]) ||
                               dSet->IsInternal(ids[tri.y]) ||
                               dSet->IsInternal(ids[tri.z]));
                }
            #endif
                acceptor(tri, opp[s], tetIndex, s);
            }
        }
    });

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
            i3 face = faces[s];

            if(checkRadius){
                Float rs[3] = {
                    fRadius[dSet->GetTrueId(ids[face.t[0]])],
                    fRadius[dSet->GetTrueId(ids[face.t[1]])],
                    fRadius[dSet->GetTrueId(ids[face.t[2]])]
                };

                bool valid_face = check_triangle(face, ids, pSet, dSet, rs);
                if(!valid_face)
                    continue;
            }

            if(info[s] == 0 || info[s] == 2){
                vec3ui tri = matching_orientation(face, tet);
            #if 0
                bool accept = true;
                if(extended){
                    accept = !(dSet->IsInternal(ids[tri.x]) ||
                               dSet->IsInternal(ids[tri.y]) ||
                               dSet->IsInternal(ids[tri.z]));
                }
            #endif
                acceptor(tri, opp[s], tetIndex, s);
            }
        }
    });

}

bb_cpu_gpu __forceinline__
float __length(float x0, float y0, float z0){
#if defined(__CUDA_ARCH__)
    return __fsqrt_rn( __fmaf_rn(x0, x0,
                       __fmaf_rn(y0, y0,
                       __fmaf_rn(z0, z0, 0.f))) );
#else
    return std::sqrt(x0 * x0 + y0 * y0 + z0 * z0);
#endif
}

bb_cpu_gpu __forceinline__
float __distance(float x0, float y0, float z0,
                 float x1, float y1, float z1)
{
    float dx, dy, dz;
#if defined(__CUDA_ARCH__)
    dx = __fsub_rn(x0, x1);
    dy = __fsub_rn(y0, y1);
    dz = __fsub_rn(z0, z1);
    return __fsqrt_rn( __fmaf_rn(dx, dx,
                       __fmaf_rn(dy, dy,
                       __fmaf_rn(dz, dz, 0.f))) );
#else
    dx = x0 - x1;
    dy = y0 - y1;
    dz = z0 - z1;
    return std::sqrt(dx * dx + dy * dy + dz * dz);
#endif
}

bb_cpu_gpu __forceinline__
void __cross(float x1, float y1, float z1,
             float x2, float y2, float z2,
             float *x, float *y, float *z)
{
#if defined(__CUDA_ARCH__)
    *x = __fmaf_rn(y1, z2, __fmaf_rn(-z1, y2, 0.f));
    *y = __fmaf_rn(z1, x2, __fmaf_rn(-x1, z2, 0.f));
    *z = __fmaf_rn(x1, y2, __fmaf_rn(-y1, x2, 0.f));
#else
    *x = (y1 * z2) - (z1 * y2);
    *y = (z1 * x2) - (x1 * z2);
    *z = (x1 * y2) - (y1 * x2);
#endif
}

#define _distance(v0, v1) __distance(v0.x, v0.y, v0.z, v1.x, v1.y, v1.z)
#define _length(v0) __length(v0.x, v0.y, v0.z)
#define _cross(v1, v2, p) __cross(v1.x, v1.y, v1.z, v2.x, v2.y, v2.z, &p.x, &p.y, &p.z)

// NOTE: Leverage 1/α and build on δ/α as testing factor
//       for R > δ/α instead of R'> 1/α. This way we can
//       prevent 1/δ yielding nan, and can have more stable tests.
bb_cpu_gpu
Float accept_alpha_test(vec3f v0, vec3f v1, vec3f v2, vec3f v3, Float invAlpha){
    vec3f u2u3, u3u1, u1u2;
    Float l01 = _distance(v0, v1);
    Float l02 = _distance(v0, v2);
    Float l03 = _distance(v0, v3);

    vec3f u1 = v1 - v0;
    vec3f u2 = v2 - v0;
    vec3f u3 = v3 - v0;

    _cross(u2, u3, u2u3);
    _cross(u3, u1, u3u1);
    _cross(u1, u2, u1u2);

    Float delta = 2.f * Dot(u1, u2u3);

    vec3f O = ( (l01 * l01 * u2u3) +
                (l02 * l02 * u3u1) +
                (l03 * l03 * u1u2) );

    return !(_length(O) > (delta * invAlpha));
}


static void
DelaunayWorkQueueAlphaShape(GpuDel *triangulator, DelaunayTriangulation &triangulation,
                            ParticleSet3 *pSet, DelaunaySet *dSet, uint32_t *ids,
                            Float alpha, DelaunayWorkQueue *delQ, TimerList &timer)
{
    Tet *kernTet = toKernelPtr(triangulator->_tetVec);
    TetOpp *kernTetOpp = toKernelPtr(triangulator->_oppVec);
    char *kernChar = toKernelPtr(triangulator->_tetInfoVec);

    uint32_t size = triangulator->_tetVec.size();
    uint32_t pLen = triangulation.pLen;

    char *tetFlags = cudaAllocateUnregisterVx(char, size);
    char *tetFaces = cudaAllocateUnregisterVx(char, size);
    int *extraCounter = cudaAllocateUnregisterVx(int, 1);
    *extraCounter = 0;

    Float invAlpha = 1.f / alpha;

    timer.Start("Tetrahedra Filtering - Alpha Shapes");
    AutoParallelFor("Delaunay_Filter", size, AutoLambda(int i){
        Tet tet = kernTet[i];
        const TetOpp botOpp = loadOpp(kernTetOpp, i);
        int A = tet._v[0], B = tet._v[1],
            C = tet._v[2], D = tet._v[3];

        tetFlags[i] = -1;
        tetFaces[i] = 0;
        if(DelaunayIsTetValid(tet, botOpp, kernChar[i], pLen)){
            int idA = ids[A], idB = ids[B],
                idC = ids[C], idD = ids[D];

            vec3f pA = DelaunayPosition(pSet, dSet, idA);
            vec3f pB = DelaunayPosition(pSet, dSet, idB);
            vec3f pC = DelaunayPosition(pSet, dSet, idC);
            vec3f pD = DelaunayPosition(pSet, dSet, idD);

            if(accept_alpha_test(pA, pB, pC, pD, invAlpha)){
                tetFlags[i] = 0xf;
            }
        }
    });

    auto fn = AutoLambda(vec3ui tri, int opp, int tetIndex, int faceIndex){
        tetFaces[tetIndex] |= (1 << faceIndex);
        atomic_increase(extraCounter);
    };

    DelaunayForEachFinalTriangle(triangulator, triangulation, pSet, dSet, ids, tetFlags,
                                 nullptr, false, false, fn);


    if(*extraCounter == 0){
        printf("[BUG] Delaunay gave no valid triangle\n");
    }else{
        delQ->SetSlots(*extraCounter, false);

        AutoParallelFor("Delaunay_UniqueTris_Push", size, AutoLambda(int tetIndex){
            Tet tet = kernTet[tetIndex];
            int A = tet._v[0], B = tet._v[1],
                C = tet._v[2], D = tet._v[3];
            i3 faces[] = { i3(A, B, C), i3(A, B, D), i3(A, D, C), i3(B, D, C) };
            int opp[] = { D, C, B, A };
            char pickFlag = tetFaces[tetIndex];

            for(int s = 0; s < 4; s++){
                if(pickFlag & (1 << s)){
                    vec3ui tri = matching_orientation(faces[s], tet);
                    delQ->Push({tri, opp[s]});
                }
            }
        });
    }

    cudaFree(extraCounter);
    cudaFree(tetFlags);
    cudaFree(tetFaces);
}

static void
DelaunayWorkQueueFindSurface(GpuDel *triangulator, DelaunayTriangulation &triangulation,
                             ParticleSet3 *pSet, DelaunaySet *dSet, uint32_t *ids,
                             Float *fRadius, DelaunayWorkQueue *delQ,
                             bool extended, TimerList &timer)
{
    Tet *kernTet = toKernelPtr(triangulator->_tetVec);
    TetOpp *kernTetOpp = toKernelPtr(triangulator->_oppVec);
    char *kernChar = toKernelPtr(triangulator->_tetInfoVec);

    uint32_t size = triangulator->_tetVec.size();
    uint32_t pLen = triangulation.pLen;

    char *tetFlags = cudaAllocateUnregisterVx(char, size);
    char *tetFaces = cudaAllocateUnregisterVx(char, size);
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
        tetFaces[i] = 0;

        if(DelaunayIsTetValid(tet, botOpp, kernChar[i], pLen)){
            int idA = ids[A], idB = ids[B],
                idC = ids[C], idD = ids[D];

            vec3f pA = DelaunayPosition(pSet, dSet, idA);
            vec3f pB = DelaunayPosition(pSet, dSet, idB);
            vec3f pC = DelaunayPosition(pSet, dSet, idC);
            vec3f pD = DelaunayPosition(pSet, dSet, idD);

            Float rA = fRadius[dSet->GetTrueId(idA)];
            Float rB = fRadius[dSet->GetTrueId(idB)];
            Float rC = fRadius[dSet->GetTrueId(idC)];
            Float rD = fRadius[dSet->GetTrueId(idD)];

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
                //tetFlags[i] = 0;
            }else
                tetFlags[i] = 0xf;
        }
    });

    auto fn = AutoLambda(vec3ui tri, int opp, int tetIndex, int faceIndex){
        tetFaces[tetIndex] |= (1 << faceIndex);
        atomic_increase(extraCounter);
    };

    DelaunayForEachFinalTriangle(triangulator, triangulation, pSet, dSet, ids, tetFlags,
                                 fRadius, extended, true, fn);

    //DelaunayMarkNonManifold(triangulator, &tetFaces, &tetFlags);

    if(*extraCounter == 0){
        printf("[BUG] Delaunay gave no valid triangle\n");
    }else{
        delQ->SetSlots(*extraCounter, false);

        AutoParallelFor("Delaunay_UniqueTris_Push", size, AutoLambda(int tetIndex){
            Tet tet = kernTet[tetIndex];
            int A = tet._v[0], B = tet._v[1],
                C = tet._v[2], D = tet._v[3];
            i3 faces[] = { i3(A, B, C), i3(A, B, D), i3(A, D, C), i3(B, D, C) };
            int opp[] = { D, C, B, A };
            char pickFlag = tetFaces[tetIndex];

            for(int s = 0; s < 4; s++){
                if(pickFlag & (1 << s)){
                    vec3ui tri = matching_orientation(faces[s], tet);
                    delQ->Push({tri, opp[s]});
                }
            }
        });
    }


    cudaFree(extraCounter);
    cudaFree(tetFlags);
    cudaFree(tetFaces);
}

/*
* NOTE: This is completely unnecessary and could be replaced by a single variable
* r = μλ. The only reason it is not is because someone might ask about 1-NN.
* Having this not only makes everything slow because we don't actually have a point
* hash scheme for the delaunay point-set but we will have to constantly query this
* memory during reconstruction making CUDA want to kill me. I'm sorry I promise I'll
* do better next time.
*/
Float *DelaunayGetRadius(ParticleSet3 *pSet, Grid3 *domain,  DelaunaySet *dSet,
                         uint32_t *ids, Float mu, Float spacing, bool with_ext,
                         bool fixed=true)
{
    int size = dSet->nPos;
    Float *minRadius = cudaAllocateUnregisterVx(Float, size);

    if(fixed){
        Float r = mu * spacing;
        for(int i = 0; i < size; i++)
            minRadius[i] = r;
    }else{
        if(with_ext){ // NOTE: braceyourself this is going to take several minutes
            AutoParallelFor("Bruteforce-1NN", size, AutoLambda(int i){
                Point3 pi = dSet->positions[i];
                Float radius = Infinity;
                for(int j = 0; j < dSet->nPos; j++){
                    if(i == j)
                        continue;

                    Point3 pj = dSet->positions[j];
                    Float dx = pi._p[0] - pj._p[0];
                    Float dy = pi._p[1] - pj._p[1];
                    Float dz = pi._p[2] - pj._p[2];

                    Float dij = std::sqrt(dx * dx + dy * dy + dz * dz);
                    radius = dij < radius ? dij : radius;
                }

                minRadius[i] = mu * radius;
            });
        }else{
            // NOTE: If this run is without extension we can levarage the particle
            // domain for hashing so we dont have to wait for the sun to cooldown.
            AutoParallelFor("Compute-1NN", size, AutoLambda(int i){
                int *neighbors = nullptr;
                int p_id = ids[i];
                vec3f pi = pSet->GetParticlePosition(p_id);
                unsigned int cellId = domain->GetLinearHashedPosition(pi);
                int count = domain->GetNeighborsOf(cellId, &neighbors);
                int counter = 0;

                Float radius = Infinity;
                for(int i = 0; i < count; i++){
                    Cell3 *cell = domain->GetCell(neighbors[i]);
                    ParticleChain *pChain = cell->GetChain();
                    int size = cell->GetChainLength();
                    for(int j = 0; j < size; j++){
                        if(pChain->pId != p_id){
                            vec3f pj = pSet->GetParticlePosition(pChain->pId);
                            Float dij = Distance(pi, pj);
                            radius = radius < dij ? radius : dij;
                            counter += 1;
                        }

                        pChain = pChain->next;
                    }
                }

                if(counter == 0){
                    printf("Distribution might not be uniform or very large "
                           ", spacing need to be adjusted\n");
                    // fallback to μλ
                    radius = spacing;
                }

                minRadius[i] = mu * radius;
            });
        }
    }

    return minRadius;
}

void
DelaunaySurface(DelaunayTriangulation &triangulation, SphParticleSet3 *sphSet,
                Grid3 *domain, DelaunayOptions opts, TimerList &timer)
{
    GpuDel triangulator;
    vec3i *u_tri = nullptr;
    vec3f *ext_pos = nullptr;
    vec3f *int_pos = nullptr;
    uint32_t *u_ids = nullptr;
    Float *radius = nullptr;
    DelaunayWorkQueue *delQ = nullptr;
    int pointNum = 0;

    Float mu = opts.mu;
    Float spacing = opts.spacing;

    DelaunaySet *dSet = nullptr;
    ParticleSet3 *pSet = sphSet->GetParticleSet();

    printf(" * Delaunay: [μ = %g] [h = %g] ( %d )\n", mu, spacing,
           pSet->GetParticleCount());

    std::cout << " - Adjusting domain..." << std::flush;

    int totalCount = 0;
    int supports = 0;
    int extensionSize = 0;
    int intensionSize = 0;
    for(int i = 0; i < pSet->GetParticleCount(); i++){
        if(opts.extendBoundary){
            if(pSet->GetParticleV0(i) > 0){
                supports += 3;
                totalCount += 4;
                extensionSize += 4;
            }else if(opts.withInteriorParticles){
                totalCount += 1;
                intensionSize += 1;
            }
        }else{
            totalCount += 1;
        }
    }

    dSet = cudaAllocateUnregisterVx(DelaunaySet, 1);
    u_ids = cudaAllocateUnregisterVx(uint32_t, totalCount);

    if(DELAUNAY_OUTPUT_PARTICLES){
        ext_pos = cudaAllocateUnregisterVx(vec3f, extensionSize);
        int_pos = cudaAllocateUnregisterVx(vec3f, intensionSize);
    }

    dSet->positions = cudaAllocateUnregisterVx(Point3, totalCount);
    dSet->totalSize = totalCount;
    dSet->offset = pSet->GetParticleCount();

    int *iIndex = cudaAllocateUnregisterVx(int, 1);
    int *refIndex = cudaAllocateUnregisterVx(int, 1);
    *iIndex = 0;
    *refIndex = 0;

    timer.Start("Particle Filter and Spawn");

    if(opts.withInteriorParticles){
        AutoParallelFor("Delaunay_Particle-Filter", pSet->GetParticleCount(),
        AutoLambda(int pId)
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

                if(DELAUNAY_OUTPUT_PARTICLES){
                    int_pos[key] = pi;
                }
            }
        });
    }

    *refIndex = *iIndex;
    dSet->internalRefId = *refIndex;
    *iIndex = 0;

    AutoParallelFor("Delaunay_Tet-Spawn", pSet->GetParticleCount(), AutoLambda(int pId){
        int bi = pSet->GetParticleV0(pId);

        if(bi <= 0)
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

        if(acceptpart == 1 && opts.extendBoundary){
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

            if(DELAUNAY_OUTPUT_PARTICLES){
                ext_pos[4 * key + 0] = p0;
                ext_pos[4 * key + 1] = p1;
                ext_pos[4 * key + 2] = p2;
                ext_pos[4 * key + 3] = p3;
            }

            dSet->positions[*refIndex + 4 * key + 0] = {p0.x, p0.y, p0.z};
            dSet->positions[*refIndex + 4 * key + 1] = {p1.x, p1.y, p1.z};
            dSet->positions[*refIndex + 4 * key + 2] = {p2.x, p2.y, p2.z};
            dSet->positions[*refIndex + 4 * key + 3] = {p3.x, p3.y, p3.z};
            u_ids[*refIndex + 4 * key + 0] = id + 0;
            u_ids[*refIndex + 4 * key + 1] = id + 1;
            u_ids[*refIndex + 4 * key + 2] = id + 2;
            u_ids[*refIndex + 4 * key + 3] = id + 3;
        }else if(acceptpart == 1){
            int key = atomic_increase_get(iIndex);
            int id = pSet->GetParticleCount() + *refIndex + key;
            dSet->positions[*refIndex + key] = {pi.x, pi.y, pi.z};
            u_ids[*refIndex + key] = id;
        }
    });

    timer.Stop();

    if(DELAUNAY_OUTPUT_PARTICLES){
        dump_particles(ext_pos, 4 * (*iIndex), "boundary.txt");
        dump_particles(int_pos, *refIndex, "interior.txt");
    }

    if(opts.extendBoundary){
        dSet->nPos = *refIndex + 4 * (*iIndex);
    }else{
        dSet->nPos = *refIndex + (*iIndex);
    }

    radius = DelaunayGetRadius(pSet, domain, dSet, u_ids, mu,
                               spacing, !opts.use_1nn_radius);

    triangulation.pointVec.resize(dSet->nPos);

    for(int i = 0, j = 0; i < dSet->nPos; i++){
        triangulation.pointVec[j++] = dSet->positions[i];
    }

    pointNum = triangulation.pointVec.size();
    Float frac = 100.0 * supports / pSet->GetParticleCount();
    std::cout << "done\n - Support points " << supports <<
                 " ( " << frac << "% )" << std::endl;

    std::cout << " - Total set " << dSet->nPos << std::endl;

    std::cout << " - Running delaunay triangulation..." << std::flush;

    timer.Start("Delaunay Triangulation");
    triangulator.compute(triangulation.pointVec, &triangulation.output);
    timer.Stop();
    triangulation.pLen = pointNum;
    std::cout << "done" << std::endl;

    delQ = cudaAllocateUnregisterVx(DelaunayWorkQueue, 1);

    if(opts.outType == GatherConvexHull){
        DelaunayWorkQueueConvexHull(&triangulator, triangulation, delQ, timer);
    }else if(opts.outType == GatherEverything){
        DelaunayWorkQueueAll(&triangulator, triangulation, delQ, timer);
    }else if(opts.use_alpha_shapes){ // α-shapes
        std::cout << " - Surface Extraction by Alpha Shapes [ " << opts.alpha << " ]" << std::endl;
        DelaunayWorkQueueAlphaShape(&triangulator, triangulation, pSet, dSet,
                                    u_ids, opts.alpha, delQ, timer);
    }else{ // SIG
        std::cout << " - Surface Extraction by SIG" << std::endl;
        DelaunayWorkQueueFindSurface(&triangulator, triangulation, pSet, dSet,
                                     u_ids, radius, delQ, opts.extendBoundary, timer);
    }

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
    cudaFree(radius);
    cudaFree(refIndex);
    if(DELAUNAY_OUTPUT_PARTICLES){
        cudaFree(ext_pos);
        cudaFree(int_pos);
    }
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
        vec3f laplacianSum(0.f, 0.f, 0.f);
        int count = 0;
        for(size_t fi = 0; fi < faceCount; fi++){
            vec3i face = faceBuf[fi];
            if(shiftface(face, index)){
                vec3f vj = inOutBuf[face[0]];
                vec3f vl = inOutBuf[face[1]];
                laplacianSum += vj + vl;
                count += 2;
            }
        }

        if(count > 0){
            Float inv = 1.f / (Float)count;
            tmpBuf[index] = inOutBuf[index] +
                        lambda * (laplacianSum * inv - inOutBuf[index]);
        }else{
            tmpBuf[index] = inOutBuf[index];
        }
    });

    AutoParallelFor("Taubin Smooth - Backwards", verticeCount, AutoLambda(int index){
        vec3f laplacianSum(0.f, 0.f, 0.f);
        int count = 0;
        for(size_t fi = 0; fi < faceCount; fi++){
            vec3i face = faceBuf[fi];
            if(shiftface(face, index)){
                vec3f vj = tmpBuf[face[0]];
                vec3f vl = tmpBuf[face[1]];
                laplacianSum += vj + vl;
                count += 2;
            }
        }

        if(count > 0){
            Float inv = 1.f / (Float)count;
            inOutBuf[index] = tmpBuf[index] +
                            mu * (laplacianSum * inv - tmpBuf[index]);
        }else{
            inOutBuf[index] = tmpBuf[index];
        }
    });
}

// Trivial laplacian
void LaplacianSmoothOnce(vec3f *readBuf, vec3f *writeBuf, size_t verticeCount,
                         vec3i *faces, size_t faceCount, Float lambda)
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
                vec3f vj = sourceBuf[face[0]];
                vec3f vl = sourceBuf[face[1]];
                laplacianSum += vj + vl;
                count += 2;
            }
        }

        if(count > 0){
            Float inv = 1.f / (Float)count;
            destBuf[index] = sourceBuf[index] +
                                lambda * (laplacianSum * inv - sourceBuf[index]);
        }else{
            destBuf[index] = sourceBuf[index];
        }
    });
}

// TODO: Is it worth to refactor filtering to do the smooth in the kernel like
//       before? I'm gonna let this here as smoothing is optional.
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

            LaplacianSmoothOnce(readBuf, writeBuf, vsize, faceBuf, fsize, opts.lambda);
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

