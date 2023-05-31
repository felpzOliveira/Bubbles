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

    // TODO: The small bucket thing is not being used, I'm letting here
    //       just in case we ever want to do more processing on them
    DelaunaySmallBucket tbucket;
    DelaunaySmallBucket *bucket = &tbucket;

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
    if(density < threshold)
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
        ParticleFlag(pSet, domain, i, radius, threshold);
    });
}

void DumpPoints(DelaunaySetBuilder &dBuilder, ParticleSet3 *pSet){
    FILE *fp = fopen("test_points.txt", "wb");
    if(fp){
        int nCount = pSet->GetParticleCount();
        int dCount = dBuilder.Size();
        int totalP = nCount + dCount;
        fprintf(fp, "%d\n", totalP);
        for(int i = 0; i < nCount; i++){
            vec3f pi = pSet->GetParticlePosition(i);
            int vi = pSet->GetParticleV0(i);
            fprintf(fp, "%g %g %g %d\n", pi.x, pi.y, pi.z, vi);
        }
        for(int i = 0; i < dCount; i++){
            vec3f pi = dBuilder.pos[i];
            fprintf(fp, "%g %g %g 2\n", pi.x, pi.y, pi.z);
        }
        fclose(fp);
    }
}

void SpawnParticles(vec3f pi, int pId, vec3f n, int id, DelaunaySetBuilder &dBuilder,
                    std::vector<uint32_t> &ids, Point3HVec &pointVec, unsigned int cellId,
                    Grid3 *grid, ParticleSet3 *pSet)
{
    int *neighbors = nullptr;
    int count = grid->GetNeighborsOf(cellId, &neighbors);
    Float radius = 1.1f * 0.02f;
    Float scaledRadius = 0.5 * radius;
    vec3f ps[120];
    int max_n = sizeof(ps) / sizeof(vec3f);
    int c = 0;

    int added = 0;
    for(int i = 0; i < count; i++){
        Cell3 *cell = grid->GetCell(neighbors[i]);
        ParticleChain *pChain = cell->GetChain();
        int size = cell->GetChainLength();

        for(int j = 0; j < size; j++){
            if(pId != pChain->pId){
                vec3f pj = pSet->GetParticlePosition(pChain->pId);
                Float d = Distance(pi, pj);
                if(d < scaledRadius && d > 1e-4){
                    added += 1;
                }else{
                    if(c < max_n)
                        ps[c++] = pj;
                }
            }
            pChain = pChain->next;
        }
    }

    if(added == 0){
        for(int i = 0; i < 6; i++){
            Float theta = i * Pi / 6.f;
            Float phi = i * TwoPi / 6.f;
            Float x = scaledRadius * sinf(theta) * cosf(phi);
            Float y = scaledRadius * sinf(theta) * sinf(phi);
            Float z = scaledRadius * cosf(theta);
            vec3f pn = pi + vec3f(x, y, z) * 0.6;
            bool accept = true;
            for(int s = 0; s < c && accept; s++){
                if(Distance(pn, ps[s]) < 1e-4)
                    accept = false;
            }
            if(accept){
                ids.push_back(id + added);
                pointVec.push_back({pn.x, pn.y, pn.z});
                dBuilder.PushParticle(pn, n);
                added += 1;
            }
        }
    }else{
        ids.push_back(pId);
        pointVec.push_back({pi.x, pi.y, pi.z});
    }
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

void SpawnTetra2(vec3f p0, int pId, vec3f n, int id, DelaunaySetBuilder &dBuilder,
                 std::vector<uint32_t> &ids, Point3HVec &pointVec, Float edgeLen)
{
    Matrix3x3 I;
    const vec3f u0 = vec3f(0.57735, 0.57735, 0.57735);
    const vec3f u1 = vec3f(0.57735, -0.57735, -0.57735);
    const vec3f u2 = vec3f(-0.57735, 0.57735, -0.57735);
    const vec3f u3 = vec3f(-0.57735, -0.57735, 0.57735);
    if(IsZero(n.LengthSquared())){
        SpawnTetra1(p0, pId, n, id, dBuilder, ids, pointVec, edgeLen);
        return;
    }

    vec3f dir = -n;
    vec3f edir = Normalize(u1 - u0);
    vec3f v = Cross(edir, dir);
    Float c = Dot(edir, dir);

    if(c == -1){
        vec3f dir1 = -edir;
        vec3f dir2 = Normalize(u2 - u1);
        vec3f dir3 = Normalize(u3 - u1);

        vec3f _np1 = p0 + dir1 * edgeLen;
        vec3f _np2 = p0 + dir2 * edgeLen;
        vec3f _np3 = p0 + dir3 * edgeLen;

        ids.push_back(pId);
        ids.push_back(id + 0);
        ids.push_back(id + 1);
        ids.push_back(id + 2);

        pointVec.push_back({p0.x, p0.y, p0.z});
        pointVec.push_back({_np1.x, _np1.y, _np1.z});
        pointVec.push_back({_np2.x, _np2.y, _np2.z});
        pointVec.push_back({_np3.x, _np3.y, _np3.z});

        dBuilder.PushParticle(_np1, n);
        dBuilder.PushParticle(_np2, n);
        dBuilder.PushParticle(_np3, n);

        return;
    }

    // Efficiently Building a Matrix to Rotate One Vector to Another
    //    Tomas Moller and John F. Hughes
    Matrix3x3 vx( 0,   -v.z,  v.y,
                  v.z,  0,   -v.x,
                 -v.y,  v.x,  0 );
    Matrix3x3 vx2 = Matrix3x3::Mul(vx, vx);
    Matrix3x3 R = I + vx + vx2 * (1.f / (1.f + c));

    vec3f ru1 = R.Vec(u1);
    vec3f ru2 = R.Vec(u2);
    vec3f ru3 = R.Vec(u3);

    vec3f dir_u1 = Normalize(ru1); // TODO: isn't this -n itself?
    vec3f dir_u2 = Normalize(ru2);
    vec3f dir_u3 = Normalize(ru3);

    vec3f np1 = p0 + dir_u1 * edgeLen;
    vec3f np2 = p0 + dir_u2 * edgeLen;
    vec3f np3 = p0 + dir_u3 * edgeLen;

    ids.push_back(pId);
    ids.push_back(id + 0);
    ids.push_back(id + 1);
    ids.push_back(id + 2);

    pointVec.push_back({p0.x, p0.y, p0.z});
    pointVec.push_back({np1.x, np1.y, np1.z});
    pointVec.push_back({np2.x, np2.y, np2.z});
    pointVec.push_back({np3.x, np3.y, np3.z});

    dBuilder.PushParticle(np1, n);
    dBuilder.PushParticle(np2, n);
    dBuilder.PushParticle(np3, n);
}

__bidevice__
void MoveParticles(ParticleSet3 *pSet, Grid3 *grid, int pId, vec3f &po){
    vec3f pi = pSet->GetParticlePosition(pId);
    int vi = pSet->GetParticleV0(pId);
    Float spacing = 0.01f;
    Float halfSpacing = spacing * 0.5f;
    Float invSpacing = 1.f / spacing;
    po = pi;

    if(vi == 0)
        return;

    int x_id = std::floor(pi.x * invSpacing);
    int y_id = std::floor(pi.y * invSpacing);
    int z_id = std::floor(pi.z * invSpacing);

    vec3f center =
            vec3f((Float)x_id, (Float)y_id, (Float)z_id) * spacing + vec3f(halfSpacing);
    const vec3f vertices[] = {
        vec3f(-1, -1, -1), vec3f(-1, -1, +1),
        vec3f(-1, +1, -1), vec3f(-1, +1, +1),
        vec3f(+1, -1, -1), vec3f(+1, -1, +1),
        vec3f(+1, +1, -1), vec3f(+1, +1, +1),

        vec3f(-1.0, +0.5, +0.0), vec3f(+1.0, +0.5, +0.0),
        vec3f(+0.0, +0.5, +1.0), vec3f(+0.0, +0.5, -1.0),
        vec3f(+0.0, +1.0, +0.0), vec3f(+0.0, -1.0, +0.0),
    };

    int vcount = sizeof(vertices) / sizeof(vertices[0]);
    Float minDist = Infinity;
    vec3f ps = pi;
    for(int i = 0; i < vcount; i++){
        vec3f pn = center + vertices[i] * halfSpacing;
        Float dist = Distance(pn, pi);
        if(minDist > dist){
            ps = pn;
            minDist = dist;
        }
    }

    po = ps;
}

__bidevice__
void MoveParticles2(ParticleSet3 *pSet, Grid3 *grid, int pId, vec3f &po){
    vec3f pi = pSet->GetParticlePosition(pId);
    int vi = pSet->GetParticleV0(pId);
    po = pi;

    if(vi == 0)
        return;

    vec3f ni = pSet->GetParticleNormal(pId);
    if(IsZero(ni.LengthSquared()))
        return;

    int *neighbors = nullptr;
    unsigned int cellId = grid->GetLinearHashedPosition(pi);
    int count = grid->GetNeighborsOf(cellId, &neighbors);

    Float wsum = 0.f;
    vec3f dsum(0.f);
    for(int i = 0; i < count; i++){
        Cell3 *cell = grid->GetCell(neighbors[i]);
        ParticleChain *pChain = cell->GetChain();
        int size = cell->GetChainLength();

        for(int j = 0; j < size; j++){
            if(pChain->pId != pId){
                vec3f pj = pSet->GetParticlePosition(pChain->pId);
                dsum = dsum + pj;
                wsum += 1.f;
            }
        }
    }

    if(wsum > 0){
        Float relax = 0.25f;
        dsum = dsum * (1.f / wsum);
        vec3f disp = dsum - pi;
        po = pi + disp * relax;
    }
}

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
            //ids.push_back(pId);
            //pointVec.push_back({pi.x, pi.y, pi.z});
            //SpawnTetra2(pi, pId, n, totalP, dBuilder, ids, pointVec, 0.9 * dist);
            SpawnTetra1(pi, pId, n, totalP, dBuilder, ids, pointVec, 0.9 * dist);
            //SpawnParticles(pi, pId, n, totalP, dBuilder, ids, pointVec, cellId, grid, pSet);
        }else{
            ids.push_back(pId);
            pointVec.push_back({pi.x, pi.y, pi.z});
        }
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
    char *tetFlags = cudaAllocateUnregisterVx(char, size);

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

    cudaFree(tetFlags);
}

void
DelaunaySurface(DelaunayTriangulation &triangulation, SphParticleSet3 *sphSet,
                Float spacing, Float mu, Grid3 *domain, SphSolver3 *solver)
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

#if 1
    vec3f *u_ps = cudaAllocateUnregisterVx(vec3f, pSet->GetParticleCount());

    int iterations = 1;
    for(int s = 0; s < iterations; s++){
        AutoParallelFor("Delaunay_MoveParticles", pSet->GetParticleCount(),
            AutoLambda(int i)
        {
            vec3f pl;
            MoveParticles(pSet, domain, i, pl);

            u_ps[i] = pl;
        });

        for(int i = 0; i < pSet->GetParticleCount(); i++){
            pSet->SetParticlePosition(i, u_ps[i]);
        }
    }

    cudaFree(u_ps);

    UpdateGridDistributionGPU(solver->solverData);
#endif

    for(int i = 0; i < pSet->GetParticleCount(); i++){
        CheckPart(pSet, domain, i, dBuilder, triangulation.ids,
                  triangulation.pointVec, dist);
        u_bound[i] = 0;
    }

    //DumpPoints(dBuilder, pSet);

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
    std::cout << "done ( boundary: " << b_len << " / " << n_it << " ) " << std::endl;
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

