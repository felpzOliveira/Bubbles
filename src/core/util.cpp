#include <serializer.h>
#include <geometry.h>
#include <util.h>
#include <gDel3D/GpuDelaunay.h>
#include <gDel3D/CPU/PredWrapper.h>
#include <gDel3D/CommonTypes.h>
#include <deque>
#include <fstream>
#include <sstream>
#include <unordered_set>
#include <unordered_map>
#include <shape.h>

#define PushPosition(p, c, pp, pc, id) do{\
    pp[3 * id + 0] = p.x;\
    pp[3 * id + 1] = p.y;\
    pp[3 * id + 2] = p.z;\
    pc[3 * id + 0] = c.x;\
    pc[3 * id + 1] = c.y;\
    pc[3 * id + 2] = c.z;\
    id++;\
}while(0)

ParsedMesh *UtilGDel3DToParsedMesh(std::vector<vec3i> *tris, Point3HVec *pointVec,
                                   GDelOutput *output)
{
    PredWrapper predWrapper;
    predWrapper.init(*pointVec, output->ptInfty);
    int size = predWrapper.pointNum();
    ParsedMesh *mesh = cudaAllocateVx(ParsedMesh, 1);

    mesh->nVertices = size;
    mesh->nTriangles = tris->size();
    mesh->nNormals = 0;
    mesh->nUvs = 0;
    mesh->uv = nullptr;
    mesh->s = nullptr;
    mesh->n = nullptr;

    mesh->p = cudaAllocateVx(Point3f, mesh->nVertices);
    mesh->indices = cudaAllocateVx(Point3i, 3 * tris->size());
    for(int i = 0; i < mesh->nVertices; i++){
        const Point3 pt = predWrapper.getPoint(i);
        mesh->p[i] = Point3f(pt._p[0], pt._p[1], pt._p[2]);
    }

    for(int i = 0; i < tris->size(); i++){
        vec3i val = tris->at(i);
        mesh->indices[3 * i + 0] = Point3i(val.x, 0, 0);
        mesh->indices[3 * i + 1] = Point3i(val.y, 0, 0);
        mesh->indices[3 * i + 2] = Point3i(val.z, 0, 0);
    }

    mesh->transform = Transform();
    mesh->allocator = AllocatorType::GPU;
    sprintf(mesh->name, "gdel3mesh");

    return mesh;
}

__host__ int UtilGenerateBoxPoints(float *posBuffer, float *colBuffer, vec3f col,
                                   vec3f length, int nPoints, Transform transform)
{
    int perEdge = (int)Floor((Float)nPoints / 12);
    vec3f hlen = 0.5 * length;
    Float hx = length.x * (1.0 / (Float)perEdge);
    Float hy = length.y * (1.0 / (Float)perEdge);
    Float hz = length.z * (1.0 / (Float)perEdge);
    vec3f f0, f1, p;

    int it = 0;
    f0 = vec3f(hlen.x, 0, 0);
    f1 = vec3f(-hlen.x, 0, 0);
    for(int i = 0; i <= perEdge; i++){ // depth
        p = f0 + vec3f(0, -hlen.y, -hlen.z + i * hz);
        p = transform.Point(p);
        PushPosition(p, col, posBuffer, colBuffer, it);

        p = f0 + vec3f(0, +hlen.y, -hlen.z + i * hz);
        p = transform.Point(p);
        PushPosition(p, col, posBuffer, colBuffer, it);

        p = f1 + vec3f(0, -hlen.y, -hlen.z + i * hz);
        p = transform.Point(p);
        PushPosition(p, col, posBuffer, colBuffer, it);

        p = f1 + vec3f(0, +hlen.y, -hlen.z + i * hz);
        p = transform.Point(p);
        PushPosition(p, col, posBuffer, colBuffer, it);
    }

    f0 = vec3f(0, hlen.y, 0);
    f1 = vec3f(0, -hlen.y, 0);
    for(int i = 0; i <= perEdge; i++){ // horizontal
        p = f0 + vec3f(-hlen.x + i * hx, 0, -hlen.z);
        p = transform.Point(p);
        PushPosition(p, col, posBuffer, colBuffer, it);

        p = f0 + vec3f(-hlen.x + i * hx, 0, +hlen.z);
        p = transform.Point(p);
        PushPosition(p, col, posBuffer, colBuffer, it);

        p = f1 + vec3f(-hlen.x + i * hx, 0, -hlen.z);
        p = transform.Point(p);
        PushPosition(p, col, posBuffer, colBuffer, it);

        p = f1 + vec3f(-hlen.x + i * hx, 0, +hlen.z);
        p = transform.Point(p);
        PushPosition(p, col, posBuffer, colBuffer, it);
    }

    f0 = vec3f(-hlen.x, 0, 0);
    f1 = vec3f(hlen.x, 0, 0);
    for(int i = 0; i <= perEdge; i++){ // vertical
        p = f0 + vec3f(0, -hlen.y + i * hy, -hlen.z);
        p = transform.Point(p);
        PushPosition(p, col, posBuffer, colBuffer, it);

        p = f0 + vec3f(0, -hlen.y + i * hy, +hlen.z);
        p = transform.Point(p);
        PushPosition(p, col, posBuffer, colBuffer, it);

        p = f1 + vec3f(0, -hlen.y + i * hy, -hlen.z);
        p = transform.Point(p);
        PushPosition(p, col, posBuffer, colBuffer, it);

        p = f1 + vec3f(0, -hlen.y + i * hy, +hlen.z);
        p = transform.Point(p);
        PushPosition(p, col, posBuffer, colBuffer, it);
    }

    return it;
}

__host__ int UtilGenerateSquarePoints(float *posBuffer, float *colBuffer, vec3f col,
                                      Transform2 transform, vec2f len, int nPoints)
{
    int perSide = (int)Floor((Float)nPoints / 4);
    vec2f hlen = 0.5 * len;
    int it = 0;
    Float xstep = len.x / (Float)perSide;
    Float ystep = len.y / (Float)perSide;
    Float x0 = -hlen.x;
    Float x1 = hlen.x;
    Float y0 = -hlen.y;
    Float y1 = hlen.y;
    for(int i = 0; i <= perSide; i++){
        Float y = -hlen.y + i * ystep;
        vec2f p(x0, y);
        vec2f q = transform.Point(p);
        posBuffer[3 * it + 0] = q.x;
        posBuffer[3 * it + 1] = q.y;
        posBuffer[3 * it + 2] = 0;
        colBuffer[3 * it + 0] = col.x;
        colBuffer[3 * it + 1] = col.y;
        colBuffer[3 * it + 2] = col.z;
        it++;

        p = vec2f(x1, y);
        q = transform.Point(p);
        posBuffer[3 * it + 0] = q.x;
        posBuffer[3 * it + 1] = q.y;
        posBuffer[3 * it + 2] = 0;
        colBuffer[3 * it + 0] = col.x;
        colBuffer[3 * it + 1] = col.y;
        colBuffer[3 * it + 2] = col.z;
        it++;

        Float x = -hlen.x + i * xstep;
        p = vec2f(x, y0);
        q = transform.Point(p);
        posBuffer[3 * it + 0] = q.x;
        posBuffer[3 * it + 1] = q.y;
        posBuffer[3 * it + 2] = 0;
        colBuffer[3 * it + 0] = col.x;
        colBuffer[3 * it + 1] = col.y;
        colBuffer[3 * it + 2] = col.z;
        it++;

        p = vec2f(x, y1);
        q = transform.Point(p);
        posBuffer[3 * it + 0] = q.x;
        posBuffer[3 * it + 1] = q.y;
        posBuffer[3 * it + 2] = 0;
        colBuffer[3 * it + 0] = col.x;
        colBuffer[3 * it + 1] = col.y;
        colBuffer[3 * it + 2] = col.z;
        it++;
    }

    return it;
}

__host__ int UtilGenerateSpherePoints(float *posBuffer, float *colBuffer, vec3f col,
                                      Float rad, int nPoints, Transform transform)
{
    int it = 0;
    Float area = 4.0 * Pi * rad * rad / (Float)nPoints;
    Float d = std::sqrt(area);
    Float mtheta = std::round(Pi / d);
    Float dtheta = Pi / mtheta;
    Float dphi = area / dtheta;

    for(Float m = 0; m < mtheta; m++){
        Float theta = Pi * (m + 0.5) / mtheta;
        Float mphi = std::round(2 * Pi * std::sin(theta) / dphi);
        for(Float n = 0; n < mphi; n++){
            Float phi = 2 * Pi * n / mphi;
            Float x = rad * std::sin(theta) * std::cos(phi);
            Float y = rad * std::sin(theta) * std::sin(phi);
            Float z = rad * std::cos(theta);
            vec3f p = transform.Point(vec3f(x, y, z));
            posBuffer[3 * it + 0] = p.x;
            posBuffer[3 * it + 1] = p.y;
            posBuffer[3 * it + 2] = p.z;
            if(colBuffer){
                colBuffer[3 * it + 0] = col.x;
                colBuffer[3 * it + 1] = col.y;
                colBuffer[3 * it + 2] = col.z;
            }
            if(it >= nPoints) return it-1;
            it++;
        }
    }

    return it;
}

__host__ int UtilGenerateCirclePoints(float *posBuffer, float *colBuffer, vec3f col,
                                      vec2f center, Float rad, int nPoints)
{
    Float step = 2.0 * Pi / (Float)nPoints;
    Float alpha = 0;
    for(int i = 0; i < nPoints; i++, alpha += step){
        posBuffer[3 * i + 0] = rad * std::cos(alpha) + center.x;
        posBuffer[3 * i + 1] = rad * std::sin(alpha) + center.y;
        posBuffer[3 * i + 2] = 0;
        colBuffer[3 * i + 0] = col.x;
        colBuffer[3 * i + 1] = col.y;
        colBuffer[3 * i + 2] = col.z;
    }

    return nPoints;
}

__bidevice__ int ParticleBySDF(const vec3f &p, FieldGrid3f *sdf, Float sdfThreshold,
                               int absolute=1)
{
    int rv = 0;
    Float sample = sdf->Sample(p);
    if(absolute){
        rv = Absf(sample - sdfThreshold) < Epsilon ? 1 : 0;
    }else{
        rv = sample < sdfThreshold ? 1 : 0;
    }

    return rv;
}

__host__ int UtilGetSDFParticles(FieldGrid3f *sdf, std::vector<vec3f> *particles,
                                 Float sdfThreshold, Float spacing, int absolute)
{
    Float h = spacing;
    int added = 0;
    if(sdf){
        vec3f p0 = sdf->bounds.pMin;
        vec3f p1 = sdf->bounds.pMax;
        for(Float x = p0.x; x < p1.x; x += h){
            for(Float y = p0.y; y < p1.y; y += h){
                for(Float z = p0.z; z < p1.z; z += h){
                    vec3f p(x, y, z);
                    if(ParticleBySDF(p, sdf, sdfThreshold, absolute)){
                        particles->push_back(p);
                        added++;
                    }
                }
            }
        }
    }
    return added;
}

__host__ Bounds3f _UtilComputePointsBounds(vec3f *points, int size){
    vec3f pi = points[0];
    Bounds3f bounds(pi, pi);
    for(int i = 1; i < size; i++){
        pi = points[i];
        bounds = Union(bounds, pi);
    }

    return bounds;
}

__host__ Bounds3f UtilComputeMeshBounds(ParsedMesh *mesh){
    AssertA(mesh->nVertices > 0 && mesh->p != nullptr, "Invalid mesh given");
    return _UtilComputePointsBounds(mesh->p, mesh->nVertices);
}

__host__ Transform UtilComputeFitTransform(ParsedMesh *mesh, Float maximumAxisLength,
                                           Float *scaleValue)
{
    Bounds3f bounds = UtilComputeMeshBounds(mesh);
    int shrinkAxis = bounds.MaximumExtent();
    Float length = bounds.ExtentOn(shrinkAxis);
    Float scale = maximumAxisLength / length;
    if(scaleValue) *scaleValue = scale;
    return Scale(scale);
}

__host__ Grid3 *UtilBuildGridForDomain(Bounds3f domain, Float spacing,
                                       Float spacingScale)
{
    vec3f pMin(0), pMax(0), half(0);
    Float length = spacing * spacingScale;
    Float hlen = 0.5 * length;
    Float invLen = 1.0f / length;
    int mx = (int)std::ceil(domain.ExtentOn(0) * invLen);
    int my = (int)std::ceil(domain.ExtentOn(1) * invLen);
    int mz = (int)std::ceil(domain.ExtentOn(2) * invLen);
    mx += (mx % 2) * length;
    my += (my % 2) * length;
    mz += (mz % 2) * length;

    half = vec3f(mx * hlen, my * hlen, mz * hlen);
    pMin = domain.Center() - half;
    pMax = domain.Center() + half;
    return MakeGrid(vec3ui(mx, my, mz), pMin, pMax);
}

__host__ Grid2 *UtilBuildGridForDomain(Bounds2f domain, Float spacing,
                                       Float spacingScale)
{
    vec2f pMin(0), pMax(0), half(0);
    Float length = spacing * spacingScale;
    Float hlen = 0.5 * length;
    Float invLen = 1.0f / length;
    int mx = (int)std::ceil(domain.ExtentOn(0) * invLen);
    int my = (int)std::ceil(domain.ExtentOn(1) * invLen);
    mx += (mx % 2) * length;
    my += (my % 2) * length;

    half = vec2f(mx * hlen, my * hlen);
    pMin = domain.Center() - half;
    pMax = domain.Center() + half;
    return MakeGrid(vec2ui(mx, my), pMin, pMax);
}

__host__ Grid3 *UtilBuildGridForBuilder(ParticleSetBuilder3 *builder, Float spacing,
                                        Float spacingScale)
{
    int size = builder->positions.size();
    vec3f p = builder->positions[0];
    Bounds3f container(p, p);
    for(int i = 1; i < size; i++){
        container = Union(container, builder->positions[i]);
    }

    container.Expand(spacing);
    return UtilBuildGridForDomain(container, spacing, spacingScale);
}

__host__ Bounds3f UtilParticleSetBuilder3FromBB(const char *path, ParticleSetBuilder3 *builder,
                                                int legacy, Transform transform,
                                                vec3f centerAt, vec3f initialVelocity)
{
    std::vector<vec3f> points;
    int size = 0;
    int flags = SERIALIZER_POSITION;
    if(legacy){
        SerializerLoadLegacySystem3(&points, path, flags);
    }else{
        SerializerLoadPoints3(&points, path, flags);
    }
    size = points.size();
    printf("Psize : %d\n", (int)size);

    // compute bounds center and generates the transformed particles
    Bounds3f bounds = _UtilComputePointsBounds(points.data(), size);
    Transform gtransform = Translate(centerAt - bounds.Center()) * transform;

    for(int i = 0; i < size; i++){
        vec3f pi = points[i];
        builder->AddParticle(gtransform.Point(pi), initialVelocity);
    }

    return gtransform(bounds);
}

__host__ Bounds3f UtilComputeBoundsAfter(ParsedMesh *mesh, Transform transform){
    vec3f pi = transform.Point(mesh->p[0]);
    Bounds3f bounds(pi, pi);
    for(int i = 1; i < mesh->nVertices; i++){
        pi = transform.Point(mesh->p[i]);
        bounds = Union(bounds, pi);
    }

    return bounds;
}

__host__ int UtilIsEmitterOverlapping(VolumeParticleEmitter3 *emitter,
                                      ColliderSet3 *colliderSet)
{
    int emitter_collider_overlaps = 0;
    Bounds3f bound = emitter->shape->GetBounds();
    for(int j = 0; j < colliderSet->nColiders; j++){
        Collider3 *collider = colliderSet->colliders[j];
        Bounds3f boundj = collider->shape->GetBounds();
        if(Overlaps(bound, boundj) && !collider->shape->reverseOrientation){
            emitter_collider_overlaps = 1;
            break;
        }
    }

    return emitter_collider_overlaps;
}

__host__ void UtilSphDataToFieldGrid2f(SphSolverData2 *solverData, FieldGrid2f *field){
    vec2ui resolution;
    vec2f gridSpacing;
    vec2f origin;
    Float kernelRadius = solverData->sphpSet->GetKernelRadius();
    // Compute density, make sure this thing has a density value
    UpdateGridDistributionCPU(solverData);
    ComputeDensityCPU(solverData);

    // Build the field
    Float scale = 0.5;
    Float invScale = 1.0 / scale;
    gridSpacing = vec2f(kernelRadius * scale);
    origin = solverData->domain->minPoint;
    resolution = solverData->domain->GetIndexCount() * invScale;
    field->Build(resolution, gridSpacing, origin, VertexCentered);

    for(int i = 0; i < field->total; i++){
        vec2ui u = DimensionalIndex(i, field->resolution, 2);
        vec2f pi = field->GetDataPosition(u);
        Float di = ComputeDensityForPoint(solverData, pi);

        Float methodDi = 0.5 - di;//TODO
        field->SetValueAt(methodDi, u);
    }
}

__host__ int UtilIsEmitterOverlapping(VolumeParticleEmitterSet3 *emitterSet,
                                      ColliderSet3 *colliderSet)
{
    int emitter_overlaps = 0;
    int emitter_collider_overlaps = 0;
    std::vector<VolumeParticleEmitter3 *> emitters = emitterSet->emitters;

    for(int i = 0; i < emitters.size() && !emitter_overlaps; i++){
        VolumeParticleEmitter3 *emitter = emitters[i];
        Bounds3f boundi = emitter->shape->GetBounds();
        for(int j = 0; j < emitters.size(); j++){
            if(i != j){
                VolumeParticleEmitter3 *otherEmitter = emitters[j];
                Bounds3f boundj = otherEmitter->shape->GetBounds();
                if(Overlaps(boundi, boundj)){
                    emitter_overlaps = 1;
                    break;
                }
            }
        }
    }

    for(int i = 0; i < emitters.size() && !emitter_collider_overlaps; i++){
        VolumeParticleEmitter3 *emitter = emitters[i];
        Bounds3f boundi = emitter->shape->GetBounds();
        for(int j = 0; j < colliderSet->nColiders; j++){
            Collider3 *collider = colliderSet->colliders[j];
            Bounds3f boundj = collider->shape->GetBounds();
            if(Overlaps(boundi, boundj) && !collider->shape->reverseOrientation){
                emitter_collider_overlaps = 1;
                break;
            }
        }
    }

    if(emitter_overlaps){
        printf("**************************************************************************\n");
        printf("Warning: You have overlapping emitters, while this is not\n");
        printf("         an error it might make particles too close and forces to scale up\n");
        printf("**************************************************************************\n");
    }

    if(emitter_collider_overlaps){
        printf("********************************************************************\n");
        printf("Warning: You have overlapping emitter x collider, this will trigger\n");
        printf("         unexpected behaviour during distribution, consider adjusting.\n");
        printf("********************************************************************\n");
    }

    return emitter_collider_overlaps + emitter_overlaps;
}

#define PLY_ADD_TRI(a, b, c, triSet, stream)do{\
    if(triSet.find(i3(a, b, c)) == triSet.end()){\
        stream << "3 " << a << " " << b << " " << c << " \n";\
        triSet.insert(i3(a, b, c));\
    }\
}while(0)

inline void GDel3D_WriteVertex(PredWrapper *predWrapper, std::ofstream &ofs){
    for(int i = 0; i < (int)predWrapper->pointNum(); i++){
        const Point3 pt = predWrapper->getPoint(i);
        for(int vi = 0; vi < 3; vi++){
            ofs << pt._p[vi] << " ";
        }
        ofs << "\n";
    }
}

__host__ void UtilGDel3DWritePly(std::vector<vec3i> *tris, Point3HVec *pointVec,
                                 GDelOutput *output, const char *path)
{
    PredWrapper predWrapper;
    predWrapper.init(*pointVec, output->ptInfty);
    std::ofstream ofs(path);
    ofs << "ply\n";
    ofs << "format ascii 1.0\n";
    ofs << "element vertex " << (int) predWrapper.pointNum() << "\n";
    ofs << "property double x\n";
    ofs << "property double y\n";
    ofs << "property double z\n";
    ofs << "element face " << tris->size() << "\n";
    ofs << "property list uchar int vertex_index\n";
    ofs << "end_header\n";

    GDel3D_WriteVertex(&predWrapper, ofs);

    for(int i = 0; i < tris->size(); i++){
        vec3i val = tris->at(i);
        ofs << "3 " << val[0] << " " << val[1] << " " << val[2] << " \n";
    }
    ofs.close();
}

__host__ void UtilGDel3DWritePly(Point3HVec *pointVec, GDelOutput *output, int pLen,
                                 const char *path, bool tetras)
{
    PredWrapper predWrapper;
    std::unordered_set<i3, i3Hasher, i3IsSame> triSet;
    predWrapper.init(*pointVec, output->ptInfty);

    // - write header
    std::ofstream ofs(path);
    ofs << "ply\n";
    ofs << "format ascii 1.0\n";
    ofs << "element vertex " << (int) predWrapper.pointNum() << "\n";
    ofs << "property double x\n";
    ofs << "property double y\n";
    ofs << "property double z\n";

    // - write geometry
    if(tetras){
        static int warned = 0;
        if(warned == 0){
            std::cout << "[PLY] Warning: Output is 4-indexed and " <<
                        "represents a tetrahedron, NOT a quad." << std::endl;
            warned = 1;
        }
        ofs << "element face " << GDel3D_RealTetraCount(output, pLen) << "\n";
        ofs << "property list uchar int vertex_index\n";
        ofs << "end_header\n";

        // write vertices
        GDel3D_WriteVertex(&predWrapper, ofs);

        GDel3D_ForEachRealTetra(output, pLen, [&](Tet tet, TetOpp botOpp, int i) -> void{
            ofs << "4 "<< tet._v[0] << " " << tet._v[1] << " " << tet._v[2] << " "
                    << tet._v[3] << std::endl;
        });
    }else{
        std::stringstream ss;
        GDel3D_ForEachRealTetra(output, pLen, [&](Tet tet, TetOpp botOpp, int i) -> void{
            uint32_t A = tet._v[0], B = tet._v[1],
                     C = tet._v[2], D = tet._v[3];
            // triangles are: ACB, ABD, ADC, BDC
            PLY_ADD_TRI(A, C, B, triSet, ss);
            PLY_ADD_TRI(A, B, D, triSet, ss);
            PLY_ADD_TRI(A, D, C, triSet, ss);
            PLY_ADD_TRI(B, D, C, triSet, ss);
        });

        ofs << "element face " << triSet.size() << "\n";
        ofs << "property list uchar int vertex_index\n";
        ofs << "end_header\n";

        // write vertices
        GDel3D_WriteVertex(&predWrapper, ofs);

        ofs << ss.str();
    }

    ofs.close();
}

#define TRI_MAP_ADD(a, b, c, u_map)do{\
    i3 i_val(a, b, c);\
    if(u_map.find(i_val) == u_map.end()){\
        u_map[i_val] = 1;\
    }else{\
        u_map[i_val] += 1;\
    }\
}while(0)

__host__ void UtilGDel3DUniqueTris(std::vector<i3> &tris, Point3HVec *pointVec,
                                   GDelOutput *output, int pLen)
{
    std::unordered_map<i3, int, i3Hasher, i3IsSame> triMap;
    GDel3D_ForEachRealTetra(output, pLen, [&](Tet tet, TetOpp botOpp, int i) -> void{
        uint32_t A = tet._v[0], B = tet._v[1],
                 C = tet._v[2], D = tet._v[3];
            // triangles are: ACB, ABD, ADC, BDC
            TRI_MAP_ADD(A, C, B, triMap);
            TRI_MAP_ADD(A, B, D, triMap);
            TRI_MAP_ADD(A, D, C, triMap);
            TRI_MAP_ADD(B, D, C, triMap);
    });

    for(auto it = triMap.begin(); it != triMap.end(); it++){
        if(it->second == 1){
            tris.push_back(it->first);
        }
    }
}


uint32_t GDel3D_TetraCount(GDelOutput *output){
    return output->tetVec.size();
}

uint32_t GDel3D_RealTetraCount(GDelOutput *output, uint32_t pLen){
    uint32_t counter = 0;
    GDel3D_ForEachRealTetra(output, pLen, [&](Tet &, const TetOpp &, int){
        counter += 1;
    });
    return counter;
}

