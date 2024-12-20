#include <serializer.h>
#include <geometry.h>
#include <util.h>
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

int UtilGenerateBoxPoints(float *posBuffer, float *colBuffer, vec3f col,
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
    for(int i = 0; i < perEdge; i++){ // depth
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
    for(int i = 0; i < perEdge; i++){ // horizontal
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
    for(int i = 0; i < perEdge; i++){ // vertical
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

int UtilGenerateSquarePoints(float *posBuffer, float *colBuffer, vec3f col,
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

int UtilGenerateSpherePoints(float *posBuffer, float *colBuffer, vec3f col,
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

int UtilGenerateCirclePoints(float *posBuffer, float *colBuffer, vec3f col,
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

bb_cpu_gpu int ParticleBySDF(const vec3f &p, FieldGrid3f *sdf, Float sdfThreshold,
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

int UtilGetSDFParticles(FieldGrid3f *sdf, std::vector<vec3f> *particles,
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

Bounds3f _UtilComputePointsBounds(vec3f *points, int size){
    vec3f pi = points[0];
    Bounds3f bounds(pi, pi);
    for(int i = 1; i < size; i++){
        pi = points[i];
        bounds = Union(bounds, pi);
    }

    return bounds;
}

Bounds3f UtilComputeMeshBounds(ParsedMesh *mesh){
    AssertA(mesh->nVertices > 0 && mesh->p != nullptr, "Invalid mesh given");
    return _UtilComputePointsBounds(mesh->p, mesh->nVertices);
}

Transform UtilComputeFitTransform(ParsedMesh *mesh, Float maximumAxisLength,
                                           Float *scaleValue)
{
    Bounds3f bounds = UtilComputeMeshBounds(mesh);
    int shrinkAxis = bounds.MaximumExtent();
    Float length = bounds.ExtentOn(shrinkAxis);
    Float scale = maximumAxisLength / length;
    if(scaleValue) *scaleValue = scale;
    return Scale(scale);
}

Grid3 *UtilBuildGridForDomain(Bounds3f domain, Float spacing,
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

Grid2 *UtilBuildGridForDomain(Bounds2f domain, Float spacing,
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

Grid3 *UtilBuildGridForBuilder(ParticleSetBuilder3 *builder, Float spacing,
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

Bounds3f UtilParticleSetBuilder3FromBB(const char *path, ParticleSetBuilder3 *builder,
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

Bounds3f UtilComputeBoundsAfter(ParsedMesh *mesh, Transform transform){
    vec3f pi = transform.Point(mesh->p[0]);
    Bounds3f bounds(pi, pi);
    for(int i = 1; i < mesh->nVertices; i++){
        pi = transform.Point(mesh->p[i]);
        bounds = Union(bounds, pi);
    }

    return bounds;
}

int UtilIsEmitterOverlapping(VolumeParticleEmitter3 *emitter, ColliderSet3 *colliderSet){
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

void UtilSphDataToFieldGrid2f(SphSolverData2 *solverData, FieldGrid2f *field){
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

int UtilIsEmitterOverlapping(VolumeParticleEmitterSet3 *emitterSet,
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

std::string modelsResources;
std::string outputResources;
void UtilSetGlobalModelPath(const char *path){
    modelsResources = std::string(path);
}

void UtilSetGlobalOutputPath(const char *path){
    outputResources = std::string(path);
}

