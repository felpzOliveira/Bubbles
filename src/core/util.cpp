#include <serializer.h>
#include <geometry.h>
#include <util.h>

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
    for(int i = 0; i < perSide; i++){
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
        rv = Absf(sample - sdfThreshold) < 0.01 ? 1 : 0;
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
    vec3ui resolution(mx, my, mz);
    return MakeGrid(resolution, pMin, pMax);
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
    vec2ui resolution(mx, my);
    return MakeGrid(resolution, pMin, pMax);
}

__host__ Bounds3f UtilParticleSetBuilder3FromBB(const char *path, ParticleSetBuilder3 *builder,
                                                Transform transform, vec3f centerAt,
                                                vec3f initialVelocity)
{
    std::vector<vec3f> points;
    int size = 0;
    SerializerLoadPoints3(&points, path, SERIALIZER_POSITION);
    size = points.size();
    
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
        printf("Warning: You have overlapping emitters, while this is not\n");
        printf("         an error it might make particles too close and forces to scale up\n");
    }
    
    if(emitter_collider_overlaps){
        printf("Warning: You have overlapping emitter x collider, this will trigger\n");
        printf("         unexpected behaviour during distribution, consider adjusting.\n");
    }
    
    return emitter_collider_overlaps + emitter_overlaps;
}
