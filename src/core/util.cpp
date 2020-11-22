#include <serializer.h>
#include <geometry.h>
#include <util.h>

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
    int axis = domain.MinimumExtent();
    Float length = domain.ExtentOn(axis);
    vec3f pMin = domain.pMin - vec3f(spacing);
    vec3f pMax = domain.pMax + vec3f(spacing);
    int resolution = (int)std::ceil(length / (spacing * spacingScale));
    return MakeGrid(vec3ui(resolution), pMin, pMax);;
}

//TODO: Test me!
__host__ Grid3 *UtilBuildLNMGridForDomain(Bounds3f domain, Float spacing, 
                                          Float spacingScale)
{
    vec3f pMin, pMax, half;
    Float lenght = spacing * spacingScale; // ideal size for LNM
    Float invLength = 1.0 / lenght;
    Float hlen = lenght * 0.5f;
    int mx = (int)std::ceil(domain.ExtentOn(0) * invLength);
    int my = (int)std::ceil(domain.ExtentOn(1) * invLength);
    int mz = (int)std::ceil(domain.ExtentOn(2) * invLength);
    
    mx += (mx % 2) * lenght;
    my += (my % 2) * lenght;
    mz += (mz % 2) * lenght;
    
    half = vec3f(mx * hlen, my * hlen, mz * hlen);
    
    pMin = domain.Center() - half;
    pMax = domain.Center() + half;
    vec3ui resolution(mx, my, mz);
    
    Grid3 *grid = MakeGrid(resolution, pMin, pMax);
    printf("Generating gid: {%d x %d x %d}\n", resolution.x, resolution.y, resolution.z);
    grid->bounds.PrintSelf();
    printf("\n");
    return grid;
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