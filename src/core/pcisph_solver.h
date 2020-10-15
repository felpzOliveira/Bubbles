#pragma once

#include <sph_solver.h>

typedef struct{
    SphSolverData2 *sphData;
    vec2f *tempPositions;
    vec2f *tempVelocities;
    vec2f *pressureForces;
    Float *densityErrors;
    Float *densityPredicted;
    vec2f *refMemory;
}PciSphSolverData2;

typedef struct{
    SphSolverData3 *sphData;
    vec3f *tempPositions;
    vec3f *tempVelocities;
    vec3f *pressureForces;
    Float *densityErrors;
    Float *densityPredicted;
    vec3f *refMemory;
}PciSphSolverData3;

class PciSphSolver2{
    public:
    Float maxErrorDensity;
    unsigned int maxIterations;
    Float massOverTargetDensitySquared;
    Float deltaDenom;
    PciSphSolverData2 *solverData;
    
    __bidevice__ PciSphSolver2();
    __host__ void Initialize(SphSolverData2 *data);
    __host__ void Setup(Float targetDensity, Float targetSpacing, Float relativeRadius,
                        Grid2 *domain, SphParticleSet2 *pSet);
    __host__ void SetColliders(ColliderSet2 *colliders);
    __bidevice__ SphSolverData2 *GetSphSolverData();
    __bidevice__ SphParticleSet2 *GetSphParticleSet();
    __host__ void Advance(Float timeIntervalInSeconds);
    __host__ void Cleanup();
    
    __host__ Float ComputeBeta(Float timeIntervalInSeconds);
    __host__ Float ComputeDelta(Float timeIntervalInSeconds);
    __host__ Float ComputeDeltaDenom();
};

class PciSphSolver3{
    public:
    Float maxErrorDensity;
    unsigned int maxIterations;
    Float massOverTargetDensitySquared;
    Float deltaDenom;
    PciSphSolverData3 *solverData;
    
    __bidevice__ PciSphSolver3();
    __host__ void Initialize(SphSolverData3 *data);
    __host__ void Setup(Float targetDensity, Float targetSpacing, Float relativeRadius,
                        Grid3 *domain, SphParticleSet3 *pSet);
    __host__ void SetColliders(ColliderSet3 *colliders);
    __bidevice__ SphSolverData3 *GetSphSolverData();
    __bidevice__ SphParticleSet3 *GetSphParticleSet();
    __host__ void Advance(Float timeIntervalInSeconds);
    __host__ void Cleanup();
    
    __host__ Float ComputeBeta(Float timeIntervalInSeconds);
    __host__ Float ComputeDelta(Float timeIntervalInSeconds);
    __host__ Float ComputeDeltaDenom();
};

__host__ void ComputePressureForceAndIntegrate(PciSphSolverData2 *data, 
                                               Float timeIntervalInSeconds, 
                                               Float maxDensityErrorRatio, 
                                               Float delta, int maxIt,
                                               int is_cpu=0);

__host__ void ComputePressureForceAndIntegrate(PciSphSolverData3 *data, 
                                               Float timeIntervalInSeconds, 
                                               Float maxDensityErrorRatio, 
                                               Float delta, int maxIt,
                                               int is_cpu=0);