#pragma once
#include <vector>
#include <sph_solver.h>
#include <functional>
#include <statics.h>

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
    Float stepInterval;
    LNMStats lnmStats;

    PciSphSolver2();
    void Initialize(SphSolverData2 *data);
    void Setup(Float targetDensity, Float targetSpacing, Float relativeRadius,
                        Grid2 *domain, SphParticleSet2 *pSet);
    void SetColliders(ColliderSet2 *colliders);
    void SetViscosityCoefficient(Float viscosityCoefficient);
    bb_cpu_gpu SphSolverData2 *GetSphSolverData();
    bb_cpu_gpu SphParticleSet2 *GetSphParticleSet();
    void Advance(Float timeIntervalInSeconds);
    void UpdateDensity();

    Float ComputeBeta(Float timeIntervalInSeconds);
    Float ComputeDelta(Float timeIntervalInSeconds);
    Float ComputeDeltaDenom();
    LNMStats GetLNMStats();
    Float GetAdvanceTime();
    int GetParticleCount();
};

class PciSphSolver3{
    public:
    Float maxErrorDensity;
    unsigned int maxIterations;
    Float massOverTargetDensitySquared;
    Float deltaDenom;
    PciSphSolverData3 *solverData;
    Float stepInterval;
    LNMStats lnmStats;

    PciSphSolver3();
    void Initialize(SphSolverData3 *data);
    void Setup(Float targetDensity, Float targetSpacing, Float relativeRadius,
                        Grid3 *domain, SphParticleSet3 *pSet);
    void SetColliders(ColliderSet3 *colliders);
    void SetViscosityCoefficient(Float viscosityCoefficient);
    bb_cpu_gpu SphSolverData3 *GetSphSolverData();
    bb_cpu_gpu SphParticleSet3 *GetSphParticleSet();
    void Advance(Float timeIntervalInSeconds);
    ColliderSet3 *GetColliders();

    Float ComputeBeta(Float timeIntervalInSeconds);
    Float ComputeDelta(Float timeIntervalInSeconds);
    Float ComputeDeltaDenom();
    LNMStats GetLNMStats();
    Float GetAdvanceTime();
    int GetParticleCount();
};

void ComputePressureForceAndIntegrate(PciSphSolverData2 *data, Float timeIntervalInSeconds,
                                      Float maxDensityErrorRatio, Float delta, int maxIt,
                                      int is_cpu=0);

void ComputePressureForceAndIntegrate(PciSphSolverData3 *data, Float timeIntervalInSeconds,
                                      Float maxDensityErrorRatio, Float delta, int maxIt,
                                      int is_cpu=0);

int EmptyCallback(int);

void PciSphRunSimulation3(PciSphSolver3 *solver, Float spacing, vec3f origin, vec3f target,
                          Float targetInterval, std::vector<Shape*> sdfs={},
                          const std::function<int(int )> &callback=EmptyCallback);

void PciSphRunSimulation2(PciSphSolver2 *solver, Float spacing, vec2f lower, vec2f upper,
                          Float targetInterval,
                          const std::function<int(int )> &callback=EmptyCallback);
