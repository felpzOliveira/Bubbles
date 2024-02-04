#pragma once

#include <geometry.h>
#include <particle.h>
#include <grid.h>
#include <collider.h>
#include <statics.h>
#include <lnm.h>

typedef struct{
    SphParticleSet2 *sphpSet;
    Grid2 *domain;
    ColliderSet2 *collider;
    vec2f *smoothedVelocities;
    ConstantInteraction2 *cInteractions;
    FunctionalInteraction2 *fInteractions;
    int fInteractionsCount;
    int cInteractionsCount;
    Float dragCoefficient;
    Float eosExponent;
    Float negativePressureScale;
    Float viscosity;
    Float pseudoViscosity;
    Float soundSpeed;
    Float timestepScale;
    Float Tmin, Tmax;
    Float Tamb;
    int frame_index;
}SphSolverData2;

typedef struct{
    SphParticleSet3 *sphpSet;
    Grid3 *domain;
    ConstantInteraction3 *cInteractions;
    FunctionalInteraction3 *fInteractions;
    int fInteractionsCount;
    int cInteractionsCount;
    ColliderSet3 *collider;
    vec3f *smoothedVelocities;
    Float dragCoefficient;
    Float eosExponent;
    Float negativePressureScale;
    Float viscosity;
    Float pseudoViscosity;
    Float soundSpeed;
    Float timestepScale;
    Float Tmin, Tmax;
    Float Tamb;
    int frame_index;
}SphSolverData3;


void SphSolverData2SetupFor(SphSolverData2 *solverData, int expectedParticleCount);
void SphSolverData3SetupFor(SphSolverData3 *solverData, int expectedParticleCount);

class SphSolver2{
    public:
    SphSolverData2 *solverData;
    Float stepInterval;
    bb_cpu_gpu SphSolver2();
    bb_cpu_gpu void Initialize(SphSolverData2 *data);
    void Setup(Float targetDensity, Float targetSpacing, Float relativeRadius,
               Grid2 *domain, SphParticleSet2 *pSet);

    void SetColliders(ColliderSet2 *colliders);
    bb_cpu_gpu SphSolverData2 *GetSphSolverData();
    void SetViscosityCoefficient(Float viscosityCoefficient);
    void SetPseudoViscosityCoefficient(Float pseudoViscosityCoefficient);

    bb_cpu_gpu Float GetKernelRadius();
    bb_cpu_gpu SphParticleSet2 *GetSphParticleSet();
    void UpdateDensity();
    void Advance(Float timeIntervalInSeconds);
};

class SphSolver3{
    public:
    SphSolverData3 *solverData;
    Float stepInterval;
    bb_cpu_gpu SphSolver3();
    bb_cpu_gpu void Initialize(SphSolverData3 *data);
    void Setup(Float targetDensity, Float targetSpacing, Float relativeRadius,
               Grid3 *domain, SphParticleSet3 *pSet);

    void SetColliders(ColliderSet3 *colliders);
    ColliderSet3 *GetColliders();

    bb_cpu_gpu SphSolverData3 *GetSphSolverData();
    void SetViscosityCoefficient(Float viscosityCoefficient);
    void SetPseudoViscosityCoefficient(Float pseudoViscosityCoefficient);

    bb_cpu_gpu Float GetKernelRadius();
    bb_cpu_gpu SphParticleSet3 *GetSphParticleSet();

    void Advance(Float timeIntervalInSeconds);
};

class SphGasSolver2{
    public:
    SphSolver2 *solver;

    bb_cpu_gpu SphGasSolver2();

    void Initialize();

    void Setup(Float targetDensity, Float targetSpacing, Float relativeRadius,
               Grid2 *domain, SphParticleSet2 *pSet);

    void SetColliders(ColliderSet2 *colliders);

    void SetViscosityCoefficient(Float viscosityCoefficient);
    void SetPseudoViscosityCoefficient(Float pseudoViscosityCoefficient);

    bb_cpu_gpu Float GetKernelRadius();

    bb_cpu_gpu SphSolverData2 *GetSphSolverData();

    void UpdateDensity();
    void Advance(Float timeIntervalInSeconds);
};

// Generic SphSolver routines

// Compute standard sph density, for the simple formulation can also compute
// pressure in a single pass increasing performance.
bb_cpu_gpu void ComputeDensityFor(SphSolverData2 *data, int particleId,
                                  int compute_pressure = 1);
bb_cpu_gpu void ComputeDensityFor(SphSolverData3 *data, int particleId,
                                  int compute_pressure = 1);
// Compute pressure for a given particle with density di
bb_cpu_gpu void ComputePressureFor(SphSolverData2 *data, int particleId, Float di);
bb_cpu_gpu void ComputePressureFor(SphSolverData3 *data, int particleId, Float di);

// Compute viscosity, gravity and drag forces to particle force vector
bb_cpu_gpu void ComputeNonPressureForceFor(SphSolverData2 *data, int particleId);
bb_cpu_gpu void ComputeNonPressureForceFor(SphSolverData3 *data, int particleId);
// Compute viscosity/gravity/drag/pressure forces in a single call
bb_cpu_gpu void ComputeAllForcesFor(SphSolverData2 *data, int particleId,
                                    Float timeStep, int extended = 0, int integrate = 1);
bb_cpu_gpu void ComputeAllForcesFor(SphSolverData3 *data, int particleId,
                                    Float timeStep, int extended = 0);
// Evolve particles
bb_cpu_gpu void TimeIntegrationFor(SphSolverData2 *data, int particleId,
                                   Float timeStep, int extended);
bb_cpu_gpu void TimeIntegrationFor(SphSolverData3 *data, int particleId,
                                   Float timeStep, int extended);
bb_cpu_gpu void ComputeInitialTemperatureFor(SphSolverData2 *data, int particleId,
                                             Float Tmin, Float Tmax, int maxLevel);

// Generic calls for sph solvers wrappers
// Perform computation on CPU {easy debug}
void ComputeDensityCPU(SphSolverData2 *data, int compute_pressure = 1);
void ComputeDensityCPU(SphSolverData3 *data, int compute_pressure = 1);
void ComputeNormalCPU(SphSolverData3 *data);
void ComputeNormalCPU(SphSolverData2 *data);
void ComputePressureForceCPU(SphSolverData2 *data, Float timeStep,
                                          int integrate=1);
void ComputePressureForceCPU(SphSolverData3 *data, Float timeStep);
void ComputeNonPressureForceCPU(SphSolverData2 *data);
void ComputeNonPressureForceCPU(SphSolverData3 *data);
void ComputeParticleInteractionCPU(SphSolverData2 *data);
void ComputeParticleInteractionCPU(SphSolverData3 *data);
void TimeIntegrationCPU(SphSolverData2 *data, Float timeStep, int extended=0);
void TimeIntegrationCPU(SphSolverData3 *data, Float timeStep, int extended=0);
void ComputeInitialTemperatureMapCPU(SphSolverData2 *data, Float Tmin,
                                     Float Tmax, int maxLevel);
void ComputePseudoViscosityInterpolationCPU(SphSolverData2 *data, Float timeStep);
void ComputePseudoViscosityInterpolationCPU(SphSolverData3 *data, Float timeStep);

// Perform computation on GPU
void ComputeDensityGPU(SphSolverData2 *data, int compute_pressure = 1);
void ComputeDensityGPU(SphSolverData3 *data, int compute_pressure = 1);
void ComputeNormalGPU(SphSolverData3 *data);
void ComputeNormalGPU(SphSolverData2 *data);
void ComputePressureForceGPU(SphSolverData2 *data, Float timeStep, int integrate=1);
void ComputePressureForceGPU(SphSolverData3 *data, Float timeStep);
void ComputeNonPressureForceGPU(SphSolverData2 *data);
void ComputeNonPressureForceGPU(SphSolverData3 *data);
void ComputeParticleInteractionGPU(SphSolverData2 *data);
void ComputeParticleInteractionGPU(SphSolverData3 *data);
void TimeIntegrationGPU(SphSolverData2 *data, Float timeStep, int extended=0);
void TimeIntegrationGPU(SphSolverData3 *data, Float timeStep, int extended=0);
void ComputeInitialTemperatureMapGPU(SphSolverData2 *data, Float Tmin,
                                     Float Tmax, int maxLevel);
void ComputePseudoViscosityInterpolationGPU(SphSolverData2 *data, Float timeStep);
void ComputePseudoViscosityInterpolationGPU(SphSolverData3 *data, Float timeStep);

// Computes the average temperature of all particles
bb_cpu_gpu Float ComputeAverageTemperature(SphSolverData2 *data);

// Generic update grid methods
void UpdateGridDistributionCPU(SphSolverData2 *data);
void UpdateGridDistributionGPU(SphSolverData2 *data);
void UpdateGridDistributionCPU(SphSolverData3 *data);
void UpdateGridDistributionGPU(SphSolverData3 *data);

// Sph utilities for other modules
bb_cpu_gpu Float ComputeDensityForPoint(SphSolverData2 *data, const vec2f &p);

// Generate a set of default SPH properties
SphSolverData2 *DefaultSphSolverData2(bool with_gravity=true);
SphSolverData3 *DefaultSphSolverData3(bool with_gravity=true);

// Display routines
void Debug_GraphyDisplaySolverParticles(ParticleSet2 *pSet);
void Debug_GraphyDisplaySolverParticles(ParticleSet2 *pSet, float *buffer);
void Debug_GraphyDisplaySolverParticles(ParticleSet2 *pSet, float *buffer, float *colors);
void Debug_GraphyDisplayParticles(int n, float *buffer, float *colors, Float pSize = 5.0);
