#pragma once

#include <geometry.h>
#include <particle.h>
#include <grid.h>
#include <collider.h>
#include <statics.h>
#include <cnm.h>

typedef struct{
    SphParticleSet2 *sphpSet;
    Grid2 *domain;
    ColliderSet2 *collider;
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
    ColliderSet3 *collider;
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


class SphSolver2{
    public:
    SphSolverData2 *solverData;
    
    __bidevice__ SphSolver2();
    __bidevice__ void Initialize(SphSolverData2 *data);
    __host__ void Setup(Float targetDensity, Float targetSpacing, Float relativeRadius, 
                        Grid2 *domain, SphParticleSet2 *pSet);
    
    __host__ void SetColliders(ColliderSet2 *colliders);
    __bidevice__ SphSolverData2 *GetSphSolverData();
    __host__ void SetViscosityCoefficient(Float viscosityCoefficient);
    __host__ void SetPseudoViscosityCoefficient(Float pseudoViscosityCoefficient);
    
    __bidevice__ Float GetKernelRadius();
    __bidevice__ SphParticleSet2 *GetSphParticleSet();
    __host__ void UpdateDensity();
    __host__ void Advance(Float timeIntervalInSeconds);
    __host__ void Cleanup();
};

class SphSolver3{
    public:
    SphSolverData3 *solverData;
    
    __bidevice__ SphSolver3();
    __bidevice__ void Initialize(SphSolverData3 *data);
    __host__ void Setup(Float targetDensity, Float targetSpacing, Float relativeRadius, 
                        Grid3 *domain, SphParticleSet3 *pSet);
    
    __host__ void SetColliders(ColliderSet3 *colliders);
    __bidevice__ SphSolverData3 *GetSphSolverData();
    __host__ void SetViscosityCoefficient(Float viscosityCoefficient);
    __host__ void SetPseudoViscosityCoefficient(Float pseudoViscosityCoefficient);
    
    __bidevice__ Float GetKernelRadius();
    __bidevice__ SphParticleSet3 *GetSphParticleSet();
    
    __host__ void Advance(Float timeIntervalInSeconds);
    __host__ void Cleanup();
};

class SphGasSolver2{
    public:
    SphSolver2 *solver;
    
    __bidevice__ SphGasSolver2();
    
    __host__ void Initialize();
    
    __host__ void Setup(Float targetDensity, Float targetSpacing, Float relativeRadius, 
                        Grid2 *domain, SphParticleSet2 *pSet);
    
    __host__ void SetColliders(ColliderSet2 *colliders);
    
    __host__ void SetViscosityCoefficient(Float viscosityCoefficient);
    __host__ void SetPseudoViscosityCoefficient(Float pseudoViscosityCoefficient);
    
    __bidevice__ Float GetKernelRadius();
    
    __bidevice__ SphSolverData2 *GetSphSolverData();
    
    __host__ void UpdateDensity();
    __host__ void Advance(Float timeIntervalInSeconds);
    __host__ void Cleanup();
};

// Generic SphSolver routines

// Compute standard sph density, for the simple formulation can also compute
// pressure in a single pass increasing performance.
__bidevice__ void ComputeDensityFor(SphSolverData2 *data, int particleId,
                                    int compute_pressure = 1);
// Compute pressure for a given particle with density di
__bidevice__ void ComputePressureFor(SphSolverData2 *data, int particleId, Float di);
__bidevice__ void ComputePressureFor(SphSolverData3 *data, int particleId, Float di);

// Compute viscosity, gravity and drag forces to particle force vector
__bidevice__ void ComputeNonPressureForceFor(SphSolverData2 *data, int particleId);
__bidevice__ void ComputeNonPressureForceFor(SphSolverData3 *data, int particleId);
// Compute viscosity/gravity/drag/pressure forces in a single call
__bidevice__ void ComputeAllForcesFor(SphSolverData2 *data, int particleId,
                                      Float timeStep, int extended = 0);
__bidevice__ void ComputeAllForcesFor(SphSolverData3 *data, int particleId,
                                      Float timeStep, int extended = 0);
// Evolve particles
__bidevice__ void TimeIntegrationFor(SphSolverData2 *data, int particleId, 
                                     Float timeStep, int extended);
__bidevice__ void TimeIntegrationFor(SphSolverData3 *data, int particleId, 
                                     Float timeStep, int extended);
__bidevice__ void ComputeInitialTemperatureFor(SphSolverData2 *data, int particleId,
                                               Float Tmin, Float Tmax, int maxLevel);

// Generic calls for sph solvers wrappers
// Perform computation on CPU {easy debug}
__bidevice__ void ComputeDensityCPU(SphSolverData2 *data, int compute_pressure = 1);
__bidevice__ void ComputeDensityCPU(SphSolverData3 *data, int compute_pressure = 1);
__bidevice__ void ComputePressureForceCPU(SphSolverData2 *data, Float timeStep);
__bidevice__ void ComputePressureForceCPU(SphSolverData3 *data, Float timeStep);
__bidevice__ void TimeIntegrationCPU(SphSolverData2 *data, Float timeStep, int extended=0);
__bidevice__ void TimeIntegrationCPU(SphSolverData3 *data, Float timeStep, int extended=0);
__bidevice__ void ComputeInitialTemperatureMapCPU(SphSolverData2 *data, Float Tmin, 
                                                  Float Tmax, int maxLevel);

// Perform computation on GPU
__host__ void ComputeDensityGPU(SphSolverData2 *data, int compute_pressure = 1);
__host__ void ComputeDensityGPU(SphSolverData3 *data, int compute_pressure = 1);
__host__ void ComputePressureForceGPU(SphSolverData2 *data, Float timeStep);
__host__ void ComputePressureForceGPU(SphSolverData3 *data, Float timeStep);
__host__ void ComputeNonPressureForceGPU(SphSolverData2 *data);
__host__ void ComputeNonPressureForceGPU(SphSolverData3 *data);
__host__ void TimeIntegrationGPU(SphSolverData2 *data, Float timeStep, int extended=0);
__host__ void TimeIntegrationGPU(SphSolverData3 *data, Float timeStep, int extended=0);
__host__ void ComputeInitialTemperatureMapGPU(SphSolverData2 *data, Float Tmin, 
                                              Float Tmax, int maxLevel);

// Computes the average temperature of all particles
__bidevice__ Float ComputeAverageTemperature(SphSolverData2 *data);

// Generic update grid methods
__host__ void UpdateGridDistributionCPU(SphSolverData2 *data);
__host__ void UpdateGridDistributionGPU(SphSolverData2 *data);
__host__ void UpdateGridDistributionCPU(SphSolverData3 *data);
__host__ void UpdateGridDistributionGPU(SphSolverData3 *data);

// Generate a set of default SPH properties
__host__ SphSolverData2 *DefaultSphSolverData2();
__host__ SphSolverData3 *DefaultSphSolverData3();

// Display routines
__host__ void Debug_GraphyDisplaySolverParticles(ParticleSet2 *pSet);
__host__ void Debug_GraphyDisplaySolverParticles(ParticleSet2 *pSet, float *buffer);
__host__ void Debug_GraphyDisplaySolverParticles(ParticleSet2 *pSet, float *buffer,
                                                 float *colors);
__host__ void Debug_GraphyDisplayParticles(int n, float *buffer, float *colors, 
                                           Float pSize = 5.0);
