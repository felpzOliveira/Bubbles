/* date = December 2nd 2020 7:33 pm */

#ifndef PBF_SOLVER_H
#define PBF_SOLVER_H

/*
* I think it is working, but I don't like it. Very unstable, not gonna do a 3D.
*/

#include <vector>
#include <sph_solver.h>
#include <functional>
#include <statics.h>

typedef struct{
    SphSolverData2 *sphData;
    vec2f *originalPositions;
    Float *densities;
    Float *lambdas;
    Float *w;
    Float lambdaRelax;
    Float antiClustDenom;
    Float antiClustStr;
    Float antiClustExp;
}PbfSolverData2;

class PbfSolver2{
    public:
    PbfSolverData2 *solverData;
    unsigned int predictIterations;
    Float stepInterval;
    LNMStats lnmStats;
    
    __host__ PbfSolver2();
    __host__ void Initialize(SphSolverData2 *data);
    __host__ void Setup(Float targetDensity, Float targetSpacing, Float relativeRadius,
                        Grid2 *domain, SphParticleSet2 *pSet);
    __host__ void SetColliders(ColliderSet2 *colliders);
    __bidevice__ SphSolverData2 *GetSphSolverData();
    __bidevice__ SphParticleSet2 *GetSphParticleSet();
    __host__ void Advance(Float timeIntervalInSeconds);
    __host__ LNMStats GetLNMStats();
    __host__ Float GetAdvanceTime();
    __host__ int GetParticleCount();
};

__host__ void ComputePredictedPositionsCPU(PbfSolverData2 *data,
                                           Float timeIntervalInSeconds);
__host__ void ComputePredictedPositionsGPU(PbfSolverData2 *data,
                                           Float timeIntervalInSeconds);

__host__ void ComputeLambdaGPU(PbfSolverData2 *data);
__host__ void ComputeLambdaCPU(PbfSolverData2 *data);

__host__ void ComputeDeltaPCPU(PbfSolverData2 *data, Float timeIntervalInSeconds);
__host__ void ComputeDeltaPGPU(PbfSolverData2 *data, Float timeIntervalInSeconds);

__host__ void AdvancePBF(PbfSolverData2 *data, Float timeIntervalInSeconds,
                         unsigned int predictIterations, int use_cpu);

#endif //PBF_SOLVER_H
