#include <sph_solver.h>

void SphSolverData3SetupFor(SphSolverData3 *solverData, int expectedParticleCount){
    solverData->smoothedVelocities = cudaAllocateVx(vec3f, expectedParticleCount);
}

bb_cpu_gpu SphSolver3::SphSolver3(){ solverData = nullptr; }

bb_cpu_gpu void SphSolver3::Initialize(SphSolverData3 *data){
    solverData = data;
}

bb_cpu_gpu SphSolverData3 *SphSolver3::GetSphSolverData(){
    return solverData;
}

void SphSolver3::SetViscosityCoefficient(Float viscosityCoefficient){
    AssertA(solverData, "Invalid solverData for {SetViscosityCoefficient}");
    solverData->viscosity = Max(0, viscosityCoefficient);
}

void SphSolver3::SetPseudoViscosityCoefficient(Float pseudoViscosityCoefficient){
    AssertA(solverData, "Invalid solverData for {SetPseudoViscosityCoefficient}");
    solverData->pseudoViscosity = Max(0, pseudoViscosityCoefficient);
}

bb_cpu_gpu Float SphSolver3::GetKernelRadius(){
    AssertA(solverData, "Invalid solverData for {GetKernelRadius}");
    AssertA(solverData->sphpSet, "No Particle set in solver data");
    return solverData->sphpSet->GetKernelRadius();
}

void SphSolver3::SetColliders(ColliderSet3 *col){
    AssertA(solverData, "Invalid solverData for {SetColliders}");
    solverData->collider = col;
}

ColliderSet3 *SphSolver3::GetColliders(){
    AssertA(solverData, "Invalid solverData for {GetColliders}");
    return solverData->collider;
}

bb_cpu_gpu SphParticleSet3 *SphSolver3::GetSphParticleSet(){
    AssertA(solverData, "Invalid solverData for {GetSphParticleSet}");
    return solverData->sphpSet;
}

void AdvanceTimeStep(SphSolver3 *solver, Float timeStep, int use_cpu = 0){
    SphSolverData3 *data = solver->solverData;
    if(use_cpu)
        UpdateGridDistributionCPU(data);
    else
        UpdateGridDistributionGPU(data);

    data->sphpSet->ResetHigherLevel();

    if(use_cpu)
        ComputeDensityCPU(data);
    else
        ComputeDensityGPU(data);

    if(use_cpu)
        ComputePressureForceCPU(data, timeStep);
    else
        ComputePressureForceGPU(data, timeStep);

    if(use_cpu)
        ComputePseudoViscosityInterpolationCPU(data, timeStep);
    else
        ComputePseudoViscosityInterpolationGPU(data, timeStep);
}

void SphSolver3::Advance(Float timeIntervalInSeconds){
    TimerList lnmTimer;
    unsigned int numberOfIntervals = 0;
    unsigned int numberOfIntervalsRunned = 0;
    Float remainingTime = timeIntervalInSeconds;

    SphSolverData3 *data = solverData;
    ParticleSet3 *pSet = data->sphpSet->GetParticleSet();
    Float h = data->sphpSet->GetTargetSpacing();

    ProfilerBeginStep();
    while(remainingTime > Epsilon){
        SphParticleSet3 *sphpSet = solverData->sphpSet;
        numberOfIntervals = sphpSet->ComputeNumberOfTimeSteps(remainingTime,
                                                              solverData->soundSpeed);
        Float timeStep = remainingTime / (Float)numberOfIntervals;
        AdvanceTimeStep(this, timeStep, GetSystemUseCPU());
        remainingTime -= timeStep;
        numberOfIntervalsRunned += 1;
        ProfilerIncreaseStepIteration();
    }

    ProfilerEndStep();
    stepInterval = ProfilerGetStepInterval();

    pSet->ClearDataBuffer(&pSet->v0s);
    data->domain->UpdateQueryState();

    lnmTimer.Start();
    LNMBoundary(pSet, data->domain, h, 0);
    lnmTimer.Stop();
}

void SphSolver3::Setup(Float targetDensity, Float targetSpacing, Float relativeRadius,
                       Grid3 *dom, SphParticleSet3 *pSet)
{
    solverData->domain = dom;
    solverData->sphpSet = pSet;
    AssertA(solverData->domain && solverData->sphpSet, "Invalid SphSolver3 initialization");
    solverData->sphpSet->SetTargetDensity(targetDensity);
    solverData->sphpSet->SetTargetSpacing(targetSpacing);
    solverData->sphpSet->SetRelativeKernelRadius(relativeRadius);
    ParticleSet3 *pData = solverData->sphpSet->GetParticleSet();

    Float rad  = pData->GetRadius();
    Float mass = pData->GetMass();
    int pCount = pData->GetReservedSize();
    SphSolverData3SetupFor(solverData, pCount);

    printf("[SPH SOLVER]Radius : %g Spacing: %g, Particle Count: %d\n",
           rad, targetSpacing, pCount);

    // Perform a particle distribution so that distribution
    // during simulation can be optmized
    for(int i = 0; i < solverData->domain->GetCellCount(); i++){
        solverData->domain->DistributeResetCell(i);
    }
    solverData->domain->DistributeByParticle(pData);
    solverData->sphpSet->ResetHigherLevel();
    solverData->frame_index = 1;
}

SphSolverData3 *DefaultSphSolverData3(){
    SphSolverData3 *data = cudaAllocateVx(SphSolverData3, 1);
    data->eosExponent = 7.0;
    data->negativePressureScale = 0.0;
    data->viscosity = 0.04;
    data->pseudoViscosity = 10.0;
    data->soundSpeed = 100.0;
    data->timestepScale = 1.0;
    data->sphpSet = nullptr;
    data->collider = nullptr;
    data->dragCoefficient = 0.0001;
    data->frame_index = 0;
    data->Tmin = 1;
    data->Tmax = 20;
    data->Tamb = 0;
    return data;
}
