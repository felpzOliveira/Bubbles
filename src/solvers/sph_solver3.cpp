#include <sph_solver.h>

__bidevice__ SphSolver3::SphSolver3(){ solverData = nullptr; }

__bidevice__ void SphSolver3::Initialize(SphSolverData3 *data){
    solverData = data;
}

__bidevice__ SphSolverData3 *SphSolver3::GetSphSolverData(){
    return solverData;
}

__host__ void SphSolver3::SetViscosityCoefficient(Float viscosityCoefficient){
    AssertA(solverData, "Invalid solverData for {SetViscosityCoefficient}");
    solverData->viscosity = Max(0, viscosityCoefficient);
}

__host__ void SphSolver3::SetPseudoViscosityCoefficient(Float pseudoViscosityCoefficient){
    AssertA(solverData, "Invalid solverData for {SetPseudoViscosityCoefficient}");
    solverData->pseudoViscosity = Max(0, pseudoViscosityCoefficient);
}

__bidevice__ Float SphSolver3::GetKernelRadius(){
    AssertA(solverData, "Invalid solverData for {GetKernelRadius}");
    AssertA(solverData->sphpSet, "No Particle set in solver data");
    return solverData->sphpSet->GetKernelRadius();
}

__host__ void SphSolver3::SetColliders(ColliderSet3 *col){
    AssertA(solverData, "Invalid solverData for {SetColliders}");
    solverData->collider = col;
}

__bidevice__ SphParticleSet3 *SphSolver3::GetSphParticleSet(){
    AssertA(solverData, "Invalid solverData for {GetSphParticleSet}");
    return solverData->sphpSet;
}


__host__ void PrintSphTimers(TimerList *timers);
__host__ void AdvanceTimeStep(SphSolver3 *solver, Float timeStep, 
                              int use_cpu = 0)
{
    TimerList timers;
    SphSolverData3 *data = solver->solverData;
    timers.Start();
    if(use_cpu)
        UpdateGridDistributionCPU(data);
    else
        UpdateGridDistributionGPU(data);
    timers.StopAndNext();
    
    data->sphpSet->ResetHigherLevel();
    
    if(use_cpu)
        ComputeDensityCPU(data);
    else
        ComputeDensityGPU(data);
    
    timers.StopAndNext();
    if(use_cpu)
        ComputePressureForceCPU(data, timeStep);
    else
        ComputePressureForceGPU(data, timeStep);
    timers.Stop();
    
#if defined(PRINT_TIMER)
    PrintSphTimers(&timers);
#endif
    timers.Reset();
}

__host__ void SphSolver3::Advance(Float timeIntervalInSeconds){
    unsigned int numberOfIntervals = 0;
    unsigned int numberOfIntervalsRunned = 0;
    Float remainingTime = timeIntervalInSeconds;
    
    TimerList timers;
    SphSolverData3 *data = solverData;
    timers.Start();
    while(remainingTime > Epsilon){
        SphParticleSet3 *sphpSet = solverData->sphpSet;
        numberOfIntervals = sphpSet->ComputeNumberOfTimeSteps(remainingTime,
                                                              solverData->soundSpeed);
        Float timeStep = remainingTime / (Float)numberOfIntervals;
        AdvanceTimeStep(this, timeStep);
        remainingTime -= timeStep;
        numberOfIntervalsRunned += 1;
    }
    
    CNMInvalidateCells(data->domain);
    
    timers.StopAndNext();
    CNMClassifyLazyGPU(data->domain);
    timers.Stop();
    
#ifdef PRINT_TIMER
    Float adv = timers.GetElapsedCPU(0);
    Float cnm = timers.GetElapsedCPU(1);
    Float pct = cnm * 100.0 / adv;
    printf("\nAdvance [%d] {%g} CNM {%g} [%g%%]\n", numberOfIntervalsRunned, adv, cnm, pct);
    fflush(stdout);
#endif
    
    timers.Reset();
}

__host__ void SphSolver3::Setup(Float targetDensity, Float targetSpacing, 
                                Float relativeRadius, Grid3 *dom, SphParticleSet3 *pSet)
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
    int pCount = pData->GetParticleCount();
    
    printf("Radius : %g  Mass: %g  Density: %g  Spacing: %g, Particle Count: %d\n", 
           rad, mass, targetDensity, targetSpacing, pCount);
    
    // TODO: Add multi-cell neighbor querying
    vec3f len = solverData->domain->GetCellSize();
    Float minLen = Min(len[0], Min(len[1], len[2]));
    AssertA(minLen > solverData->sphpSet->GetKernelRadius(),
            "Spacing is too large for single neighbor querying");
    
    // Perform a particle distribution so that distribution
    // during simulation can be optmized
    for(int i = 0; i < solverData->domain->GetCellCount(); i++){
        solverData->domain->DistributeResetCell(i);
    }
    solverData->domain->DistributeByParticle(pData);
    solverData->sphpSet->ResetHigherLevel();
    solverData->frame_index = 1;
}

__host__ SphSolverData3 *DefaultSphSolverData3(){
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
