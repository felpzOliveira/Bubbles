#include <pbf_solver.h>

__host__ PbfSolver2::PbfSolver2(){}

__host__ void PbfSolver2::Initialize(SphSolverData2 *data){
    solverData = cudaAllocateVx(PbfSolverData2, 1);
    memset(solverData, 0x00, sizeof(PbfSolverData2));
    solverData->sphData = data;
    data->pseudoViscosity = 1.0;
    solverData->lambdaRelax = 10.0;
    solverData->antiClustDenom = 0.2;
    solverData->antiClustStr = 1e-6;
    solverData->antiClustExp = 4.0;
    solverData->vorticityStr = 4.0;
    predictIterations = 10;
}

__bidevice__ SphSolverData2 *PbfSolver2::GetSphSolverData(){
    AssertA(solverData, "Invalid solverData for {GetSphSolverData}");
    return solverData->sphData;
}

__bidevice__ SphParticleSet2 *PbfSolver2::GetSphParticleSet(){
    AssertA(solverData, "Invalid solverData for {GetSphSolverData}");
    AssertA(solverData->sphData, "Invalid solverData for {GetSphSolverData}");
    return solverData->sphData->sphpSet;
}

__host__ void PbfSolver2::SetColliders(ColliderSet2 *colliders){
    AssertA(solverData, "Invalid solverData for {SetColliders}");
    AssertA(solverData->sphData, "Invalid solverData for {SetColliders}");
    solverData->sphData->collider = colliders;
}

__host__ LNMStats PbfSolver2::GetLNMStats(){
    return lnmStats;
}

__host__ Float PbfSolver2::GetAdvanceTime(){
    return stepInterval;
}

__host__ int PbfSolver2::GetParticleCount(){
    return solverData->sphData->sphpSet->GetParticleSet()->GetParticleCount();
}

__host__ void PbfSolver2::Setup(Float targetDensity, Float targetSpacing,
                                Float relativeRadius, Grid2 *domain, 
                                SphParticleSet2 *pSet)
{
    AssertA(solverData, "Invalid call to {PbfSolver2::Setup}");
    SphSolverData2 *sphData = solverData->sphData;
    sphData->domain = domain;
    sphData->sphpSet = pSet;
    AssertA(sphData->domain && sphData->sphpSet, "Invalid PbfSolver2 initialization");
    
    sphData->sphpSet->SetTargetDensity(targetDensity);
    sphData->sphpSet->SetTargetSpacing(targetSpacing);
    sphData->sphpSet->SetRelativeKernelRadius(relativeRadius);
    
    ParticleSet2 *pData = sphData->sphpSet->GetParticleSet();
    Float mass = pData->GetMass();
    int pCount = pData->GetReservedSize();
    int actualCount = pData->GetParticleCount();
    
    SphSolverData2SetupFor(sphData, pCount);
    
    Float *mem  = cudaAllocateVx(Float, 3 * pCount);
    vec2f *vmem = cudaAllocateVx(vec2f, pCount);
    solverData->originalPositions = &vmem[0];
    solverData->densities         = &mem[0];
    solverData->lambdas           = &mem[pCount];
    solverData->w                 = &mem[2 * pCount];
    
    printf("[PBF SOLVER]Spacing: %g, Particle Count: %d\n", targetSpacing, pCount);
    
    for(int i = 0; i < sphData->domain->GetCellCount(); i++){
        sphData->domain->DistributeResetCell(i);
    }
    sphData->domain->DistributeByParticle(pData);
    sphData->frame_index = 1;
}


__host__ void AdvanceTimeStep(PbfSolver2 *solver, Float timeStep, 
                              int use_cpu = 0)
{
    PbfSolverData2 *pbfData = solver->solverData;
    SphSolverData2 *data = pbfData->sphData;
    
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
        ComputePressureForceCPU(data, timeStep, 0);
    else
        ComputePressureForceGPU(data, timeStep, 0);
    
    AdvancePBF(pbfData, timeStep, solver->predictIterations, use_cpu);
}


__host__ void PbfSolver2::Advance(Float timeIntervalInSeconds){
    TimerList lnmTimer;
    unsigned int numberOfIntervals = 0;
    Float remainingTime = timeIntervalInSeconds;
    
    SphSolverData2 *data = solverData->sphData;
    ParticleSet2 *pSet = data->sphpSet->GetParticleSet();
    Float h = data->sphpSet->GetTargetSpacing();
    
    ProfilerBeginStep();
    while(remainingTime > Epsilon){
        SphParticleSet2 *sphpSet = data->sphpSet;
        numberOfIntervals = sphpSet->ComputeNumberOfTimeSteps(remainingTime,
                                                              data->soundSpeed, 2);
        Float timeStep = remainingTime / (Float)numberOfIntervals;
        AdvanceTimeStep(this, timeStep);
        remainingTime -= timeStep;
        ProfilerIncreaseStepIteration();
    }
    
    ProfilerEndStep();
    stepInterval = ProfilerGetStepInterval();
    
    pSet->ClearDataBuffer(&pSet->v0s);
    data->domain->UpdateQueryState();
    
    lnmTimer.Start();
    LNMBoundary(pSet, data->domain, h, 0);
    lnmTimer.Stop();
    
    Float lnm = lnmTimer.GetElapsedGPU(0);
    Float pct = lnm * 100.0 / stepInterval;
    lnmStats.Add(LNMData(lnm, pct));
}
