#include <pcisph_solver.h>
#include <util.h>

extern const Float kDefaultTimeStepLimitScale;

__host__ PciSphSolver3::PciSphSolver3(){}

__host__ void PciSphSolver3::Initialize(SphSolverData3 *data){
    solverData = cudaAllocateVx(PciSphSolverData3, 1);
    solverData->sphData = data;
    maxIterations = 5;
    maxErrorDensity = 0.01;
}

__bidevice__ SphSolverData3 *PciSphSolver3::GetSphSolverData(){
    AssertA(solverData, "Invalid solverData for {GetSphSolverData}");
    return solverData->sphData;
}

__bidevice__ SphParticleSet3 *PciSphSolver3::GetSphParticleSet(){
    AssertA(solverData, "Invalid solverData for {GetSphSolverData}");
    AssertA(solverData->sphData, "Invalid solverData for {GetSphSolverData}");
    return solverData->sphData->sphpSet;
}

__host__ void PciSphSolver3::SetColliders(ColliderSet3 *colliders){
    AssertA(solverData, "Invalid solverData for {SetColliders}");
    AssertA(solverData->sphData, "Invalid solverData for {SetColliders}");
    solverData->sphData->collider = colliders;
}

__host__ void PrintPciSphTimers(TimerList *timers);

__host__ void AdvanceTimeStep(PciSphSolver3 *solver, Float timeStep, 
                              int use_cpu = 0)
{
    SphSolverData3 *data = solver->solverData->sphData;
    solver->stepTimer.Reset();
    
    solver->stepTimer.Start();
    if(use_cpu)
        UpdateGridDistributionCPU(data);
    else
        UpdateGridDistributionGPU(data);
    solver->stepTimer.StopAndNext();
    
    data->sphpSet->ResetHigherLevel();
    
    if(use_cpu)
        ComputeDensityCPU(data);
    else
        ComputeDensityGPU(data);
    solver->stepTimer.StopAndNext();
    
    if(use_cpu)
        ComputeNonPressureForceCPU(data);
    else
        ComputeNonPressureForceGPU(data);
    
    solver->stepTimer.StopAndNext();
    
    Float delta = solver->ComputeDelta(timeStep);
    ComputePressureForceAndIntegrate(solver->solverData, timeStep, 
                                     0.01, delta, 5, use_cpu);
    
    solver->stepTimer.Stop();
}

__host__ void PciSphSolver3::Advance(Float timeIntervalInSeconds){
    TimerList cnmTimer;
    unsigned int numberOfIntervals = 0;
    Float remainingTime = timeIntervalInSeconds;
    SphSolverData3 *data = solverData->sphData;
    ParticleSet3 *pSet = data->sphpSet->GetParticleSet();
    Float h = data->sphpSet->GetKernelRadius();
    
    advanceTimer.Reset();
    advanceTimer.Start();
    while(remainingTime > Epsilon){
        SphParticleSet3 *sphpSet = data->sphpSet;
        numberOfIntervals = sphpSet->ComputeNumberOfTimeSteps(remainingTime,
                                                              data->soundSpeed,
                                                              kDefaultTimeStepLimitScale);
        Float timeStep = remainingTime / (Float)numberOfIntervals;
        AdvanceTimeStep(this, timeStep);
        remainingTime -= timeStep;
    }
    
    advanceTimer.Stop();
    
    data->domain->UpdateQueryState();
    CNMInvalidateCells(data->domain);
    pSet->ClearDataBuffer(&pSet->v0s);
    
    Float elapsed = advanceTimer.GetElapsedCPU(0);
    cnmTimer.Start();
    CNMBoundary(pSet, data->domain, h, 0);
    cnmTimer.Stop();
    
    Float cnm = cnmTimer.GetElapsedGPU(0);
    Float pct = cnm * 100.0 / elapsed;
    cnmStats.Add(CNMData(cnm, pct));
}

__host__ Float PciSphSolver3::GetAdvanceTime(){
    return advanceTimer.GetElapsedCPU(0);
}

__host__ CNMStats PciSphSolver3::GetCNMStats(){
    return cnmStats;
}

__host__ int PciSphSolver3::GetParticleCount(){
    return solverData->sphData->sphpSet->GetParticleSet()->GetParticleCount();
}

__host__ void PciSphSolver3::Setup(Float targetDensity, Float targetSpacing,
                                   Float relativeRadius, Grid3 *domain, SphParticleSet3 *pSet)
{
    AssertA(solverData, "Invalid call to {PciSphSolver3::Setup}");
    SphSolverData3 *sphData = solverData->sphData;
    sphData->domain = domain;
    sphData->sphpSet = pSet;
    AssertA(sphData->domain && sphData->sphpSet, "Invalid PciSphSolver3 initialization");
    sphData->sphpSet->SetTargetDensity(targetDensity);
    sphData->sphpSet->SetTargetSpacing(targetSpacing);
    sphData->sphpSet->SetRelativeKernelRadius(relativeRadius);
    
    ParticleSet3 *pData = sphData->sphpSet->GetParticleSet();
    Float rad  = pData->GetRadius();
    Float mass = pData->GetMass();
    int pCount = pData->GetReservedSize();
    int actualCount = pData->GetParticleCount();
    
    vec3f len = sphData->domain->GetCellSize();
    Float minLen = Min(len[0], Min(len[1], len[2]));
    AssertA(minLen > sphData->sphpSet->GetKernelRadius(),
            "Spacing is too large for single neighbor querying");
    
    solverData->refMemory = cudaAllocateVx(vec3f, 3 * pCount);
    solverData->densityErrors    = cudaAllocateVx(Float, 2 * pCount);
    solverData->densityPredicted = &solverData->densityErrors[pCount];
    solverData->tempPositions    = &solverData->refMemory[0];
    solverData->tempVelocities   = &solverData->refMemory[1*pCount];
    solverData->pressureForces   = &solverData->refMemory[2*pCount];
    
    massOverTargetDensitySquared = mass / targetDensity;
    massOverTargetDensitySquared *= massOverTargetDensitySquared;
    deltaDenom = ComputeDeltaDenom();
    
    printf("[PCISPH SOLVER]Radius : %g Spacing: %g, Particle Count: %d, Delta: %g\n", 
           rad, targetSpacing, actualCount, deltaDenom);
    
    // Perform a particle distribution so that distribution
    // during simulation can be optmized
    for(int i = 0; i < sphData->domain->GetCellCount(); i++){
        sphData->domain->DistributeResetCell(i);
    }
    sphData->domain->DistributeByParticle(pData);
    sphData->sphpSet->ResetHigherLevel();
    sphData->frame_index = 1;
}

__host__ Float PciSphSolver3::ComputeDeltaDenom(){
    BccLatticePointGenerator generator;
    std::vector<vec3f> points;
    
    SphParticleSet3 *sphData = solverData->sphData->sphpSet;
    Float kernelRadius = sphData->GetKernelRadius();
    Float spacing = sphData->GetTargetSpacing();
    Float kernelRadius2 = kernelRadius * kernelRadius;
    
    Bounds3f domain(vec3f(-1.5 * kernelRadius), vec3f(1.5 * kernelRadius));
    generator.Generate(domain, spacing, &points);
    
    SphSpikyKernel3 kernel(kernelRadius);
    Float denom = 0;
    Float denom2 = 0;
    vec3f denom1;
    
    for(int i = 0; i < points.size(); i++){
        vec3f point = points[i];
        Float distanceSquared = point.LengthSquared();
        if(distanceSquared < kernelRadius2){
            Float distance = sqrt(distanceSquared);
            vec3f direction = (distance > 0) ? point / distance : vec3f(0);
            
            vec3f gradij = kernel.gradW(distance, direction);
            denom1 += gradij;
            denom2 += Dot(gradij, gradij);
        }
    }
    
    denom += -Dot(denom1, denom1) - denom2;
    return denom;
}

__host__ Float PciSphSolver3::ComputeDelta(Float timeIntervalInSeconds){
    return Absf(deltaDenom) > 0 ? -1 / (ComputeBeta(timeIntervalInSeconds) * deltaDenom) : 0;
}

__host__ Float PciSphSolver3::ComputeBeta(Float timeIntervalInSeconds){
    Float timeStepSquare = timeIntervalInSeconds * timeIntervalInSeconds;
    return 2.0 * massOverTargetDensitySquared * timeStepSquare;
}

__host__ int EmptyCallback(int){ return 1; }

__host__ void PciSphRunSimulation3(PciSphSolver3 *solver, Float spacing,
                                   vec3f origin, vec3f target, 
                                   Float targetInterval, std::vector<Shape*> sdfs,
                                   const std::function<int(int )> &callback)
{
    SphParticleSet3 *sphSet = solver->GetSphParticleSet();
    ParticleSet3 *pSet = sphSet->GetParticleSet();
    return UtilRunSimulation3<PciSphSolver3, ParticleSet3>(solver, pSet,  spacing, 
                                                           origin, target, 
                                                           targetInterval, sdfs,
                                                           callback);
}