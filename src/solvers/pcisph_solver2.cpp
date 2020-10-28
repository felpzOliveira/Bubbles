#include <pcisph_solver.h>
#include <shape.h>
#include <point_generator.h>

Float kDefaultTimeStepLimitScale = 5.0;

__bidevice__ PciSphSolver2::PciSphSolver2(){}

__host__ void PciSphSolver2::Initialize(SphSolverData2 *data){
    solverData = cudaAllocateVx(PciSphSolverData2, 1);
    solverData->sphData = data;
    maxIterations = 5;
    maxErrorDensity = 0.001;
}

__bidevice__ SphSolverData2 *PciSphSolver2::GetSphSolverData(){
    AssertA(solverData, "Invalid solverData for {GetSphSolverData}");
    return solverData->sphData;
}

__bidevice__ SphParticleSet2 *PciSphSolver2::GetSphParticleSet(){
    AssertA(solverData, "Invalid solverData for {GetSphSolverData}");
    AssertA(solverData->sphData, "Invalid solverData for {GetSphSolverData}");
    return solverData->sphData->sphpSet;
}

__host__ void PciSphSolver2::SetColliders(ColliderSet2 *colliders){
    AssertA(solverData, "Invalid solverData for {SetColliders}");
    AssertA(solverData->sphData, "Invalid solverData for {SetColliders}");
    solverData->sphData->collider = colliders;
}

__host__ void PrintPciSphTimers(TimerList *timers){
    Float update      = timers->GetElapsedGPU(0);
    Float density     = timers->GetElapsedGPU(1);
    Float nonPressure = timers->GetElapsedGPU(2);
    Float pressure    = timers->GetElapsedGPU(3);
    
    printf("\rUpdate {%g} Density {%g} NPress {%g} Press {%g}    ",
           update, density, nonPressure, pressure);
    fflush(stdout);
}

__host__ void AdvanceTimeStep(PciSphSolver2 *solver, Float timeStep, 
                              int use_cpu = 0)
{
    SphSolverData2 *data = solver->solverData->sphData;
    TimerList timers;
    
    timers.Start();
    if(use_cpu)
        UpdateGridDistributionCPU(data);
    else
        UpdateGridDistributionGPU(data);
    timers.StopAndNext();
    
    if(use_cpu)
        ComputeDensityCPU(data);
    else
        ComputeDensityGPU(data);
    timers.StopAndNext();
    
    //TODO: Implement CPU version for 2D?
    ComputeNonPressureForceGPU(data);
    
    timers.StopAndNext();
    
    Float delta = solver->ComputeDelta(timeStep);
    ComputePressureForceAndIntegrate(solver->solverData, timeStep, 0.01, delta, 5);
    
    timers.Stop();
    
#if defined(PRINT_TIMER)
    //PrintPciSphTimers(&timers);
#endif
    
}


__host__ void PciSphSolver2::Advance(Float timeIntervalInSeconds){
    unsigned int numberOfIntervals = 0;
    unsigned int numberOfIntervalsRunned = 0;
    Float remainingTime = timeIntervalInSeconds;
    
    SphSolverData2 *data = solverData->sphData;
    TimerList timers;
    timers.Start();
    while(remainingTime > Epsilon){
        SphParticleSet2 *sphpSet = data->sphpSet;
        numberOfIntervals = sphpSet->ComputeNumberOfTimeSteps(remainingTime,
                                                              data->soundSpeed,
                                                              kDefaultTimeStepLimitScale);
        Float timeStep = remainingTime / (Float)numberOfIntervals;
        AdvanceTimeStep(this, timeStep);
        remainingTime -= timeStep;
        numberOfIntervalsRunned += 1;
    }
    
    CNMInvalidateCells(data->domain);
    timers.StopAndNext();
    int n = CNMClassifyLazyGPU(data->domain);
    timers.Stop();
#ifdef PRINT_TIMER
    Float adv = timers.GetElapsedCPU(0);
    Float cnm = timers.GetElapsedCPU(1);
    Float pct = cnm * 100.0 / adv;
    printf("\rAdvance [%d] {%g} CNM [%d] {%g} [%g%%]    ", 
           numberOfIntervalsRunned, adv, n, cnm, pct);
    fflush(stdout);
#endif
}

__host__ void PciSphSolver2::Setup(Float targetDensity, Float targetSpacing,
                                   Float relativeRadius, Grid2 *domain, SphParticleSet2 *pSet)
{
    AssertA(solverData, "Invalid call to {PciSphSolver2::Setup}");
    SphSolverData2 *sphData = solverData->sphData;
    sphData->domain = domain;
    sphData->sphpSet = pSet;
    AssertA(sphData->domain && sphData->sphpSet, "Invalid PciSphSolver2 initialization");
    sphData->sphpSet->SetTargetDensity(targetDensity);
    sphData->sphpSet->SetTargetSpacing(targetSpacing);
    sphData->sphpSet->SetRelativeKernelRadius(relativeRadius);
    
    ParticleSet2 *pData = sphData->sphpSet->GetParticleSet();
    Float rad  = pData->GetRadius();
    Float mass = pData->GetMass();
    int pCount = pData->GetReservedSize();
    int actualCount = pData->GetParticleCount();
    
    vec2f len = sphData->domain->GetCellSize();
    Float minLen = Min(len[0], len[1]);
    AssertA(minLen > sphData->sphpSet->GetKernelRadius(),
            "Spacing is too large for single neighbor querying");
    
    solverData->refMemory = cudaAllocateVx(vec2f, 3 * pCount);
    solverData->densityErrors    = cudaAllocateVx(Float, 2 * pCount);
    solverData->densityPredicted = &solverData->densityErrors[pCount];
    solverData->tempPositions    = &solverData->refMemory[0];
    solverData->tempVelocities   = &solverData->refMemory[1*pCount];
    solverData->pressureForces   = &solverData->refMemory[2*pCount];
    
    massOverTargetDensitySquared = mass / targetDensity;
    massOverTargetDensitySquared *= massOverTargetDensitySquared;
    deltaDenom = ComputeDeltaDenom();
    
    printf("Radius : %g  Mass: %g  Density: %g  Spacing: %g, Particle Count: %d, Delta: %g\n", 
           rad, mass, targetDensity, targetSpacing, actualCount, deltaDenom);
    
    // Perform a particle distribution so that distribution
    // during simulation can be optmized
    for(int i = 0; i < sphData->domain->GetCellCount(); i++){
        sphData->domain->DistributeResetCell(i);
    }
    sphData->domain->DistributeByParticle(pData);
    sphData->frame_index = 1;
}

__host__ Float PciSphSolver2::ComputeDeltaDenom(){
    TrianglePointGenerator generator;
    std::vector<vec2f> points;
    
    SphParticleSet2 *sphData = solverData->sphData->sphpSet;
    Float kernelRadius = sphData->GetKernelRadius();
    Float spacing = sphData->GetTargetSpacing();
    Float kernelRadius2 = kernelRadius * kernelRadius;
    
    Bounds2f domain(vec2f(-1.5 * kernelRadius), vec2f(1.5 * kernelRadius));
    generator.Generate(domain, spacing, &points);
    
    SphSpikyKernel2 kernel(kernelRadius);
    Float denom = 0;
    Float denom2 = 0;
    vec2f denom1;
    
    
    for(int i = 0; i < points.size(); i++){
        vec2f point = points[i];
        Float distanceSquared = point.LengthSquared();
        if(distanceSquared < kernelRadius2){
            Float distance = sqrt(distanceSquared);
            vec2f direction = (distance > 0) ? point / distance : vec2f(0,0);
            
            vec2f gradij = kernel.gradW(distance, direction);
            denom1 += gradij;
            denom2 += Dot(gradij, gradij);
        }
    }
    
    denom += -Dot(denom1, denom1) - denom2;
    return denom;
}

__host__ Float PciSphSolver2::ComputeDelta(Float timeIntervalInSeconds){
    return Absf(deltaDenom) > 0 ? -1 / (ComputeBeta(timeIntervalInSeconds) * deltaDenom) : 0;
}

__host__ Float PciSphSolver2::ComputeBeta(Float timeIntervalInSeconds){
    Float timeStepSquare = timeIntervalInSeconds * timeIntervalInSeconds;
    return 2.0 * massOverTargetDensitySquared * timeStepSquare;
}
