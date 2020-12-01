#include <sph_solver.h>
#include <cutil.h>
#include <shape.h>

__host__ void SphSolverData2SetupFor(SphSolverData2 *solverData,
                                     int expectedParticleCount)
{
    solverData->smoothedVelocities = cudaAllocateVx(vec2f, expectedParticleCount);
}

__bidevice__ SphSolver2::SphSolver2(){ solverData = nullptr; }

__bidevice__ void SphSolver2::Initialize(SphSolverData2 *data){
    solverData = data;
}

__bidevice__ SphSolverData2 *SphSolver2::GetSphSolverData(){
    return solverData;
}

__host__ void SphSolver2::SetViscosityCoefficient(Float viscosityCoefficient){
    AssertA(solverData, "Invalid solverData for {SetViscosityCoefficient}");
    solverData->viscosity = Max(0, viscosityCoefficient);
}

__host__ void SphSolver2::SetPseudoViscosityCoefficient(Float pseudoViscosityCoefficient){
    AssertA(solverData, "Invalid solverData for {SetPseudoViscosityCoefficient}");
    solverData->pseudoViscosity = Max(0, pseudoViscosityCoefficient);
}

__bidevice__ Float SphSolver2::GetKernelRadius(){
    AssertA(solverData, "Invalid solverData for {GetKernelRadius}");
    AssertA(solverData->sphpSet, "No Particle set in solver data");
    return solverData->sphpSet->GetKernelRadius();
}

__host__ void SphSolver2::SetColliders(ColliderSet2 *col){
    AssertA(solverData, "Invalid solverData for {SetColliders}");
    solverData->collider = col;
}

__bidevice__ SphParticleSet2 *SphSolver2::GetSphParticleSet(){
    AssertA(solverData, "Invalid solverData for {GetSphParticleSet}");
    return solverData->sphpSet;
}

__host__ void AdvanceTimeStep(SphSolver2 *solver, Float timeStep, 
                              int use_cpu = 0)
{
    SphSolverData2 *data = solver->solverData;
    
    if(use_cpu)
        UpdateGridDistributionCPU(data);
    else
        UpdateGridDistributionGPU(data);
    
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

__host__ void SphSolver2::Advance(Float timeIntervalInSeconds){
    TimerList lnmTimer;
    unsigned int numberOfIntervals = 0;
    unsigned int numberOfIntervalsRunned = 0;
    Float remainingTime = timeIntervalInSeconds;
    
    SphSolverData2 *data = solverData;
    ParticleSet2 *pSet = data->sphpSet->GetParticleSet();
    Float h = data->sphpSet->GetTargetSpacing();
    
    ProfilerBeginStep();
    
    while(remainingTime > Epsilon){
        SphParticleSet2 *sphpSet = solverData->sphpSet;
        numberOfIntervals = sphpSet->ComputeNumberOfTimeSteps(remainingTime,
                                                              solverData->soundSpeed);
        Float timeStep = remainingTime / (Float)numberOfIntervals;
        AdvanceTimeStep(this, timeStep);
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


__host__ void SphSolver2::Setup(Float targetDensity, Float targetSpacing, 
                                Float relativeRadius, Grid2 *dom, SphParticleSet2 *pSet)
{
    solverData->domain = dom;
    solverData->sphpSet = pSet;
    AssertA(solverData->domain && solverData->sphpSet, "Invalid SphSolver2 initialization");
    solverData->sphpSet->SetTargetDensity(targetDensity);
    solverData->sphpSet->SetTargetSpacing(targetSpacing);
    solverData->sphpSet->SetRelativeKernelRadius(relativeRadius);
    ParticleSet2 *pData = solverData->sphpSet->GetParticleSet();
    
    Float rad  = pData->GetRadius();
    Float mass = pData->GetMass();
    int pCount = pData->GetReservedSize();
    SphSolverData2SetupFor(solverData, pCount);
    
    printf("[SPH SOLVER]Radius : %g Spacing: %g, Particle Count: %d\n", 
           rad, targetSpacing, pCount);
    
    // Perform a particle distribution so that distribution
    // during simulation can be optmized
    solverData->domain->DistributeByParticle(pData);
    solverData->frame_index = 1;
}

__host__ void SphSolver2::UpdateDensity(){
    UpdateGridDistributionCPU(solverData);
    ComputeDensityCPU(solverData);
}

__host__ SphSolverData2 *DefaultSphSolverData2(){
    SphSolverData2 *data = cudaAllocateVx(SphSolverData2, 1);
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


#include <graphy.h>
__host__ void Debug_GraphyDisplayParticles(int n, float *buffer, float *colors, Float pSize){
    graphy_render_points_size(buffer, colors, pSize, n, -1, 1, 1, -1);
}

__host__ void Debug_GraphyDisplaySolverParticles(ParticleSet2 *pSet, float *buffer){
    AssertA(pSet, "Invalid SPHParticle pointer for Debug Display");
    for(int i = 0; i < pSet->GetParticleCount(); i++){
        vec2f pi = pSet->GetParticlePosition(i);
        buffer[3 * i + 0] = pi[0];
        buffer[3 * i + 1] = pi[1];
        buffer[3 * i + 2] = 0;
    }
    
    float rgb[3] = {1,0,0};
    // TODO: Domain size
    graphy_render_points(buffer, rgb, pSet->GetParticleCount(),-2,2,2,-2);
    printf("Press anything...");
    getchar();
}

__host__ void Debug_GraphyDisplaySolverParticles(ParticleSet2 *pSet, float *buffer,
                                                 float *colors)
{
    AssertA(pSet, "Invalid SPHParticle pointer for Debug Display");
    Float pSize = 2.5;
    
    for(int i = 0; i < pSet->GetParticleCount(); i++){
        vec2f pi = pSet->GetParticlePosition(i);
        buffer[3 * i + 0] = pi[0];
        buffer[3 * i + 1] = pi[1];
        buffer[3 * i + 2] = 0;
    }
    
    // TODO: Domain size
    graphy_render_points_size(buffer, colors, pSize, pSet->GetParticleCount(),-1,1,1,-1);
}

__host__ void Debug_GraphyDisplaySolverParticles(ParticleSet2 *pSet){
    AssertA(pSet, "Invalid SPHParticle pointer for Debug Display");
    float *position = new float[pSet->GetParticleCount() * 3];
    Debug_GraphyDisplaySolverParticles(pSet, position);
    delete[] position;
}