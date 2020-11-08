#include <sph_solver.h>
#include <cutil.h>
#include <shape.h>

__bidevice__ SphGasSolver2::SphGasSolver2(){ solver = nullptr; }

__host__ void SphGasSolver2::Initialize(){
    solver = cudaAllocateVx(SphSolver2, 1);
    solver->Initialize(DefaultSphSolverData2());
}

__host__ void SphGasSolver2::SetViscosityCoefficient(Float viscosityCoefficient){
    solver->SetViscosityCoefficient(viscosityCoefficient);
}

__host__ void SphGasSolver2::SetPseudoViscosityCoefficient(Float pseudoViscosityCoefficient){
    solver->SetPseudoViscosityCoefficient(pseudoViscosityCoefficient);
}

__bidevice__ Float SphGasSolver2::GetKernelRadius(){
    return solver->GetKernelRadius();
}

__host__ void SphGasSolver2::SetColliders(ColliderSet2 *col){
    solver->SetColliders(col);
}

__bidevice__ SphSolverData2 *SphGasSolver2::GetSphSolverData(){
    return solver->solverData;
}

__bidevice__ void ComputeDensityInterpolationFor(SphSolverData2 *data, int particleId){
    int *neighbors = nullptr;
    ParticleSet2 *pSet = data->sphpSet->GetParticleSet();
    vec2f pi = pSet->GetParticlePosition(particleId);
    Float di = pSet->GetParticleDensity(particleId);
    unsigned int cellId = data->domain->GetLinearHashedPosition(pi);
    
    Float facK = 0.6 * data->sphpSet->GetKernelRadius(); // TODO: Fix this
    
    Float mass = pSet->GetMass();
    
    int count = data->domain->GetNeighborsOf(cellId, &neighbors);
    SphStdKernel2 kernel(data->sphpSet->GetKernelRadius());
    
    // V0 computation, page 4.
    vec2f ni(0, 0);
    for(int i = 0; i < count; i++){
        Cell<Bounds2f> *cell = data->domain->GetCell(neighbors[i]);
        ParticleChain *pChain = cell->GetChain();
        int size = cell->GetChainLength();
        
        for(int j = 0; j < size; j++){
            vec2f pj = pSet->GetParticlePosition(pChain->pId);
            Float dj = pSet->GetParticleDensity(pChain->pId);
            Float dist = Distance(pi, pj);
            if(dist > 0 && !IsZero(dist)){
                AssertA(!IsZero(dj), "Zero density");
                vec2f dir = (pj - pi) / dist;
                vec2f gradij = kernel.gradW(dist, dir);
                
                ni += (mass / dj) * gradij;
            }
            
            pChain = pChain->next;
        }
    }
    
    if(IsZero(ni.LengthSquared())){
        DBG_PRINT("Zero Normal: {%g %g}\n", ni.x, ni.y);
        ni = vec2f(1,0);
    }
    
    vec2f gradik = kernel.gradW(facK, ni);
    AssertA(!IsZero(gradik.LengthSquared()), "Zero gradient");
    Float v0 = ni.Length() / gradik.Length();
    Float wik = kernel.W(facK);
    
    // NOTE: Now we have that:
    //         mk = v0 * di;
    //         di = di * (1 + v0 * Wik);
    
    Float dii = di * (1 + v0 * wik);
    AssertA(!IsZero(dii), "Zero density interpolation");
    pSet->SetParticleDensityEx(particleId, dii);
    pSet->SetParticleV0(particleId, v0);
}

// Does not *technically* need to swap densities, can just perform with ex instead
__bidevice__ void SwapDensitiesFor(SphSolverData2 *data, int particleId){
    ParticleSet2 *pSet = data->sphpSet->GetParticleSet();
    Float di = pSet->GetParticleDensityEx(particleId);
    Float dii = pSet->GetParticleDensity(particleId);
    
    pSet->SetParticleDensity(particleId, di);
    pSet->SetParticleDensityEx(particleId, dii);
    Float dif = Absf(di - dii);
    DBG_PRINT("Dif[%d] = %g\n", particleId, dif);
}

__global__ void ComputeExtendedDensityKernel(SphSolverData2 *data){
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    ParticleSet2 *pSet = data->sphpSet->GetParticleSet();
    if(i < pSet->GetParticleCount()){
        ComputeDensityInterpolationFor(data, i);
    }
}

__host__ void ComputeExtendedDensityGPU(SphSolverData2 *data){
    ParticleSet2 *pSet = data->sphpSet->GetParticleSet();
    int N = pSet->GetParticleCount();
    GPULaunch(N, ComputeExtendedDensityKernel, data);
}

__host__ void ComputeExtendedDensityCPU(SphSolverData2 *data){
    SphParticleSet2 *sphSet = data->sphpSet;
    ParticleSet2 *pSet = sphSet->GetParticleSet();
    int count = pSet->GetParticleCount();
    
    // Compute interpolated density
    for(int i = 0; i < count; i++){
        ComputeDensityInterpolationFor(data, i);
    }
}

__global__ void ComputeExtendedPressureKernel(SphSolverData2 *data){
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    ParticleSet2 *pSet = data->sphpSet->GetParticleSet();
    if(i < pSet->GetParticleCount()){
        Float di = pSet->GetParticleDensityEx(i);
        ComputePressureFor(data, i, di);
        SwapDensitiesFor(data, i);
    }
}

__host__ void ComputePressureGPU(SphSolverData2 *data){
    ParticleSet2 *pSet = data->sphpSet->GetParticleSet();
    int N = pSet->GetParticleCount();
    GPULaunch(N, ComputeExtendedPressureKernel, data);
}

__host__ void ComputePressureCPU(SphSolverData2 *data){
    ParticleSet2 *pSet = data->sphpSet->GetParticleSet();
    int count = pSet->GetParticleCount();
    
    for(int i = 0; i < count; i++){
        Float di = pSet->GetParticleDensityEx(i);
        ComputePressureFor(data, i, di);
        SwapDensitiesFor(data, i);
    }
}

__global__ void ComputeExtendedPressureForceKernel(SphSolverData2 *data){
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    ParticleSet2 *pSet = data->sphpSet->GetParticleSet();
    if(i < pSet->GetParticleCount()){
        ComputeAllForcesFor(data, i, 1);
    }
}

__host__ void ComputeExtendedPressureForceGPU(SphSolverData2 *data){
    ParticleSet2 *pSet = data->sphpSet->GetParticleSet();
    int N = pSet->GetParticleCount();
    GPULaunch(N, ComputeExtendedPressureForceKernel, data);
}

__host__ void ComputeExtendedPressureForceCPU(SphSolverData2 *data){
    SphParticleSet2 *sphSet = data->sphpSet;
    ParticleSet2 *pSet = sphSet->GetParticleSet();
    int count = pSet->GetParticleCount();
    
    // Compute forces considering virtual particle
    for(int i = 0; i < count; i++){
        ComputeAllForcesFor(data, i, 1);
    }
}


__host__ void AdvanceTimeStep(SphGasSolver2 *solver, Float timeStep, int use_cpu = 0){
    SphSolverData2 *data = solver->GetSphSolverData();
    //StaticsCompute *sCompute = data->statsCompute;
    //StaticsStep *step = nullptr;
    if(use_cpu)
        UpdateGridDistributionCPU(data);
    else
        UpdateGridDistributionGPU(data);
    
    //sCompute->NextStep();
    if(use_cpu)
        ComputeDensityCPU(data, 0);
    else 
        ComputeDensityGPU(data, 0);
    
    //sCompute->FinishStep(&step);
    //StaticsStepPrintInfo(step);
    
    ComputeExtendedDensityGPU(data);
    
    if(use_cpu){
        ComputePressureCPU(data);
        ComputeExtendedPressureForceCPU(data);
    }else{
        ComputePressureGPU(data);
        ComputeExtendedPressureForceGPU(data);
    }
    
    //data->Tamb = ComputeAverageTemperature(data);
    
    if(use_cpu)
        TimeIntegrationCPU(data, timeStep, 1);
    else
        TimeIntegrationGPU(data, timeStep, 1);
}

__host__ void SphGasSolver2::Advance(Float timeIntervalInSeconds){
    unsigned int numberOfIntervals = 0;
    unsigned int numberOfIntervalsRunned = 0;
    Float remainingTime = timeIntervalInSeconds;
    
    SphSolverData2 *data = solver->solverData;
    while(remainingTime > Epsilon){
        SphParticleSet2 *sphpSet = solver->GetSphParticleSet();
        
        numberOfIntervals = sphpSet->ComputeNumberOfTimeSteps(remainingTime,
                                                              data->soundSpeed);
        Float timeStep = remainingTime / (Float)numberOfIntervals;
        AdvanceTimeStep(this, timeStep);
        remainingTime -= timeStep;
        numberOfIntervalsRunned += 1;
    }
    
    CNMClassifyLazyGPU(data->domain);
}


__host__ void SphGasSolver2::Setup(Float targetDensity, Float targetSpacing, 
                                   Float relativeRadius, Grid2 *dom, SphParticleSet2 *pSet)
{
    SphSolverData2 *data = GetSphSolverData();
    //StaticsCompute *sCompute = data->statsCompute;
    //sCompute->Allocate(5, pSet->GetParticleSet()->GetParticleCount());
    solver->Setup(targetDensity, targetSpacing, relativeRadius, dom, pSet);
    
    int use_cpu = 0;
    int maxLevel = 0;
    
    if(use_cpu)
        UpdateGridDistributionCPU(data);
    else
        UpdateGridDistributionGPU(data);
    
    maxLevel = CNMClassifyLazyGPU(data->domain);
    
    if(use_cpu)
        ComputeInitialTemperatureMapCPU(data, data->Tmin, data->Tmax, maxLevel);
    else
        ComputeInitialTemperatureMapGPU(data, data->Tmin, data->Tmax, maxLevel);
}

__host__ void SphGasSolver2::UpdateDensity(){
    solver->UpdateDensity();
}
