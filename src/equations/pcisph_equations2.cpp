#include <pcisph_solver.h>


__bidevice__ void PredictVelocityAndPositionFor(PciSphSolverData2 *data, int particleId,
                                                Float timeIntervalInSeconds, int is_first)
{
    ParticleSet2 *pSet = data->sphData->sphpSet->GetParticleSet();
    vec2f vi = pSet->GetParticlePosition(particleId);
    vec2f pi = pSet->GetParticlePosition(particleId);
    vec2f fi = pSet->GetParticleForce(particleId);
    Float mass = pSet->GetMass();
    vec2f tmpVi;
    vec2f tmpPi;
    if(is_first){ // make initialization
        Float di = pSet->GetParticleDensity(particleId);
        data->densityPredicted[particleId] = di;
        data->pressureForces[particleId] = vec2f(0,0);
        data->densityErrors[particleId] = 0;
        pSet->SetParticlePressure(particleId, 0);
    }
    
    tmpVi = vi + (timeIntervalInSeconds / mass) * (fi + data->pressureForces[particleId]);
    tmpPi = pi + timeIntervalInSeconds * tmpVi;
    
    data->sphData->collider->ResolveCollision(pSet->GetRadius(), 0.75, &tmpPi, &tmpVi);
    
    data->tempVelocities[particleId] = tmpVi;
    data->tempPositions[particleId] = tmpPi;
}

__host__ void PredictVelocityAndPositionCPU(PciSphSolverData2 *data, 
                                            Float timeIntervalInSeconds, int is_first=0)
{
    ParticleSet2 *pSet = data->sphData->sphpSet->GetParticleSet();
    int count = pSet->GetParticleCount();
    for(int i = 0; i < count; i++){
        PredictVelocityAndPositionFor(data, i, timeIntervalInSeconds, is_first);
    }
}

__global__ void PredictVelocityAndPositionKernel(PciSphSolverData2 *data, 
                                                 Float timeIntervalInSeconds, int is_first)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    ParticleSet2 *pSet = data->sphData->sphpSet->GetParticleSet();
    if(i < pSet->GetParticleCount()){
        PredictVelocityAndPositionFor(data, i, timeIntervalInSeconds, is_first);
    }
}

__host__ void PredictVelocityAndPositionGPU(PciSphSolverData2 *data, 
                                            Float timeIntervalInSeconds, int is_first=0)
{
    ParticleSet2 *pSet = data->sphData->sphpSet->GetParticleSet();
    int N = pSet->GetParticleCount();
    GPULaunch(N, PredictVelocityAndPositionKernel, data, timeIntervalInSeconds, is_first);
}

// NOTE: Potential error as tmpPi was not reditributed in the grid
//       neighbor querying might not be correct
__bidevice__ void PredictPressureFor(PciSphSolverData2 *pciData, Float delta, int particleId){
    int *neighbors = nullptr;
    SphSolverData2 *data = pciData->sphData;
    ParticleSet2 *pSet = data->sphpSet->GetParticleSet();
    vec2f pi = pciData->tempPositions[particleId];
    Float ppi = pSet->GetParticlePressure(particleId);
    
    unsigned int cellId = data->domain->GetLinearHashedPosition(pi);
    int count = data->domain->GetNeighborsOf(cellId, &neighbors);
    
    SphStdKernel2 kernel(data->sphpSet->GetKernelRadius());
    
    Float weightSum = 0;
    for(int i = 0; i < count; i++){
        Cell<Bounds2f> *cell = data->domain->GetCell(neighbors[i]);
        ParticleChain *pChain = cell->GetChain();
        int size = cell->GetChainLength();
        for(int j = 0; j < size; j++){
            vec2f pj = pciData->tempPositions[pChain->pId];
            Float dist = Distance(pi, pj);
            weightSum += kernel.W(dist);
            pChain = pChain->next;
        }
    }
    
    Float density = weightSum * pSet->GetMass();
    Float densityError = (density - data->sphpSet->GetTargetDensity());
    Float pressure = delta * densityError;
    
    if(pressure < 0){
        pressure *= data->negativePressureScale;
        densityError *= data->negativePressureScale;
    }
    
    ppi += pressure;
    pSet->SetParticlePressure(particleId, ppi);
    pciData->densityPredicted[particleId] = density;
}

__host__ void PredictPressureCPU(PciSphSolverData2 *data, Float delta){
    ParticleSet2 *pSet = data->sphData->sphpSet->GetParticleSet();
    int count = pSet->GetParticleCount();
    for(int i = 0; i < count; i++){
        PredictPressureFor(data, delta, i);
    }
}

__global__ void PredictPressureKernel(PciSphSolverData2 *data, Float delta){
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    ParticleSet2 *pSet = data->sphData->sphpSet->GetParticleSet();
    if(i < pSet->GetParticleCount()){
        PredictPressureFor(data, delta, i);
    }
}

__host__ void PredictPressureGPU(PciSphSolverData2 *data, Float delta){
    ParticleSet2 *pSet = data->sphData->sphpSet->GetParticleSet();
    int N = pSet->GetParticleCount();
    GPULaunch(N, PredictPressureKernel, data, delta);
}

__bidevice__ void PredictPressureForceFor(PciSphSolverData2 *pciData, int particleId){
    int *neighbors = nullptr;
    SphSolverData2 *data = pciData->sphData;
    ParticleSet2 *pSet = data->sphpSet->GetParticleSet();
    
    vec2f pi = pSet->GetParticlePosition(particleId);
    unsigned int cellId = data->domain->GetLinearHashedPosition(pi);
    int count = data->domain->GetNeighborsOf(cellId, &neighbors);
    
    Float mass = pSet->GetMass();
    Float mass2 = pSet->GetMass() * pSet->GetMass();
    Float sphRadius = data->sphpSet->GetKernelRadius();
    SphSpikyKernel2 kernel(sphRadius);
    
    Float poi = pSet->GetParticlePressure(particleId);
    Float di = pciData->densityPredicted[particleId];
    Float di2 = di * di;
    
    vec2f fi(0, 0);
    vec2f ti(0,0);
    for(int i = 0; i < count; i++){
        Cell<Bounds2f> *cell = data->domain->GetCell(neighbors[i]);
        ParticleChain *pChain = cell->GetChain();
        int size = cell->GetChainLength();
        
        for(int j = 0; j < size; j++){
            if(particleId != pChain->pId){
                vec2f pj = pSet->GetParticlePosition(pChain->pId);
                Float dj = pciData->densityPredicted[pChain->pId];
                Float dj2 = dj * dj;
                Float poj = pSet->GetParticlePressure(pChain->pId);
                
                Float dist = Distance(pi, pj);
                bool valid = IsWithinSpiky(dist, sphRadius);
                
                if(valid){
                    AssertA(!IsZero(dj2), "Zero neighbor density {ComputePressureForceFor}");
                    if(dist > 0 && !IsZero(dist)){
                        vec2f dir = (pj - pi) / dist;
                        vec2f gradij = kernel.gradW(dist, dir);
                        ti += mass2 * (poi / di2 + poj / dj2) * gradij;
                    }
                }
            }
            
            pChain = pChain->next;
        }
    }
    
    fi = fi - ti;
    pciData->pressureForces[particleId] = fi;
}

__host__ void PredictPressureForceCPU(PciSphSolverData2 *data){
    ParticleSet2 *pSet = data->sphData->sphpSet->GetParticleSet();
    int count = pSet->GetParticleCount();
    for(int i = 0; i < count; i++){
        PredictPressureForceFor(data, i);
    }
}

__global__ void PredictPressureForceKernel(PciSphSolverData2 *data){
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    ParticleSet2 *pSet = data->sphData->sphpSet->GetParticleSet();
    if(i < pSet->GetParticleCount()){
        PredictPressureForceFor(data, i);
    }
}

__host__ void PredictPressureForceGPU(PciSphSolverData2 *data){
    ParticleSet2 *pSet = data->sphData->sphpSet->GetParticleSet();
    int N = pSet->GetParticleCount();
    GPULaunch(N, PredictPressureForceKernel, data);
}

__bidevice__ void AccumulateAndIntegrateFor(PciSphSolverData2 *pciData, int particleId,
                                            Float timeStep)
{
    SphSolverData2 *data = pciData->sphData;
    ParticleSet2 *pSet = data->sphpSet->GetParticleSet();
    vec2f fi = pSet->GetParticleForce(particleId);
    
    fi += pciData->pressureForces[particleId];
    pSet->SetParticleForce(particleId, fi);
    TimeIntegrationFor(data, particleId, timeStep, 0);
}

__host__ void AccumulateAndIntegrateCPU(PciSphSolverData2 *data, Float timeStep){
    ParticleSet2 *pSet = data->sphData->sphpSet->GetParticleSet();
    int count = pSet->GetParticleCount();
    for(int i = 0; i < count; i++){
        AccumulateAndIntegrateFor(data, i, timeStep);
    }
}

__global__ void AccumulateAndIntegrateKernel(PciSphSolverData2 *data, Float timeStep){
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    ParticleSet2 *pSet = data->sphData->sphpSet->GetParticleSet();
    if(i < pSet->GetParticleCount()){
        AccumulateAndIntegrateFor(data, i, timeStep);
    }
}

__host__ void AccumulateAndIntegrateGPU(PciSphSolverData2 *data, Float timeStep){
    ParticleSet2 *pSet = data->sphData->sphpSet->GetParticleSet();
    int N = pSet->GetParticleCount();
    GPULaunch(N, AccumulateAndIntegrateKernel, data, timeStep);
}

__bidevice__ inline Float AbsfMax(Float x, Float y){
    return (x*x > y*y) ? x : y;
}

__host__ void ComputePressureForceAndIntegrate(PciSphSolverData2 *data, 
                                               Float timeIntervalInSeconds, 
                                               Float maxDensityErrorRatio, 
                                               Float delta, int maxIt,
                                               int use_cpu)
{
    ParticleSet2 *pSet = data->sphData->sphpSet->GetParticleSet();
    Float targetDensity = data->sphData->sphpSet->GetTargetDensity();
    int count = pSet->GetParticleCount();
    Float densityErrorRatio = 0;
    unsigned int iterations = 0;
    
    for(int k = 0; k < maxIt; k++){
        if(use_cpu)
            PredictVelocityAndPositionCPU(data, timeIntervalInSeconds, k == 0);
        else
            PredictVelocityAndPositionGPU(data, timeIntervalInSeconds, k == 0);
        
        if(use_cpu)
            PredictPressureCPU(data, delta);
        else
            PredictPressureGPU(data, delta);
        
        if(use_cpu)
            PredictPressureForceCPU(data);
        else
            PredictPressureForceGPU(data);
        
        Float maxDensityError = 0;
        for(int i = 0; i < count; i++){
            maxDensityError = AbsfMax(maxDensityError, data->densityErrors[i]);
        }
        
        densityErrorRatio = maxDensityError / targetDensity;
        if(Absf(densityErrorRatio) < maxDensityErrorRatio){
            break;
        }
        
        iterations++;
    }
    
    if(use_cpu)
        AccumulateAndIntegrateCPU(data, timeIntervalInSeconds);
    else
        AccumulateAndIntegrateGPU(data, timeIntervalInSeconds);
}