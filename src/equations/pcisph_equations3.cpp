#include <pcisph_solver.h>

__bidevice__ void PredictVelocityAndPositionFor(PciSphSolverData3 *data, int particleId,
                                                Float timeIntervalInSeconds, int is_first)
{
    ParticleSet3 *pSet = data->sphData->sphpSet->GetParticleSet();
    vec3f vi = pSet->GetParticleVelocity(particleId);
    vec3f pi = pSet->GetParticlePosition(particleId);
    vec3f fi = pSet->GetParticleForce(particleId);
    Float mass = pSet->GetMass();
    vec3f tmpVi;
    vec3f tmpPi;
    if(is_first){ // make initialization
        Float di = pSet->GetParticleDensity(particleId);
        data->densityPredicted[particleId] = di;
        data->pressureForces[particleId] = vec3f(0);
        data->densityErrors[particleId] = 0;
        pSet->SetParticlePressure(particleId, 0);
    }
    
    tmpVi = vi + (timeIntervalInSeconds / mass) * (fi + data->pressureForces[particleId]);
    tmpPi = pi + timeIntervalInSeconds * tmpVi;
    
    data->sphData->collider->ResolveCollision(pSet->GetRadius(), 0, &tmpPi, &tmpVi);
    
    data->tempVelocities[particleId] = tmpVi;
    data->tempPositions[particleId] = tmpPi;
}

__host__ void PredictVelocityAndPositionCPU(PciSphSolverData3 *data, 
                                            Float timeIntervalInSeconds, int is_first=0)
{
    ParticleSet3 *pSet = data->sphData->sphpSet->GetParticleSet();
    int count = pSet->GetParticleCount();
    for(int i = 0; i < count; i++){
        PredictVelocityAndPositionFor(data, i, timeIntervalInSeconds, is_first);
    }
}

__global__ void PredictVelocityAndPositionKernel(PciSphSolverData3 *data, 
                                                 Float timeIntervalInSeconds, int is_first)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    ParticleSet3 *pSet = data->sphData->sphpSet->GetParticleSet();
    if(i < pSet->GetParticleCount()){
        PredictVelocityAndPositionFor(data, i, timeIntervalInSeconds, is_first);
    }
}

__host__ void PredictVelocityAndPositionGPU(PciSphSolverData3 *data, 
                                            Float timeIntervalInSeconds, int is_first=0)
{
    ParticleSet3 *pSet = data->sphData->sphpSet->GetParticleSet();
    int N = pSet->GetParticleCount();
    GPULaunch(N, PredictVelocityAndPositionKernel, data, timeIntervalInSeconds, is_first);
}

// NOTE: Potential error as tmpPi was not reditributed in the grid
//       neighbor querying might not be correct
__bidevice__ void PredictPressureFor(PciSphSolverData3 *pciData, Float delta, int particleId)
{
    SphSolverData3 *data = pciData->sphData;
    ParticleSet3 *pSet = data->sphpSet->GetParticleSet();
    vec3f pi = pciData->tempPositions[particleId];
    Float ppi = pSet->GetParticlePressure(particleId);
    Bucket *bucket = pSet->GetParticleBucket(particleId);
    
    SphStdKernel3 kernel(data->sphpSet->GetKernelRadius());
    
    Float weightSum = 0;
    for(int i = 0; i < bucket->Count(); i++){
        int j = bucket->Get(i);
        vec3f pj = pciData->tempPositions[j];
        Float dist = Distance(pi, pj);
        weightSum += kernel.W(dist);
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

__host__ void PredictPressureCPU(PciSphSolverData3 *data, Float delta){
    ParticleSet3 *pSet = data->sphData->sphpSet->GetParticleSet();
    int count = pSet->GetParticleCount();
    for(int i = 0; i < count; i++){
        PredictPressureFor(data, delta, i);
    }
}

__global__ void PredictPressureKernel(PciSphSolverData3 *data, Float delta){
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    ParticleSet3 *pSet = data->sphData->sphpSet->GetParticleSet();
    if(i < pSet->GetParticleCount()){
        PredictPressureFor(data, delta, i);
    }
}

__host__ void PredictPressureGPU(PciSphSolverData3 *data, Float delta){
    ParticleSet3 *pSet = data->sphData->sphpSet->GetParticleSet();
    int N = pSet->GetParticleCount();
    GPULaunch(N, PredictPressureKernel, data, delta);
}

__bidevice__ void PredictPressureForceFor(PciSphSolverData3 *pciData, int particleId){
    SphSolverData3 *data = pciData->sphData;
    ParticleSet3 *pSet = data->sphpSet->GetParticleSet();
    vec3f pi = pSet->GetParticlePosition(particleId);
    Bucket *bucket = pSet->GetParticleBucket(particleId);
    
    Float mass = pSet->GetMass();
    Float mass2 = pSet->GetMass() * pSet->GetMass();
    Float sphRadius = data->sphpSet->GetKernelRadius();
    SphSpikyKernel3 kernel(sphRadius);
    
    Float poi = pSet->GetParticlePressure(particleId);
    Float di = pciData->densityPredicted[particleId];
    Float di2 = di * di;
    
    vec3f fi(0);
    vec3f ti(0);
    for(int i = 0; i < bucket->Count(); i++){
        int j = bucket->Get(i);
        if(particleId != j){
            vec3f pj = pSet->GetParticlePosition(j);
            Float dj = pciData->densityPredicted[j];
            Float dj2 = dj * dj;
            Float poj = pSet->GetParticlePressure(j);
            
            Float dist = Distance(pi, pj);
            bool valid = IsWithinSpiky(dist, sphRadius);
            
            if(valid){
                AssertA(!IsZero(dj2), "Zero neighbor density {ComputePressureForceFor}");
                if(dist > 0 && !IsZero(dist)){
                    vec3f dir = (pj - pi) / dist;
                    vec3f gradij = kernel.gradW(dist, dir);
                    ti += mass2 * (poi / di2 + poj / dj2) * gradij;
                }
            }
        }
    }
    
    fi = fi - ti;
    pciData->pressureForces[particleId] = fi;
}

__host__ void PredictPressureForceCPU(PciSphSolverData3 *data){
    ParticleSet3 *pSet = data->sphData->sphpSet->GetParticleSet();
    int count = pSet->GetParticleCount();
    for(int i = 0; i < count; i++){
        PredictPressureForceFor(data, i);
    }
}

__global__ void PredictPressureForceKernel(PciSphSolverData3 *data){
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    ParticleSet3 *pSet = data->sphData->sphpSet->GetParticleSet();
    if(i < pSet->GetParticleCount()){
        PredictPressureForceFor(data, i);
    }
}

__host__ void PredictPressureForceGPU(PciSphSolverData3 *data){
    ParticleSet3 *pSet = data->sphData->sphpSet->GetParticleSet();
    int N = pSet->GetParticleCount();
    GPULaunch(N, PredictPressureForceKernel, data);
}

__bidevice__ void AccumulateForcesFor(PciSphSolverData3 *pciData, int particleId,
                                      Float timeStep)
{
    SphSolverData3 *data = pciData->sphData;
    ParticleSet3 *pSet = data->sphpSet->GetParticleSet();
    vec3f fi = pSet->GetParticleForce(particleId);
    
    fi += pciData->pressureForces[particleId];
    pSet->SetParticleForce(particleId, fi);
}

__host__ void AccumulateAndIntegrateCPU(PciSphSolverData3 *data, Float timeStep){
    ParticleSet3 *pSet = data->sphData->sphpSet->GetParticleSet();
    int count = pSet->GetParticleCount();
    for(int i = 0; i < count; i++){
        AccumulateForcesFor(data, i, timeStep);
        TimeIntegrationFor(data->sphData, i, timeStep, 0);
    }
}

__global__ void AccumulateAndIntegrateKernel(PciSphSolverData3 *data, Float timeStep){
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    ParticleSet3 *pSet = data->sphData->sphpSet->GetParticleSet();
    if(i < pSet->GetParticleCount()){
        AccumulateForcesFor(data, i, timeStep);
    }
}

__host__ void AccumulateAndIntegrateGPU(PciSphSolverData3 *data, Float timeStep){
    ParticleSet3 *pSet = data->sphData->sphpSet->GetParticleSet();
    int N = pSet->GetParticleCount();
    GPULaunch(N, AccumulateAndIntegrateKernel, data, timeStep);
    TimeIntegrationGPU(data->sphData, timeStep, 0);
}

__bidevice__ inline Float AbsfMax(Float x, Float y){
    return (x*x > y*y) ? x : y;
}

__host__ void ComputePressureForceAndIntegrate(PciSphSolverData3 *data, 
                                               Float timeIntervalInSeconds, 
                                               Float maxDensityErrorRatio, 
                                               Float delta, int maxIt,
                                               int use_cpu)
{
    ParticleSet3 *pSet = data->sphData->sphpSet->GetParticleSet();
    Float targetDensity = data->sphData->sphpSet->GetTargetDensity();
    int count = pSet->GetParticleCount();
    Float densityErrorRatio = 0;
    
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
    }
    
    if(use_cpu)
        AccumulateAndIntegrateCPU(data, timeIntervalInSeconds);
    else
        AccumulateAndIntegrateGPU(data, timeIntervalInSeconds);

    if(use_cpu)
        ComputePseudoViscosityInterpolationCPU(data->sphData, timeIntervalInSeconds);
    else
        ComputePseudoViscosityInterpolationGPU(data->sphData, timeIntervalInSeconds);

}
