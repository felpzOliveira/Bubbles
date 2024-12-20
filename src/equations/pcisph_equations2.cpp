#include <pcisph_solver.h>


bb_cpu_gpu void PredictVelocityAndPositionFor(PciSphSolverData2 *data, int particleId,
                                                Float timeIntervalInSeconds, int is_first)
{
    ParticleSet2 *pSet = data->sphData->sphpSet->GetParticleSet();
    vec2f vi = pSet->GetParticleVelocity(particleId);
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

void PredictVelocityAndPositionCPU(PciSphSolverData2 *data,
                                   Float timeIntervalInSeconds, int is_first=0)
{
    ParticleSet2 *pSet = data->sphData->sphpSet->GetParticleSet();
    int count = pSet->GetParticleCount();
    ParallelFor(0, count, [&](int i){
        PredictVelocityAndPositionFor(data, i, timeIntervalInSeconds, is_first);
    });
}

bb_kernel void PredictVelocityAndPositionKernel(PciSphSolverData2 *data,
                                                 Float timeIntervalInSeconds, int is_first)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    ParticleSet2 *pSet = data->sphData->sphpSet->GetParticleSet();
    if(i < pSet->GetParticleCount()){
        PredictVelocityAndPositionFor(data, i, timeIntervalInSeconds, is_first);
    }
}

void PredictVelocityAndPositionGPU(PciSphSolverData2 *data,
                                   Float timeIntervalInSeconds, int is_first=0)
{
    ParticleSet2 *pSet = data->sphData->sphpSet->GetParticleSet();
    int N = pSet->GetParticleCount();
    GPULaunch(N, PredictVelocityAndPositionKernel, data, timeIntervalInSeconds, is_first);
}

// NOTE: Potential error as tmpPi was not reditributed in the grid and
//       neighbor querying might not be correct, to avoid having to
//       reditribute we assume the bucket stays intact, which is incorrect.
bb_cpu_gpu void PredictPressureFor(PciSphSolverData2 *pciData, Float delta, int particleId)
{
    SphSolverData2 *data = pciData->sphData;
    ParticleSet2 *pSet = data->sphpSet->GetParticleSet();
    vec2f pi = pciData->tempPositions[particleId];
    Float ppi = pSet->GetParticlePressure(particleId);
    Bucket *bucket = pSet->GetParticleBucket(particleId);

    SphStdKernel2 kernel(data->sphpSet->GetKernelRadius());

    Float weightSum = 0;
    for(int i = 0; i < bucket->Count(); i++){
        int j = bucket->Get(i);
        vec2f pj = pciData->tempPositions[j];
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

void PredictPressureCPU(PciSphSolverData2 *data, Float delta){
    ParticleSet2 *pSet = data->sphData->sphpSet->GetParticleSet();
    int count = pSet->GetParticleCount();
    ParallelFor(0, count, [&](int i){
        PredictPressureFor(data, delta, i);
    });
}

bb_kernel void PredictPressureKernel(PciSphSolverData2 *data, Float delta){
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    ParticleSet2 *pSet = data->sphData->sphpSet->GetParticleSet();
    if(i < pSet->GetParticleCount()){
        PredictPressureFor(data, delta, i);
    }
}

void PredictPressureGPU(PciSphSolverData2 *data, Float delta){
    ParticleSet2 *pSet = data->sphData->sphpSet->GetParticleSet();
    int N = pSet->GetParticleCount();
    GPULaunch(N, PredictPressureKernel, data, delta);
}

bb_cpu_gpu void PredictPressureForceFor(PciSphSolverData2 *pciData, int particleId){
    SphSolverData2 *data = pciData->sphData;
    ParticleSet2 *pSet = data->sphpSet->GetParticleSet();
    vec2f pi = pSet->GetParticlePosition(particleId);
    Bucket *bucket = pSet->GetParticleBucket(particleId);

    Float mass = pSet->GetMass();
    Float mass2 = pSet->GetMass() * pSet->GetMass();
    Float sphRadius = data->sphpSet->GetKernelRadius();
    SphSpikyKernel2 kernel(sphRadius);

    Float poi = pSet->GetParticlePressure(particleId);
    Float di = pciData->densityPredicted[particleId];
    Float di2 = di * di;

    vec2f fi(0, 0);
    vec2f ti(0,0);
    for(int i = 0; i < bucket->Count(); i++){
        int j = bucket->Get(i);
        if(particleId != j){
            vec2f pj = pSet->GetParticlePosition(j);
            Float dj = pciData->densityPredicted[j];
            Float dj2 = dj * dj;
            Float poj = pSet->GetParticlePressure(j);

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
    }

    fi = fi - ti;
    pciData->pressureForces[particleId] = fi;
}

void PredictPressureForceCPU(PciSphSolverData2 *data){
    ParticleSet2 *pSet = data->sphData->sphpSet->GetParticleSet();
    int count = pSet->GetParticleCount();
    ParallelFor(0, count, [&](int i){
        PredictPressureForceFor(data, i);
    });
}

bb_kernel void PredictPressureForceKernel(PciSphSolverData2 *data){
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    ParticleSet2 *pSet = data->sphData->sphpSet->GetParticleSet();
    if(i < pSet->GetParticleCount()){
        PredictPressureForceFor(data, i);
    }
}

void PredictPressureForceGPU(PciSphSolverData2 *data){
    ParticleSet2 *pSet = data->sphData->sphpSet->GetParticleSet();
    int N = pSet->GetParticleCount();
    GPULaunch(N, PredictPressureForceKernel, data);
}

bb_cpu_gpu void AccumulateAndIntegrateFor(PciSphSolverData2 *pciData, int particleId,
                                          Float timeStep)
{
    SphSolverData2 *data = pciData->sphData;
    ParticleSet2 *pSet = data->sphpSet->GetParticleSet();
    vec2f fi = pSet->GetParticleForce(particleId);

    fi += pciData->pressureForces[particleId];
    pSet->SetParticleForce(particleId, fi);
    TimeIntegrationFor(data, particleId, timeStep, 0);
}

void AccumulateAndIntegrateCPU(PciSphSolverData2 *data, Float timeStep){
    ParticleSet2 *pSet = data->sphData->sphpSet->GetParticleSet();
    int count = pSet->GetParticleCount();
    ParallelFor(0, count, [&](int i){
        AccumulateAndIntegrateFor(data, i, timeStep);
    });
}

bb_kernel void AccumulateAndIntegrateKernel(PciSphSolverData2 *data, Float timeStep){
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    ParticleSet2 *pSet = data->sphData->sphpSet->GetParticleSet();
    if(i < pSet->GetParticleCount()){
        AccumulateAndIntegrateFor(data, i, timeStep);
    }
}

void AccumulateAndIntegrateGPU(PciSphSolverData2 *data, Float timeStep){
    ParticleSet2 *pSet = data->sphData->sphpSet->GetParticleSet();
    int N = pSet->GetParticleCount();
    GPULaunch(N, AccumulateAndIntegrateKernel, data, timeStep);
}

bb_cpu_gpu inline Float AbsfMax(Float x, Float y){
    return (x*x > y*y) ? x : y;
}

void ComputePressureForceAndIntegrate(PciSphSolverData2 *data,
                                      Float timeIntervalInSeconds,
                                      Float maxDensityErrorRatio,
                                      Float delta, int maxIt, int use_cpu)
{
    ParticleSet2 *pSet = data->sphData->sphpSet->GetParticleSet();
    Float targetDensity = data->sphData->sphpSet->GetTargetDensity();
    int count = pSet->GetParticleCount();
    Float densityErrorRatio = 0;
    unsigned int iterations = 0;

    for(int k = 0; k < maxIt; k++){
        if(use_cpu){
            PredictVelocityAndPositionCPU(data, timeIntervalInSeconds, k == 0);
            PredictPressureCPU(data, delta);
            PredictPressureForceCPU(data);
        }else{
            PredictVelocityAndPositionGPU(data, timeIntervalInSeconds, k == 0);
            PredictPressureGPU(data, delta);
            PredictPressureForceGPU(data);
        }

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

    if(use_cpu){
        AccumulateAndIntegrateCPU(data, timeIntervalInSeconds);
        ComputePseudoViscosityInterpolationCPU(data->sphData, timeIntervalInSeconds);
    }else{
        AccumulateAndIntegrateGPU(data, timeIntervalInSeconds);
        ComputePseudoViscosityInterpolationGPU(data->sphData, timeIntervalInSeconds);
    }
}
