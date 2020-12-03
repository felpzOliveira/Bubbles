#include <pbf_solver.h>

/**************************************************************/
//             P O S I T I O N     P R E D I C T I O N        //
/**************************************************************/
__bidevice__ void ComputePredictedPositionsFor(PbfSolverData2 *data, int particleId,
                                               Float timeIntervalInSeconds)
{
    ParticleSet2 *pSet = data->sphData->sphpSet->GetParticleSet();
    vec2f pi = pSet->GetParticlePosition(particleId);
    vec2f oi = pi;
    vec2f vi = pSet->GetParticleVelocity(particleId);
    vec2f fi = pSet->GetParticleForce(particleId);
    Float mass = pSet->GetMass();
    
    data->originalPositions[particleId] = pi;
    
    vec2f ai = fi / mass;
    vi += timeIntervalInSeconds * ai;
    pi += timeIntervalInSeconds * vi;
    
    data->sphData->collider->ResolveCollision(pSet->GetRadius(), 0.75, &pi, &vi);
    pSet->SetParticlePosition(particleId, pi);
    pSet->SetParticleVelocity(particleId, vi);
    
    vec2f len = data->sphData->domain->GetCellSize();
    Float minLen = MinComponent(len);
    Float traveled = Distance(pi, oi);
    if(traveled > minLen){
        data->sphData->sphpSet->SetHigherLevel();
    }
}

__host__ void ComputePredictedPositionsCPU(PbfSolverData2 *data,
                                           Float timeIntervalInSeconds)
{
    ParticleSet2 *pSet = data->sphData->sphpSet->GetParticleSet();
    for(int i = 0; i < pSet->GetParticleCount(); i++){
        ComputePredictedPositionsFor(data, i, timeIntervalInSeconds);
    }
}

__global__ void ComputePredictedPositionsKernel(PbfSolverData2 *data,
                                                Float timeIntervalInSeconds)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    ParticleSet2 *pSet = data->sphData->sphpSet->GetParticleSet();
    if(i < pSet->GetParticleCount()){
        ComputePredictedPositionsFor(data, i, timeIntervalInSeconds);
    }
}

__host__ void ComputePredictedPositionsGPU(PbfSolverData2 *data,
                                           Float timeIntervalInSeconds)
{
    ParticleSet2 *pSet = data->sphData->sphpSet->GetParticleSet();
    int N = pSet->GetParticleCount();
    GPULaunch(N, ComputePredictedPositionsKernel, data, timeIntervalInSeconds);
}

/**************************************************************/
//                L A M B D A    C O M P U T A T I O N        //
/**************************************************************/
__bidevice__ void ComputeLambdaFor(PbfSolverData2 *data, int particleId){
    ParticleSet2 *pSet = data->sphData->sphpSet->GetParticleSet();
    vec2f pi = pSet->GetParticlePosition(particleId);
    Float di = pSet->GetParticleDensity(particleId);
    Float pho0 = data->sphData->sphpSet->GetTargetDensity();
    Bucket *bucket = pSet->GetParticleBucket(particleId);
    
    Float sphRadius = data->sphData->sphpSet->GetKernelRadius();
    SphSpikyKernel2 spiky(sphRadius);
    
    vec2f sumGrad(0);
    Float sumGradNorm = 0;
    Float ci = di / pho0 - 1.0;
    for(int i = 0; i < bucket->Count(); i++){
        int j = bucket->Get(i);
        vec2f pj = pSet->GetParticlePosition(j);
        Float distance = Distance(pi, pj);
        if(distance > 0){
            vec2f dir = (pj - pi) / distance;
            vec2f gradW = spiky.gradW(distance, dir);
            sumGrad += gradW;
            if(particleId != j)
                sumGradNorm += Dot(gradW, gradW);
        }
    }
    
    sumGradNorm += Dot(sumGrad, sumGrad);
    sumGradNorm  = sumGradNorm / pho0;
    
    data->lambdas[particleId] = -ci / (sumGradNorm + data->lambdaRelax);
}

__host__ void ComputeLambdaCPU(PbfSolverData2 *data){
    ParticleSet2 *pSet = data->sphData->sphpSet->GetParticleSet();
    for(int i = 0; i < pSet->GetParticleCount(); i++){
        ComputeLambdaFor(data, i);
    }
}

__global__ void ComputeLambdaKernel(PbfSolverData2 *data){
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    ParticleSet2 *pSet = data->sphData->sphpSet->GetParticleSet();
    if(i < pSet->GetParticleCount()){
        ComputeLambdaFor(data, i);
    }
}

__host__ void ComputeLambdaGPU(PbfSolverData2 *data){
    ParticleSet2 *pSet = data->sphData->sphpSet->GetParticleSet();
    int N = pSet->GetParticleCount();
    GPULaunch(N, ComputeLambdaKernel, data);
}

/**************************************************************/
//              D E L T A P    C O M P U T A T I O N          //
/**************************************************************/
__bidevice__ void ComputeDeltaPFor(PbfSolverData2 *data, Float timeIntervalInSeconds, 
                                   int particleId)
{
    ParticleSet2 *pSet = data->sphData->sphpSet->GetParticleSet();
    Float sphRadius = data->sphData->sphpSet->GetKernelRadius();
    Float h = data->sphData->sphpSet->GetTargetSpacing();
    vec2f pi = pSet->GetParticlePosition(particleId);
    vec2f vi = pSet->GetParticleVelocity(particleId);
    Float pho0 = data->sphData->sphpSet->GetTargetDensity();
    Bucket *bucket = pSet->GetParticleBucket(particleId);
    
    SphStdKernel2 kernel(sphRadius);
    SphSpikyKernel2 spiky(sphRadius);
    Float Wdq = kernel.W(data->antiClustDenom * h);
    Float lambdai = data->lambdas[particleId];
    vec2f sum(0);
    
    for(int i = 0; i < bucket->Count(); i++){
        int j = bucket->Get(i);
        vec2f pj = pSet->GetParticlePosition(j);
        Float distance = Distance(pi, pj);
        if(distance > 0){
            Float lambdaj = data->lambdas[j];
            Float mult = pow(kernel.W(distance) / Wdq, data->antiClustExp);
            Float sCorr = -data->antiClustStr * mult;
            vec2f dir = (pj - pi) / distance;
            vec2f gradW = spiky.gradW(distance, dir);
            
            sum += (lambdai + lambdaj + sCorr) * gradW;
        }
    }
    
    pi += sum / pho0;
    data->sphData->collider->ResolveCollision(pSet->GetRadius(), 0.75, &pi, &vi);
    
    vi = (pi - data->originalPositions[particleId]) / timeIntervalInSeconds;
    
    pSet->SetParticlePosition(particleId, pi);
    pSet->SetParticleVelocity(particleId, vi);
    
    vec2f len = data->sphData->domain->GetCellSize();
    Float minLen = MinComponent(len);
    Float traveled = Distance(pi, data->originalPositions[particleId]);
    if(traveled > minLen){
        data->sphData->sphpSet->SetHigherLevel();
    }
}

__host__ void ComputeDeltaPCPU(PbfSolverData2 *data, Float timeIntervalInSeconds){
    ParticleSet2 *pSet = data->sphData->sphpSet->GetParticleSet();
    for(int i = 0; i < pSet->GetParticleCount(); i++){
        ComputeDeltaPFor(data, i, timeIntervalInSeconds);
    }
}

__global__ void ComputeDeltaPKernel(PbfSolverData2 *data, Float timeIntervalInSeconds){
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    ParticleSet2 *pSet = data->sphData->sphpSet->GetParticleSet();
    if(i < pSet->GetParticleCount()){
        ComputeDeltaPFor(data, i, timeIntervalInSeconds);
    }
}

__host__ void ComputeDeltaPGPU(PbfSolverData2 *data, Float timeIntervalInSeconds){
    ParticleSet2 *pSet = data->sphData->sphpSet->GetParticleSet();
    int N = pSet->GetParticleCount();
    GPULaunch(N, ComputeDeltaPKernel, data, timeIntervalInSeconds);
}

/**************************************************************/
//          V O R T I C I T Y  C O M P U T A T I O N          //
/**************************************************************/
__bidevice__ void ComputeVorticityFor(PbfSolverData2 *data, int particleId){
    ParticleSet2 *pSet = data->sphData->sphpSet->GetParticleSet();
    vec2f vi = pSet->GetParticleVelocity(particleId);
    vec2f pi = pSet->GetParticlePosition(particleId);
    Bucket *bucket = pSet->GetParticleBucket(particleId);
    Float sphRadius = data->sphData->sphpSet->GetKernelRadius();
    Float pho0 = data->sphData->sphpSet->GetTargetDensity();
    Float mass = pSet->GetMass();
    
    SphSpikyKernel2 spiky(sphRadius);
    
    Float wi = 0;
    for(int i = 0; i < bucket->Count(); i++){
        int j = bucket->Get(i);
        vec2f pj = pSet->GetParticlePosition(j);
        Float distance = Distance(pj, pi);
        if(distance > 0){
            vec2f vj = pSet->GetParticleVelocity(j);
            vec2f vij = (vj - vi);
            vec2f dir = (pj - pi) / distance;
            vec2f gradW = spiky.gradW(distance, dir);
            wi += Cross(vij, gradW);
        }
    }
    
    wi *= mass / pho0; // ?
    
    data->w[particleId] = wi;
}

__bidevice__ void ComputeVorticityForceFor(PbfSolverData2 *data, int particleId){
    ParticleSet2 *pSet = data->sphData->sphpSet->GetParticleSet();
    Bucket *bucket = pSet->GetParticleBucket(particleId);
    vec2f pi = pSet->GetParticlePosition(particleId);
    Float pho0 = data->sphData->sphpSet->GetTargetDensity();
    Float mass = pSet->GetMass();
    Float sphRadius = data->sphData->sphpSet->GetKernelRadius();
    SphSpikyKernel2 spiky(sphRadius);
    
    vec2f gradVorticity(0);
    for(int i = 0; i < bucket->Count(); i++){
        int j = bucket->Get(i);
        vec2f pj = pSet->GetParticlePosition(j);
        Float distance = Distance(pj, pi);
        if(distance > 0){
            vec2f dir = (pj - pi) / distance;
            vec2f gradW = spiky.gradW(distance, dir);
            gradVorticity += Absf(data->w[j]) * gradW; // ?
        }
    }
    
    gradVorticity *= mass / pho0;
    if(gradVorticity.LengthSquared() > 0){
        vec2f fi = pSet->GetParticleForce(particleId);
        Float d = 1.0 / gradVorticity.Length();
        vec2f n = gradVorticity * d;
        fi += data->vorticityStr * data->w[particleId] * vec2f(n.y, -n.x);
        
        pSet->SetParticleForce(particleId, fi);
    }
}

__host__ void ComputeVorticityForceCPU(PbfSolverData2 *data){
    ParticleSet2 *pSet = data->sphData->sphpSet->GetParticleSet();
    for(int i = 0; i < pSet->GetParticleCount(); i++){
        ComputeVorticityForceFor(data, i);
    }
}

__global__ void ComputeVorticityForceKernel(PbfSolverData2 *data){
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    ParticleSet2 *pSet = data->sphData->sphpSet->GetParticleSet();
    if(i < pSet->GetParticleCount()){
        ComputeVorticityForceFor(data, i);
    }
}

__host__ void ComputeVorticityForceGPU(PbfSolverData2 *data){
    ParticleSet2 *pSet = data->sphData->sphpSet->GetParticleSet();
    int N = pSet->GetParticleCount();
    GPULaunch(N, ComputeVorticityForceKernel, data);
}

__host__ void ComputeVorticityCPU(PbfSolverData2 *data){
    ParticleSet2 *pSet = data->sphData->sphpSet->GetParticleSet();
    for(int i = 0; i < pSet->GetParticleCount(); i++){
        ComputeVorticityFor(data, i);
    }
}

__global__ void ComputeVorticityKernel(PbfSolverData2 *data){
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    ParticleSet2 *pSet = data->sphData->sphpSet->GetParticleSet();
    if(i < pSet->GetParticleCount()){
        ComputeVorticityFor(data, i);
    }
}

__host__ void ComputeVorticityGPU(PbfSolverData2 *data){
    ParticleSet2 *pSet = data->sphData->sphpSet->GetParticleSet();
    int N = pSet->GetParticleCount();
    GPULaunch(N, ComputeVorticityKernel, data);
}


__host__ void AdvancePBF(PbfSolverData2 *data, Float timeIntervalInSeconds,
                         unsigned int predictIterations, int use_cpu)
{
    if(data->vorticityStr > 0){
        if(use_cpu){
            ComputeVorticityCPU(data);
            ComputeVorticityForceCPU(data);
        }else{
            ComputeVorticityGPU(data);
            ComputeVorticityForceGPU(data);
        }
    }
    
    if(use_cpu){
        ComputePredictedPositionsCPU(data, timeIntervalInSeconds);
        UpdateGridDistributionCPU(data->sphData);
        //ComputeDensityCPU(data->sphData);
    }else{
        ComputePredictedPositionsGPU(data, timeIntervalInSeconds);
        UpdateGridDistributionGPU(data->sphData);
        //ComputeDensityGPU(data->sphData);
    }
    
    data->sphData->sphpSet->ResetHigherLevel();
    
    for(unsigned int i = 0; i < predictIterations; i++){
        if(use_cpu){
            ComputeLambdaCPU(data);
            ComputeDeltaPCPU(data, timeIntervalInSeconds);
        }else{
            ComputeLambdaGPU(data);
            ComputeDeltaPGPU(data, timeIntervalInSeconds);
        }
    }
    
    if(use_cpu)
        ComputePseudoViscosityInterpolationCPU(data->sphData, timeIntervalInSeconds);
    else
        ComputePseudoViscosityInterpolationGPU(data->sphData, timeIntervalInSeconds);
}