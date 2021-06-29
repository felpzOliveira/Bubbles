#include <sph_solver.h>

__bidevice__ Float ComputeAverageTemperature(SphSolverData2 *data){
    ParticleSet2 *pSet = data->sphpSet->GetParticleSet();
    int count = pSet->GetParticleCount();
    Float Tt = 0;
    for(int i = 0; i < count; i++){
        Tt += pSet->GetParticleTemperature(i);
    }
    
    return Tt / (Float)count;
}

/**************************************************************/
//      D E N S I T Y     A N D    P R E S S U R E            //
/**************************************************************/
__bidevice__ Float ComputePressureValue(SphSolverData2 *data, Float di){
    ParticleSet2 *pSet = data->sphpSet->GetParticleSet();
    Float targetDensity = data->sphpSet->GetTargetDensity();
    Float eosScale = targetDensity * data->soundSpeed * data->soundSpeed;
    Float eosExponent = data->eosExponent;
    Float p = eosScale / eosExponent * (std::pow((di / targetDensity), eosExponent) - 1.0);
    if(p < 0){
        p *= data->negativePressureScale;
    }
    
    return p;
}

__bidevice__ void ComputePressureFor(SphSolverData2 *data, int particleId, Float di){
    ParticleSet2 *pSet = data->sphpSet->GetParticleSet();
    pSet->SetParticlePressure(particleId, ComputePressureValue(data, di));
}

__bidevice__ Float ComputeDensityForPoint(SphSolverData2 *data, const vec2f &pi){
    int *neighbors = nullptr;
    ParticleSet2 *pSet = data->sphpSet->GetParticleSet();
    unsigned int cellId = data->domain->GetLinearHashedPosition(pi);
    
    int count = data->domain->GetNeighborsOf(cellId, &neighbors);
    Float sphRadius = data->sphpSet->GetKernelRadius();
    SphStdKernel2 kernel(sphRadius);
    
    Float sum = 0;
    for(int i = 0; i < count; i++){
        Cell2 *cell = data->domain->GetCell(neighbors[i]);
        ParticleChain *pChain = cell->GetChain();
        int size = cell->GetChainLength();
        
        for(int j = 0; j < size; j++){
            vec2f pj = pSet->GetParticlePosition(pChain->pId);
            Float distance = Distance(pi, pj);
            sum += kernel.W(distance);
            pChain = pChain->next;
        }
    }
    
    Float di = sum * pSet->GetMass();
    return di;
}

__bidevice__ Float ComputeDensityForParticle(SphSolverData2 *data, int particleId){
    ParticleSet2 *pSet = data->sphpSet->GetParticleSet();
    vec2f pi = pSet->GetParticlePosition(particleId);
    Bucket *bucket = pSet->GetParticleBucket(particleId);
    
    Float sphRadius = data->sphpSet->GetKernelRadius();
    SphStdKernel2 kernel(sphRadius);
    
    Float sum = 0;
    for(int i = 0; i < bucket->Count(); i++){
        int j = bucket->Get(i);
        vec2f pj = pSet->GetParticlePosition(j);
        Float distance = Distance(pi, pj);
        sum += kernel.W(distance);
    }

    if(IsZero(sum)){
        printf("Zero summation for particle %d\n", particleId);
        //sum = kernel.W(0);
    }

    Float di = sum * pSet->GetMass();
    return di;
}

__bidevice__ void ComputeDensityFor(SphSolverData2 *data, int particleId,
                                    int compute_pressure)
{
    ParticleSet2 *pSet = data->sphpSet->GetParticleSet();
    Float di = ComputeDensityForParticle(data, particleId);
    if(compute_pressure != 0){
        ComputePressureFor(data, particleId, di);
    }
    
    AssertA(!IsZero(di), "Zero density");
    
    pSet->SetParticleDensity(particleId, di);
}

/**************************************************************/
//            F O R C E S    C O M P U T A T I O N            //
/**************************************************************/
__bidevice__ void ComputeNonPressureForceFor(SphSolverData2 *data, int particleId){
    ParticleSet2 *pSet = data->sphpSet->GetParticleSet();
    vec2f pi = pSet->GetParticlePosition(particleId);
    vec2f vi = pSet->GetParticleVelocity(particleId);
    Bucket *bucket = pSet->GetParticleBucket(particleId);
    
    Float mass = pSet->GetMass();
    Float mass2 = mass * mass;
    Float sphRadius = data->sphpSet->GetKernelRadius();
    SphSpikyKernel2 kernel(sphRadius);
    
    vec2f fi(0,0);
    const vec2f Gravity2D(0, -9.8);
    fi += mass * Gravity2D;
    fi += -data->dragCoefficient * vi;
    
    for(int i = 0; i < bucket->Count(); i++){
        int j = bucket->Get(i);
        vec2f pj = pSet->GetParticlePosition(j);
        vec2f vj = pSet->GetParticleVelocity(j);
        Float dj = pSet->GetParticleDensity(j);
        
        Float dist = Distance(pi, pj);
        fi += data->viscosity * mass2 * (vj - vi) * kernel.d2W(dist) / dj;
    }
    
    pSet->SetParticleForce(particleId, fi);
}

__bidevice__ void ComputePressureForceFor(SphSolverData2 *data, int particleId){
    ParticleSet2 *pSet = data->sphpSet->GetParticleSet();
    vec2f pi = pSet->GetParticlePosition(particleId);
    Bucket *bucket = pSet->GetParticleBucket(particleId);
    
    Float mass = pSet->GetMass();
    Float mass2 = pSet->GetMass() * pSet->GetMass();
    Float sphRadius = data->sphpSet->GetKernelRadius();
    SphSpikyKernel2 kernel(sphRadius);
    
    vec2f fi = pSet->GetParticleForce(particleId);
    Float poi = pSet->GetParticlePressure(particleId);
    Float di = pSet->GetParticleDensity(particleId);
    Float di2 = di * di;
    
    AssertA(!IsZero(di2), "Zero source density {ComputePressureForceFor}");
    vec2f ti(0,0);
    for(int i = 0; i < bucket->Count(); i++){
        int j = bucket->Get(i);
        if(particleId != j){
            vec2f pj = pSet->GetParticlePosition(j);
            Float dj = pSet->GetParticleDensity(j);
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
    pSet->SetParticleForce(particleId, fi);
}

__bidevice__ void ComputeAllForcesFor(SphSolverData2 *data, int particleId, 
                                      Float timeStep, int extended, int integrate)
{
    ParticleSet2 *pSet = data->sphpSet->GetParticleSet();
    vec2f pi = pSet->GetParticlePosition(particleId);
    Bucket *bucket = pSet->GetParticleBucket(particleId);
    
    Float mass = pSet->GetMass();
    Float mass2 = pSet->GetMass() * pSet->GetMass();
    Float sphRadius = data->sphpSet->GetKernelRadius();
    SphSpikyKernel2 kernel(sphRadius);
    
    vec2f fi(0,0);
    Float poi = pSet->GetParticlePressure(particleId);
    Float di = pSet->GetParticleDensity(particleId);
    Float di2 = di * di;
    Float Ti = 0;
    if(extended) Ti = pSet->GetParticleTemperature(particleId);
    
    vec2f vi = pSet->GetParticleVelocity(particleId);
    Float Tout = Ti;
    
    const vec2f Gravity2D(0, -9.8);
    fi += mass * Gravity2D;
    fi += -data->dragCoefficient * vi;
    
    AssertA(!IsZero(di2), "Zero source density {ComputePressureForceFor}");
    vec2f ti(0,0);
    vec2f ni(0,0);
    Float Tidt = 0;
    Float Dc = 0.15; // Helium
    for(int i = 0; i < bucket->Count(); i++){
        int j = bucket->Get(i);
        if(particleId != j){
            vec2f pj = pSet->GetParticlePosition(j);
            Float dj = pSet->GetParticleDensity(j);
            Float dj2 = dj * dj;
            vec2f vj = pSet->GetParticleVelocity(j);
            Float poj = pSet->GetParticlePressure(j);
            Float Tj = 0;
            if(extended) Tj = pSet->GetParticleTemperature(j);
            
            Float dist = Distance(pi, pj);
            bool valid = IsWithinSpiky(dist, sphRadius);
            
            if(valid){
                AssertA(!IsZero(dj2), "Zero neighbor density {ComputePressureForceFor}");
                if(dist > 0 && !IsZero(dist)){
                    vec2f dir = (pj - pi) / dist;
                    vec2f gradij = kernel.gradW(dist, dir);
                    ti += mass2 * (poi / di2 + poj / dj2) * gradij;
                    ni += (mass / dj) * gradij;
                    
                    if(extended){
                        Float pdotgrad = Dot(pi - pj, gradij);
                        pdotgrad /= (dist * dist + 0.0001);
                        Float dt = mass2 / (di * dj) * Dc * (Ti - Tj) * pdotgrad;
                        Tidt += dt;
                    }
                }
                
                fi += data->viscosity * mass2 * (vj - vi) * kernel.d2W(dist) / dj;
            }
        }
    }
    
    if(extended){
        SphStdKernel2 iKernel(data->sphpSet->GetKernelRadius());
        Float dk = di;
        Float dk2 = dk * dk;
        Float v0 = pSet->GetParticleV0(particleId);
        Float pok = ComputePressureValue(data, dk);
        Float facK = 0.6 * data->sphpSet->GetKernelRadius(); // TODO: Fix this
        vec2f gradik = iKernel.gradW(facK, ni);
        
        // TODO: Radiation term?
        // Tidt = -Ti / Dr ?
        
        Float mk = v0 * di;
        Float pa = 1.0;
        vec2f pp = (mk * mass) * (poi / dk2 + pok / di2) * gradik;
        ti = ti - pp;
        fi = fi - pa * ni;
        Tout += Tidt;
        AssertA(!IsNaN(Tout), "NaN temperature");
        pSet->SetParticleTemperature(particleId, Tout);
    }
    
    fi = fi - ti;
    pSet->SetParticleForce(particleId, fi);
    if(pSet->HasNormal())
        pSet->SetParticleNormal(particleId, ni);
    
    if(integrate)
        TimeIntegrationFor(data, particleId, timeStep, extended);
}

/**************************************************************/
//               T I M E    I N T E G R A T I O N             //
/**************************************************************/
__bidevice__ void TimeIntegrationFor(SphSolverData2 *data, int particleId, 
                                     Float timeStep, int extended)
{
    ParticleSet2 *pSet = data->sphpSet->GetParticleSet();
    vec2f vi = pSet->GetParticleVelocity(particleId);
    vec2f fi = pSet->GetParticleForce(particleId);
    vec2f pi = pSet->GetParticlePosition(particleId);
    Float di = pSet->GetParticleDensity(particleId);
    Float mass = pSet->GetMass();
    vec2f oi = pi;
    
    Float Cb = 100.0;
    vec2f aTi(0,0);
    if(extended){
        Float Ti = pSet->GetParticleTemperature(particleId);
        aTi = Cb * Ti * vec2f(0, 1);
    }
    
    vec2f aFi = fi / mass;
    
    vec2f at = aTi + aFi;
    
    vi += timeStep * at;
    pi += timeStep * vi;

    vec2f len = data->domain->GetCellSize();
    Float dist = Distance(pi, oi);
    Float minLen = Min(len[0], len[1]);
    if(dist >= minLen * 0.5){
        data->sphpSet->SetHigherLevel();
    }
    
    //TODO: Figure out restitution coefficient
    data->collider->ResolveCollision(pSet->GetRadius(), 0.75, &pi, &vi);
    pSet->SetParticlePosition(particleId, pi);
    pSet->SetParticleVelocity(particleId, vi);
}

__bidevice__ void ComputePseudoViscosityAggregationKernelFor(SphSolverData2 *data,
                                                             int particleId)
{
    ParticleSet2 *pSet = data->sphpSet->GetParticleSet();
    vec2f vi = pSet->GetParticleVelocity(particleId);
    vec2f pi = pSet->GetParticlePosition(particleId);
    Bucket *bucket = pSet->GetParticleBucket(particleId);
    Float mass = pSet->GetMass();
    Float sphRadius = data->sphpSet->GetKernelRadius();
    SphSpikyKernel2 kernel(sphRadius);
    
    vec2f smoothedVi(0);
    Float weightSum = 0;
    for(int i = 0; i < bucket->Count(); i++){
        int j = bucket->Get(i);
        vec2f pj = pSet->GetParticlePosition(j);
        Float dj = pSet->GetParticleDensity(j);
        vec2f vj = pSet->GetParticleVelocity(j);
        Float dist = Distance(pi, pj);
        Float wj = mass / dj * kernel.W(dist);
        weightSum += wj;
        
        smoothedVi += wj * vj;
    }
    
    if(weightSum > 0){
        smoothedVi *= (1.0 / weightSum);
    }
    
    data->smoothedVelocities[particleId] = smoothedVi;
}

__bidevice__ void ComputePseudoViscosityInterpolationKernelFor(SphSolverData2 *data,
                                                               int particleId,
                                                               Float timeStep)
{
    ParticleSet2 *pSet = data->sphpSet->GetParticleSet();
    vec2f vi = pSet->GetParticleVelocity(particleId);
    vec2f smoothedVi = data->smoothedVelocities[particleId];
    Float smoothFactor = Clamp(timeStep * data->pseudoViscosity, 0.0, 1.0);
    vi = Lerp(vi, smoothedVi, smoothFactor);
    pSet->SetParticleVelocity(particleId, vi);
}

/**************************************************************/
//                 G R I D     D I S T R I B U T I O N        //
/**************************************************************/
__host__ void UpdateGridDistributionCPU(SphSolverData2 *data){
    Grid2 *grid = data->domain;
    ParticleSet2 *pSet = data->sphpSet->GetParticleSet();
    AssertA(grid, "SphSolver2 has no domain for UpdateGridDistribution");
    if(data->frame_index == 0){
        for(int i = 0; i < grid->GetCellCount(); i++){
            grid->DistributeToCell(pSet, i);
        }
    }else{
        for(int i = 0; i < grid->GetCellCount(); i++){
            grid->DistributeToCellOpt(pSet, i);
        }
        
        for(int i = 0; i < grid->GetCellCount(); i++){
            grid->SwapCellList(i);
        }
    }
    
    int pCount = pSet->GetParticleCount();
    Float kernelRadius = data->sphpSet->GetKernelRadius();
    for(int i = 0; i < pCount; i++){
        grid->DistributeParticleBucket(pSet, i, kernelRadius);
    }
    
    data->frame_index = 1;
}

__global__ void UpdateGridDistributionKernel(Grid2 *grid, ParticleSet2 *pSet, int index){
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if(i < grid->GetCellCount()){
        if(index == 0)
            grid->DistributeToCell(pSet, i);
        else
            grid->DistributeToCellOpt(pSet, i);
    }
}

__global__ void SwapGridStatesKernel(Grid2 *grid){
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if(i < grid->GetCellCount()){
        grid->SwapCellList(i);
    }
}

__global__ void UpdateParticlesBuckets(Grid2 *grid, ParticleSet2 *pSet, Float kernelRadius){
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if(i < pSet->GetParticleCount()){
        grid->DistributeParticleBucket(pSet, i, kernelRadius);
    }
}

__host__ void UpdateGridDistributionGPU(SphSolverData2 *data){
    Grid2 *grid = data->domain;
    ParticleSet2 *pSet = data->sphpSet->GetParticleSet();
    AssertA(grid, "SphSolver2 has no domain for UpdateGridDistribution");
    if(data->sphpSet->requiresHigherLevelUpdate){
        for(int i = 0; i < data->domain->GetCellCount(); i++){
            data->domain->DistributeResetCell(i);
        }
        data->domain->DistributeByParticle(pSet);
    }else{
        int N = grid->GetCellCount();
        int index = data->frame_index;
        GPULaunch(N, UpdateGridDistributionKernel, grid, pSet, index);
        
        if(data->frame_index == 1){
            GPULaunch(N, SwapGridStatesKernel, grid);
        }
    }
    
    int pCount = pSet->GetParticleCount();
    Float kernelRadius = data->sphpSet->GetKernelRadius();
    GPULaunch(pCount, UpdateParticlesBuckets, grid, pSet, kernelRadius);
    
    data->frame_index = 1;
}

__host__ void ComputeDensityCPU(SphSolverData2 *data, int compute_pressure){
    ParticleSet2 *pSet = data->sphpSet->GetParticleSet();
    AssertA(pSet, "SphSolver2 has no valid particle set for ComputeDensity");
    AssertA(pSet->GetParticleCount() > 0, "SphSolver2 has no particles for ComputeDensity");
    for(int i = 0; i < pSet->GetParticleCount(); i++){
        ComputeDensityFor(data, i, compute_pressure);
    }
}

__global__ void ComputeDensityKernel(SphSolverData2 *data, int compute_pressure){
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    ParticleSet2 *pSet = data->sphpSet->GetParticleSet();
    if(i < pSet->GetParticleCount()){
        ComputeDensityFor(data, i, compute_pressure);
    }
}

__host__ void ComputeDensityGPU(SphSolverData2 *data, int compute_pressure){
    ParticleSet2 *pSet = data->sphpSet->GetParticleSet();
    int N = pSet->GetParticleCount();
    GPULaunch(N, ComputeDensityKernel, data, compute_pressure);
}

__global__ void ComputePseudoViscosityAggregationKernel(SphSolverData2 *data)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    ParticleSet2 *pSet = data->sphpSet->GetParticleSet();
    if(i < pSet->GetParticleCount()){
        ComputePseudoViscosityAggregationKernelFor(data, i);
    }
}

__global__ void ComputePseudoViscosityInterpolationKernel(SphSolverData2 *data,
                                                          Float timeStep)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    ParticleSet2 *pSet = data->sphpSet->GetParticleSet();
    if(i < pSet->GetParticleCount()){
        ComputePseudoViscosityInterpolationKernelFor(data, i, timeStep);
    }
}

__host__ void ComputePseudoViscosityInterpolationGPU(SphSolverData2 *data, Float timeStep){
    Float scale = data->pseudoViscosity * timeStep;
    if(scale > 0.1){
        ParticleSet2 *pSet = data->sphpSet->GetParticleSet();
        int N = pSet->GetParticleCount();
        GPULaunch(N, ComputePseudoViscosityAggregationKernel, data);
        
        GPULaunch(N, ComputePseudoViscosityInterpolationKernel, data, timeStep);
    }
}

__host__ void ComputePseudoViscosityInterpolationCPU(SphSolverData2 *data, Float timeStep){
    Float scale = data->pseudoViscosity * timeStep;
    if(scale > 0.1){
        ParticleSet2 *pSet = data->sphpSet->GetParticleSet();
        int N = pSet->GetParticleCount();
        for(int i = 0; i < N; i++){
            ComputePseudoViscosityAggregationKernelFor(data, i);
        }
        
        for(int i = 0; i < N; i++){
            ComputePseudoViscosityInterpolationKernelFor(data, i, timeStep);
        }
    }
}

__host__ void ComputePressureForceCPU(SphSolverData2 *data, Float timeStep,
                                          int integrate)
{
    ParticleSet2 *pSet = data->sphpSet->GetParticleSet();
    AssertA(pSet, "SphSolver2 has no valid particle set for ComputePressureForce");
    AssertA(pSet->GetParticleCount() > 0,
            "SphSolver2 has no particles for ComputePressureForce");
    for(int i = 0; i < pSet->GetParticleCount(); i++){
        ComputeAllForcesFor(data, i, timeStep, 0, integrate);
    }
}

__host__ void ComputeNonPressureForceCPU(SphSolverData2 *data){
    ParticleSet2 *pSet = data->sphpSet->GetParticleSet();
    AssertA(pSet, "SphSolver2 has no valid particle set for ComputeNonPressureForce");
    AssertA(pSet->GetParticleCount() > 0,
            "SphSolver2 has no particles for ComputeNonPressureForce");
    for(int i = 0; i < pSet->GetParticleCount(); i++){
        ComputeNonPressureForceFor(data, i);
    }
}


__global__ void ComputeNonPressureForceKernel(SphSolverData2 *data){
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    ParticleSet2 *pSet = data->sphpSet->GetParticleSet();
    if(i < pSet->GetParticleCount()){
        ComputeNonPressureForceFor(data, i);
    }
}

__host__ void ComputeNonPressureForceGPU(SphSolverData2 *data){
    ParticleSet2 *pSet = data->sphpSet->GetParticleSet();
    int N = pSet->GetParticleCount();
    GPULaunch(N, ComputeNonPressureForceKernel, data);
}

__global__ void ComputePressureForceKernel(SphSolverData2 *data, Float timeStep,
                                           int integrate)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    ParticleSet2 *pSet = data->sphpSet->GetParticleSet();
    if(i < pSet->GetParticleCount()){
        ComputeAllForcesFor(data, i, timeStep, 0, integrate);
    }
}

__host__ void ComputePressureForceGPU(SphSolverData2 *data, Float timeStep, int integrate){
    ParticleSet2 *pSet = data->sphpSet->GetParticleSet();
    AssertA(pSet, "SphSolver2 has no valid particle set for ComputePressureForce");
    AssertA(pSet->GetParticleCount() > 0,
            "SphSolver2 has no particles for ComputePressureForce");
    int N = pSet->GetParticleCount();
    GPULaunch(N, ComputePressureForceKernel, data, timeStep, integrate);
}

__host__ void TimeIntegrationCPU(SphSolverData2 *data, Float timeStep, int extended){
    ParticleSet2 *pSet = data->sphpSet->GetParticleSet();
    AssertA(pSet, "SphSolver2 has no valid particle set for TimeIntegration");
    AssertA(pSet->GetParticleCount() > 0, "SphSolver2 has no particles for TimeIntegration");
    for(int i = 0; i < pSet->GetParticleCount(); i++){
        TimeIntegrationFor(data, i, timeStep, extended);
    }
}

__global__ void TimeIntegrationKernel(SphSolverData2 *data, Float timeStep, int extended){
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    ParticleSet2 *pSet = data->sphpSet->GetParticleSet();
    if(i < pSet->GetParticleCount()){
        TimeIntegrationFor(data, i, timeStep, extended);
    }
}

__host__ void TimeIntegrationGPU(SphSolverData2 *data, Float timeStep, int extended){
    ParticleSet2 *pSet = data->sphpSet->GetParticleSet();
    int N = pSet->GetParticleCount();
    GPULaunch(N, TimeIntegrationKernel, data, timeStep, extended);
}

__bidevice__ void ComputeInitialTemperatureFor(SphSolverData2 *data, int particleId,
                                               Float Tmin, Float Tmax, int maxLevel)
{
    ParticleSet2 *pSet = data->sphpSet->GetParticleSet();
    vec2f pi = pSet->GetParticlePosition(particleId);
    unsigned int cellId = data->domain->GetLinearHashedPosition(pi);
    Cell<Bounds2f> *cell = data->domain->GetCell(cellId);
    int level = cell->GetLevel();
    AssertA(level > 0 && maxLevel > 1, "Zero level cell");
    Float alpha = (Float)(level - 1.0) / (Float)(maxLevel - 1.0);
    Float Ti = Lerp(alpha, Tmin, Tmax);
    
    pSet->SetParticleTemperature(particleId, Ti);
}

__host__ void ComputeInitialTemperatureMapCPU(SphSolverData2 *data, Float Tmin,
                                                  Float Tmax, int maxLevel)
{
    ParticleSet2 *pSet = data->sphpSet->GetParticleSet();
    int count = pSet->GetParticleCount();
    for(int i = 0; i < count; i++){
        ComputeInitialTemperatureFor(data, i, Tmin, Tmax, maxLevel);
    }
}

__global__ void ComputeInitialTemperatureMapKernel(SphSolverData2 *data, Float Tmin, 
                                                   Float Tmax, int maxLevel)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    ParticleSet2 *pSet = data->sphpSet->GetParticleSet();
    if(i < pSet->GetParticleCount()){
        ComputeInitialTemperatureFor(data, i, Tmin, Tmax, maxLevel);
    }
}

__host__ void ComputeInitialTemperatureMapGPU(SphSolverData2 *data, Float Tmin, 
                                              Float Tmax, int maxLevel)
{
    ParticleSet2 *pSet = data->sphpSet->GetParticleSet();
    int N = pSet->GetParticleCount();
    GPULaunch(N, ComputeInitialTemperatureMapKernel, data, Tmin, Tmax, maxLevel);
}
