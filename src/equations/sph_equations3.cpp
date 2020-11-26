#include <sph_solver.h>
#include <profiler.h>

/**************************************************************/
//      D E N S I T Y     A N D    P R E S S U R E            //
/**************************************************************/
__bidevice__ Float ComputePressureValue(SphSolverData3 *data, Float di){
    ParticleSet3 *pSet = data->sphpSet->GetParticleSet();
    Float targetDensity = data->sphpSet->GetTargetDensity();
    Float eosScale = targetDensity * data->soundSpeed * data->soundSpeed;
    Float eosExponent = data->eosExponent;
    Float p = eosScale / eosExponent * (std::pow((di / targetDensity), eosExponent) - 1.0);
    if(p < 0){
        p *= data->negativePressureScale;
    }
    
    return p;
}

__bidevice__ void ComputePressureFor(SphSolverData3 *data, int particleId, Float di){
    ParticleSet3 *pSet = data->sphpSet->GetParticleSet();
    pSet->SetParticlePressure(particleId, ComputePressureValue(data, di));
}

__bidevice__ void ComputeDensityFor(SphSolverData3 *data, int particleId,
                                    int compute_pressure)
{
    ParticleSet3 *pSet = data->sphpSet->GetParticleSet();
    vec3f pi = pSet->GetParticlePosition(particleId);
    Bucket *bucket = pSet->GetParticleBucket(particleId);
    AssertA(bucket != nullptr, "Invalid bucket received");
    
    Float sphRadius = data->sphpSet->GetKernelRadius();
    SphStdKernel3 kernel(sphRadius);
    
    Float sum = 0;
    
    /* profiler interactions statics */
    int scount = 0;
    for(int i = 0; i < bucket->Count(); i++){
        int j = bucket->Get(i);
        AssertA(j < pSet->GetParticleCount(), "Invalid id returned by bucket");
        vec3f pj = pSet->GetParticlePosition(j);
        Float distance = Distance(pi, pj);
        sum += kernel.W(distance);
        scount ++;
    }
    
    Float di = sum * pSet->GetMass();
    if(compute_pressure != 0){
        ComputePressureFor(data, particleId, di);
    }
    
    ProfilerUpdate(scount, particleId);
    AssertA(!IsZero(di), "Zero density");
    
    pSet->SetParticleDensity(particleId, di);
}

/**************************************************************/
//            F O R C E S    C O M P U T A T I O N            //
/**************************************************************/
__bidevice__ void ComputeNonPressureForceFor(SphSolverData3 *data, int particleId){
    ParticleSet3 *pSet = data->sphpSet->GetParticleSet();
    vec3f pi = pSet->GetParticlePosition(particleId);
    vec3f vi = pSet->GetParticleVelocity(particleId);
    
    Bucket *bucket = pSet->GetParticleBucket(particleId);
    
    Float mass = pSet->GetMass();
    Float mass2 = mass * mass;
    Float sphRadius = data->sphpSet->GetKernelRadius();
    SphSpikyKernel3 kernel(sphRadius);
    
    vec3f fi(0);
    const vec3f Gravity3D(0, -9.8, 0);
    fi += mass * Gravity3D;
    fi += -data->dragCoefficient * vi;
    
    for(int i = 0; i < bucket->Count(); i++){
        int j = bucket->Get(i);
        
        vec3f pj = pSet->GetParticlePosition(j);
        vec3f vj = pSet->GetParticleVelocity(j);
        Float dj = pSet->GetParticleDensity(j);
        
        Float dist = Distance(pi, pj);
        fi += data->viscosity * mass2 * (vj - vi) * kernel.d2W(dist) / dj;
    }
    
    pSet->SetParticleForce(particleId, fi);
}

__bidevice__ void ComputePressureForceFor(SphSolverData3 *data, int particleId){
    ParticleSet3 *pSet = data->sphpSet->GetParticleSet();
    vec3f pi = pSet->GetParticlePosition(particleId);
    Bucket *bucket = pSet->GetParticleBucket(particleId);
    
    Float mass = pSet->GetMass();
    Float mass2 = pSet->GetMass() * pSet->GetMass();
    Float sphRadius = data->sphpSet->GetKernelRadius();
    SphSpikyKernel3 kernel(sphRadius);
    
    vec3f fi = pSet->GetParticleForce(particleId);
    Float poi = pSet->GetParticlePressure(particleId);
    Float di = pSet->GetParticleDensity(particleId);
    Float di2 = di * di;
    
    AssertA(!IsZero(di2), "Zero source density {ComputePressureForceFor}");
    vec3f ti(0);
    for(int i = 0; i < bucket->Count(); i++){
        int j = bucket->Get(i);
        if(particleId != j){
            vec3f pj = pSet->GetParticlePosition(j);
            Float dj = pSet->GetParticleDensity(j);
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
    pSet->SetParticleForce(particleId, fi);
}

__bidevice__ void ComputeNormalFor(SphSolverData3 *data, int particleId){
    vec3f ni(0);
    ParticleSet3 *pSet = data->sphpSet->GetParticleSet();
    vec3f pi = pSet->GetParticlePosition(particleId);
    Bucket *bucket = pSet->GetParticleBucket(particleId);
    
    Float mass = pSet->GetMass();
    Float sphRadius = data->sphpSet->GetKernelRadius();
    
    SphSpikyKernel3 kernel(sphRadius);
    
    for(int i = 0; i < bucket->Count(); i++){
        int j = bucket->Get(i);
        vec3f pj = pSet->GetParticlePosition(j);
        Float dj = pSet->GetParticleDensity(j);
        Float dist = Distance(pi, pj);
        bool valid = IsWithinSpiky(dist, sphRadius);
        if(valid){
            if(dist > 0 && !IsZero(dist)){
                vec3f dir = (pj - pi) / dist;
                vec3f gradij = kernel.gradW(dist, dir);
                ni += (mass / dj) * gradij;
            }
        }
    }
    
    Float sqni = ni.LengthSquared();
    if(sqni > 1e-8){
        ni = Normalize(ni);
    }
    pSet->SetParticleNormal(particleId, ni);
}

__bidevice__ void ComputeAllForcesFor(SphSolverData3 *data, int particleId, 
                                      Float timeStep, int extended)
{
    ParticleSet3 *pSet = data->sphpSet->GetParticleSet();
    vec3f pi = pSet->GetParticlePosition(particleId);
    Bucket *bucket = pSet->GetParticleBucket(particleId);
    
    Float mass = pSet->GetMass();
    Float mass2 = pSet->GetMass() * pSet->GetMass();
    Float sphRadius = data->sphpSet->GetKernelRadius();
    SphSpikyKernel3 kernel(sphRadius);
    
    vec3f fi(0);
    Float poi = pSet->GetParticlePressure(particleId);
    Float di = pSet->GetParticleDensity(particleId);
    Float di2 = di * di;
    Float Ti = 0;
    if(extended) Ti = pSet->GetParticleTemperature(particleId);
    
    vec3f vi = pSet->GetParticleVelocity(particleId);
    Float Tout = Ti;
    
    const vec3f Gravity3D(0, -9.8, 0);
    fi += mass * Gravity3D;
    fi += -data->dragCoefficient * vi;
    
    AssertA(!IsZero(di2), "Zero source density {ComputePressureForceFor}");
    vec3f ti(0);
    vec3f ni(0);
    Float Tidt = 0;
    Float Dc = 0.15; // Helium
    for(int i = 0; i < bucket->Count(); i++){
        int j = bucket->Get(i);
        if(particleId != j){
            vec3f pj = pSet->GetParticlePosition(j);
            Float dj = pSet->GetParticleDensity(j);
            Float dj2 = dj * dj;
            vec3f vj = pSet->GetParticleVelocity(j);
            Float poj = pSet->GetParticlePressure(j);
            Float Tj = 0;
            if(extended) Tj = pSet->GetParticleTemperature(j);
            
            Float dist = Distance(pi, pj);
            bool valid = IsWithinSpiky(dist, sphRadius);
            
            if(valid){
                AssertA(!IsZero(dj2), "Zero neighbor density {ComputePressureForceFor}");
                if(dist > 0 && !IsZero(dist)){
                    vec3f dir = (pj - pi) / dist;
                    vec3f gradij = kernel.gradW(dist, dir);
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
        SphStdKernel3 iKernel(data->sphpSet->GetKernelRadius());
        Float dk = di;
        Float dk2 = dk * dk;
        Float v0 = pSet->GetParticleV0(particleId);
        Float pok = ComputePressureValue(data, dk);
        Float facK = 0.6 * data->sphpSet->GetKernelRadius(); // TODO: Fix this
        vec3f gradik = iKernel.gradW(facK, ni);
        
        // TODO: Radiation term?
        // Tidt = -Ti / Dr ?
        
        Float mk = v0 * di;
        Float pa = 1.0;
        vec3f pp = (mk * mass) * (poi / dk2 + pok / di2) * gradik;
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
    
    TimeIntegrationFor(data, particleId, timeStep, extended);
}

/**************************************************************/
//               T I M E    I N T E G R A T I O N             //
/**************************************************************/
__bidevice__ void TimeIntegrationFor(SphSolverData3 *data, int particleId, 
                                     Float timeStep, int extended)
{
    ParticleSet3 *pSet = data->sphpSet->GetParticleSet();
    vec3f vi = pSet->GetParticleVelocity(particleId);
    vec3f fi = pSet->GetParticleForce(particleId);
    vec3f pi = pSet->GetParticlePosition(particleId);
    Float di = pSet->GetParticleDensity(particleId);
    Float mass = pSet->GetMass();
    vec3f oi = pi;
    
    Float Cb = 100.0;
    vec3f aTi(0);
    if(extended){
        Float Ti = pSet->GetParticleTemperature(particleId);
        aTi = Cb * Ti * vec3f(0, 1, 0);
    }
    
    vec3f aFi = fi / mass;
    
    vec3f at = aTi + aFi;
    
    vi += timeStep * at;
    pi += timeStep * vi;
    
    vec3f opi = pi;
    
    data->collider->ResolveCollision(pSet->GetRadius(), 0.6, &pi, &vi);
    
    if(!Inside(pi, data->domain->bounds)){
        vec3f pMin = data->domain->bounds.pMin;
        vec3f pMax = data->domain->bounds.pMax;
        printf("Point pi outside: {%g %g %g}, {%g %g %g} x {%g %g %g} [%g %g %g]\n",
               pi.x, pi.y, pi.z, pMin.x, pMin.y, pMin.z, 
               pMax.x, pMax.y, pMax.z, opi.x, opi.y, opi.z);
        
    }
    
    AssertA(Inside(pi, data->domain->bounds), "Particle outside domain");
    
    vec3f len = data->domain->GetCellSize();
    Float dist = Distance(pi, oi);
    Float minLen = Min(len[0], Min(len[1], len[2]));
    if(dist > minLen){
        data->sphpSet->SetHigherLevel();
    }
    
    pSet->SetParticlePosition(particleId, pi);
    pSet->SetParticleVelocity(particleId, vi);
}

/**************************************************************/
//                   C P U    W R A P P E R S                 //
/**************************************************************/
__host__ void UpdateGridDistributionCPU(SphSolverData3 *data){
    Grid3 *grid = data->domain;
    AssertA(grid, "SphSolver3 has no domain for UpdateGridDistribution");
    ParticleSet3 *pSet = data->sphpSet->GetParticleSet();
    if(data->sphpSet->requiresHigherLevelUpdate){
        //printf("Performing full distribution by excessive delta\n");
        for(int i = 0; i < data->domain->GetCellCount(); i++){
            data->domain->DistributeResetCell(i);
        }
        data->domain->DistributeByParticle(pSet);
    }else{
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
    }
    
    int pCount = pSet->GetParticleCount();
    Float kernelRadius = data->sphpSet->GetKernelRadius();
    for(int i = 0; i < pCount; i++){
        grid->DistributeParticleBucket(pSet, i, kernelRadius);
    }
    
    data->frame_index = 1;
}

__bidevice__ void ComputeNormalCPU(SphSolverData3 *data){
    ParticleSet3 *pSet = data->sphpSet->GetParticleSet();
    AssertA(pSet, "SphSolver3 has no valid particle set for ComputeDensity");
    AssertA(pSet->GetParticleCount() > 0, "SphSolver3 has no particles for ComputeDensity");
    for(int i = 0; i < pSet->GetParticleCount(); i++){
        ComputeNormalFor(data, i);
    }
}

__bidevice__ void ComputeDensityCPU(SphSolverData3 *data, int compute_pressure){
    ParticleSet3 *pSet = data->sphpSet->GetParticleSet();
    AssertA(pSet, "SphSolver3 has no valid particle set for ComputeDensity");
    AssertA(pSet->GetParticleCount() > 0, "SphSolver3 has no particles for ComputeDensity");
    for(int i = 0; i < pSet->GetParticleCount(); i++){
        ComputeDensityFor(data, i, compute_pressure);
    }
}

__bidevice__ void ComputePressureForceCPU(SphSolverData3 *data, Float timeStep){
    ParticleSet3 *pSet = data->sphpSet->GetParticleSet();
    AssertA(pSet, "SphSolver3 has no valid particle set for ComputePressureForce");
    AssertA(pSet->GetParticleCount() > 0,
            "SphSolver3 has no particles for ComputePressureForce");
    for(int i = 0; i < pSet->GetParticleCount(); i++){
        ComputeAllForcesFor(data, i, timeStep);
    }
}

__bidevice__ void TimeIntegrationCPU(SphSolverData3 *data, Float timeStep, int extended){
    ParticleSet3 *pSet = data->sphpSet->GetParticleSet();
    AssertA(pSet, "SphSolver3 has no valid particle set for TimeIntegration");
    AssertA(pSet->GetParticleCount() > 0, "SphSolver3 has no particles for TimeIntegration");
    for(int i = 0; i < pSet->GetParticleCount(); i++){
        TimeIntegrationFor(data, i, timeStep, extended);
    }
}

__bidevice__ void ComputeNonPressureForceCPU(SphSolverData3 *data){
    ParticleSet3 *pSet = data->sphpSet->GetParticleSet();
    for(int i = 0; i < pSet->GetParticleCount(); i++){
        ComputeNonPressureForceFor(data, i);
    }
}

/**************************************************************/
//                   G P U    W R A P P E R S                 //
/**************************************************************/
__global__ void UpdateGridDistributionKernel(Grid3 *grid, ParticleSet3 *pSet, int index){
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if(i < grid->GetCellCount()){
        if(index == 0)
            grid->DistributeToCell(pSet, i);
        else
            grid->DistributeToCellOpt(pSet, i);
    }
}

__global__ void SwapGridStatesKernel(Grid3 *grid){
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if(i < grid->GetCellCount()){
        grid->SwapCellList(i);
    }
}

__global__ void UpdateParticlesBuckets(Grid3 *grid, ParticleSet3 *pSet, Float kernelRadius){
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if(i < pSet->GetParticleCount()){
        grid->DistributeParticleBucket(pSet, i, kernelRadius);
    }
}

__host__ void UpdateGridDistributionGPU(SphSolverData3 *data){
    Grid3 *grid = data->domain;
    ParticleSet3 *pSet = data->sphpSet->GetParticleSet();
    AssertA(grid, "SphSolver3 has no domain for UpdateGridDistribution");
    
    //TODO(Felipe): This is a bug - debug with dragon pool with spacing 0.05!
    if(data->sphpSet->requiresHigherLevelUpdate || 0){
        //printf("Performing full distribution by excessive delta\n");
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

__global__ void ComputeDensityKernel(SphSolverData3 *data, int compute_pressure){
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    ParticleSet3 *pSet = data->sphpSet->GetParticleSet();
    if(i < pSet->GetParticleCount()){
        ComputeDensityFor(data, i, compute_pressure);
    }
}

__host__ void ComputeDensityGPU(SphSolverData3 *data, int compute_pressure){
    ParticleSet3 *pSet = data->sphpSet->GetParticleSet();
    int N = pSet->GetParticleCount();
    GPULaunch(N, ComputeDensityKernel, data, compute_pressure);
}

__global__ void ComputePressureForceKernel(SphSolverData3 *data, Float timeStep){
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    ParticleSet3 *pSet = data->sphpSet->GetParticleSet();
    if(i < pSet->GetParticleCount()){
        ComputeAllForcesFor(data, i, timeStep);
    }
}

__global__ void ComputeNormalKernel(SphSolverData3 *data){
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    ParticleSet3 *pSet = data->sphpSet->GetParticleSet();
    if(i < pSet->GetParticleCount()){
        ComputeNormalFor(data, i);
    }
}

__host__ void ComputeNormalGPU(SphSolverData3 *data){
    ParticleSet3 *pSet = data->sphpSet->GetParticleSet();
    int N = pSet->GetParticleCount();
    GPULaunch(N, ComputeNormalKernel, data);
}

__host__ void ComputePressureForceGPU(SphSolverData3 *data, Float timeStep){
    ParticleSet3 *pSet = data->sphpSet->GetParticleSet();
    AssertA(pSet, "SphSolver3 has no valid particle set for ComputePressureForce");
    AssertA(pSet->GetParticleCount() > 0,
            "SphSolver3 has no particles for ComputePressureForce");
    int N = pSet->GetParticleCount();
    GPULaunch(N, ComputePressureForceKernel, data, timeStep);
}

__global__ void ComputeNonPressureForceKernel(SphSolverData3 *data){
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    ParticleSet3 *pSet = data->sphpSet->GetParticleSet();
    if(i < pSet->GetParticleCount()){
        ComputeNonPressureForceFor(data, i);
    }
}

__host__ void ComputeNonPressureForceGPU(SphSolverData3 *data){
    ParticleSet3 *pSet = data->sphpSet->GetParticleSet();
    int N = pSet->GetParticleCount();
    GPULaunch(N, ComputeNonPressureForceKernel, data);
}

__global__ void TimeIntegrationKernel(SphSolverData3 *data, Float timeStep, int extended){
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    ParticleSet3 *pSet = data->sphpSet->GetParticleSet();
    if(i < pSet->GetParticleCount()){
        TimeIntegrationFor(data, i, timeStep, extended);
    }
}

__host__ void TimeIntegrationGPU(SphSolverData3 *data, Float timeStep, int extended){
    ParticleSet3 *pSet = data->sphpSet->GetParticleSet();
    int N = pSet->GetParticleCount();
    GPULaunch(N, TimeIntegrationKernel, data, timeStep, extended);
}