#include <sph_solver.h>

bb_cpu_gpu Float ComputeAverageTemperature(SphSolverData2 *data){
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
bb_cpu_gpu Float ComputePressureValue(SphSolverData2 *data, Float di){
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

bb_cpu_gpu void ComputePressureFor(SphSolverData2 *data, int particleId, Float di){
    ParticleSet2 *pSet = data->sphpSet->GetParticleSet();
    pSet->SetParticlePressure(particleId, ComputePressureValue(data, di));
}

bb_cpu_gpu Float ComputeDensityForPoint(SphSolverData2 *data, const vec2f &pi){
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

bb_cpu_gpu Float ComputeDensityForParticle(SphSolverData2 *data, int particleId){
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

bb_cpu_gpu void ComputeDensityFor(SphSolverData2 *data, int particleId,
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
bb_cpu_gpu void ComputeParticleInteraction(SphSolverData2 *data, int particleId){
    vec2f acc(0.f, 0.f);
    ParticleSet2 *pSet = data->sphpSet->GetParticleSet();
    vec2f pi = pSet->GetParticlePosition(particleId);

    // TODO: Consider splitting by type
    for(int i = 0; i < data->cInteractionsCount; i++){
        acc += SampleInteraction(&data->cInteractions[i], pi);
    }

    for(int i = 0; i < data->fInteractionsCount; i++){
        acc += SampleInteraction(&data->fInteractions[i], pi);
    }

    pSet->SetParticleInteraction(particleId, acc);
}

bb_cpu_gpu void ComputeNonPressureForceFor(SphSolverData2 *data, int particleId){
    ParticleSet2 *pSet = data->sphpSet->GetParticleSet();
    vec2f intr = pSet->GetParticleInteraction(particleId);
    vec2f pi = pSet->GetParticlePosition(particleId);
    vec2f vi = pSet->GetParticleVelocity(particleId);
    Bucket *bucket = pSet->GetParticleBucket(particleId);

    Float mass = pSet->GetMass();
    Float mass2 = mass * mass;
    Float sphRadius = data->sphpSet->GetKernelRadius();
    SphSpikyKernel2 kernel(sphRadius);

    vec2f fi(0,0);

    fi += mass * intr;
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

bb_cpu_gpu void ComputePressureForceFor(SphSolverData2 *data, int particleId){
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

bb_cpu_gpu void ComputeNormalFor(SphSolverData2 *data, int particleId){
    vec2f ni(0);
    ParticleSet2 *pSet = data->sphpSet->GetParticleSet();
    vec2f pi = pSet->GetParticlePosition(particleId);
    Bucket *bucket = pSet->GetParticleBucket(particleId);

    Float sphRadius = data->sphpSet->GetKernelRadius();
    SphStdKernel2 kernel(sphRadius);

    for(int i = 0; i < bucket->Count(); i++){
        int j = bucket->Get(i);
        if(particleId == j) continue;

        vec2f pj = pSet->GetParticlePosition(j);
        Float dist = Distance(pi, pj);
        if(!IsZero(dist)){
            vec2f dir = (pj - pi) / dist;
            vec2f gradij = kernel.gradW(dist, dir);
            ni += gradij;
        }
    }

    ni = -ni;
    if(ni.LengthSquared() > 1e-8){
        ni = Normalize(ni);
    }
    pSet->SetParticleNormal(particleId, ni);
}


bb_cpu_gpu void ComputeAllForcesFor(SphSolverData2 *data, int particleId,
                                    Float timeStep, int extended, int integrate)
{
    ParticleSet2 *pSet = data->sphpSet->GetParticleSet();
    vec2f intr = pSet->GetParticleInteraction(particleId);
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

    fi += mass * intr;
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
bb_cpu_gpu void TimeIntegrationFor(SphSolverData2 *data, int particleId,
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
    if(!Inside(pi, data->domain->bounds)){
        pi = data->domain->bounds.Clamped(pi, pSet->GetRadius());
    }
    pSet->SetParticlePosition(particleId, pi);
    pSet->SetParticleVelocity(particleId, vi);
}

bb_cpu_gpu void ComputePseudoViscosityAggregationKernelFor(SphSolverData2 *data,
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

bb_cpu_gpu void ComputePseudoViscosityInterpolationKernelFor(SphSolverData2 *data,
                                                             int particleId, Float timeStep)
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
void UpdateGridDistributionCPU(SphSolverData2 *data){
    Grid2 *grid = data->domain;
    ParticleSet2 *pSet = data->sphpSet->GetParticleSet();
    AssertA(grid, "SphSolver2 has no domain for UpdateGridDistribution");
    int cellCount = data->domain->GetCellCount();
    if(data->frame_index == 0){
        ParallelFor(0, cellCount, [&](int i){
            data->domain->DistributeResetCell(i);
        });
    }else{
        ParallelFor(0, cellCount, [&](int i){
            grid->DistributeToCellOpt(pSet, i);
        });

        ParallelFor(0, cellCount, [&](int i){
            grid->SwapCellList(i);
        });
    }

    int pCount = pSet->GetParticleCount();
    Float kernelRadius = data->sphpSet->GetKernelRadius();
    ParallelFor(0, pCount, [&](int i){
        grid->DistributeParticleBucket(pSet, i, kernelRadius);
    });

    data->frame_index = 1;
}

bb_kernel void UpdateGridDistributionKernel(Grid2 *grid, ParticleSet2 *pSet, int index){
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if(i < grid->GetCellCount()){
        if(index == 0)
            grid->DistributeToCell(pSet, i);
        else
            grid->DistributeToCellOpt(pSet, i);
    }
}

bb_kernel void SwapGridStatesKernel(Grid2 *grid){
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if(i < grid->GetCellCount()){
        grid->SwapCellList(i);
    }
}

bb_kernel void UpdateParticlesBuckets(Grid2 *grid, ParticleSet2 *pSet, Float kernelRadius){
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if(i < pSet->GetParticleCount()){
        grid->DistributeParticleBucket(pSet, i, kernelRadius);
    }
}

void UpdateGridDistributionGPU(SphSolverData2 *data){
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

void ComputeDensityCPU(SphSolverData2 *data, int compute_pressure){
    ParticleSet2 *pSet = data->sphpSet->GetParticleSet();
    AssertA(pSet, "SphSolver2 has no valid particle set for ComputeDensity");
    AssertA(pSet->GetParticleCount() > 0, "SphSolver2 has no particles for ComputeDensity");
    ParallelFor(0, pSet->GetParticleCount(), [&](int i){
        ComputeDensityFor(data, i, compute_pressure);
    });
}

bb_kernel void ComputeDensityKernel(SphSolverData2 *data, int compute_pressure){
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    ParticleSet2 *pSet = data->sphpSet->GetParticleSet();
    if(i < pSet->GetParticleCount()){
        ComputeDensityFor(data, i, compute_pressure);
    }
}

void ComputeDensityGPU(SphSolverData2 *data, int compute_pressure){
    ParticleSet2 *pSet = data->sphpSet->GetParticleSet();
    int N = pSet->GetParticleCount();
    GPULaunch(N, ComputeDensityKernel, data, compute_pressure);
}

bb_kernel void ComputePseudoViscosityAggregationKernel(SphSolverData2 *data)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    ParticleSet2 *pSet = data->sphpSet->GetParticleSet();
    if(i < pSet->GetParticleCount()){
        ComputePseudoViscosityAggregationKernelFor(data, i);
    }
}

bb_kernel void ComputePseudoViscosityInterpolationKernel(SphSolverData2 *data,
                                                          Float timeStep)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    ParticleSet2 *pSet = data->sphpSet->GetParticleSet();
    if(i < pSet->GetParticleCount()){
        ComputePseudoViscosityInterpolationKernelFor(data, i, timeStep);
    }
}

void ComputePseudoViscosityInterpolationGPU(SphSolverData2 *data, Float timeStep){
    Float scale = data->pseudoViscosity * timeStep;
    if(scale > 0.1){
        ParticleSet2 *pSet = data->sphpSet->GetParticleSet();
        int N = pSet->GetParticleCount();
        GPULaunch(N, ComputePseudoViscosityAggregationKernel, data);

        GPULaunch(N, ComputePseudoViscosityInterpolationKernel, data, timeStep);
    }
}

void ComputePseudoViscosityInterpolationCPU(SphSolverData2 *data, Float timeStep){
    Float scale = data->pseudoViscosity * timeStep;
    if(scale > 0.1){
        ParticleSet2 *pSet = data->sphpSet->GetParticleSet();
        int N = pSet->GetParticleCount();
        ParallelFor(0, N, [&](int i){
            ComputePseudoViscosityAggregationKernelFor(data, i);
        });

        ParallelFor(0, N, [&](int i){
            ComputePseudoViscosityInterpolationKernelFor(data, i, timeStep);
        });
    }
}

void ComputePressureForceCPU(SphSolverData2 *data, Float timeStep, int integrate){
    ParticleSet2 *pSet = data->sphpSet->GetParticleSet();
    AssertA(pSet, "SphSolver2 has no valid particle set for ComputePressureForce");
    AssertA(pSet->GetParticleCount() > 0,
            "SphSolver2 has no particles for ComputePressureForce");
    ParallelFor(0, pSet->GetParticleCount(), [&](int i){
        ComputeAllForcesFor(data, i, timeStep, 0, integrate);
    });
}

void ComputeNonPressureForceCPU(SphSolverData2 *data){
    ParticleSet2 *pSet = data->sphpSet->GetParticleSet();
    AssertA(pSet, "SphSolver2 has no valid particle set for ComputeNonPressureForce");
    AssertA(pSet->GetParticleCount() > 0,
            "SphSolver2 has no particles for ComputeNonPressureForce");
    ParallelFor(0, pSet->GetParticleCount(), [&](int i){
        ComputeNonPressureForceFor(data, i);
    });
}

bb_kernel void ComputeNonPressureForceKernel(SphSolverData2 *data){
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    ParticleSet2 *pSet = data->sphpSet->GetParticleSet();
    if(i < pSet->GetParticleCount()){
        ComputeNonPressureForceFor(data, i);
    }
}

void ComputeNonPressureForceGPU(SphSolverData2 *data){
    ParticleSet2 *pSet = data->sphpSet->GetParticleSet();
    int N = pSet->GetParticleCount();
    GPULaunch(N, ComputeNonPressureForceKernel, data);
}

bb_kernel void ComputePressureForceKernel(SphSolverData2 *data, Float timeStep,
                                           int integrate)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    ParticleSet2 *pSet = data->sphpSet->GetParticleSet();
    if(i < pSet->GetParticleCount()){
        ComputeAllForcesFor(data, i, timeStep, 0, integrate);
    }
}

void ComputePressureForceGPU(SphSolverData2 *data, Float timeStep, int integrate){
    ParticleSet2 *pSet = data->sphpSet->GetParticleSet();
    AssertA(pSet, "SphSolver2 has no valid particle set for ComputePressureForce");
    AssertA(pSet->GetParticleCount() > 0,
            "SphSolver2 has no particles for ComputePressureForce");
    int N = pSet->GetParticleCount();
    GPULaunch(N, ComputePressureForceKernel, data, timeStep, integrate);
}

void ComputeParticleInteractionCPU(SphSolverData2 *data){
    ParticleSet2 *pSet = data->sphpSet->GetParticleSet();
    ParallelFor(0, pSet->GetParticleCount(), [&](int i){
        ComputeParticleInteraction(data, i);
    });
}

bb_kernel void ComputeParticleInteractionKernel(SphSolverData2 *data){
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    ParticleSet2 *pSet = data->sphpSet->GetParticleSet();
    if(i < pSet->GetParticleCount()){
        ComputeParticleInteraction(data, i);
    }
}

void ComputeParticleInteractionGPU(SphSolverData2 *data){
    ParticleSet2 *pSet = data->sphpSet->GetParticleSet();
    int N = pSet->GetParticleCount();
    GPULaunch(N, ComputeParticleInteractionKernel, data);
}

void TimeIntegrationCPU(SphSolverData2 *data, Float timeStep, int extended){
    ParticleSet2 *pSet = data->sphpSet->GetParticleSet();
    AssertA(pSet, "SphSolver2 has no valid particle set for TimeIntegration");
    AssertA(pSet->GetParticleCount() > 0, "SphSolver2 has no particles for TimeIntegration");
    ParallelFor(0, pSet->GetParticleCount(), [&](int i){
        TimeIntegrationFor(data, i, timeStep, extended);
    });
}

bb_kernel void TimeIntegrationKernel(SphSolverData2 *data, Float timeStep, int extended){
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    ParticleSet2 *pSet = data->sphpSet->GetParticleSet();
    if(i < pSet->GetParticleCount()){
        TimeIntegrationFor(data, i, timeStep, extended);
    }
}

void TimeIntegrationGPU(SphSolverData2 *data, Float timeStep, int extended){
    ParticleSet2 *pSet = data->sphpSet->GetParticleSet();
    int N = pSet->GetParticleCount();
    GPULaunch(N, TimeIntegrationKernel, data, timeStep, extended);
}

void ComputeNormalCPU(SphSolverData2 *data){
    ParticleSet2 *pSet = data->sphpSet->GetParticleSet();
    AssertA(pSet, "SphSolver2 has no valid particle set for ComputeDensity");
    AssertA(pSet->GetParticleCount() > 0, "SphSolver2 has no particles for ComputeDensity");
    ParallelFor(0, pSet->GetParticleCount(), [&](int i){
        ComputeNormalFor(data, i);
    });
}

bb_kernel void ComputeNormalKernel(SphSolverData2 *data){
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    ParticleSet2 *pSet = data->sphpSet->GetParticleSet();
    if(i < pSet->GetParticleCount()){
        ComputeNormalFor(data, i);
    }
}

void ComputeNormalGPU(SphSolverData2 *data){
    ParticleSet2 *pSet = data->sphpSet->GetParticleSet();
    int N = pSet->GetParticleCount();
    GPULaunch(N, ComputeNormalKernel, data);
}

bb_cpu_gpu void ComputeInitialTemperatureFor(SphSolverData2 *data, int particleId,
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

void ComputeInitialTemperatureMapCPU(SphSolverData2 *data, Float Tmin,
                                     Float Tmax, int maxLevel)
{
    ParticleSet2 *pSet = data->sphpSet->GetParticleSet();
    int count = pSet->GetParticleCount();
    ParallelFor(0, count, [&](int i){
        ComputeInitialTemperatureFor(data, i, Tmin, Tmax, maxLevel);
    });
}

bb_kernel void ComputeInitialTemperatureMapKernel(SphSolverData2 *data, Float Tmin,
                                                   Float Tmax, int maxLevel)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    ParticleSet2 *pSet = data->sphpSet->GetParticleSet();
    if(i < pSet->GetParticleCount()){
        ComputeInitialTemperatureFor(data, i, Tmin, Tmax, maxLevel);
    }
}

void ComputeInitialTemperatureMapGPU(SphSolverData2 *data, Float Tmin,
                                     Float Tmax, int maxLevel)
{
    ParticleSet2 *pSet = data->sphpSet->GetParticleSet();
    int N = pSet->GetParticleCount();
    GPULaunch(N, ComputeInitialTemperatureMapKernel, data, Tmin, Tmax, maxLevel);
}
