#include <sph_solver.h>
#include <profiler.h>

/**************************************************************/
//      D E N S I T Y     A N D    P R E S S U R E            //
/**************************************************************/
bb_cpu_gpu Float ComputePressureValue(SphSolverData3 *data, Float di){
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

bb_cpu_gpu void ComputePressureFor(SphSolverData3 *data, int particleId, Float di){
    ParticleSet3 *pSet = data->sphpSet->GetParticleSet();
    pSet->SetParticlePressure(particleId, ComputePressureValue(data, di));
}

bb_cpu_gpu void ComputeDensityFor(SphSolverData3 *data, int particleId,
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
bb_cpu_gpu void ComputeParticleInteraction(SphSolverData3 *data, int particleId){
    ParticleSet3 *pSet = data->sphpSet->GetParticleSet();
    vec3f pi = pSet->GetParticlePosition(particleId);

    // TODO: Adjust for all components
    vec3f extAcc = SampleInteraction(data->cInteractions, pi);
    //if(particleId == 0)
        //printf("{%g %g %g}\n", extAcc.x, extAcc.y, extAcc.z);

    pSet->SetParticleInteraction(particleId, extAcc);
}

bb_cpu_gpu void ComputeNonPressureForceFor(SphSolverData3 *data, int particleId){
    ParticleSet3 *pSet = data->sphpSet->GetParticleSet();
    vec3f intr = pSet->GetParticleInteraction(particleId);
    vec3f pi = pSet->GetParticlePosition(particleId);
    vec3f vi = pSet->GetParticleVelocity(particleId);

    Bucket *bucket = pSet->GetParticleBucket(particleId);

    Float mass = pSet->GetMass();
    Float mass2 = mass * mass;
    Float sphRadius = data->sphpSet->GetKernelRadius();
    SphSpikyKernel3 kernel(sphRadius);

    vec3f fi(0);

    fi += mass * intr;
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

bb_cpu_gpu void ComputePressureForceFor(SphSolverData3 *data, int particleId){
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

bb_cpu_gpu void ComputeNormalFor(SphSolverData3 *data, int particleId){
    vec3f ni(0);
    ParticleSet3 *pSet = data->sphpSet->GetParticleSet();
    vec3f pi = pSet->GetParticlePosition(particleId);
    Bucket *bucket = pSet->GetParticleBucket(particleId);

    Float sphRadius = data->sphpSet->GetKernelRadius();
    SphStdKernel3 kernel(sphRadius);

    for(int i = 0; i < bucket->Count(); i++){
        int j = bucket->Get(i);
        if(particleId == j) continue;

        vec3f pj = pSet->GetParticlePosition(j);
        Float dist = Distance(pi, pj);
        if(!IsZero(dist)){
            vec3f dir = (pj - pi) / dist;
            vec3f gradij = kernel.gradW(dist, dir);
            ni += gradij;
        }
    }

    ni = -ni;
    if(ni.LengthSquared() > 1e-8){
        ni = Normalize(ni);
    }
    pSet->SetParticleNormal(particleId, ni);
}

bb_cpu_gpu void ComputeAllForcesFor(SphSolverData3 *data, int particleId,
                                    Float timeStep, int extended)
{
    ParticleSet3 *pSet = data->sphpSet->GetParticleSet();
    vec3f intr = pSet->GetParticleInteraction(particleId);
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

    fi += mass * intr;
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
bb_cpu_gpu void TimeIntegrationFor(SphSolverData3 *data, int particleId,
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

#ifdef DEBUG
    vec3f opi = pi;
#endif

    data->collider->ResolveCollision(pSet->GetRadius(), 0.6, &pi, &vi);

    // miss configuration on spacing x domain size is hard to detect
    // during scene setup so let's push particles inside in case they
    // somehow got pushed out by collider collision solver
    if(!Inside(pi, data->domain->bounds)){
#if DEBUG
        vec3f pMin = data->domain->bounds.pMin;
        vec3f pMax = data->domain->bounds.pMax;
        printf("Point pi outside: {%g %g %g}, {%g %g %g} x {%g %g %g} [%g %g %g]\n",
               pi.x, pi.y, pi.z, pMin.x, pMin.y, pMin.z,
               pMax.x, pMax.y, pMax.z, opi.x, opi.y, opi.z);
#endif
        pi = data->domain->bounds.Clamped(pi, pSet->GetRadius());
    }

    AssertA(Inside(pi, data->domain->bounds), "Particle outside domain");

    vec3f len = data->domain->GetCellSize();
    Float dist = Distance(pi, oi);
    Float minLen = Min(len[0], Min(len[1], len[2]));
    if(dist >= minLen * 0.9){
        data->sphpSet->SetHigherLevel();
    }

    pSet->SetParticlePosition(particleId, pi);
    pSet->SetParticleVelocity(particleId, vi);
}

bb_cpu_gpu void ComputePseudoViscosityAggregationKernelFor(SphSolverData3 *data,
                                                           int particleId)
{
    ParticleSet3 *pSet = data->sphpSet->GetParticleSet();
    vec3f vi = pSet->GetParticleVelocity(particleId);
    vec3f pi = pSet->GetParticlePosition(particleId);
    Bucket *bucket = pSet->GetParticleBucket(particleId);
    Float mass = pSet->GetMass();
    Float sphRadius = data->sphpSet->GetKernelRadius();
    SphSpikyKernel3 kernel(sphRadius);

    vec3f smoothedVi(0);
    Float weightSum = 0;
    for(int i = 0; i < bucket->Count(); i++){
        int j = bucket->Get(i);
        vec3f pj = pSet->GetParticlePosition(j);
        Float dj = pSet->GetParticleDensity(j);
        vec3f vj = pSet->GetParticleVelocity(j);
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

bb_cpu_gpu void ComputePseudoViscosityInterpolationKernelFor(SphSolverData3 *data,
                                                             int particleId, Float timeStep)
{
    ParticleSet3 *pSet = data->sphpSet->GetParticleSet();
    vec3f vi = pSet->GetParticleVelocity(particleId);
    vec3f smoothedVi = data->smoothedVelocities[particleId];
    Float smoothFactor = Clamp(timeStep * data->pseudoViscosity, 0.0, 1.0);
    vi = Lerp(vi, smoothedVi, smoothFactor);
    pSet->SetParticleVelocity(particleId, vi);
}


/**************************************************************/
//                   C P U    W R A P P E R S                 //
/**************************************************************/
void UpdateGridDistributionCPU(SphSolverData3 *data){
    Grid3 *grid = data->domain;
    AssertA(grid, "SphSolver3 has no domain for UpdateGridDistribution");
    ParticleSet3 *pSet = data->sphpSet->GetParticleSet();
    int cellCount = data->domain->GetCellCount();
    if(data->sphpSet->requiresHigherLevelUpdate){
        //printf("Performing full distribution by excessive delta\n");
        ParallelFor(0, cellCount, [&](int i){
            data->domain->DistributeResetCell(i);
        });

        data->domain->DistributeByParticle(pSet);
    }else{
        if(data->frame_index == 0){
            ParallelFor(0, cellCount, [&](int i){
                grid->DistributeToCell(pSet, i);
            });
        }else{
            ParallelFor(0, cellCount, [&](int i){
                grid->DistributeToCellOpt(pSet, i);
            });

            ParallelFor(0, cellCount, [&](int i){
                grid->SwapCellList(i);
            });
        }
    }

    int pCount = pSet->GetParticleCount();
    Float kernelRadius = data->sphpSet->GetKernelRadius();
    ParallelFor(0, pCount, [&](int i){
        grid->DistributeParticleBucket(pSet, i, kernelRadius);
    });

    data->frame_index = 1;
}

void ComputeNormalCPU(SphSolverData3 *data){
    ParticleSet3 *pSet = data->sphpSet->GetParticleSet();
    AssertA(pSet, "SphSolver3 has no valid particle set for ComputeDensity");
    AssertA(pSet->GetParticleCount() > 0, "SphSolver3 has no particles for ComputeDensity");
    ParallelFor(0, pSet->GetParticleCount(), [&](int i){
        ComputeNormalFor(data, i);
    });
}

void ComputeDensityCPU(SphSolverData3 *data, int compute_pressure){
    ParticleSet3 *pSet = data->sphpSet->GetParticleSet();
    AssertA(pSet, "SphSolver3 has no valid particle set for ComputeDensity");
    AssertA(pSet->GetParticleCount() > 0, "SphSolver3 has no particles for ComputeDensity");
    ParallelFor(0, pSet->GetParticleCount(), [&](int i){
        ComputeDensityFor(data, i, compute_pressure);
    });
}

void ComputePressureForceCPU(SphSolverData3 *data, Float timeStep){
    ParticleSet3 *pSet = data->sphpSet->GetParticleSet();
    AssertA(pSet, "SphSolver3 has no valid particle set for ComputePressureForce");
    AssertA(pSet->GetParticleCount() > 0,
            "SphSolver3 has no particles for ComputePressureForce");
    ParallelFor(0, pSet->GetParticleCount(), [&](int i){
        ComputeAllForcesFor(data, i, timeStep);
    });
}

void TimeIntegrationCPU(SphSolverData3 *data, Float timeStep, int extended){
    ParticleSet3 *pSet = data->sphpSet->GetParticleSet();
    AssertA(pSet, "SphSolver3 has no valid particle set for TimeIntegration");
    AssertA(pSet->GetParticleCount() > 0, "SphSolver3 has no particles for TimeIntegration");
    ParallelFor(0, pSet->GetParticleCount(), [&](int i){
        TimeIntegrationFor(data, i, timeStep, extended);
    });
}

void ComputeNonPressureForceCPU(SphSolverData3 *data){
    ParticleSet3 *pSet = data->sphpSet->GetParticleSet();
    ParallelFor(0, pSet->GetParticleCount(), [&](int i){
        ComputeNonPressureForceFor(data, i);
    });
}

void ComputePseudoViscosityInterpolationCPU(SphSolverData3 *data, Float timeStep){
    Float scale = data->pseudoViscosity * timeStep;
    if(scale > 0.1){
        ParticleSet3 *pSet = data->sphpSet->GetParticleSet();
        int N = pSet->GetParticleCount();
        ParallelFor(0, N, [&](int i){
            ComputePseudoViscosityAggregationKernelFor(data, i);
        });

        ParallelFor(0, N, [&](int i){
            ComputePseudoViscosityInterpolationKernelFor(data, i, timeStep);
        });
    }
}

/**************************************************************/
//                   G P U    W R A P P E R S                 //
/**************************************************************/
bb_kernel void UpdateGridDistributionKernel(Grid3 *grid, ParticleSet3 *pSet, int index){
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if(i < grid->GetCellCount()){
        if(index == 0)
            grid->DistributeToCell(pSet, i);
        else
            grid->DistributeToCellOpt(pSet, i);
    }
}

bb_kernel void SwapGridStatesKernel(Grid3 *grid){
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if(i < grid->GetCellCount()){
        grid->SwapCellList(i);
    }
}

bb_kernel void UpdateParticlesBuckets(Grid3 *grid, ParticleSet3 *pSet, Float kernelRadius){
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if(i < pSet->GetParticleCount()){
        grid->DistributeParticleBucket(pSet, i, kernelRadius);
    }
}

void UpdateGridDistributionGPU(SphSolverData3 *data){
    Grid3 *grid = data->domain;
    ParticleSet3 *pSet = data->sphpSet->GetParticleSet();
    AssertA(grid, "SphSolver3 has no domain for UpdateGridDistribution");
    int cellCount = data->domain->GetCellCount();

    if(data->sphpSet->requiresHigherLevelUpdate){
        //printf("Performing full distribution by excessive delta\n");
        ParallelFor(0, cellCount, [&](int i){
            data->domain->DistributeResetCell(i);
        });

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

bb_kernel void ComputeDensityKernel(SphSolverData3 *data, int compute_pressure){
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    ParticleSet3 *pSet = data->sphpSet->GetParticleSet();
    if(i < pSet->GetParticleCount()){
        ComputeDensityFor(data, i, compute_pressure);
    }
}

void ComputeDensityGPU(SphSolverData3 *data, int compute_pressure){
    ParticleSet3 *pSet = data->sphpSet->GetParticleSet();
    int N = pSet->GetParticleCount();
    GPULaunch(N, ComputeDensityKernel, data, compute_pressure);
}

bb_kernel void ComputePressureForceKernel(SphSolverData3 *data, Float timeStep){
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    ParticleSet3 *pSet = data->sphpSet->GetParticleSet();
    if(i < pSet->GetParticleCount()){
        ComputeAllForcesFor(data, i, timeStep);
    }
}

bb_kernel void ComputeNormalKernel(SphSolverData3 *data){
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    ParticleSet3 *pSet = data->sphpSet->GetParticleSet();
    if(i < pSet->GetParticleCount()){
        ComputeNormalFor(data, i);
    }
}

void ComputeNormalGPU(SphSolverData3 *data){
    ParticleSet3 *pSet = data->sphpSet->GetParticleSet();
    int N = pSet->GetParticleCount();
    GPULaunch(N, ComputeNormalKernel, data);
}


void ComputeParticleInteractionCPU(SphSolverData3 *data){
    ParticleSet3 *pSet = data->sphpSet->GetParticleSet();
    ParallelFor(0, pSet->GetParticleCount(), [&](int i){
        ComputeParticleInteraction(data, i);
    });
}

bb_kernel void ComputeParticleInteractionKernel(SphSolverData3 *data){
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    ParticleSet3 *pSet = data->sphpSet->GetParticleSet();
    if(i < pSet->GetParticleCount()){
        ComputeParticleInteraction(data, i);
    }
}

void ComputeParticleInteractionGPU(SphSolverData3 *data){
    ParticleSet3 *pSet = data->sphpSet->GetParticleSet();
    int N = pSet->GetParticleCount();
    GPULaunch(N, ComputeParticleInteractionKernel, data);
}

void ComputePressureForceGPU(SphSolverData3 *data, Float timeStep){
    ParticleSet3 *pSet = data->sphpSet->GetParticleSet();
    AssertA(pSet, "SphSolver3 has no valid particle set for ComputePressureForce");
    AssertA(pSet->GetParticleCount() > 0,
            "SphSolver3 has no particles for ComputePressureForce");
    int N = pSet->GetParticleCount();
    GPULaunch(N, ComputePressureForceKernel, data, timeStep);
}

bb_kernel void ComputeNonPressureForceKernel(SphSolverData3 *data){
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    ParticleSet3 *pSet = data->sphpSet->GetParticleSet();
    if(i < pSet->GetParticleCount()){
        ComputeNonPressureForceFor(data, i);
    }
}

void ComputeNonPressureForceGPU(SphSolverData3 *data){
    ParticleSet3 *pSet = data->sphpSet->GetParticleSet();
    int N = pSet->GetParticleCount();
    GPULaunch(N, ComputeNonPressureForceKernel, data);
}

bb_kernel void TimeIntegrationKernel(SphSolverData3 *data, Float timeStep, int extended){
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    ParticleSet3 *pSet = data->sphpSet->GetParticleSet();
    if(i < pSet->GetParticleCount()){
        TimeIntegrationFor(data, i, timeStep, extended);
    }
}

void TimeIntegrationGPU(SphSolverData3 *data, Float timeStep, int extended){
    ParticleSet3 *pSet = data->sphpSet->GetParticleSet();
    int N = pSet->GetParticleCount();
    GPULaunch(N, TimeIntegrationKernel, data, timeStep, extended);
}

bb_kernel void ComputePseudoViscosityAggregationKernel(SphSolverData3 *data)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    ParticleSet3 *pSet = data->sphpSet->GetParticleSet();
    if(i < pSet->GetParticleCount()){
        ComputePseudoViscosityAggregationKernelFor(data, i);
    }
}

bb_kernel void ComputePseudoViscosityInterpolationKernel(SphSolverData3 *data,
                                                          Float timeStep)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    ParticleSet3 *pSet = data->sphpSet->GetParticleSet();
    if(i < pSet->GetParticleCount()){
        ComputePseudoViscosityInterpolationKernelFor(data, i, timeStep);
    }
}

void ComputePseudoViscosityInterpolationGPU(SphSolverData3 *data, Float timeStep){
    Float scale = data->pseudoViscosity * timeStep;
    if(scale > 0.1){
        ParticleSet3 *pSet = data->sphpSet->GetParticleSet();
        int N = pSet->GetParticleCount();
        GPULaunch(N, ComputePseudoViscosityAggregationKernel, data);

        GPULaunch(N, ComputePseudoViscosityInterpolationKernel, data, timeStep);
    }
}
