#include <espic_solver.h>

bb_cpu_gpu void TimeIntegrationFor(NodeEdgeGrid2v *ef, SpecieSet2 *pSet,
                                     Float dt, ColliderSet2 *collider, int particleId)
{
    vec2f v = pSet->GetParticleVelocity(particleId);
    vec2f x = pSet->GetParticlePosition(particleId);
    vec2f E = ef->NodesToParticle(x);
    int id = pSet->GetFamilyId();
    Float inv = 1.0 / pSet->GetMass();
    vec2f F = pSet->GetCharge() * E;
    vec2f a = F * inv;

    v += a * dt ;
    x += v * dt;

    // TODO: Figure out collider radius
    collider->ResolveCollision(0.001, 0.75, &x, &v);

    pSet->SetParticlePosition(particleId, x);
    pSet->SetParticleVelocity(particleId, v);
}

bb_kernel void TimeIntegrationKernel(NodeEdgeGrid2v *ef, SpecieSet2 *pSet,
                                      ColliderSet2 *collider, Float dt)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if(i < pSet->GetParticleCount()){
        TimeIntegrationFor(ef, pSet, dt, collider, i);
    }
}

void TimeIntegrationCPU(NodeEdgeGrid2v *ef, SpecieSet2 *pSet,
                        ColliderSet2 *collider, Float dt)
{
    int n = pSet->GetParticleCount();
    for(int i = 0; i < n; i++){
        TimeIntegrationFor(ef, pSet, dt, collider, i);
    }
}

void TimeIntegrationGPU(NodeEdgeGrid2v *ef, SpecieSet2 *pSet,
                        ColliderSet2 *collider, Float dt)
{
    int N = pSet->GetParticleCount();
    GPULaunch(N, TimeIntegrationKernel, ef, pSet, collider, dt);
}

void TimeIntegration(NodeEdgeGrid2v *ef, SpecieSet2 *pSet,
                     ColliderSet2 *collider, Float dt, int is_cpu)
{
    if(is_cpu) TimeIntegrationCPU(ef, pSet, collider, dt);
    else TimeIntegrationGPU(ef, pSet, collider, dt);
}

////////////////////////////////////////////////////////////////////////////////////////////

bb_cpu_gpu void ComputeChargeDensityFor(NodeEdgeGrid2f *rho, SpecieSet2 **ppSet,
                                        NodeEdgeGrid2f **dens, int n, unsigned int nodeId)
{
    Float cRho = 0;
    for(int i = 0; i < n; i++){
        SpecieSet2 *pSet = ppSet[i];
        NodeEdgeGrid2f *den = dens[i];
        Float d = den->GetValue(nodeId);
        cRho += d * pSet->charge;
    }
    rho->SetValue(nodeId, cRho);
}

bb_kernel void ComputeChargeDensityKernel(NodeEdgeGrid2f *rho, SpecieSet2 **ppSet,
                                           NodeEdgeGrid2f **dens, int n)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if(i < rho->GetNodeCount()){
        ComputeChargeDensityFor(rho, ppSet, dens, n, i);
    }
}

void ComputeChargeDensityCPU(NodeEdgeGrid2f *rho, SpecieSet2 **ppSet,
                             NodeEdgeGrid2f **dens, int n)
{
    unsigned int N = rho->GetNodeCount();
    for(unsigned int i = 0; i < N; i++){
        ComputeChargeDensityFor(rho, ppSet, dens, n, i);
    }
}

void ComputeChargeDensityGPU(NodeEdgeGrid2f *rho, SpecieSet2 **ppSet,
                             NodeEdgeGrid2f **dens, int n)
{
    unsigned int N = rho->GetNodeCount();
    GPULaunch(N, ComputeChargeDensityKernel, rho, ppSet, dens, n);
}

void ComputeChargeDensity(NodeEdgeGrid2f *rho, SpecieSet2 **ppSet,
                          NodeEdgeGrid2f **dens, int n, int is_cpu)
{
    if(is_cpu) ComputeChargeDensityCPU(rho, ppSet, dens, n);
    else ComputeChargeDensityGPU(rho, ppSet, dens, n);
}

////////////////////////////////////////////////////////////////////////////////////////////

bb_cpu_gpu void ComputeNumberDensityFor(NodeEdgeGrid2f *den, SpecieSet2 *pSet,
                                        unsigned int nodeId)
{
    int cells[4];
    int count = den->GetCellsFrom(nodeId, cells);
    unsigned int fid = pSet->GetFamilyId();
    den->SetValue(nodeId, 0);
    for(int i = 0; i < count; i++){
        Cell2 *cell = den->GetCell(cells[i]);
        ParticleChain *pChain = cell->GetChain();
        int size = cell->GetChainLength();
        for(int j = 0; j < size; j++){
            if(pChain->sId == fid){
                vec2f p = pSet->GetParticlePosition(pChain->pId);
                Float mpw = pSet->GetParticleMPW(pChain->pId);
                den->ParticleToNodes(p, mpw, nodeId);
            }

            pChain = pChain->next;
        }
    }

    Float vol = den->NodeVolume(nodeId);
    Float val = den->GetValue(nodeId);
    den->SetValue(nodeId, val / vol);
}

bb_kernel void ComputeNumberDensityKernel(NodeEdgeGrid2f *den, SpecieSet2 *ppSet){
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if(i < den->GetNodeCount()){
        ComputeNumberDensityFor(den, ppSet, i);
    }
}

void ComputeNumberDensityCPU(NodeEdgeGrid2f *den, SpecieSet2 *ppSet){
    unsigned int h = den->GetNodeCount();
    for(unsigned int i = 0; i < h; i++){
        ComputeNumberDensityFor(den, ppSet, i);
    }
}

void ComputeNumberDensityGPU(NodeEdgeGrid2f *den, SpecieSet2 *ppSet){
    int N = den->GetNodeCount();
    GPULaunch(N, ComputeNumberDensityKernel, den, ppSet);
}

void ComputeNumberDensity(NodeEdgeGrid2f *den, SpecieSet2 *ppSet, int is_cpu){
    if(is_cpu) ComputeNumberDensityCPU(den, ppSet);
    else ComputeNumberDensityGPU(den, ppSet);
}

////////////////////////////////////////////////////////////////////////////////////////////

bb_cpu_gpu void ComputeElectricFieldFor(NodeEdgeGrid2v *ef, NodeEdgeGrid2f *phi,
                                        unsigned int x, unsigned int y, vec2ui count)
{
    vec2f h = ef->GetSpacing();
    Float i2hx = 1.0 / (2.0 * h.x);
    Float i2hy = 1.0 / (2.0 * h.y);

    vec2f efij = ef->GetValue(vec2ui(x, y));
    Float phiij = phi->GetValue(vec2ui(x, y));
    if(x == 0){
        efij[0] = -(-3*phi->GetValue(vec2ui(x,y))
                    +4*phi->GetValue(vec2ui(x+1,y))
                    -1*phi->GetValue(vec2ui(x+2,y))) * i2hx;
    }else if(x == count.x-1){
        efij[0] = -(+1*phi->GetValue(vec2ui(x-2,y))
                    -4*phi->GetValue(vec2ui(x-1,y))
                    +3*phi->GetValue(vec2ui(x,y))) * i2hx;
    }else{
        efij[0] = -(phi->GetValue(vec2ui(x+1,y))-phi->GetValue(vec2ui(x-1,y))) * i2hx;
    }

    if(y == 0){
        efij[1] = -(-3*phi->GetValue(vec2ui(x,y))
                    +4*phi->GetValue(vec2ui(x,y+1))
                    -1*phi->GetValue(vec2ui(x,y+2))) * i2hy;
    }else if(y == count.y-1){
        efij[1] = -(+1*phi->GetValue(vec2ui(x,y-2))
                    -4*phi->GetValue(vec2ui(x,y-1))
                    +3*phi->GetValue(vec2ui(x,y))) * i2hy;
    }else{
        efij[1] = -(phi->GetValue(vec2ui(x,y+1))-phi->GetValue(vec2ui(x,y-1))) * i2hy;
    }

    ef->SetValue(vec2ui(x, y), efij);
}

bb_kernel void ComputeElectricFieldKernel(NodeEdgeGrid2v *ef, NodeEdgeGrid2f *phi){
    unsigned int i = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int j = threadIdx.y + blockIdx.y * blockDim.y;
    vec2ui count = ef->GetNodeIndexCount();
    if(i < count.x && j < count.y){
        ComputeElectricFieldFor(ef, phi, i, j, count);
    }
}

void ComputeElectricFieldCPU(NodeEdgeGrid2v *ef, NodeEdgeGrid2f *phi){
    vec2ui count = ef->GetNodeIndexCount();
    for(unsigned int x = 0; x < count.x; x++){
        for(unsigned int y = 0; y < count.y; y++){
            ComputeElectricFieldFor(ef, phi, x, y, count);
        }
    }
}

void ComputeElectricFieldGPU(NodeEdgeGrid2v *ef, NodeEdgeGrid2f *phi){
    vec2ui count = ef->GetNodeIndexCount();
    int tx = 8;
    int ty = 8;
    dim3 blocks(count.x / tx + 1, count.y / ty + 1);
    dim3 threads(tx, ty);
    ComputeElectricFieldKernel<<<blocks, threads>>>(ef, phi);
    cudaDeviceAssert("ComputeElectricFieldKernel");
}

// Compute E = -grad(phi)
void ComputeElectricField(NodeEdgeGrid2v *ef, NodeEdgeGrid2f *phi, int is_cpu){
    if(is_cpu) ComputeElectricFieldCPU(ef, phi);
    else ComputeElectricFieldGPU(ef, phi);
}

////////////////////////////////////////////////////////////////////////////////////////////

// Apply Gauss-Seidel SOR to compute L(phi) = -rho / e
// I have no idea how to solve this with CUDA, see later the solution for Fluids
void ComputePotentialField(NodeEdgeGrid2f *phi, NodeEdgeGrid2f *rho,
                           Float tolerance, unsigned int max_it)
{
    Float L2 = 0;
    bool converged = false;
    Float EPS0_ = 1.0 / PermittivityEPS;
    vec2f h = rho->GetSpacing();
    vec2ui count = rho->GetNodeIndexCount();
    Float icxy = 1.0 / (Float)(count.x * count.y);
    Float ihx2 = 1.0 / (h.x * h.x);
    Float ihy2 = 1.0 / (h.y * h.y);
    Float ih2 = 2.0 * (ihx2 + ihy2);
    Float iih2 = 1.0 / ih2;

    for(unsigned int it = 0; it < max_it; it++){
        for(int x = 1; x < count.x-1; x++){
            for(int y = 1; y < count.y-1; y++){
                Float phi_new = 0;
                Float phiij = phi->GetValue(vec2ui(x, y));
                Float rhoij = rho->GetValue(vec2ui(x, y));

                phi_new  = rhoij * EPS0_;

                phi_new += ihx2 * (phi->GetValue(vec2ui(x-1, y)) +
                                   phi->GetValue(vec2ui(x+1, y)));

                phi_new += ihy2 * (phi->GetValue(vec2ui(x,y-1)) +
                                   phi->GetValue(vec2ui(x,y+1)));

                phi_new = phi_new * iih2;

                phiij = phiij + 1.4 * (phi_new - phiij);
                phi->SetValue(vec2ui(x, y), phiij);
            }
        }

        if(it % 100 == 0 && it > 1){
            Float sum = 0;
            for(int x = 1; x < count.x-1; x++){
                for(int y = 1; y < count.y-1; y++){
                    Float phiij = phi->GetValue(vec2ui(x, y));
                    Float rhoij = rho->GetValue(vec2ui(x, y));

                    Float R = -phiij * ih2;

                    R += rhoij * EPS0_;

                    R += ihx2 * (phi->GetValue(vec2ui(x-1, y)) +
                                 phi->GetValue(vec2ui(x+1, y)));

                    R += ihy2 * (phi->GetValue(vec2ui(x,y-1)) +
                                 phi->GetValue(vec2ui(x,y+1)));

                    sum += R * R;
                }
            }

            L2 = sqrt(sum * icxy);
            if(L2 < tolerance){
                converged = true;
                break;
            }
        }
    }

    if(!converged) printf("Poisson did not converged, L2 = %g\n", L2);
}
