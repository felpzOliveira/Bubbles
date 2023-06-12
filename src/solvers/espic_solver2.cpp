#include <espic_solver.h>

bb_kernel void SwapGridStates(Grid2 *grid){
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if(i < grid->GetCellCount()){
        grid->SwapCellList(i);
    }
}

bb_kernel void UpdateGridSpeciesKernel(Grid2 *grid, SpecieSet2 **ppSet, int n){
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if(i < grid->GetCellCount()){
        grid->DistributeToCellOpt(ppSet, n, i);
    }
}

void UpdateGridDistribution(Grid2 *grid, SpecieSet2 **ppSet, int n){
    int N = grid->GetCellCount();
    GPULaunch(N, UpdateGridSpeciesKernel, grid, ppSet, n);
    GPULaunch(N, SwapGridStates, grid);
}

bb_cpu_gpu EspicSolver2::EspicSolver2(){}

void EspicSolver2::Advance(Float dt){
    // Compute number density
    for(int i = 0; i < spCount; i++){
        SpecieSet2 *pSet = spSet[i];
        NodeEdgeGrid2f *pDen = den[i];
        ComputeNumberDensity(pDen, pSet);
    }

    // Compute charge density
    ComputeChargeDensity(rho, spSet, den, spCount);

    // Compute Potential field
    ComputePotentialField(phi, rho, 1e-2, 10000);

    // Compute electric field
    ComputeElectricField(ef, phi);

    // Perform time integration
    for(int i = 0; i < spCount; i++){
        SpecieSet2 *pSet = spSet[i];
        TimeIntegration(ef, pSet, collider, dt);
    }

    // Distribute particles after particle movement
    UpdateGridDistribution(ef->grid, spSet, spCount);
}

void EspicSolver2::Setup(Grid2 *domain, SpecieSet2 **species, int speciesCount){
    spSet = species;
    spCount = speciesCount;
    phi = cudaAllocateVx(NodeEdgeGrid2f, 1);
    rho = cudaAllocateVx(NodeEdgeGrid2f, 1);
    ef  = cudaAllocateVx(NodeEdgeGrid2v, 1);
    den = cudaAllocateVx(NodeEdgeGrid2f*, speciesCount);
    phi->Build(domain);
    rho->Build(domain);
    ef->Build(domain);

    for(int i = 0; i < speciesCount; i++){
        den[i] = cudaAllocateVx(NodeEdgeGrid2f, 1);
        den[i]->Build(domain);
    }

    // distribute particles once so that we don't have to bother
    // choosing which algorithm to use later
    for(int i = 0; i < domain->GetCellCount(); i++){
        domain->DistributeResetCell(i);

        for(int j = 0; j < speciesCount; j++){
            domain->DistributeAddToCell(species[j], i);
        }
    }

    // Initialize fields once
    ComputePotentialField(phi, rho, 1e-2, 10000);
    ComputeElectricField(ef, phi);
}

void EspicSolver2::SetColliders(ColliderSet2 *coll){
    collider = coll;
}
