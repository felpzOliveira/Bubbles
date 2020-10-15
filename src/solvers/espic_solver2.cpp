#include <espic_solver.h>

__global__ void SwapGridStates(Grid2 *grid){
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if(i < grid->GetCellCount()){
        grid->SwapCellList(i);
    }
}

__global__ void UpdateGridSpeciesKernel(Grid2 *grid, SpecieSet2 **ppSet, int n){
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if(i < grid->GetCellCount()){
        grid->DistributeToCellOpt(ppSet, n, i);
    }
}

__host__ void UpdateGridDistribution(Grid2 *grid, SpecieSet2 **ppSet, int n){
    int N = grid->GetCellCount();
    int nThreads = CUDA_THREADS_PER_BLOCK;
    UpdateGridSpeciesKernel<<<(N + nThreads - 1) / nThreads, nThreads>>>(grid, ppSet, n);
    cudaDeviceAssert();
    
    SwapGridStates<<<(N + nThreads - 1) / nThreads, nThreads>>>(grid);
    cudaDeviceAssert();
}

__bidevice__ EspicSolver2::EspicSolver2(){}

__host__ void EspicSolver2::Advance(Float dt){
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

__host__ void EspicSolver2::Setup(Grid2 *domain, SpecieSet2 **species, int speciesCount){
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

__host__ void EspicSolver2::SetColliders(ColliderSet2 *coll){
    collider = coll;
}

__host__ void EspicSolver2::Release(){
    if(den){
        for(int i = 0; i < spCount; i++){
            if(den[i]){
                den[i]->Release();
                cudaFree(den[i]);
            }
        }
        cudaFree(den);
    }
    
    if(phi){
        phi->Release();
        cudaFree(phi);
    }
    
    if(rho){
        rho->Release();
        cudaFree(rho);
    }
    
    if(ef){
        ef->Release();
        cudaFree(ef);
    }
}