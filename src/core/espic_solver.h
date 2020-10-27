#pragma once

#include <grid.h>
#include <collider.h>

class EspicSolver2{
    public:
    NodeEdgeGrid2f *phi; // potential
    NodeEdgeGrid2f *rho; // charge density
    NodeEdgeGrid2v *ef; // electric field
    NodeEdgeGrid2f **den; // number density
    SpecieSet2 **spSet; // species being simulated
    int spCount; // species count
    ColliderSet2 *collider;
    
    __bidevice__ EspicSolver2();
    __host__ void Setup(Grid2 *domain, SpecieSet2 **species, int speciesCount);
    __host__ void SetColliders(ColliderSet2 *collider);
    __host__ void Advance(Float dt);
};

// Apply Gauss-Seidel SOR to compute L(phi) = -rho / e
__host__ void ComputePotentialField(NodeEdgeGrid2f *phi, NodeEdgeGrid2f *rho,
                                    Float tolerance, unsigned int max_it=5000);

// Compute E = -grad(phi)
__host__ void ComputeElectricField(NodeEdgeGrid2v *ef, NodeEdgeGrid2f *phi, int is_cpu=0);

// Compute number density for a given SpecieSet2
__host__ void ComputeNumberDensity(NodeEdgeGrid2f *den, SpecieSet2 *ppSet, int is_cpu=0);

// Compute charge density
__host__ void ComputeChargeDensity(NodeEdgeGrid2f *rho, SpecieSet2 **ppSet, 
                                   NodeEdgeGrid2f **dens, int n, int is_cpu=0);
// Advance particles
__host__ void TimeIntegration(NodeEdgeGrid2v *ef, SpecieSet2 *pSet, 
                              ColliderSet2 *collider, Float dt, int is_cpu=0);

__host__ void UpdateGridDistribution(Grid2 *grid, SpecieSet2 **ppSet, int n);