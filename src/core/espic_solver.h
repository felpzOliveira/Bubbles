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

    bb_cpu_gpu EspicSolver2();
    void Setup(Grid2 *domain, SpecieSet2 **species, int speciesCount);
    void SetColliders(ColliderSet2 *collider);
    void Advance(Float dt);
};

// Apply Gauss-Seidel SOR to compute L(phi) = -rho / e
void ComputePotentialField(NodeEdgeGrid2f *phi, NodeEdgeGrid2f *rho,
                                    Float tolerance, unsigned int max_it=5000);

// Compute E = -grad(phi)
void ComputeElectricField(NodeEdgeGrid2v *ef, NodeEdgeGrid2f *phi, int is_cpu=0);

// Compute number density for a given SpecieSet2
void ComputeNumberDensity(NodeEdgeGrid2f *den, SpecieSet2 *ppSet, int is_cpu=0);

// Compute charge density
void ComputeChargeDensity(NodeEdgeGrid2f *rho, SpecieSet2 **ppSet,
                                   NodeEdgeGrid2f **dens, int n, int is_cpu=0);
// Advance particles
void TimeIntegration(NodeEdgeGrid2v *ef, SpecieSet2 *pSet,
                              ColliderSet2 *collider, Float dt, int is_cpu=0);

void UpdateGridDistribution(Grid2 *grid, SpecieSet2 **ppSet, int n);
