/* date = October 20th 2022 23:33 */
#pragma once
#include <explicit_grid.h>
#include <vgrid.h>
#include <pressuresolver.h>

template<typename ExplicitGrid> struct _MacSolverData2{
    ExplicitGrid *d;
    ExplicitGrid *u;
    ExplicitGrid *v;
    Float hx; // cell size
    Float density; // target density
    MaterialGridData2 *materialGrid; // description of the grid states
    GridIndexVector fluidCells;

    //////////////////test
    MACVelocityGrid2 *macGrid;
};

typedef _MacSolverData2<CatmullMarkerAndCellGrid2> MacSolverData2;
typedef _MacSolverData2<LinearMarkerAndCellGrid2> LinearMacSolverData2;


class MacSolver2{
    public:
    MacSolverData2 solverData;
    PressureSolver2 *pSolver;

    /*
     * Since all operations run under the ExplicitGrid we don't actually need a Grid2
     * to keep track of domain and distribute stuff so just keep a vec2ui for the domain
     * resolution.
     * TODO: Maybe use Bounds2f so we can simulate under a region and not
     *       [0, width] x [0, height]
     */
    vec2ui domain;

    MacSolver2() = default;
    ~MacSolver2() = default;

    __host__ void Initialize(vec2ui res, Float targetDensity);
    __host__ void Initialize(vec2ui res, Float targetDensity,
                             PressureSolver2 *pressureSolver);
    __host__ void SetPressureSolver(PressureSolver2 *pressureSolver);
    __host__ void AddInflow(vec2f p, vec2f length, Float d, Float u, Float v);

    __host__ void Advance(Float timeIntervalInSeconds);
    __bidevice__ MacSolverData2 *GetSolverData(){ return &solverData; }
};
