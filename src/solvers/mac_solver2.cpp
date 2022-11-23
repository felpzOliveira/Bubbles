#include <mac_solver.h>
#include <pressuresolver.h>

#define MacSolverAssertPressureSolver(solver)do{\
    if(solver->pSolver == nullptr){\
        printf("Did not build a pressure solver, cannot execute %s!\n", __func__);\
        exit(0);\
    }\
}while(0)

constexpr int kPressureSolverIterations = 1000;

static void MacSolverProject(MacSolver2 *solver, int limit, Float timestep){
    MacSolverAssertPressureSolver(solver);
    MacSolverData2 *data = &solver->solverData;
    solver->pSolver->SolvePressure(data->u, data->v, data->d, limit,
                                   data->density, timestep);
}

static void MacSolverProject2(MacSolver2 *solver, int limit, Float timestep){
    MacSolverAssertPressureSolver(solver);
    MacSolverData2 *data = &solver->solverData;
    TmpPressureSolver pSolver;

    PressureSolverParameters params;
    params.cellwidth = data->hx;
    params.deltaTime = timestep;
    params.density = data->density;
    params.fluidCells = &data->fluidCells;
    params.materialGrid = data->materialGrid;
    params.velocityField = data->macGrid;

    VectorXd pressures((int)data->fluidCells.size());
    pSolver.solve(params, pressures);

    Float *p = solver->pSolver->Pressure();
    size_t items = solver->domain.x * solver->domain.y;
    for(size_t i = 0; i < items; i++){
        p[i] = 0;
    }

    GridIndex2 g;
    for(size_t i = 0; i < pressures.size(); i++){
        g = data->fluidCells[i];
        Accessor2D(p, g.i, g.j, solver->domain.x) = pressures[i];
    }
}

static void MacSolverUpdateMaterialGrid(MacSolver2 *solver){
    MacSolverData2 *data = &solver->solverData;
    MaterialGridData2 *materialGrid = data->materialGrid;
    CatmullMarkerAndCellGrid2 *d = data->d;
    GridIndexVector *fluidCells = &data->fluidCells;

    size_t items = materialGrid->total;
    int w = materialGrid->nx;

    fluidCells->clear();

    //AutoParallelFor("Update materials", items, AutoLambda(size_t index){
    for(size_t index = 0; index < items; index++){
        int x = index % w;
        int y = index / w;

        Material mat;
        mat.id = 0;
        if(true){
            if(d->at(x, y) > 0){
                mat.type = Fluid;
                fluidCells->push_back(GridIndex2(x, y));
            }else
                mat.type = Air;
        }

        materialGrid->Set(x, y, mat);
    }
    //);


    CatmullMarkerAndCellGrid2 *u = data->u;
    CatmullMarkerAndCellGrid2 *v = data->v;

    for(int i = 0; i < u->resolution.x; i++){
        for(int j = 0; j < u->resolution.y; j++){
            data->macGrid->SetU(i, j, u->at(i, j));
        }
    }

    for(int i = 0; i < v->resolution.x; i++){
        for(int j = 0; j < v->resolution.y; j++){
            data->macGrid->SetV(i, j, v->at(i, j));
        }
    }
}

static void MacSolverApplyPressure(MacSolver2 *solver, Float timestep){
    MacSolverAssertPressureSolver(solver);
    MacSolverData2 *data = &solver->solverData;
    Float scale = timestep/(data->density * data->hx);
    int w = solver->domain.x;
    int h = solver->domain.y;
    size_t items = w * h;

    Float *p = solver->pSolver->Pressure();
    CatmullMarkerAndCellGrid2 *u = data->u;
    CatmullMarkerAndCellGrid2 *v = data->v;

    //ParallelFor((size_t)0, items, [&](int i) -> void
    for(size_t i = 0; i < items; i++)
    {
        int x = i % w;
        int y = i / w;
        Float pi = p[i];
        Float pxpy = 0;
        Float pxyp = 0;
        if(x >= 1){
            pxpy = Accessor2D(p, x-1, y, w);
        }

        if(y >= 1){
            pxyp = Accessor2D(p, x, y-1, w);
        }

        u->at(x, y) -= scale*(pi - pxpy);
        v->at(x, y) -= scale*(pi - pxyp);
    }
    //);

    for(int i = 0; i < Max(h, w); i++){
        if(i < h){
            u->at(0, i) = u->at(w, i) = 0.0;
        }
        if(i < w){
            v->at(i, 0) = v->at(i, h) = 0.0;
        }
    }
}

__host__ void MacSolver2::Initialize(vec2ui res, Float targetDensity){
    Initialize(res, targetDensity, nullptr);
}

__host__ void MacSolver2::Initialize(vec2ui res, Float targetDensity,
                                     PressureSolver2 *pressureSolver)
{
    Float hx = 1.f / Min(res.x, res.y);
    domain = res;
    solverData.hx = hx;
    solverData.density = targetDensity;
    solverData.d = CatmullMarkerAndCellGrid2::Create(res.x    , res.y    , 0.5, 0.5, hx);
    solverData.u = CatmullMarkerAndCellGrid2::Create(res.x + 1, res.y    , 0.0, 0.5, hx);
    solverData.v = CatmullMarkerAndCellGrid2::Create(res.x    , res.y + 1, 0.5, 0.0, hx);

    solverData.materialGrid = cudaAllocateVx(MaterialGridData2, 1);
    solverData.materialGrid->Build(res, 0.f);
    solverData.materialGrid->SetGeometry(vec2f(0.f), hx, vec2f(0.5));
    solverData.fluidCells = GridIndexVector(res.x, res.y);

    if(pressureSolver){
        SetPressureSolver(pressureSolver);
    }

    /////////////////// test
    solverData.macGrid = cudaAllocateVx(MACVelocityGrid2, 1);
    solverData.macGrid->Init(vec2f(0.f), res, hx, false);
}

__host__ void MacSolver2::SetPressureSolver(PressureSolver2 *pressureSolver){
    pSolver = pressureSolver;
    if(!pSolver->IsBuilt()){
        Float hx = 1.f / Min(domain.x, domain.y);
        size_t n = domain.x * domain.y;
        pSolver->BuildSolver(n, hx, domain);
    }
}

__host__ void MacSolver2::AddInflow(vec2f p, vec2f length, Float d, Float u, Float v){
    MacSolverData2 *data = &solverData;

    data->d->addInflow(p.x, p.y, p.x + length.x, p.y + length.y, d);
    data->u->addInflow(p.x, p.y, p.x + length.x, p.y + length.y, u);
    data->v->addInflow(p.x, p.y, p.x + length.x, p.y + length.y, v);
}

__host__ void MacSolver2::Advance(Float timeIntervalInSeconds){
    MacSolverData2 *data = &solverData;

    MacSolverUpdateMaterialGrid(this);

    //std::cout << *(data->materialGrid) << std::endl;

    MacSolverProject(this, kPressureSolverIterations, timeIntervalInSeconds);
    MacSolverApplyPressure(this, timeIntervalInSeconds);

    Advect(timeIntervalInSeconds, data->d, data->u, data->v);
    Advect(timeIntervalInSeconds, data->u, data->u, data->v);
    Advect(timeIntervalInSeconds, data->v, data->u, data->v);

    data->d->flip();
    data->u->flip();
    data->v->flip();
}
