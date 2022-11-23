#include <explicit_grid.h>

// eigen has a const infinity somewhere that clashes with our define
#if defined(Infinity)
    #undef Infinity
#endif
#include <eigen/Eigen/SparseCore>
#include <eigen/Eigen/IterativeLinearSolvers>

struct PressureSolverEigenData2{
    vec2ui res;
    GridData2f fluidGrid;
    Eigen::ConjugateGradient<Eigen::SparseMatrix<double>> cgSolver;
};

PressureSolverEigen2::PressureSolverEigen2() : memory(nullptr){}

PressureSolverEigen2::~PressureSolverEigen2(){
    if(memory){
        PressureSolverEigenData2 *data =
                static_cast<PressureSolverEigenData2 *>(memory);
        delete data;
    }
}

__host__ void PressureSolverEigen2::BuildSolver(size_t n, Float _dx, vec2ui res){
    size_t size = res.x * res.y;
    if(n != size){
        printf("Error: Resolution does not match ( %u != %u )!\n",
                (unsigned int)n, (unsigned int)size);
        exit(0);
    }

    Set(res, _dx, n);
    PressureSolverEigenData2 *data = new PressureSolverEigenData2;
    data->res = res;
    data->fluidGrid.Build(res, -1);
    memory = (void *)data;
}

__host__ bool IsCellSolid(int i, int j, int w, int h,
                          std::function<Float(int,int)> dSampler)
{
    if(i < 0 || i >= w || j < 0 || j >= h) return true;
    return false;
}

__host__ bool IsCellLiquid(int i, int j, int w, int h,
                           std::function<Float(int,int)> dSampler)
{
    if(i < 0 || i >= w || j < 0 || j >= h) return false;
    return dSampler(i, j) > 0;
}

__host__ void PressureSolverEigen2::UpdateFull(int limit, Float *p, Float density,
                                               Float timestep, Float maxErr,
                                               std::function<Float(int,int)> uSampler,
                                               std::function<Float(int,int)> vSampler,
                                               std::function<Float(int,int)> dSampler)
{
    int n_cells = 0;
    PressureSolverEigenData2 *info = static_cast<PressureSolverEigenData2 *>(memory);
    size_t items = info->res.x * info->res.y;
    Eigen::SparseMatrix<double> A;
    info->fluidGrid.Fill(-1);
    for(int i = 0; i < info->res.x; i++){
        for(int j = 0; j < info->res.y; j++){
            if(dSampler(i, j) > 0 || true){ // liquid
                info->fluidGrid.Set(i, j, n_cells);
                n_cells ++;
            }else{ // air
                info->fluidGrid.Set(i, j, -1);
            }
        }
    }

    if(n_cells == 0) return;

    A.resize(n_cells, n_cells);
    A.reserve(Eigen::VectorXi::Constant(5, n_cells));

    Eigen::VectorXd b(n_cells);

    Float invLap = timestep / (dx * dx * density);
    int h = info->res.y;
    int w = info->res.x;
    for(int j = 0; j < h; j++){
        for(int i = 0; i < w; i++){
            if(dSampler(i, j) > 0 || true){
                int idx = info->fluidGrid.At(i, j);
                int non_solids = 0;
                if(!IsCellSolid(i-1, j, w, h, dSampler)){
                    if(IsCellLiquid(i-1, j, w, h, dSampler)){
                        A.insert(info->fluidGrid.At(i-1, j), idx) = invLap;
                    }
                    non_solids += 1;
                }

                if(!IsCellSolid(i+1, j, w, h, dSampler)){
                    if(IsCellLiquid(i+1, j, w, h, dSampler)){
                        A.insert(info->fluidGrid.At(i+1, j), idx) = invLap;
                    }
                    non_solids += 1;
                }

                if(!IsCellSolid(i, j-1, w, h, dSampler)){
                    if(IsCellLiquid(i, j-1, w, h, dSampler)){
                        A.insert(info->fluidGrid.At(i, j-1), idx) = invLap;
                    }
                    non_solids += 1;
                }

                if(!IsCellSolid(i, j+1, w, h, dSampler)){
                    if(IsCellLiquid(i, j+1, w, h, dSampler)){
                        A.insert(info->fluidGrid.At(i-1, j), idx) = invLap;
                    }
                    non_solids += 1;
                }

                A.insert(idx, idx) = -(Float)non_solids * invLap;

                b[idx] = ((uSampler(i+1,j) - uSampler(i,j)) +
                          (vSampler(i,j+1) - vSampler(i,j))) / dx;
            }
        }
    }
#if 0
    float usolid = 0.0;
    float vsolid = 0.0;
    float scale= 1.f /dx;
    for(int j = 0; j < h; j++){
        for(int i = 0; i < w; i++){
            if(info->fluidGrid.At(i, j) > -.5){
                int idx = info->fluidGrid.At(i, j);
                if(IsCellSolid(i-1, j, w, h, dSampler)){
                    b[idx] -= (float)scale*(uSampler(i, j) - usolid);
                }
                if(IsCellSolid(i+1, j, w, h, dSampler)){
                    b[idx] += (float)scale*(uSampler(i+1, j) - usolid);
                }

                if(IsCellSolid(i, j-1, w, h, dSampler)){
                    b[idx] -= (float)scale*(vSampler(i, j) - vsolid);
                }
                if(IsCellSolid(i, j+1, w, h, dSampler)){
                    b[idx] += (float)scale*(vSampler(i, j+1) - vsolid);
                }
            }
        }
    }
#endif
    Eigen::VectorXd pressures(n_cells);

    info->cgSolver.compute(A);
    pressures = info->cgSolver.solve(b);

    memset(p, 0, sizeof(Float) * items);
    int counter = 0;
    for(int i = 0; i < info->res.x; i++){
        for(int j = 0; j < info->res.y; j++){
            Float at = info->fluidGrid.At(i, j);
            if(dSampler(i, j) > 0){
                counter++;
                p[i + j * info->res.x] = pressures[at];
            }
        }
    }

    printf("Iterations: %d, Error: %g, Cells: %d - %d\n", (int)info->cgSolver.iterations(),
            (Float)info->cgSolver.error(), n_cells, counter);
}

__host__ void PressureSolverEigen2::Update(int limit, Float *p, Float density,
                                           Float timestep, Float maxErr)
{
    printf("Invoked invalid pressure solver\n");
}
