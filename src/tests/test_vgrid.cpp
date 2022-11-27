#include <tests.h>
#include <vgrid.h>
#include <grid.h>
#include <mac_grid.h>
#include <memory.h>
#include <advection.h>
#include <explicit_grid.h>
#include <pressuresolver.h>
#include <graphy-inl.h>

template<typename Interpolator>
void AddInflow(GridData2f *grid, vec2f p, vec2f length, Float val){
    Interpolator interpolator;
    Float hx = grid->spacing;
    Float x0 = p.x;
    Float y0 = p.y;
    Float y1 = p.y + length.y;
    Float x1 = p.x + length.x;
    int ix0 = (int)(x0/hx - grid->dataOffset.x);
    int iy0 = (int)(y0/hx - grid->dataOffset.y);
    int ix1 = (int)(x1/hx - grid->dataOffset.x);
    int iy1 = (int)(y1/hx - grid->dataOffset.y);

    if(IsHighpZero(val)) return;
    Float v = val;
    for (int y = max(iy0, 0); y < min(iy1, grid->ny); y++){
        for(int x = max(ix0, 0); x < min(ix1, grid->nx); x++){
            Float px = (2.0*(x + 0.5) * hx - (x0 + x1)) / (x1 - x0);
            Float py = (2.0*(y + 0.5) * hx - (y0 + y1)) / (y1 - y0);
            Float pl = vec2f(px, py).Length();
            Float vi = interpolator.Pulse(pl) * v;
            Float oldValue = grid->At(x, y);
            if(fabs(oldValue) < fabs(vi)){
                grid->Set(x, y, vi);
            }
        }
    }
}

void test_copy_explict_to_vgrid(vec2ui res, vec2f off, Float hx,
                                vec2f p, vec2f length, Float val,
                                GridData2f **vgrid, CatmullMarkerAndCellGrid2 **dd)
{
    std::vector<vec2ui> non_zeros;
    GridData2f *grid = cudaAllocateVx(GridData2f, 1);
     CatmullMarkerAndCellGrid2 *d =
        CatmullMarkerAndCellGrid2::Create(res.x, res.y, off.x, off.y, hx);

    if(val > 0)
        d->addInflow(p.x, p.y, p.x + length.x, p.y + length.y, val);

    grid->Build(res, 0.f);
    grid->BuildAuxiliary();
    grid->SetGeometry(vec2f(0.f), hx, off);

    AddInflow<MonotonicCatmull>(grid, p, length, val);

    TEST_CHECK(grid->nx == d->resolution.x &&
               grid->ny == d->resolution.y, "Invalid grid resolution");

    for(int i = 0; i < res.x; i++){
        for(int j = 0; j < res.y; j++){
            //grid->Set(i, j, d->at(i, j));
            if(d->at(i, j) > 0){
                non_zeros.push_back(vec2ui(i, j));
            }
        }
    }

    for(int i = 0; i < res.x; i++){
        for(int j = 0; j < res.y; j++){
            TEST_CHECK(grid->At(i, j) == d->at(i, j), "Grid initialization failed");
        }
    }

    *vgrid = grid;
    *dd = d;
}

void test_mac_(){
    MACVelocityGrid2 grid;
    GridAdvectionSolver2 advectionSolver;
    Float timestep = 1.f;
    Float dx = 0.1f;

    vec2ui res(128);
    GridData2f *d, *u, *v;
    CatmullMarkerAndCellGrid2 *d_, *u_, *v_;

    test_copy_explict_to_vgrid(res, vec2f(0.5f), dx, vec2f(0.1f),
                               vec2f(0.6f), 15.f, &d, &d_);

    test_copy_explict_to_vgrid(vec2ui(res.x + 1, res.y), vec2f(0.f, 0.5f), dx,
                               vec2f(0.1f), vec2f(0.6), 0.01, &u, &u_);

    test_copy_explict_to_vgrid(vec2ui(res.x, res.y + 1), vec2f(0.5f, 0.f), dx,
                               vec2f(0.1f), vec2f(0.6), 0.01, &v, &v_);

    grid._us[0] = u;
    grid._us[1] = v;
    int it = 0;
    GridData2f resGrid;
    resGrid.Build(res, 0.f);
    while(it++ < 30){
        advectionSolver.Advect(&grid, d, timestep);
        Advect(timestep, d_, u_, v_);

        d->Flip();
        d_->flip();

        bool error = false;

        Float err = 0.f;
        for(int i = 0; i < res.x; i++){
            for(int j = 0; j < res.y; j++){
                Float f = Absf(d->At(i, j) - d_->at(i, j));
                resGrid.Set(i, j, f);
                if(!IsZero(f) && !error){
                    err = f;
                    error = true;
                }
            }
        }
        std::cout << *d << std::endl;
        if(error){
            std::cout << resGrid << std::endl;
            printf("Not zero convergence ( %d ) Error = %g\n", it, err);
            exit(0);
        }
    }

    MaterialGridData2 matGrid;
    Material solid;
    Material fluid;
    solid.type = Solid;
    fluid.type = Fluid;
    matGrid.Build(res, 0.f);

    for(int j = 0; j < d->ny; j++){
        d->Set(0, j, 0.f);
        d->Set(d->nx-1, j, 0.f);
        matGrid.Set(0, j, solid);
        matGrid.Set(d->nx-1, j, solid);
    }

    for(int i = 0; i < d->nx; i++){
        d->Set(i, 0, 0.f);
        d->Set(i, d->ny-1, 0.f);
        matGrid.Set(i, 0, solid);
        matGrid.Set(i, d->ny-1, solid);
    }

    GridIndexVector fluidCells(res.x, res.y);
    for(int i = 0; i < d->nx; i++){
        for(int j = 0; j < d->ny; j++){
            if(d->At(i, j) > 0){
                matGrid.Set(i, j, fluid);
                fluidCells.push_back(GridIndex2(i, j));
            }
        }
    }

    std::cout << *d << std::endl;
    std::cout << matGrid << std::endl;

    TmpPressureSolver solver;
    PressureSolverParameters params;
    params.cellwidth = dx;
    params.deltaTime = timestep;
    params.density = 0.1;

    params.fluidCells = &fluidCells;
    params.materialGrid = &matGrid;
    params.velocityField = &grid;

    VectorXd pressures((int)fluidCells.size());

    solver.solve(params, pressures);
}

class TestSolver{
    public:
    MACVelocityGrid2 *macGrid;
    GridData2f *density;
    MaterialGridData2 *matPtr;

    TestSolver(vec2ui res, Float hx){
        macGrid = cudaAllocateVx(MACVelocityGrid2, 1);
        density = cudaAllocateVx(GridData2f, 1);
        macGrid->Init(vec2f(0.f), res, hx, true);

        density->Build(res, 0.f);
        density->BuildAuxiliary();
        density->SetGeometry(vec2f(0.f), hx, vec2f(0.5, 0.5));
    }
};

bool isFaceBorderingMaterialU(MaterialGridData2 *grid, int i, int j, MaterialType m) {
    if (i == grid->nx) { return grid->At(i - 1, j).type == m; }
    else if (i > 0) { return grid->At(i, j).type == m || grid->At(i - 1, j).type == m; }
    else { return grid->At(i, j).type == m; }
}

bool isFaceBorderingMaterialV(MaterialGridData2 *grid, int i, int j, MaterialType m) {
    if (j == grid->ny) { return grid->At(i, j - 1).type == m; }
    else if (j > 0) { return grid->At(i, j).type == m || grid->At(i, j - 1).type == m; }
    else { return grid->At(i, j).type == m; }
}

bool isFaceBorderingSolidU(MaterialGridData2 *grid, int i, int j) {
    return isFaceBorderingMaterialU(grid, i, j, MaterialType::Solid);
}

bool isFaceBorderingFluidU(MaterialGridData2 *grid, int i, int j) {
    return isFaceBorderingMaterialU(grid, i, j, MaterialType::Fluid);
}

bool isFaceBorderingSolidV(MaterialGridData2 *grid, int i, int j) {
    return isFaceBorderingMaterialV(grid, i, j, MaterialType::Solid);
}

bool isFaceBorderingFluidV(MaterialGridData2 *grid, int i, int j) {
    return isFaceBorderingMaterialV(grid, i, j, MaterialType::Fluid);
}

bool isCellSolid(MaterialGridData2 *grid, int i, int j){
    Material mat = grid->At(i, j);
    return mat.type == Solid;
}

void _applyPressureToFaceU(int i, int j, GridData2f *pressureGrid,
                           MACVelocityGrid2 *tempMACVelocity,
                           MACVelocityGrid2 *macGrid, double dt,
                           MaterialGridData2 *_materialGrid)
{
    Float _density = 0.1;
    Float _dx = 1.f / 128.f;
    double usolid = 0.0;   // solids are stationary
    double scale = dt / (_density * _dx);
    double invscale = 1.0 / scale;

    int ci = i - 1; int cj = j;

    double p0, p1;
    if (!isCellSolid(_materialGrid, ci, cj) && !isCellSolid(_materialGrid, ci + 1, cj))
    {
        p0 = pressureGrid->At(ci, cj);
        p1 = pressureGrid->At(ci + 1, cj);
    } else if (isCellSolid(_materialGrid, ci, cj)) {
        p0 = pressureGrid->At(ci + 1, cj) -
                invscale*(macGrid->U(i, j) - usolid);
        p1 = pressureGrid->At(ci + 1, cj);
    } else {
        p0 = pressureGrid->At(ci, cj);
        p1 = pressureGrid->At(ci, cj) +
                invscale*(macGrid->U(i, j) - usolid);
    }

    double unext = macGrid->U(i, j) - scale*(p1 - p0);
    tempMACVelocity->SetU(i, j, unext);
}

void _applyPressureToFaceV(int i, int j, GridData2f *pressureGrid,
                           MACVelocityGrid2 *tempMACVelocity,
                           MACVelocityGrid2 *macGrid, double dt,
                           MaterialGridData2 *_materialGrid)
{
    Float _density = 0.1;
    Float _dx = 1.f / 128.f;
    double usolid = 0.0;   // solids are stationary
    double scale = dt / (_density * _dx);
    double invscale = 1.0 / scale;

    int ci = i; int cj = j - 1;

    double p0, p1;
    if (!isCellSolid(_materialGrid, ci, cj) && !isCellSolid(_materialGrid, ci, cj + 1)){
        p0 = pressureGrid->At(ci, cj);
        p1 = pressureGrid->At(ci, cj + 1);
    }
    else if (isCellSolid(_materialGrid, ci, cj)) {
        p0 = pressureGrid->At(ci, cj + 1) -
            invscale*(macGrid->V(i, j) - usolid);
        p1 = pressureGrid->At(ci, cj + 1);
    }
    else {
        p0 = pressureGrid->At(ci, cj);
        p1 = pressureGrid->At(ci, cj) +
            invscale*(macGrid->V(i, j) - usolid);
    }

    double vnext = macGrid->V(i, j) - scale*(p1 - p0);
    tempMACVelocity->SetV(i, j, vnext);
}

void apply_pressure(TestSolver &solver, MACVelocityGrid2 *tmpGrid,
                    GridData2f *pressureGrid, Float dt)
{
    int _jsize = pressureGrid->ny;
    int _isize = pressureGrid->nx;
    MACVelocityGrid2 *macGrid = solver.macGrid;
    MaterialGridData2 *_materialGrid = solver.matPtr;

    {
        for (int j = 0; j < _jsize; j++) {
            for (int i = 0; i < _isize + 1; i++) {
                if (isFaceBorderingSolidU(_materialGrid, i, j)) {
                    tmpGrid->SetU(i, j, 0.0);
                }

                if (isFaceBorderingFluidU(_materialGrid, i, j) &&
                    !isFaceBorderingSolidU(_materialGrid, i, j))
                {
                    _applyPressureToFaceU(i, j, pressureGrid, tmpGrid, macGrid,
                                        dt, _materialGrid);
                }
            }
        }
    }

    {
        for (int j = 0; j < _jsize + 1; j++) {
            for (int i = 0; i < _isize; i++) {
                if (isFaceBorderingSolidV(_materialGrid, i, j)){
                    tmpGrid->SetV(i, j, 0.0);
                }

                if (isFaceBorderingFluidV(_materialGrid, i, j) &&
                    !isFaceBorderingSolidV(_materialGrid, i, j))
                {
                    _applyPressureToFaceV(i, j, pressureGrid, tmpGrid, macGrid,
                                        dt, _materialGrid);
                }
            }
        }
    }

    {
        for (int j = 0; j < _jsize; j++) {
            for (int i = 0; i < _isize + 1; i++) {
                if (isFaceBorderingFluidU(_materialGrid, i, j)) {
                    macGrid->SetU(i, j, tmpGrid->U(i, j));
                }
            }
        }
    }

    {
        for (int j = 0; j < _jsize + 1; j++) {
            for (int i = 0; i < _isize; i++) {
                if (isFaceBorderingFluidV(_materialGrid, i, j)) {
                    macGrid->SetV(i, j, tmpGrid->V(i, j));
                }
            }
        }
    }
}

template<typename ActualInterpolator>
struct TestInterpolator{
    Float queried_x, queried_y;
    ActualInterpolator interpolator;
    vec2ui ids[64];
    int count;

    template<typename Fn>
    __bidevice__ Float Interpolate(Float x, Float y, vec2ui resolution, const Fn &fn){
        queried_x = x;
        queried_y = y;
        count = 0;
        auto query_steal = [&](int _x, int _y) -> Float{
            ids[count++] = vec2ui(_x, _y);
            return fn(_x, _y);
        };
        return interpolator.Interpolate(x, y, resolution, query_steal);
    }
};

void test_vgrid_grid2_sample(){
    printf("===== Test VGrid 2D Sample\n");
    int nx = 512;
    int ny = 256;
    int total = nx * ny;
    Float dx = 0.25f;
    vec2f origin = vec2f(-1, -2);
    vec2f dataOffset = vec2f(0.5f, 0.5f);

    GridData2f grid(vec2ui(nx, ny), 0.f);
    grid.SetGeometry(origin, dx, dataOffset);

    int samples = 8;

    printf(" * Grid configuration:\n");
    printf("   - Resolution %d x %d\n", nx, ny);
    printf("   - Offset %g %g\n", dataOffset.x, dataOffset.y);
    printf("   - Spacing %g\n", dx);
    printf("   - Origin %g %g\n", origin.x, origin.y);

    TEST_CHECK(grid.nx == nx, "Invalid vgrid width");
    TEST_CHECK(grid.ny == ny, "Invalid vgrid height");
    TEST_CHECK(grid.total == total, "Invalid vgrid size");
    TEST_CHECK(grid.dataOffset[0] == dataOffset.x, "Invalid offset x");
    TEST_CHECK(grid.dataOffset[1] == dataOffset.y, "Invalid offset x");
    TEST_CHECK(grid.origin[0] == origin.x, "Invalid origin x");
    TEST_CHECK(grid.origin[1] == origin.y, "Invalid origin y");
    TEST_CHECK(grid.Spacing() == dx, "Invalid spacing");

    for(int i = 0; i < grid.nx; i++){
        for(int j = 0; j < grid.ny; j++){
            int val = i + j * grid.nx;
            grid.Set(i, j, (Float)val);
        }
    }

    TestInterpolator<LinearInterpolator> linear;

    using interp_type_linear = decltype(linear);

    auto value_at = [&](int i, int j) -> Float{
        return i + j * grid.nx;
    };

    auto interp_test = [&](Float x, Float y, std::vector<vec2f> &vals) -> Float{
        int iSize = nx, jSize = ny;
        int ix = (int)x;
        int iy = (int)y;
        x -= ix;
        y -= iy;

        Float x00 = value_at(ix, iy),
              x10 = value_at(Min(ix + 1, iSize - 1), iy),
              x01 = value_at(ix, Min(iy + 1, jSize - 1)),
              x11 = value_at(Min(ix + 1, iSize - 1), Min(iy + 1, jSize - 1));

        vals.push_back(vec2f(ix, iy));
        vals.push_back(vec2f(Min(ix + 1, iSize - 1), iy));
        vals.push_back(vec2f(ix, Min(iy + 1, jSize - 1)));
        vals.push_back(vec2f(Min(ix + 1, iSize - 1), Min(iy + 1, jSize - 1)));

        return Mix(Mix(x00, x10, x), Mix(x01, x11, x), y);
    };

    for(int i = 0; i < grid.nx; i++){
        for(int j = 0; j < grid.ny; j++){
            vec2f ref_pos = vec2f(i * dx, j * dx);
            vec2f data_pos = grid.DataPosition(i + j * grid.nx);
            Float x_ = origin.x + (i + dataOffset.x) * dx;
            Float y_ = origin.y + (j + dataOffset.y) * dx;
            Float x0 = x_, x1 = x0 + dx;
            Float y0 = y_, y1 = y0 + dx;

            Float f_real = i + j * grid.nx;

            TEST_CHECK(IsZero(data_pos.x - x_), "Invalid x position for sample point");
            TEST_CHECK(IsZero(data_pos.y - y_), "Invalid y position for sample point");
            TEST_CHECK(grid.At(i, j) == f_real, "Invalid value on cell");

            for(int s = 0; s < samples; s++){
                std::vector<vec2f> cells;
                vec2f p_sample, p_ref;
                vec2f u = vec2f(rand_float(), rand_float()) * dx;
                p_sample = data_pos + u;
                p_ref = ref_pos + u;

                TEST_CHECK(p_sample.x >= x0 && p_sample.x <= x1 &&
                           p_sample.y >= y0 && p_sample.y <= y1, "Invalid sample");

                Float f_sample  =
                    grid.Sample<interp_type_linear>(p_sample.x, p_sample.y, linear);

                Float f_real = interp_test(p_ref.x, p_ref.y, cells);

                vec2f q_box_0 = vec2f(i, j) * dx;
                vec2f q_box_1 = q_box_0 + vec2f(dx);

                TEST_CHECK(linear.queried_x >= q_box_0.x, "Invalid lower bound query (x)");
                TEST_CHECK(linear.queried_x <= q_box_1.x, "Invalid upper bound query (x)");
                TEST_CHECK(linear.queried_y >= q_box_0.y, "Invalid lower bound query (y)");
                TEST_CHECK(linear.queried_y <= q_box_1.y, "Invalid upper bound query (y)");
                TEST_CHECK(cells.size() == linear.count, "Invalid query count");
                TEST_CHECK((Absf(linear.queried_x - p_ref.x) < 1e-4) &&
                           (Absf(linear.queried_y - p_ref.y) < 1e-4),
                            "Query fraction does not match");

                for(int i = 0; i < cells.size(); i++){
                    vec2ui cell = cells[i];
                    vec2ui id = linear.ids[i];
                    TEST_CHECK(cell.x == id.x && cell.y == id.y,
                               "Invalid cell query");
                }
            }
        }
    }
    printf("===== OK\n");
}

void test_mac_grid2_init(){
    printf("===== Test MACGrid2 Init\n");
    CudaMemoryManagerStart(__FUNCTION__);
    vec2ui res(256, 128);
    Float hx = 0.15;
    vec2f origin(-1, -3);
    bool withCopy = true;

    MACVelocityGrid2 grid;
    grid.Init(origin, res, hx, withCopy);

    auto *u = grid.U_ptr();
    auto *v = grid.V_ptr();
    auto *w = grid.W_ptr();

    printf("Addresses: U(%p), V(%p), W(%p)\n", u, v, w);

    TEST_CHECK(u != nullptr && v != nullptr, "Did not initialize U/V");
    TEST_CHECK(w == nullptr, "W is not null");

    printf("U dimensions: %d x %d\n", u->nx, u->ny);
    printf("V dimensions: %d x %d\n", v->nx, v->ny);

    TEST_CHECK(u->nx == res.x+1, "Invalid U x-size");
    TEST_CHECK(u->ny == res.y, "Invalid U y-size");

    TEST_CHECK(v->nx == res.x, "Invalid V x-size");
    TEST_CHECK(v->ny == res.y+1, "Invalid V y-size");

    for(int i = 0; i < 2; i++){
        TEST_CHECK(IsZero(v->origin[i] - origin[i]), "Invalid V origin axis");
        TEST_CHECK(IsZero(u->origin[i] - origin[i]), "Invalid U origin axis");
    }

    TEST_CHECK(IsZero(u->spacing - hx), "Invalid U spacing");
    TEST_CHECK(IsZero(v->spacing - hx), "Invalid V spacing");

    TEST_CHECK(u->withCopy == withCopy, "Invalid U auxiliary buffer flag");
    TEST_CHECK(v->withCopy == withCopy, "Invalid V auxiliary buffer flag");

    if(withCopy){
        TEST_CHECK(u->pong != nullptr, "Invalid U auxiliary buffer");
        TEST_CHECK(v->pong != nullptr, "Invalid V auxiliary buffer");
    }

    printf("U offsets: [%g %g]\n", u->dataOffset.x, u->dataOffset.y);
    printf("V offsets: [%g %g]\n", v->dataOffset.x, v->dataOffset.y);

    TEST_CHECK(IsZero(u->dataOffset.x), "Invalid U x-offset");
    TEST_CHECK(IsZero(u->dataOffset.y - 0.5f), "Invalid U y-offset");

    TEST_CHECK(IsZero(v->dataOffset.x - 0.5f), "Invalid V x-offset");
    TEST_CHECK(IsZero(v->dataOffset.y), "Invalid V y-offset");

    CudaMemoryManagerClearCurrent();
    printf("===== OK\n");
}


void test_vgrid_grid2_math(){
    printf("===== Test VGrid 2D Set/Add\n");
    int nx = 512;
    int ny = 256;
    vec2ui res(nx, ny);
    GridData2<Float> grid_pos(res, 0.f);
    GridData2<Float> grid_neg(res, 0.f);
    GridData2<Float> grid_zero(res, 0.f);

    auto val_fn = [](int i, int j) -> Float{
        i += 1;
        j += 1;
        return i * j + i + j;
    };

    for(int i = 0; i < nx; i++){
        for(int j = 0; j < ny; j++){
            Float val = val_fn(i, j);
            grid_pos.Set(i, j, val);
            grid_neg.Set(i, j, -val);
            Float f_pos = grid_pos(i, j);
            Float f_neg = grid_neg(i, j);
            TEST_CHECK(f_pos == val, "Read incorrect positive value");
            TEST_CHECK(f_neg == -val, "Read incorrect negative value");

            grid_pos.Set(GridIndex2(i, j), val+1);
            grid_neg.Set(GridIndex2(i, j), -val-1);
            f_pos = grid_pos(i, j);
            f_neg = grid_neg(i, j);
            TEST_CHECK(f_pos == val+1, "Read incorrect positive value");
            TEST_CHECK(f_neg == -val-1, "Read incorrect negative value");
        }
    }

    std::cout << " ** Positive grid:\n";
    std::cout << grid_pos << std::endl;
    std::cout << " ** Negative grid:\n";
    std::cout << grid_neg << std::endl;

    for(int i = 0; i < nx; i++){
        for(int j = 0; j < ny; j++){
            grid_zero.Add(i, j, grid_neg(i, j));
            grid_zero.Add(i, j, grid_pos(i, j));
            TEST_CHECK(IsZero(grid_zero(i, j)), "Did not sum to 0");

            grid_zero.Add(GridIndex2(i, j), grid_neg(i, j));
            grid_zero.Add(GridIndex2(i, j), grid_pos(i, j));
            TEST_CHECK(IsZero(grid_zero(i, j)), "Did not sum to 0");
        }
    }

    std::cout << " ** Zero grid:\n";
    std::cout << grid_zero << std::endl;
    printf("===== OK\n");
}

void test_vgrid_grid2_memory(){
    printf("===== Test VGrid 2D Memory\n");
    int nx = 512;
    int ny = 256;
    Float f_real = 3;
    int total = nx * ny;
    GridData2<Float> grid(vec2ui(nx, ny), f_real);
    TEST_CHECK(grid.nx == nx, "Invalid vgrid width");
    TEST_CHECK(grid.ny == ny, "Invalid vgrid height");
    TEST_CHECK(grid.total == total, "Invalid vgrid size");
    TEST_CHECK(grid.hasOutOfRange == false, "Invalid out of range flag");

    std::cout << " ** Grid\n";
    std::cout << grid << std::endl;

    for(int i = 0; i < grid.nx; i++){
        for(int j = 0; j < grid.ny; j++){
            int id = LinearIndex(vec2ui(i, j), vec2ui(nx, ny), 2);
            GridIndex2 index(i, j);
            Float f_id = grid.data[id];
            Float f_op_id = grid(id);
            Float f_op_ij = grid(i, j);
            Float f_op_gid = grid(index);
            //printf("Freal %g, Fid %g\n", f_real, f_id);
            TEST_CHECK(id == grid.LinearIndex(i, j), "Invalid hashed id");
            TEST_CHECK(f_real == f_id, "Invalid raw value");
            TEST_CHECK(f_real == f_op_id, "Invalid operator(uint)");
            TEST_CHECK(f_real == f_op_ij, "Invalid operator(uint, uint)");
            TEST_CHECK(f_real == f_op_gid, "Invalid operator(index)");
        }
    }

    auto val_fn = [](int i, int j) -> Float{
        i += 1;
        j += 1;
        return i * j + i + j;
    };

    for(int i = 0; i < grid.nx; i++){
        for(int j = 0; j < grid.ny; j++){
            int id = LinearIndex(vec2ui(i, j), vec2ui(nx, ny), 2);
            grid.data[id] = val_fn(i, j);
        }
    }

    for(int i = 0; i < grid.nx; i++){
        for(int j = 0; j < grid.ny; j++){
            f_real = val_fn(i, j);
            int id = LinearIndex(vec2ui(i, j), vec2ui(nx, ny), 2);
            GridIndex2 index(i, j);
            Float f_id = grid.data[id];
            Float f_op_id = grid(id);
            Float f_op_ij = grid(i, j);
            Float f_op_gid = grid(index);
            TEST_CHECK(f_real == f_id, "Invalid raw value");
            TEST_CHECK(f_real == f_op_id, "Invalid operator(uint)");
            TEST_CHECK(f_real == f_op_ij, "Invalid operator(uint, uint)");
            TEST_CHECK(f_real == f_op_gid, "Invalid operator(index)");
        }
    }

    std::cout << " ** New grid\n";
    std::cout << grid << std::endl;

    int scalex = 2;
    int scaley = 3;
    int totalscale = scalex * scaley;
    Float range = -1.5;
    GridData2<Float> grid2(vec2ui(scalex * nx, scaley * ny), 1.f);
    grid2.SetOutOfRange(range);
    grid = grid2;

    TEST_CHECK(grid.nx == scalex * nx, "Invalid copied width");
    TEST_CHECK(grid.ny == scaley * ny, "Invalid copied height");
    TEST_CHECK(grid.total == totalscale * total, "Invalid copied total");

    for(int i = 0; i < grid.nx; i++){
        for(int j = 0; j < grid.ny; j++){
            f_real = 1;
            int id = LinearIndex(vec2ui(i, j), vec2ui(nx, ny), 2);
            GridIndex2 index(i, j);
            Float f_id = grid.data[id];
            Float f_op_id = grid(id);
            Float f_op_ij = grid(i, j);
            Float f_op_gid = grid(index);
            TEST_CHECK(f_real == f_id, "Invalid raw value");
            TEST_CHECK(f_real == f_op_id, "Invalid operator(uint)");
            TEST_CHECK(f_real == f_op_ij, "Invalid operator(uint, uint)");
            TEST_CHECK(f_real == f_op_gid, "Invalid operator(index)");
        }
    }

    std::cout << " ** Copied grid\n";
    std::cout << grid << std::endl;

    int neg_i = -10000;
    int neg_j = -10000;
    int pos_i = grid.nx;
    int pos_j = grid.ny;

    for(int i = 0; i < grid.nx; i++){
        Float f_neg_j = grid(i, neg_j);
        Float f_pos_j = grid(i, pos_j);
        TEST_CHECK(range == f_neg_j, "Invalid out of range for negative j");
        TEST_CHECK(range == f_pos_j, "Invalid out of range for positive j");

        f_neg_j = grid(GridIndex2(i, neg_j));
        f_pos_j = grid(GridIndex2(i, pos_j));
        TEST_CHECK(range == f_neg_j, "Invalid out of range for negative j");
        TEST_CHECK(range == f_pos_j, "Invalid out of range for positive j");
    }

    for(int j = 0; j < grid.ny; j++){
        Float f_neg_i = grid(neg_i, j);
        Float f_pos_i = grid(pos_i, j);
        TEST_CHECK(range == f_neg_i, "Invalid out of range for negative i");
        TEST_CHECK(range == f_pos_i, "Invalid out of range for positive i");

        f_neg_i = grid(GridIndex2(neg_i, j));
        f_pos_i = grid(GridIndex2(pos_i, j));
        TEST_CHECK(range == f_neg_i, "Invalid out of range for negative i");
        TEST_CHECK(range == f_pos_i, "Invalid out of range for positive i");
    }

    TEST_CHECK(grid(neg_i) == range, "Invalid out of range for (uint)");
    TEST_CHECK(grid(grid.total) == range, "Invalid out of range for (uint)");

    printf("===== OK\n");
}

void test_vgrid_index(){
    printf("===== Test VGrid Indexing\n");
    GridIndex2 index(10, 15);
    GridIndex2 index2;
    GridIndex3 index3(10, 15, 20);
    GridIndex3 index32;

    std::cout << " ** GridIndex2:" << std::endl;
    std::cout << "Index 1 : " << index << std::endl;
    std::cout << "Index 2 : " << index2 << std::endl;
    std::cout << "Is different : " << (index != index2) << std::endl;
    std::cout << "Is equal : " << (index == index2) << std::endl;
    std::cout << "Indice 0 : " << index[0] << std::endl;
    std::cout << "Indice 1 : " << index[1] << std::endl;

    index[0] = 3;
    index[1] = 19;
    std::cout << "Change 0 : " << index[0] << std::endl;
    std::cout << "Change 1 : " << index[1] << std::endl;
    std::cout << "Index 1 : " << index << std::endl;

    TEST_CHECK(index.i == 3 && index.j == 19, "Invalid index");
    TEST_CHECK(index2.i == 0 && index2.j == 0, "Invalid index");
    TEST_CHECK(index != index2, "Index are equal");
    TEST_CHECK(!(index == index2), "Index are equal");

    index2 = index;
    std::cout << "Copied :" << index2 << std::endl;
    TEST_CHECK(index == index2, "Index are different");
    TEST_CHECK(!(index != index2), "Index are different");
    TEST_CHECK(index2.i == 3 && index2.j == 19, "Invalid index");

    std::cout << " ** GridIndex3:" << std::endl;
    std::cout << "Index 1 : " << index3 << std::endl;
    std::cout << "Index 2 : " << index32 << std::endl;
    std::cout << "Is different : " << (index3 != index32) << std::endl;
    std::cout << "Is equal : " << (index3 == index32) << std::endl;
    std::cout << "Indice 0 : " << index3[0] << std::endl;
    std::cout << "Indice 1 : " << index3[1] << std::endl;
    std::cout << "Indice 2 : " << index3[2] << std::endl;

    index3[0] = 3;
    index3[1] = 19;
    index3[2] = 1;
    std::cout << "Change 0 : " << index3[0] << std::endl;
    std::cout << "Change 1 : " << index3[1] << std::endl;
    std::cout << "Change 2 : " << index3[2] << std::endl;
    std::cout << "Index 1 : " << index3 << std::endl;

    TEST_CHECK(index3.i == 3 && index3.j == 19 && index3.k == 1, "Invalid index");
    TEST_CHECK(index32.i == 0 && index32.j == 0 && index32.k == 0, "Invalid index");
    TEST_CHECK(index3 != index32, "Index are equal");
    TEST_CHECK(!(index3 == index32), "Index are equal");

    index32 = index3;
    std::cout << "Copied :" << index32 << std::endl;
    TEST_CHECK(index3 == index32, "Index are different");
    TEST_CHECK(!(index3 != index32), "Index are different");
    TEST_CHECK(index32.i == 3 && index32.j == 19 && index32.k == 1, "Invalid index");

    printf("===== OK\n");
}

void test_virtual_grid(){
    //test_mac_();
    //test_mac_grid2_init();
    return;
    test_vgrid_grid2_math();
    test_vgrid_index();
    test_vgrid_grid2_memory();
    test_vgrid_grid2_sample();
}
