#pragma once
#include <geometry.h>
#include <cutil.h>
#include <sampling.h>

/*
* Basic grid element for 2D grid solvers.
*/
class ExplicitGrid2{
    public:
    vec2ui resolution;
    vec2f spacing;
    vec2f invSpacing;
    vec2f invSpacing2;
    vec2f origin;
    Bounds2f bounds;

    __bidevice__ ExplicitGrid2();
    __bidevice__ vec2ui GetResolution();
    __bidevice__ vec2f GetSpacing();
    __bidevice__ Bounds2f GetBounds();

    /* Sets domain information about the grid */
    __bidevice__ void Set(const vec2ui &res, const vec2f &h, const vec2f &o);

    /* Get the position according to a cell centered model for (i, j) pair */
    __bidevice__ vec2f GetCellCenteredPosition(const size_t &i, const size_t &j);

    /* Get the position according to a vertex centered model for (i, j) pair */
    __bidevice__ vec2f GetVertexCenteredPosition(const size_t &i, const size_t &j);
};

template<typename T> inline __bidevice__
T GetGridDataOrigin(VertexType type, T origin, T spacing){
    switch(type){
        case VertexType::CellCentered:   return origin + 0.5 * spacing;
        case VertexType::VertexCentered: return origin;
        default:{
            printf("Unknown explicit grid type\n");
        };
    }

    return T(0);
}

template<typename U, typename ExplicitGrid> inline __bidevice__
U GetGridDataSize(ExplicitGrid *egrid){
    U resolution = egrid->grid->GetResolution();
    switch(egrid->gridType){
        case VertexType::CellCentered:   return resolution;
        case VertexType::VertexCentered: return resolution + U(1);
        default:{
            printf("Unknown explicit grid type\n");
        };
    }

    return U(0);
}

class ExplicitScalarGrid2{
    public:
    ExplicitGrid2 *grid;
    Float *data;
    vec2f dataOrigin;
    vec2ui dataSize;
    int totalSize;
    VertexType gridType;

    __bidevice__ ExplicitScalarGrid2(){ totalSize = 0; data = nullptr; grid = nullptr; }

    __host__ void Set(const vec2ui &res, const vec2f &h, const vec2f &o,
                      VertexType type, Float initialValue=0)
    {
        AssureA(type == VertexCentered || type == CellCentered,
                "Scalar Grids only support vertex/cell centered data points");

        gridType = type;
        if(grid == nullptr){
            grid = cudaAllocateVx(ExplicitGrid2, 1);
        }

        grid->Set(res, h, o);
        dataOrigin = GetGridDataOrigin(gridType, o, h);
        dataSize = GetGridDataSize<vec2ui, ExplicitScalarGrid2>(this);
        totalSize = dataSize.x * dataSize.y;
        data = cudaAllocateVx(Float, totalSize);
        for(int i = 0; i < totalSize; i++){
            data[i] = initialValue;
        }
    }

    __bidevice__ const Float &operator()(size_t i, size_t j) const{
        size_t idx = i + dataSize.x * j;
        AssertA(idx < totalSize && data != nullptr, "Invalid query index");
        return data[idx];
    }

    __bidevice__ Float &operator()(size_t i, size_t j){
        size_t idx = i + dataSize.x * j;
        AssertA(idx < totalSize && data != nullptr, "Invalid query index");
        return data[idx];
    }

    __bidevice__ Float Sample(const vec2f &p){
        return LinearGridSampler2Sample<Float, Float>(p, data, grid->spacing,
                                                      dataOrigin, dataSize);
    }

    /* Computes gradient at a data point (fixed node) */
    __bidevice__ vec2f GradientAtPoint(size_t i, size_t j){
        Float left  = Accessor2D(data, i > 0 ? i - 1 : i, j, dataSize.x);
        Float right = Accessor2D(data, i + 1 < dataSize.x ? i + 1 : i, j, dataSize.x);
        Float down  = Accessor2D(data, i, j > 0 ? j - 1 : j, dataSize.x);
        Float up    = Accessor2D(data, i, j + 1 < dataSize.y ? j + 1 : j, dataSize.x);
        return 0.5 * vec2f(right - left, up - down) * grid->invSpacing;
    }

    /* Computes gradient at a specific point inside a cell */
    __bidevice__ vec2f Gradient(const vec2f &p){
        vec2ui indices[4];
        Float weights[4];
        vec2f result(0);
        LinearGridSampler2Weights(p, grid->spacing, dataOrigin,
                                  dataSize, &indices[0], &weights[0]);

        for(int i = 0; i < 4; i++){
            vec2ui ij = indices[i];
            result += weights[i] * GradientAtPoint(ij.x, ij.y);
        }

        return result;
    }

    /* Computes laplacian at a data point (fixed node) */
    __bidevice__ Float LaplacianAtPoint(size_t i, size_t j){
        Float center = Accessor2D(data, i, j, dataSize.x);
        Float left = 0, right = 0, up = 0, down = 0;
        if(i > 0) left = center - Accessor2D(data, i - 1, j, dataSize.x);
        if(i + 1 < dataSize.x) right = Accessor2D(data, i + 1, j, dataSize.x) - center;
        if(j > 0) down = center - Accessor2D(data, i, j - 1, dataSize.x);
        if(j + 1 < dataSize.y) up = Accessor2D(data, i, j + 1, dataSize.x) - center;

        return (right - left) * grid->invSpacing2.x + (up - down) * grid->invSpacing2.y;
    }

    /* Computes laplacian at a specific point inside a cell */
    __bidevice__ Float Laplacian(const vec2f &p){
        vec2ui indices[4];
        Float weights[4];
        Float result = 0;
        LinearGridSampler2Weights(p, grid->spacing, dataOrigin,
                                  dataSize, &indices[0], &weights[0]);

        for(int i = 0; i < 4; i++){
            vec2ui ij = indices[i];
            result += weights[i] * LaplacianAtPoint(ij.x, ij.y);
        }

        return result;
    }
};

class ExplicitCollocatedVectorGrid2{
    public:
    ExplicitGrid2 *grid;
    vec2f *data;
    vec2f dataOrigin;
    vec2ui dataSize;
    int totalSize;
    VertexType gridType;

    __bidevice__ ExplicitCollocatedVectorGrid2()
    { totalSize = 0; data = nullptr; grid = nullptr; }


    __host__ void Set(const vec2ui &res, const vec2f &h, const vec2f &o,
                      VertexType type, vec2f initialValue=0)
    {
        AssureA(type == VertexCentered || type == CellCentered,
                "Collocated Grids only support vertex/cell centered data points");

        gridType = type;
        if(grid == nullptr){
            grid = cudaAllocateVx(ExplicitGrid2, 1);
        }

        grid->Set(res, h, o);
        dataOrigin = GetGridDataOrigin(gridType, o, h);
        dataSize = GetGridDataSize<vec2ui, ExplicitCollocatedVectorGrid2>(this);
        totalSize = dataSize.x * dataSize.y;
        data = cudaAllocateVx(vec2f, totalSize);
        for(int i = 0; i < totalSize; i++){
            data[i] = initialValue;
        }
    }

    __bidevice__ const vec2f &operator()(size_t i, size_t j) const{
        size_t idx = i + dataSize.x * j;
        AssertA(idx < totalSize && data != nullptr, "Invalid query index");
        return data[idx];
    }

    __bidevice__ vec2f &operator()(size_t i, size_t j){
        size_t idx = i + dataSize.x * j;
        AssertA(idx < totalSize && data != nullptr, "Invalid query index");
        return data[idx];
    }

    __bidevice__ vec2f Sample(const vec2f &p){
        return LinearGridSampler2Sample<vec2f, Float>(p, data, grid->spacing,
                                                      dataOrigin, dataSize);
    }

    /* Computes curl at a data point (fixed node) */
    __bidevice__ vec2f CurlAtPoint(size_t i, size_t j){
        vec2f left  = Accessor2D(data, i > 0 ? i - 1 : i, j, dataSize.x);
        vec2f right = Accessor2D(data, i + 1 < dataSize.x ? i + 1 : i, j, dataSize.x);
        vec2f down  = Accessor2D(data, i, j > 0 ? j - 1 : j, dataSize.x);
        vec2f up    = Accessor2D(data, i, j + 1 < dataSize.y ? j + 1 : j, dataSize.x);

        Float Fx_ym = down.x;
        Float Fx_yp = up.x;
        Float Fy_xm = left.y;
        Float Fy_xp = right.y;

        return 0.5 * (Fy_xp - Fy_xm) * grid->invSpacing.x -
            0.5 * (Fx_yp - Fx_ym) * grid->invSpacing.y;
    }

    /* Computes curl at a specific point inside a cell */
    __bidevice__ vec2f Curl(const vec2f &p){
        vec2ui indices[4];
        Float weights[4];
        vec2f result(0);
        LinearGridSampler2Weights(p, grid->spacing, dataOrigin,
                                  dataSize, &indices[0], &weights[0]);

        for(int i = 0; i < 4; i++){
            vec2ui ij = indices[i];
            result += weights[i] * CurlAtPoint(ij.x, ij.y);
        }

        return result;
    }

    /* Computes divergent at a data point (fixed node) */
    __bidevice__ Float DivergentAtPoint(size_t i, size_t j){
        Float left  = Accessor2D(data, i > 0 ? i - 1 : i, j, dataSize.x).x;
        Float right = Accessor2D(data, i + 1 < dataSize.x ? i + 1 : i, j, dataSize.x).x;
        Float down  = Accessor2D(data, i, j > 0 ? j - 1 : j, dataSize.x).y;
        Float up    = Accessor2D(data, i, j + 1 < dataSize.y ? j + 1 : j, dataSize.x).y;

        return 0.5 * (right - left) * grid->invSpacing.x +
            0.5 * (up - down) * grid->invSpacing.y;
    }

    /* Computes divergent at a specific point inside a cell */
    __bidevice__ Float Divergent(const vec2f &p){
        vec2ui indices[4];
        Float weights[4];
        Float result = 0;
        LinearGridSampler2Weights(p, grid->spacing, dataOrigin,
                                  dataSize, &indices[0], &weights[0]);

        for(int i = 0; i < 4; i++){
            vec2ui ij = indices[i];
            result += weights[i] * DivergentAtPoint(ij.x, ij.y);
        }

        return result;
    }
};

class ExplicitFaceCenteredVectorGrid2{
    public:
    ExplicitGrid2 *grid;
    Float *dataU, *dataV;
    vec2f dataOriginU, dataOriginV;
    vec2ui dataSizeU, dataSizeV;
    unsigned int usize, vsize;

    __bidevice__ ExplicitFaceCenteredVectorGrid2(){
        grid = nullptr;
        dataU = dataV = nullptr;
        dataSizeU = vec2ui(0, 0);
        dataSizeV = vec2ui(0, 0);
        usize = 0;
        vsize = 0;
        dataOriginU = vec2f(0.0, 0.5);
        dataOriginV = vec2f(0.5, 0.0);
    }

    __bidevice__ vec2f GetPositionV(size_t i, size_t j){
        AssertA(grid != nullptr, "Invalid call to GetPositionV");
        return dataOriginV + grid->spacing * vec2f((Float)i, (Float)j);
    }

    __bidevice__ vec2f GetPositionU(size_t i, size_t j){
        AssertA(grid != nullptr, "Invalid call to GetPositionU");
        return dataOriginU + grid->spacing * vec2f((Float)i, (Float)j);
    }

    __bidevice__ const Float &U(size_t i, size_t j) const{
        size_t idx = i + dataSizeU.x * j;
        return dataU[idx];
    }

    __bidevice__ Float &U(size_t i, size_t j){
        size_t idx = i + dataSizeU.x * j;
        return dataU[idx];
    }

    __bidevice__ const Float &V(size_t i, size_t j) const{
        size_t idx = i + dataSizeV.x * j;
        return dataV[idx];
    }

    __bidevice__ Float &V(size_t i, size_t j){
        size_t idx = i + dataSizeV.x * j;
        return dataV[idx];
    }

    __host__ void Set(const vec2ui &res, const vec2f &h, const vec2f &o,
                      const vec2f &initialValue = vec2f(0,0))
    {
        Float *mem = nullptr;
        unsigned int totalSize = 0;
        if(grid == nullptr){
            grid = cudaAllocateVx(ExplicitGrid2, 1);
        }

        grid->Set(res, h, o);
        dataSizeU = res + vec2ui(1, 0);
        dataSizeV = res + vec2ui(0, 1);
        dataOriginU = o + 0.5 * vec2f(0.0, h.y);
        dataOriginV = o + 0.5 * vec2f(h.x, 0.0);
        vsize = dataSizeV.x * dataSizeV.y;
        usize = dataSizeU.x * dataSizeU.y;
        totalSize = usize + vsize;
        mem = cudaAllocateVx(Float, totalSize);

        dataU = &mem[0];
        dataV = &mem[usize];

        totalSize = Max(usize, vsize);
        for(int i = 0; i < totalSize; i++){
            if(i < usize) dataU[i] = initialValue.x;
            if(i < vsize) dataV[i] = initialValue.y;
        }
    }

    __bidevice__ vec2f ValueAtCellCenter(size_t i, size_t j){
        return 0.5 * vec2f((Accessor2D(dataU, i, j, dataSizeU.x) +
                            Accessor2D(dataU, i + 1, j, dataSizeU.x)),
                           (Accessor2D(dataV, i, j, dataSizeV.x) +
                            Accessor2D(dataV, i, j + 1, dataSizeV.x)));
    }

    __bidevice__ Float DivergentAtCellCenter(size_t i, size_t j){
        Float leftU  = Accessor2D(dataU, i, j, dataSizeU.x);
        Float rightU = Accessor2D(dataU, i + 1, j, dataSizeU.x);
        Float downV  = Accessor2D(dataV, i, j, dataSizeV.x);
        Float upV    = Accessor2D(dataV, i, j + 1, dataSizeV.x);
        return (rightU - leftU) * grid->invSpacing.x + (upV - downV) * grid->invSpacing.y;
    }

    __bidevice__ Float CurlAtCellCenter(size_t i, size_t j){
        vec2ui res = grid->resolution;
        vec2f left  = ValueAtCellCenter(i > 0 ? i - 1 : i, j);
        vec2f right = ValueAtCellCenter(i + 1 < res.x ? i + 1 : i, j);
        vec2f up    = ValueAtCellCenter(i, j + 1 < res.y ? j + 1 : j);
        vec2f down  = ValueAtCellCenter(i, j > 0 ? j - 1 : j);

        Float Fx_ym = down.x;
        Float Fx_yp = up.x;
        Float Fy_xm = left.y;
        Float Fy_xp = right.y;

        return 0.5 * (Fy_xp - Fy_xm) * grid->invSpacing.x - 0.5 * (Fx_yp - Fx_ym) * grid->invSpacing.y;
    }

    __bidevice__ Float Curl(const vec2f &p){
        int i = 0, j = 0;
        Float fx = 0, fy = 0;
        vec2ui indices[4];
        Float weights[4];
        Float curl = 0;
        vec2f cellOrigin = grid->origin + 0.5 * grid->spacing;
        vec2f normalized = (p - cellOrigin) * grid->invSpacing;

        GetBarycentric(normalized.x, 0, grid->resolution.x - 1, &i, &fx);
        GetBarycentric(normalized.y, 0, grid->resolution.y - 1, &j, &fy);

        indices[0] = vec2ui(i, j);
        indices[1] = vec2ui(i + 1, j);
        indices[2] = vec2ui(i, j + 1);
        indices[3] = vec2ui(i + 1, j + 1);

        weights[0] = (1.0 - fx) * (1.0 - fy);
        weights[1] = fx * (1.0 - fy);
        weights[2] = (1.0 - fx) * fy;
        weights[3] = fx * fy;

        for(int i = 0; i < 4; i++){
            curl += weights[i] * CurlAtCellCenter(indices[i].x, indices[i].y);
        }

        return curl;
    }

    __bidevice__ Float Divergent(const vec2f &p){
        int i = 0, j = 0;
        Float fx = 0, fy = 0;
        vec2ui indices[4];
        Float weights[4];
        Float div = 0;
        vec2f cellOrigin = grid->origin + 0.5 * grid->spacing;
        vec2f normalized = (p - cellOrigin) * grid->invSpacing;

        GetBarycentric(normalized.x, 0, grid->resolution.x - 1, &i, &fx);
        GetBarycentric(normalized.y, 0, grid->resolution.y - 1, &j, &fy);

        indices[0] = vec2ui(i, j);
        indices[1] = vec2ui(i + 1, j);
        indices[2] = vec2ui(i, j + 1);
        indices[3] = vec2ui(i + 1, j + 1);

        weights[0] = (1.0 - fx) * (1.0 - fy);
        weights[1] = fx * (1.0 - fy);
        weights[2] = (1.0 - fx) * fy;
        weights[3] = fx * fy;

        for(int i = 0; i < 4; i++){
            div += weights[i] * DivergentAtCellCenter(indices[i].x, indices[i].y);
        }

        return div;
    }
};

/* Simple forward Euler method for velocity integration in time */
template<typename Interpolator>
struct EulerIntegrator{
    Float dt;
    vec2f p;
    Float h;
    Interpolator fn;

    __bidevice__ vec2f Backtrack(){
        vec2f vel = fn(p) / h;
        return p - vel * dt;
    }

};

/* Third order Runge-Kutta method for velocity integration in time */
template<typename Interpolator>
struct RK3Integrator{
    Float dt;
    vec2f p;
    Float h;
    Interpolator fn;

    __bidevice__ vec2f Backtrack(){
        vec2f k1 = fn(p) / h;
        vec2f k2 = fn(p - 0.50 * dt * k1) / h;
        vec2f k3 = fn(p - 0.75 * dt * k2) / h;
        return p - (dt / 9.) * (2 * k1 + 3 * k2 + 4 * k3);
    }
};

template<typename InterpolatorType>
class ExplicitMarkerAndCellGrid2{
    public:
    /* Memory buffers for fluid quantity */
    Float *source;
    Float *dest;

    vec2ui resolution; // grid resolution
    vec2f dataOffset; // offset for data points from top-left
    /*
     * X and Y offset from top left grid cell.
     * This is (0.5,0.5) for centered quantities such as density,
     * and (0.0, 0.5) or (0.5, 0.0) for jittered quantities like the velocity.
     */

    /* Grid cell size */
    Float hx;

    ExplicitMarkerAndCellGrid2() = default;
    ~ExplicitMarkerAndCellGrid2() = default;

    ExplicitMarkerAndCellGrid2(int w, int h, Float ox, Float oy, Float spacing){
        Set(vec2ui(w, h), vec2f(ox, oy), spacing);
    }

    __host__ void Set(vec2ui res, vec2f dataOff, Float spacing){
        Float w = Max(1, res.x);
        Float h = Max(1, res.y);
        resolution = res;
        dataOffset = dataOff;
        hx = spacing;
        source = cudaAllocateVx(Float, w * h);
        dest = cudaAllocateVx(Float, w * h);
        memset(source, 0, w * h * sizeof(Float));
    }

    __host__ void flip(){
        Swap(&source, &dest);
    }

    __bidevice__ const Float *src() const{
        return source;
    }

    /* Read-only and read-write access to grid cells */
    __bidevice__ Float at(int x, int y) const{
        return source[x + y * resolution.x];
    }

    __bidevice__ Float &at(int x, int y){
        return source[x + y * resolution.x];
    }

    /*
     * Evaluates a point (x, y) inside the domain in grid-coordinates
     * to get a estimate of the value at this location based on the
     * nodes available in the grid using the templated interpolator.
     * The point must lie in the grid-domain, i.e.: [0,0] x [Width, Height].
     */
    __bidevice__ Float Eval(Float x, Float y) const{
        InterpolatorType interpolator;
        x = Clamp(x - dataOffset.x, 0.f, resolution.x - OnePlusEpsilon);
        y = Clamp(y - dataOffset.y, 0.f, resolution.y - OnePlusEpsilon);
        auto fn = [&](int _x, int _y) -> Float{
            return at(_x, _y);
        };
        return interpolator.Interpolate(x, y, resolution, fn);
    }

    __bidevice__ void AdvectPoint(Float timestep, int i,
                                  ExplicitMarkerAndCellGrid2<InterpolatorType> *u,
                                  ExplicitMarkerAndCellGrid2<InterpolatorType> *v)
    {
        int iy = i / resolution.x;
        int ix = i % resolution.x;
        Float x = ix + dataOffset.x;
        Float y = iy + dataOffset.y;
        auto vel_at = [&](vec2f p) -> vec2f{
            return vec2f(u->Eval(p.x, p.y), v->Eval(p.x, p.y));
        };

        using InterpolatorFn = decltype(vel_at);
        RK3Integrator<InterpolatorFn> integrator{timestep, vec2f(x, y), hx, vel_at};
        //EulerIntegrator<Interpolator> integrator{timestep, vec2f(x, y), hx, vel_at};

        /* First component: Integrate in time */
        vec2f s = integrator.Backtrack();

        /* Second component: Interpolate from grid */
        dest[i] = Eval(s.x, s.y);
    }

    /* Sets fluid quantity inside the given rect to value `v' */
    void addInflow(Float x0, Float y0, Float x1, Float y1, Float v){
        InterpolatorType interpolator;
        int ix0 = (int)(x0/hx - dataOffset.x);
        int iy0 = (int)(y0/hx - dataOffset.y);
        int ix1 = (int)(x1/hx - dataOffset.x);
        int iy1 = (int)(y1/hx - dataOffset.y);
        if(IsHighpZero(v)) return;

        for (int y = max(iy0, 0); y < min(iy1, resolution.y); y++){
            for (int x = max(ix0, 0); x < min(ix1, resolution.x); x++){
                Float px = (2.0*(x + 0.5) * hx - (x0 + x1)) / (x1 - x0);
                Float py = (2.0*(y + 0.5) * hx - (y0 + y1)) / (y1 - y0);
                Float pl = vec2f(px, py).Length();
                Float vi = interpolator.Pulse(pl) * v;
                Float oldValue = source[x + y * resolution.x];
                if(fabs(oldValue) < fabs(vi)){
                    source[x + y * resolution.x] = vi;
                }
            }
        }
    }

    static ExplicitMarkerAndCellGrid2<InterpolatorType> *Create(Float _w, Float _h,
                                                   Float _ox, Float _oy, Float _hx)
    {
        ExplicitMarkerAndCellGrid2<InterpolatorType> *ptr =
                           cudaAllocateVx(ExplicitMarkerAndCellGrid2<InterpolatorType>, 1);
        ptr->Set(vec2ui(_w, _h), vec2f(_ox, _oy), _hx);
        return ptr;
    }
};

/* Advect grid in velocity field u, v with given timestep */
template<typename I>
inline __host__ void Advect(Float timestep, ExplicitMarkerAndCellGrid2<I> *q,
                            ExplicitMarkerAndCellGrid2<I> *u, ExplicitMarkerAndCellGrid2<I> *v)
{
    size_t items = q->resolution.x * q->resolution.y;
    AutoParallelFor("Advect", items, AutoLambda(size_t index){
        q->AdvectPoint(timestep, index, u, v);
    });
}

typedef ExplicitMarkerAndCellGrid2<MonotonicCatmull> CatmullMarkerAndCellGrid2;
typedef ExplicitMarkerAndCellGrid2<LinearInterpolator> LinearMarkerAndCellGrid2;

/*
 * Builds the right hand side of the pressure equation (negative divergece).
 * Recall that divergence of velocity needs to be 0, i.e.: ∂u/∂x + ∂v/∂y + ∂w/∂z = 0
 * Take the first term (∂u/∂x) for example, we apply central differences and get:
 * ∂u/∂x ~ (Ui+1 - Ui-1) / Δx. This routine computes the value of the current divergence
 * approximating the partials with the central difference method.
 */
// 2D
template<typename ExplicitGrid>
inline __host__ void PressureSolverBuildRHS(ExplicitGrid *u, ExplicitGrid *v, Float dx,
                                            Float *rhs, vec2ui resolution)
{
    Float scale = 1.0/dx;
    size_t items = resolution.x * resolution.y;
    AutoParallelFor("Divergence", items, AutoLambda(size_t index){
        int x0 = index % resolution.x;
        int y0 = index / resolution.x;
        int x1 = Min(x0 + 1, u->resolution.x - 1);
        int y1 = Min(y0 + 1, v->resolution.y - 1);
        rhs[index] = -scale * (u->at(x1, y0) - u->at(x0, y0) +
                               v->at(x0, y1) - v->at(x0, y0));
    });
}

/*
 * The pressure matrix is given by trying to find the solution to the Poisson
 * problem, i.e.: -Δt/ρ ∂ . ∂p = -∂u. Which gives us:
 *     (Δt/ρ)(4pi,j - pi+1,j - pi,j+1 - pi-1,j - pi,j-1)/Δx² =
 *            -(ui+.5,j - ui-.5,j + vi,j+.5 - vi,j-.5)/Δx
 * Recall we want to solve this equation for the values p such that the divergence
 * is 0. The routine PressureSolverBuildRHS already builds the right side of this
 * equation. Since each cell has an equation like the above we have a very big set
 * of equations that need to solved, in matrix form we can write Ap = b. With b
 * being rhs, p the vector (p0, p1, p2, p3, ...) the pressure at each cell and A
 * the coefficients. Note that each row of A represents ONE cell and as such because
 * each cell only depends on pi,j, pi+1,j, pi-1,j, pi,j+1 and pi,j-1 all other values
 * of pi,j become 0. So the matrix A is sparse since each row has many [many] 0 terms.
 * Each cell depends on 4 neighbors (8 in 3D), note that each of these relations is
 * symmetric with regards to +-i and +-j (+-k in 3D also) so we store only the positive
 * side and the diagonal.
 */

class PressureSolver{
    public:
    Float *r; // right handle side of pressure solve
    vec2ui resolution;
    Float dx;

    PressureSolver() = default;
    ~PressureSolver() = default;

    virtual __host__ void BuildSolver(size_t n, Float _dx, vec2ui res) = 0;
    virtual __host__ void Update(int limit, Float *p, Float density,
                                 Float timestep, Float maxErr=1e-5) = 0;


    __host__ void Set(vec2ui res, Float _dx, size_t n){
        resolution = res;
        r = cudaAllocateVx(Float, n);
        dx = _dx;
    }

    template<typename ExplicitGrid>
    __host__ void SolvePressure(ExplicitGrid *u, ExplicitGrid *v,
                                int limit, Float *p, Float density,
                                Float timestep, Float maxErr=1e-5)
    {
        PressureSolverBuildRHS<ExplicitGrid>(u, v, dx, r, resolution);
        Update(limit, p, density, timestep, maxErr);
    }

    __host__ Float *RHS(){ return r; }
};

/*
 * You would think PCG is much better than simple-old Jacobi. However
 * Jacobi with 600 iterations runs faster than PCG on GPU and achieves
 * pretty much the same result.
 */
class PressureSolverJacobi : public PressureSolver{
    public:
    Float *z; // auxiliary
    Float *e; // error vector
    size_t count;

    PressureSolverJacobi() = default;
    ~PressureSolverJacobi() = default;

    virtual __host__ void BuildSolver(size_t n, Float _dx, vec2ui res) override{
        size_t size = res.x * res.y;
        if(n != size){
            printf("Error: Resolution does not match ( %u != %u )!\n",
                    (unsigned int)n, (unsigned int)size);
            exit(0);
        }

        Set(res, _dx, n);

        z = cudaAllocateVx(Float, n);
        e = cudaAllocateVx(Float, n);

        count = n;
    }

    __host__ void Iterate(Float *ping, Float *pong, Float *rhs, Float *err, Float scale){
        int w = resolution.x;
        int h = resolution.y;
        size_t items = resolution.x * resolution.y;
        AutoParallelFor("Jacobi_Iteration", items, AutoLambda(size_t index){
            int x = index % w;
            int y = index / w;

            Float diag = 0.f;
            Float off = 0.f;
            if(x > 0){
                diag += scale;
                off -= scale * ping[index - 1];
            }

            if(y > 0){
                diag += scale;
                off -= scale * ping[index - w];
            }

            if(x < w - 1){
                diag += scale;
                off -= scale * ping[index + 1];
            }

            if(y < h - 1){
                diag += scale;
                off -= scale * ping[index + w];
            }

            Float pk1 = (rhs[index] - off) / diag;
            err[index] = Absf(ping[index] - pk1);

            pong[index] = pk1;
        });
    }

    virtual __host__ void Update(int limit, Float *p, Float density,
                                 Float timestep, Float maxErr=1e-5) override
    {
        size_t items = resolution.x * resolution.y;
        Float scale = timestep/(density * dx * dx);
        Float maxError = 0;

        memset(p, 0,  items * sizeof(Float));

        Float *buffers[2] = {p, z};
        int activeBuffer = 0;
        Float *rhs = r;

        Float *ping = buffers[activeBuffer];
        Float *pong = buffers[1 - activeBuffer];

        for(int iter = 0; iter < limit; iter++){
            ping = buffers[activeBuffer];
            pong = buffers[1 - activeBuffer];

            Iterate(ping, pong, rhs, e, scale);

            maxError = 0;
            for(size_t i = 0; i < items; i++){
                maxError = Max(maxError, e[i]);
            }

            if(maxError < maxErr){
                printf("Exiting solver after %d iterations, "
                       " maximum change is %f\n", iter, maxError);
                if(pong != p){
                    memcpy(p, pong, items * sizeof(Float));
                }
                return;
            }

            activeBuffer = 1 - activeBuffer;
        }

        printf("Exceeded budget of %d iterations, "
               "maximum change was %f\n", limit, maxError);
        if(pong != p){
            memcpy(p, pong, items * sizeof(Float));
        }
    }
};

class PressureSolverPCG : public PressureSolver{
    public:
    Float *z; // auxiliary
    Float *s; // search vector
    float *pre; // preconditioner

    // based on Bridson book
    Float *aDiag; // (i,j) value
    Float *aPlusX; // +i value
    Float *aPlusY; // +j value
    size_t count;

    PressureSolverPCG() = default;
    ~PressureSolverPCG() = default;

    virtual __host__ void BuildSolver(size_t n, Float _dx, vec2ui res) override{
        size_t size = res.x * res.y;
        if(n != size){
            printf("Error: Resolution does not match ( %u != %u )!\n",
                    (unsigned int)n, (unsigned int)size);
            exit(0);
        }

        Set(res, _dx, n);

        z = cudaAllocateVx(Float, n);
        s = cudaAllocateVx(Float, n);
        aDiag = cudaAllocateVx(Float, n);
        aPlusX = cudaAllocateVx(Float, n);
        aPlusY = cudaAllocateVx(Float, n);
        pre = cudaAllocateVx(Float, n);
        count = n;
    }

    /*
     * Incomplete Cholesky from Bridson. This one cannot be parallel.
     */
    __host__ void BuildPreconditioner(){
        const Float tau = 0.97;
        const Float sigma = 0.25;
        int w = resolution.x;
        int h = resolution.y;

        for(int y = 0, idx = 0; y < h; y++){
            for(int x = 0; x < w; x++, idx++){
                Float e = aDiag[idx];

                if(x > 0){
                    Float px = aPlusX[idx - 1] * pre[idx - 1];
                    Float py = aPlusY[idx - 1] * pre[idx - 1];
                    e = e - (px * px + tau * px * py);
                }

                if(y > 0){
                    Float px = aPlusX[idx - w] * pre[idx - w];
                    Float py = aPlusY[idx - w] * pre[idx - w];
                    e = e - (py * py + tau * px * py);
                }

                if(e < sigma * aDiag[idx]) e = aDiag[idx];

                pre[idx] = 1.0 / sqrt(e);
            }
        }
    }

    /*
     * Apply preconditioner to vector a and save it in dst.
     */
    __host__ void ApplyPreconditioner(Float *dst, Float *a){
        int w = resolution.x;
        int h = resolution.y;
        for(int y = 0, idx = 0; y < h; y++){
            for(int x = 0; x < w; x++, idx++){
                Float t = a[idx];

                if(x > 0)
                    t -= aPlusX[idx - 1] * pre[idx - 1] * dst[idx - 1];
                if(y > 0)
                    t -= aPlusY[idx - w] * pre[idx - w] * dst[idx - w];

                dst[idx] = t * pre[idx];
            }
        }

        for(int y = h - 1, idx = w * h - 1; y >= 0; y--){
            for(int x = w - 1; x >= 0; x--, idx--){
                idx = x + y * w;

                Float t = dst[idx];

                if (x < w - 1)
                    t -= aPlusX[idx] * pre[idx] * dst[idx + 1];
                if (y < h - 1)
                    t -= aPlusY[idx] * pre[idx] * dst[idx + w];

                dst[idx] = t * pre[idx];
            }
        }
    }

    __host__ Float DotProduct(Float *a, Float *b){
        Float result = 0;
        size_t items = resolution.x * resolution.y;
        for(size_t i = 0; i < items; i++){
            result += a[i] * b[i];
        }
        return result;
    }

    __host__ void MatrixVectorProduct(Float *dst, Float *b){
        Float *diag = aDiag;
        Float *PX = aPlusX;
        Float *PY = aPlusY;
        vec2ui res = resolution;
        size_t items = resolution.x * resolution.y;

        AutoParallelFor("Matrix_Vector_Product", items, AutoLambda(size_t index){
            int x = index % res.x;
            int y = index / res.x;

            Float t = diag[index] * b[index];
            if(x > 0) t += PX[index - 1] * b[index - 1];
            if(y > 0) t += PY[index - res.x] * b[index - res.x];
            if(x < res.x - 1) t += PX[index] * b[index + 1];
            if(y < res.y - 1) t += PY[index] * b[index + res.x];
            dst[index] = t;
        });
    }

    __host__ Float MaxNorm(Float *a){
        size_t items = resolution.x * resolution.y;
        Float maxA = 0.f;
        for(size_t i = 0; i < items; i++){
            maxA = Max(maxA, fabs(a[i]));
        }
        return maxA;
    }

    /*
     * In order to build the + side of the equation for each cell note that we have
     * for a single cell the following contributions:
     * 1 - If the cell has a cell to its left than pi-1,j contributes to its equation;
     * 2 - If the cell has a cell to its bottom than pi,j-1 contributes to its equation;
     * 3 - If the cell has a cell to its right than pi,j contributes to this cell (right);
     * 4 - If the cell has a cell to its top than pi,j contributes to this cell (top).
     * In a diagram:
     *
     *                +1 (out)
     *         -> +1 | (i,j) | -> +1
     *                +1 (in)
     * same goes for 3D.
     */
    __host__ void BuildMatrix(Float timestep, Float density){
        size_t items = resolution.x * resolution.y;
        Float scale = timestep/(density * dx * dx);
        Float *diag = aDiag;
        Float *PX = aPlusX;
        Float *PY = aPlusY;
        vec2ui res = resolution;
        memset(diag, 0, items * sizeof(Float));

        AutoParallelFor("Pressure_Matrix_Build", items, AutoLambda(size_t index){
            int x = index % res.x;
            int y = index / res.x;
            if(x > 0){
                diag[index] += scale;
            }

            if(x < res.x - 1){
                diag[index] += scale;
                PX[index] = -scale;
            }else{
                PX[index] = 0.f;
            }

            if(y > 0){
                diag[index] += scale;
            }

            if(y < res.y - 1){
                diag[index]   += scale;
                PY[index]  = -scale;
            }else{
                PY[index]  = 0.f;
            }
        });
    }

    /*
     * Computes dst = a + b * s
     */
    __host__ void ScaledAdd(Float *dst, Float *a, Float *b, Float s){
        size_t items = resolution.x * resolution.y;
        AutoParallelFor("ScaleAdd", items, AutoLambda(size_t index){
            dst[index] = a[index] + b[index] * s;
        });
    }

    __host__ void ScaledAddUpdate(Float *p, Float alpha){
        size_t items = resolution.x * resolution.y;
        Float nAlpha = -alpha;
        Float *_s = s;
        Float *_z = z;
        Float *_r = r;
        AutoParallelFor("ScaledAddUpdate", items, AutoLambda(size_t index){
            p[index] = p[index] + _s[index] * alpha;
            _r[index] = _r[index] + _z[index] * nAlpha;
        });
    }

    virtual __host__ void Update(int limit, Float *p, Float density,
                                 Float timestep, Float maxErr=1e-5) override
    {
        size_t items = resolution.x * resolution.y;
        memset(p, 0,  items * sizeof(Float));

        BuildMatrix(timestep, density);
        BuildPreconditioner();

        ApplyPreconditioner(z, r);
        memcpy(s, z, items * sizeof(Float));

        Float maxError = MaxNorm(r);
        if(maxError < maxErr) return;

        Float sigma = DotProduct(z, r);

        for(int iter = 0; iter < limit; iter++){
            MatrixVectorProduct(z, s);
            Float alpha = sigma / DotProduct(z, s);
            ScaledAddUpdate(p, alpha);

            maxError = MaxNorm(r);
            if(maxError < maxErr){
                printf("Exiting solver after %d iterations, maximum error is %f\n", iter, maxError);
                return;
            }

            ApplyPreconditioner(z, r);
            Float sigmaNew = DotProduct(z, r);
            ScaledAdd(s, z, s, sigmaNew / sigma);
            sigma = sigmaNew;
        }

        printf("Exceeded budget of %d iterations, maximum error was %f\n", limit, maxError);
    }
};
