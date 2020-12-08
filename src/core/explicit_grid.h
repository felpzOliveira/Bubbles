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