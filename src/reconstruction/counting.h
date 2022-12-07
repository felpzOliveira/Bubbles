/* date = December 2nd 2022 20:52 */
#pragma once
#include <grid.h>
#include <util.h>
#include <bound_util.h>

/**************************************************************/
//      P A R T I C L E   C O U N T I N G   M E T H O D       //
//                      3D Particle Filtering                 //
/**************************************************************/

/*
* Implementation of the particle counting method for fast SDF generation
* of particle sets. I like this method, it is very fast and can be used with
* [very] high resolution since the grid does not need to query particles. Makes
* it consume less memory and therefore explode on resolution without too much overhead.
* This method is an extension of the screen-space fluid rendering approach
* and is the method presented in:
*    Counting Particles: a simple and fast surface reconstruction
*                        method for particle-based fluids
*/

template<typename T>
inline __bidevice__ vec3f AsFloat(vec3<T> vf){
    return vec3f(Float(vf.x), Float(vf.y), Float(vf.z));
}

template<typename T>
inline __bidevice__ vec2f AsFloat(vec2<T> vf){
    return vec2f(Float(vf.x), Float(vf.y));
}

template<typename U>
inline __bidevice__ U ClampedIndex(U u, U sizes, int dims, int &clamped){
    U copy = u;
    u[0] = Clamp(u[0], 0, sizes[0]-1);
    u[1] = Clamp(u[1], 0, sizes[1]-1);
    clamped = copy[0] != u[0] || copy[1] != u[1];
    if(dims > 2){
        u[2] = Clamp(u[2], 0, sizes[2]-1);
        clamped |= copy[2] != u[2];
    }

    return u;
}

/*
* NOTE: Since surface reconstruction by particle counting does not require access to
* particle neighborhood using Grid2/3 is overkill and in fact because marching cubes
* requires a higher resolution attempting to create a high resolution Grid2/3 can easily
* run out of memory because of the neighbors lists it builds for particle querying.
* I'll create a lightweight version of Grid2/3 that can be used with higher resolution
* but does not provide the hability to query particles.
*/
template<typename T, typename U, typename Q, typename DataType>
class LightweightGrid{
    public:
    U usizes;
    unsigned int total;
    T cellsLen;
    T minPoint;
    int dimensions;
    Q bounds;
    DataType *stored;

    __bidevice__ LightweightGrid(){}
    __host__ void SetDimension(vec2f){ dimensions = 2; }
    __host__ void SetDimension(vec3f){ dimensions = 3; }
    __bidevice__ unsigned int GetCellCount(){ return total; }
    __bidevice__ Q GetBounds(){ return bounds; }
    __bidevice__ T GetCellSize(){ return cellsLen; }
    __bidevice__ U GetIndexCount(){ return usizes; }
    __bidevice__ int GetDimensions(){ return dimensions; }

    /* Get offset for dimension value 'p' on axis 'axis' in case it lies on boundary */
    __bidevice__ Float ExtremeEpsilon(Float p, int axis){
        Float eps = 0;
        Float p0 = bounds.LengthAt(0, axis);
        Float p1 = bounds.LengthAt(1, axis);
        if(IsZero(p - p0)){
            eps = Epsilon;
        }else if(IsZero(p - p1)){
            eps = -Epsilon;
        }

        return eps;
    }

    /* Get logical index of cell */
    __bidevice__ U GetCellIndex(unsigned int i){
        return DimensionalIndex(i, usizes, dimensions);
    }

    /* Hash position 'p' into a cell index */
    __bidevice__ U GetHashedPosition(const T &p){
        U u;
        if(!Inside(p, bounds)){
            printf(" [ERROR] : Requested for hash on point outside domain ");
            p.PrintSelf();
            printf(" , Bounds: ");
            bounds.PrintSelf();
            printf("\n");
        }

        AssertA(Inside(p, bounds), "Out of bounds point");
        for(int i = 0; i < dimensions; i++){
            Float pi = p[i];
            pi += ExtremeEpsilon(pi, i);

            Float dmin = minPoint[i];
            Float dlen = cellsLen[i];
            Float dp = (pi - dmin) / dlen;

            int linearId = (int)(Floor(dp));
            AssertA(linearId >= 0 && linearId < usizes[i], "Out of bounds position");
            u[i] = linearId;
        }

        return u;
    }

    /* Get the ordered cell index of a cell */
    __bidevice__ unsigned int GetLinearCellIndex(const U &u){
        unsigned int h = LinearIndex(u, usizes, dimensions);
        AssertA(h < total, "Invalid cell id computation");
        return h;
    }

    /* Hash position 'p' and get the linear cell index */
    __bidevice__ unsigned int GetLinearHashedPosition(const T &p){
        U u = GetHashedPosition(p);
        return GetLinearCellIndex(u);
    }

    /* Initialize internals of the lightweight grid */
    __host__ void Build(const U &resolution, const T &dp0, const T &dp1,
                        bool with_data=true)
    {
        SetDimension(T(0));
        T lower(Infinity), high(-Infinity);
        T p0 = Max(dp0, dp1);
        T p1 = Min(dp0, dp1);
        T maxPoint;
        total = 1;
        for(int k = 0; k < dimensions; k++){
            Float dl = p0[k];
            Float du = p1[k];
            if(dl < lower[k]) lower[k] = dl;
            if(dl > high[k]) high[k] = dl;
            if(du > high[k]) high[k] = du;
            if(du < lower[k]) lower[k] = du;
        }

        for(int k = 0; k < dimensions; k++){
            Float s = high[k] - lower[k];
            Float len = s / (Float)resolution[k];
            cellsLen[k] = len;
            usizes[k] = (int)std::ceil(s / len);
            maxPoint[k] = lower[k] + (Float)usizes[k] * cellsLen[k];
            total *= usizes[k];
        }

        minPoint = lower;
        bounds = Q(minPoint, maxPoint);
        if(with_data){
            stored = cudaAllocateVx(DataType, total);

            for(int i = 0; i < total; i++){
                stored[i] = DataType(0);
            }
        }
    }
};

typedef LightweightGrid<vec2f, vec2ui, Bounds2f, size_t> LightweightGrid2;
typedef LightweightGrid<vec3f, vec3ui, Bounds3f, size_t> LightweightGrid3;

/*
* Simple routines for explicit build operations.
*/

inline __host__ LightweightGrid2 *
MakeLightweightGrid(const vec2ui &size, const vec2f &pMin, const vec2f &pMax)
{
    LightweightGrid2 *grid = cudaAllocateVx(LightweightGrid2, 1);
    grid->Build(size, pMin, pMax);
    return grid;
}

inline __host__ LightweightGrid3 *
MakeLightweightGrid(const vec3ui &size, const vec3f &pMin, const vec3f &pMax)
{
    LightweightGrid3 *grid = cudaAllocateVx(LightweightGrid3, 1);
    grid->Build(size, pMin, pMax);
    return grid;
}

/*
* Implement box filter and gaussian filter in 2D and 3D.
*/
struct FilterSolver2D{
    Float sigmaSquared;
    Float invTwoPiSigmaSquared;

    __bidevice__ FilterSolver2D(Float _sigma){
        sigmaSquared = _sigma * _sigma;
        invTwoPiSigmaSquared = 1.f / (TwoPi * sigmaSquared);
    }

    template<typename Fn>
    __bidevice__ void Convolution(int radius, Fn fn){
        AssertA(radius % 2 != 0, "Radius is even");
        int s = (radius-1) / 2;
        for(int x = -s; x <= s; x++){
            for(int y = -s; y <= s; y++){
                fn(x, y);
            }
        }
    }

    template<typename SamplerFn>
    __bidevice__ Float Box(vec2ui u, SamplerFn sampler){
        int i = u.x, j = u.y;
        Float val = Float(0);
        Float iterations = 0;
        Convolution(3, [&](int x, int y){
            val = val + sampler(vec2ui(i + x, j + y));
            iterations += 1.f;
        });

        return val * (1.f / iterations);
    }

    template<typename SamplerFn>
    __bidevice__ Float Gaussian(vec2ui u, SamplerFn sampler){
        int i = u.x, j = u.y;
        Float val = Float(0);
        Convolution(5, [&](int x, int y){
            Float dx = Float(x);
            Float dy = Float(y);
            Float dx2 = dx * dx;
            Float dy2 = dy * dy;
            Float kernel = invTwoPiSigmaSquared *
                        (std::exp(-(dx2 + dy2) / (2.f * sigmaSquared)));
            val = val + sampler(vec2ui(i + x, j + y)) * kernel;
        });

        return val;
    }
};

struct FilterSolver3D{
    Float sigmaSquared;
    Float invTwoPiSigmaSquared;

    __bidevice__ FilterSolver3D(Float _sigma){
        sigmaSquared = _sigma * _sigma;
        invTwoPiSigmaSquared = 1.f / (TwoPi * sigmaSquared);
    }

    template<typename Fn>
    __bidevice__ void Convolution(int radius, Fn fn){
        AssertA(radius % 2 != 0, "Radius is even");
        int s = (radius-1) / 2;
        for(int x = -s; x <= s; x++){
            for(int y = -s; y <= s; y++){
                for(int z = -s; z <= s; z++){
                    fn(x, y, z);
                }
            }
        }
    }

    template<typename SamplerFn>
    __bidevice__ Float Box(vec3ui u, SamplerFn sampler){
        int i = u.x, j = u.y, k = u.z;
        Float val = Float(0);
        Float iterations = 0;
        Convolution(3, [&](int x, int y, int z){
            val = val + sampler(vec3ui(i + x, j + y, k + z));
            iterations += 1.f;
        });

        return val * (1.f / iterations);
    }

    template<typename SamplerFn>
    __bidevice__ Float Gaussian(vec3ui u, SamplerFn sampler){
        int i = u.x, j = u.y, k = u.z;
        Float val = Float(0);
        Convolution(5, [&](int x, int y, int z){
            Float dx = Float(x);
            Float dy = Float(y);
            Float dz = Float(z);
            Float dx2 = dx * dx;
            Float dy2 = dy * dy;
            Float dz2 = dz * dz;
            Float kernel = invTwoPiSigmaSquared *
                        (std::exp(-(dx2 + dy2 + dz2) / (2.f * sigmaSquared)));
            val = val + sampler(vec3ui(i + x, j + y, k + z)) * kernel;
        });

        return val;
    }
};

// Vector computation  Dimension computation  Domain computation      FilterSolver
// T = vec2f/vec3f,    U = vec2ui/vec3ui,     Q = Bounds2f/Bounds3f   Box/Gaussian solver
template<typename T, typename U, typename Q, typename FilterSolver>
class CountingGrid{
    public:

    using GridType = LightweightGrid<T, U, Q, size_t>;
    using FieldGridType = FieldGrid<T, U, Q, Float>;

    unsigned int total; // total of cells
    size_t *countField; // the field of particle counts
    size_t particleCount;

    // the underlying grid
    GridType *grid;

    // terms for the method
    Float mu_0;
    Float *fieldDIF; // the value of the sdf for the method
    Float *smoothFieldDIF;
    int dimensions;

    __bidevice__ CountingGrid(){}
    __host__ void SetDimension(vec2f){ dimensions = 2; }
    __host__ void SetDimension(vec3f){ dimensions = 3; }

    __host__ void Build(ParticleSetBuilder<T> *pBuilder, Float spacing){
        Q bounds;
        U resolution;
        T half;
        for(T &p : pBuilder->positions){
            bounds = Union(bounds, p);
        }

        bounds.Expand(20.0 * spacing);

        SetDimension(T(0));

        Float hlen = 0.5 * spacing;
        Float invLen = 1.f / spacing;
        for(int i = 0; i < dimensions; i++){
            resolution[i] = (int)std::ceil(bounds.ExtentOn(i) * invLen);
            half[i] = resolution[i] * hlen;
        }

        T pMin = bounds.Center() - half;
        T pMax = bounds.Center() + half;
        grid = MakeLightweightGrid(resolution, pMin, pMax);

        // NOTE: Is there a way to avoid having to do a double loop over this?
        for(T &p : pBuilder->positions){
            unsigned int h = grid->GetLinearHashedPosition(p);
            grid->stored[h] += 1;
        }

        Build(grid);
    }

    __host__ void Build(GridType *_grid){
        Float initialCellCount = 0;
        grid = _grid;
        particleCount = 0;

        total = grid->GetCellCount();
        fieldDIF = cudaAllocateVx(Float, total);
        smoothFieldDIF = cudaAllocateVx(Float, total);
        countField = grid->stored;

        for(int i = 0; i < total; i++){
            if(countField[i] > 0){
                particleCount += countField[i];
                initialCellCount += 1.f;
            }
            // TODO: Check if this is ok. Theoretically according to page 2
            //       S(k) = Nk / μ0, since we start assuming there are no particles
            //       the field is 0 everywhere.
            fieldDIF[i] = 0;
            smoothFieldDIF[i] = 0;
        }

        /* Equation for μ0 on page 2 */
        mu_0 = (Float)particleCount / initialCellCount;
        if(initialCellCount == 0){
            printf("Warning [ CountingGrid ]: There are no particles inside the grid distribution\n");
        }
    }

    __host__ FieldGridType *Solve(){
        FieldGridType *field = cudaAllocateVx(FieldGridType, 1);
        Q bounds = grid->GetBounds();
        T spacing = grid->GetCellSize();
        U cells = grid->GetIndexCount();

        T p0 = bounds.pMin + 0.5 * spacing;
        field->Build(cells - U(1), spacing, p0, VertexCentered);

        ComputeDIF(this);

        AssureA(field->total == total, "Invalid build");
        for(int i = 0; i < field->total; i++){
            field->field[i] = fieldDIF[i];
        }

        field->MarkFilled();
        return field;
    }

};

template<typename T, typename U, typename Q, typename FilterSolver>
inline void ComputeDIF(CountingGrid<T, U, Q, FilterSolver> *cGrid, Float sigma=1.2){
    LightweightGrid<T, U, Q, size_t> *grid = cGrid->grid;
    size_t *countField = cGrid->countField;
    Float *DIF = cGrid->fieldDIF;
    Float *smoothDIF = cGrid->smoothFieldDIF;
    Float mu_0 = cGrid->mu_0;
    U size = grid->GetIndexCount();
    int total = cGrid->total;
    int dimensions = grid->GetDimensions();

    /* Compute S(k): Equation 2, page 3 */
    AutoParallelFor("CountingGrid - DIF", total, AutoLambda(int i){
        DIF[i] = Min(mu_0, (Float)countField[i]) / mu_0;
    });

    /* Apply box filter */
    AutoParallelFor("CountingGrid - Box Filter", total, AutoLambda(int i){
        FilterSolver solver(sigma);
        U u = grid->GetCellIndex(i);

        auto sampler = [&](U pos) -> Float{
            int clamped = 0;
            pos = ClampedIndex(pos, size, dimensions, clamped);
            unsigned int where = LinearIndex(pos, size, dimensions);
            return DIF[where];
        };

        smoothDIF[i] = solver.Box(u, sampler);
    });

    /* Apply gaussian filter */
    AutoParallelFor("CountingGrid - Box Filter", total, AutoLambda(int i){
        FilterSolver solver(sigma);
        U u = grid->GetCellIndex(i);

        auto sampler = [&](U pos) -> Float{
            int clamped = 0;
            pos = ClampedIndex(pos, size, dimensions, clamped);
            unsigned int where = LinearIndex(pos, size, dimensions);
            return smoothDIF[where];
        };

        DIF[i] = solver.Gaussian(u, sampler);
    });
}

typedef CountingGrid<vec2f, vec2ui, Bounds2f, FilterSolver2D> CountingGrid2D;
typedef CountingGrid<vec3f, vec3ui, Bounds3f, FilterSolver3D> CountingGrid3D;
