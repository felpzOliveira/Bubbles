/* date = October 26th 2022 20:56 */
#pragma once
#include <geometry.h>
#include <iomanip>
#include <sstream>
#include <sampling.h>

/*
* I'm having trouble following the book, so I'll change things a little and
* use Bridson and Ryan L. Guy implementation as references.
* I'm not gonna template this thing as I'd like to remove the amount of for-loops
* required to handle dimensions.
*/

/*
* Simple abstraction for a grid index.
*/
class GridIndex2{
    public:
    int i, j;

    __bidevice__ GridIndex2() : i(0), j(0){}
    __bidevice__ GridIndex2(int _i, int _j) : i(_i), j(_j){}

    __bidevice__ bool operator==(const GridIndex2 &other) const{
        return i == other.i && j == other.j;
    }

    __bidevice__ bool operator!=(const GridIndex2 &other) const{
        return !(i == other.i && j == other.j);
    }

    __bidevice__ int operator[](int _i) const{
        if(_i > 1){
            printf("Got query for %d but index is 2-dimensional\n", _i);
        }
        AssertA(i < 2, "Invalid id for GridIndex query");
        if(_i == 0) return i;
        return j;
    }

    __bidevice__ int &operator[](int _i){
        if(_i > 1){
            printf("Got query for %d but index is 2-dimensional\n", _i);
        }
        AssertA(i < 2, "Invalid id for GridIndex query");
        if(_i == 0) return i;
        return j;
    }

    __bidevice__ int Dimensions(){ return 2; }
};

inline std::ostream &operator<<(std::ostream &out, const GridIndex2 &index){
    return out << "(" << index.i << "," << index.j << ")";
}

template<typename T>
class GridData2{
    public:
    T *data;
    T *pong;
    bool hasOutOfRange;
    T outOfRange;
    int total;
    int nx, ny;
    bool withCopy;
    vec2f dataOffset;
    Float spacing;
    vec2f origin;

    __bidevice__ GridData2() : data(nullptr), pong(nullptr), hasOutOfRange(false),
        outOfRange(T(0)), total(0), nx(0), ny(0), withCopy(false),
        dataOffset(vec2f(0)), spacing(0.f), origin(vec2f(0)){}

    // prevent conversion from int/uint/float to vec2<T> with 1 indexed constructors
    template<class S, class Q>
    __bidevice__ GridData2(S, Q) = delete;

    explicit
    __host__ GridData2(vec2ui res, T value){
        Build(res, value);
    }

    __host__ void SetGeometry(vec2f _origin, Float _spacing, vec2f _dataOff){
        dataOffset = _dataOff;
        spacing = _spacing;
        origin = _origin;
    }

    __host__ void Build(vec2ui res, T value){
        nx = res.x;
        ny = res.y;
        total = nx * ny;
        hasOutOfRange = false;
        withCopy = false;
        Allocate();
        Fill(value);
    }

    __host__ void BuildAuxiliary(){
        withCopy = true;
        if(pong) cudaFree(pong);
        pong = cudaAllocateUnregisterVx(T, total);
        for(unsigned int i = 0; i < total; i++){
            pong[i] = data[i];
        }
    }

    __host__ void Flip(){
        if(withCopy){
            Swap(&data, &pong);
        }
    }

    __host__ GridData2(const GridData2 &other){
        if(data) cudaFree(data);
        if(pong) cudaFree(pong);
        nx = other.nx;
        ny = other.ny;
        total = other.total;
        hasOutOfRange = other.hasOutOfRange;
        outOfRange = other.outOfRange;
        withCopy = other.withCopy;

        Allocate();
        for(unsigned int i = 0; i < total; i++){
            data[i] = other.data[i];
            if(withCopy)
                pong[i] = other.pong[i];
        }
    }

    __host__ void SetOutOfRange(T value){
        hasOutOfRange = true;
        outOfRange = value;
    }

    __host__ GridData2 operator=(const GridData2 &other){
        if(data) cudaFree(data);
        if(pong) cudaFree(pong);
        nx = other.nx;
        ny = other.ny;
        total = other.total;
        hasOutOfRange = other.hasOutOfRange;
        outOfRange = other.outOfRange;
        withCopy = other.withCopy;

        Allocate();
        for(unsigned int i = 0; i < total; i++){
            data[i] = other.data[i];
            if(withCopy)
                pong[i] = other.pong[i];
        }
        return *this;
    }

    __host__ void Fill(T value){
        for(int i = 0; i < total; i++){
            data[i] = value;
            if(withCopy)
                pong[i] = value;
        }
    }

    __host__ void Allocate(){
        data = cudaAllocateUnregisterVx(T, total);
        if(withCopy)
            pong = cudaAllocateUnregisterVx(T, total);
    }

    __bidevice__ Float Spacing(){
        return spacing;
    }

    __bidevice__ T operator()(int i, int j){
        bool in_range = i < nx && j < ny && i >= 0 && j >= 0;
        if(!in_range && hasOutOfRange){
            return outOfRange;
        }

        if(!in_range){
            //printf("Out of range query for (%d %d), domain is (%d %d)\n", i, j, nx, ny);
        }
        AssertA(in_range, "Out of range query");
        return data[LinearIndex(i, j)];
    }

    __bidevice__ T operator()(int i, int j) const{
        bool in_range = i < nx && j < ny && i >= 0 && j >= 0;
        if(!in_range && hasOutOfRange){
            return outOfRange;
        }

        if(!in_range){
            printf("Out of range query for (%d %d), domain is (%d %d)\n", i, j, nx, ny);
        }
        AssertA(in_range, "Out of range query");
        return data[LinearIndex(i, j)];
    }

    __bidevice__ T operator()(int id){
        bool in_range = id < total && id >= 0;
        if(!in_range && hasOutOfRange){
            return outOfRange;
        }

        if(!in_range){
            printf("Out of range query for %d, domain has %d elements\n", id, total);
        }
        AssertA(in_range, "Out of range query");
        return data[id];
    }

    __bidevice__ bool IsQueriable(const vec2f &p){
        return(p.x >= 0 && p.x < (nx + dataOffset.x) &&
               p.y >= 0 && p.y < (ny + dataOffset.y));
    }

    __bidevice__ vec2f ToGridPosition(const vec2f &p){
        return (p - origin) * 1.f / spacing;
    }

    __bidevice__ vec2f DataGridPosition(size_t index){
        int iy = index / nx;
        int ix = index % nx;
        return vec2f((Float)ix + dataOffset.x,
                     (Float)iy + dataOffset.y);
    }

    __bidevice__ vec2f DataPosition(size_t index){
        int iy = index / nx;
        int ix = index % nx;
        return vec2f((Float) (ix + dataOffset.x) * spacing,
                     (Float) (iy + dataOffset.y) * spacing) + origin;
    }

    template<typename Interpolator>
    __bidevice__ T SampleGridCoords(Float fx, Float fy, Interpolator &interpolator)
    {
        if(fx < 0 || fy < 0 || fx > nx-1+dataOffset.x || fy > ny-1+dataOffset.y){
            //printf("Point outside domain { %g %g } (%d x %d)\n",
                    //fx, fy, nx, ny);
        }

        vec2f gridCoords = vec2f(fx, fy) - dataOffset;
        fx = Clamp(gridCoords.x, 0.f, (Float)nx - OnePlusEpsilon);
        fy = Clamp(gridCoords.y, 0.f, (Float)ny - OnePlusEpsilon);

        auto at_fn = [&](int _x, int _y) -> Float{
            return At(_x, _y);
        };

        return interpolator.Interpolate(fx, fy, vec2ui(nx, ny), at_fn);
    }

    template<typename Interpolator>
    __bidevice__ T Sample(Float fx, Float fy, Interpolator &interpolator){
        vec2f gridCoords = ToGridPosition(vec2f(fx, fy));
        return SampleGridCoords<Interpolator>(gridCoords.x, gridCoords.y,
                                              interpolator);
    }

    template<typename Interpolator>
    __bidevice__ T Sample(vec2f p){
        Interpolator interpolator;
        return Sample<Interpolator>(p.x, p.y, interpolator);
    }

    __bidevice__ T LinearSample(vec2f p){
        LinearInterpolator interpolator;
        return Sample<LinearInterpolator>(p.x, p.y, interpolator);
    }

    __bidevice__ T MonotonicCatmullSample(vec2f p){
        MonotonicCatmull interpolator;
        return Sample<MonotonicCatmull>(p.x, p.y, interpolator);
    }

    __bidevice__ T operator()(GridIndex2 index){
        return this->operator()(index.i, index.j);
    }

    __bidevice__ int Length(){ return total; }

    __bidevice__ T At(int i, int j){
        return this->operator()(i, j);
    }

    __bidevice__ T At(int i, int j) const{
        return this->operator()(i, j);
    }

    __bidevice__ T At(int id){
        return this->operator()(id);
    }

    __bidevice__ T At(GridIndex2 index){
        return this->operator()(index);
    }

    __bidevice__ void Set(int i, int j, T val){
        bool in_range = i < nx && j < ny && i >= 0 && j >= 0;
        if(!in_range){
            printf("Out of range set for (%d %d), domain is (%d %d)\n",
                    i, j, nx, ny);
        }

        AssertA(in_range, "Out of range set");
        data[LinearIndex(i, j)] = val;
    }

    __bidevice__ void Set(GridIndex2 index, T val){
        Set(index.i, index.j, val);
    }

    __bidevice__ void SetNext(size_t index, T val){
        if(!(index < total)){
            printf("Out of range set for next buffer ( %ld > % d )\n",
                    index, total);
            return;
        }

        if(!pong){
            printf("Attempted to set next buffer, but no aulixiary allocated\n");
            return;
        }

        pong[index] = val;
    }

    __bidevice__ void Add(int i, int j, T val){
        bool in_range = i < nx && j < ny && i >= 0 && j >= 0;
        if(!in_range){
            printf("Out of range add for (%d %d), domain is (%d %d)\n",
                    i, j, nx, ny);
        }

        AssertA(in_range, "Out of range set");
        data[LinearIndex(i, j)] += val;
    }

    __bidevice__ void Add(GridIndex2 index, T val){
        Add(index.i, index.j, val);
    }

    __bidevice__ int LinearIndex(int i, int j){
        return i + j * nx;
    }

    __bidevice__ int LinearIndex(int i, int j) const{
        return i + j * nx;
    }

    __bidevice__ ~GridData2(){
    #if defined(__CUDA_ARCH__)
        printf(" ** Warning: deleting object in cuda code\n");
    #else
        if(data) cudaFree(data);
        if(pong) cudaFree(pong);
    #endif
    }
};

template<typename T>
inline std::string GridDataString(const GridData2<T> &grid, int w, int h, int n_w){
    std::stringstream ss;
    ss << " - Width: " << grid.nx << std::endl;
    ss << " - Height: " << grid.ny << std::endl;
    ss << " - Total: " << grid.total << std::endl;
    ss << " - HasOutOfRange: " << grid.hasOutOfRange << std::endl;
    if(grid.hasOutOfRange){
        ss << "  - OutOfRange: " << grid.outOfRange << std::endl;
    }
    ss << " - DataOffset: [ " << grid.dataOffset.x << " " <<
            grid.dataOffset.y << " ]" << std::endl;
    ss << " - WithCopy: " << grid.withCopy << std::endl;
    ss << " - Data: " << std::endl;
    ss << std::left << std::setw(n_w) << std::setfill(' ') << " ";
    for(int j = 1; j < grid.nx+1; j++){
        std::string num("(");
        num += std::to_string(j);
        num += ")";
        ss << std::left << std::setw(n_w) << std::setfill(' ') << num;
        if(j == w && j < grid.nx-1){
            ss << std::left << std::setw(n_w) << std::setfill(' ') << "...";
            j = grid.nx-1;
        }
    }

    ss << std::endl;
    for(int j = 0; j < grid.ny; j++){
        std::string num("(");
        num += std::to_string(j+1);
        num += ")";
        ss << std::left << std::setw(n_w) << std::setfill(' ') << num;
        for(int i = 0; i < grid.nx; i++){
            if(i < w || i == grid.nx-1){
                ss << std::left << std::setw(n_w) << std::setfill(' ') <<
                    std::fixed << std::setprecision(4) << grid(i, j);
            }else if(i == w && i < grid.nx-1){
                ss << std::left << std::setw(n_w) << std::setfill(' ') << "...";
                i = grid.nx-2;
            }
        }

        if(j < grid.ny-1)
            ss << std::endl;

        if(j == h && j < grid.ny-1){
            ss << "..." << std::endl;
            j = grid.ny-2;
        }
    }
    return ss.str();
}

template<typename T>
inline std::ostream &operator<<(std::ostream &out, const GridData2<T> &grid){
    const int w = 8;
    const int h = 6;
    const int n_w = 8;
    out << GridDataString(grid, w, h, n_w);
    return out;
}

typedef GridData2<Float> GridData2f;

typedef enum{
    Air=0,
    Solid,
    Fluid
}MaterialType;

struct Material{
    MaterialType type;
    unsigned int id;

    __bidevice__ Material(){type = Air; id = 0;}

    __bidevice__ Material(Float){
        type = Air;
        id = 0;
    }
};

inline std::ostream &operator<<(std::ostream &out, const Material &mat){
    if(mat.type == Solid) out << "Solid";
    else if(mat.type == Fluid) out << "Fluid";
    else out << "Air";
    return out;
}

typedef GridData2<Material> MaterialGridData2;

class GridIndex3{
    public:
    int i, j, k;

    __bidevice__ GridIndex3() : i(0), j(0), k(0){}
    __bidevice__ GridIndex3(int _i, int _j, int _k):
        i(_i), j(_j), k(_k){}

    __bidevice__ bool operator==(const GridIndex3 &other) const{
        return i == other.i && j == other.j && k == other.k;
    }

    __bidevice__ bool operator!=(const GridIndex3 &other) const{
        return !(i == other.i && j == other.j && k == other.k);
    }

    __bidevice__ int operator[](int _i) const{
        if(_i > 2){
            printf("Got query for %d but index is 3-dimensional\n", _i);
        }
        AssertA(i < 3, "Invalid id for GridIndex query");
        if(_i == 0) return i;
        else if(_i == 1) return j;
        return k;
    }

    __bidevice__ int &operator[](int _i){
        if(_i > 2){
            printf("Got query for %d but index is 2-dimensional\n", _i);
        }
        AssertA(i < 3, "Invalid id for GridIndex query");
        if(_i == 0) return i;
        else if(_i == 1) return j;
        return k;
    }

    __bidevice__ int Dimensions(){ return 3; }
};

inline std::ostream &operator<<(std::ostream &out, const GridIndex3 &index){
    return out << "(" << index.i << "," << index.j << "," << index.k << ")";
}

