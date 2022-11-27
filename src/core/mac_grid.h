/* date = October 26th 2022 21:02 */
#pragma once
#include <vgrid.h>
#include <geometry.h>
#include <sampling.h>

#define enabled_for_2D template<typename Up = Us,\
    typename std::enable_if<std::is_same<Up, vec2ui>::value, bool>::type* = nullptr>

#define enabled_for_3D template<typename Up = Us,\
    typename std::enable_if<std::is_same<Up, vec3ui>::value, bool>::type* = nullptr>

template<typename Type>
inline __host__ void Inflow2D(Type *curr, vec2ui resolution, vec4f rect, Type val){
    Float lower_x = rect.x,
          lower_y = rect.y,
          upper_x = rect.z,
          upper_y = rect.w;
    vec2ui res = resolution;
    size_t items = res.x * res.y;

    AutoParallelFor("Inflow2D", items, AutoLambda(size_t index){
        int i = index % res.x;
        int j = index / res.x;
        if(i >= lower_x && i <= upper_x && j >= lower_y && j <= upper_y){
            curr[index] = val;
        }
    });
}

template<typename T, typename Us, typename Type, int dims>
class MemorySlab{
    public:
    Type *curr;
    Type *next;
    Us resolution;
    T offset;
    size_t total;

    __bidevice__ MemorySlab(): curr(nullptr), next(nullptr), resolution(Us(0)),
        offset(T(0)), total(0){}

    __host__ MemorySlab(const Us &res, const Type &initial=Type(0)){
        total = 1;
        for(int i = 0; i < dims; i++){
            total *= res[i];
            offset[i] = 0.5;
        }
        AssertA(total > 0, "Zero length resolution");
        Type *p_curr = cudaAllocateVx(Type, total);
        Type *p_next = cudaAllocateVx(Type, total);
        resolution = res;

        for(size_t i = 0; i < total; i++){
            p_curr[i] = initial;
            p_next[i] = initial;
        }

        curr = p_curr;
        next = p_next;
    }

    __host__ void SetOrigin(const T &off){
        offset = off;
    }

    __host__ void Flip(){
        Type *tmp = curr;
        curr = next;
        next = tmp;
    }

    __host__ Type Average(){
        Type val = Type(0);
        Float inv = 1.f / (Float)total;
        for(size_t i = 0; i < total; i++)
            val = val + (curr[i] * inv);
        return val;
    }

    enabled_for_2D
    __host__ void Inflow(vec4f rect, Type val){
        Inflow2D<Type>(curr, resolution, rect, val);
    }

    enabled_for_2D
    __bidevice__ Type At(int i, int j, Type def){
        if(i >= resolution.x || j >= resolution.y || i < 0 || j < 0)
            return def;
        return curr[i + j * resolution.x];
    }

    enabled_for_2D
    __bidevice__ vec2f DataPoint(vec2ui index){
        return vec2f((Float)index.x + offset.x, (Float)index.y + offset.y);
    }

    enabled_for_2D
    __bidevice__ Type Closest(vec2ui p){
        int i = Max(0, Min(int(p.x), resolution.x-1));
        int j = Max(0, Min(int(p.y), resolution.y-1));
        return curr[i + j * resolution.x];
    }

    enabled_for_2D
    __bidevice__ Type Sample(vec2f p){
        Float s = p.x - offset.x, t = p.y - offset.y;
        int iu = int(s), iv = int(t);
        Float fu = s - iu, fv = t - iv;
        Type f00 = Closest(vec2ui(iu+0, iv+0));
        Type f10 = Closest(vec2ui(iu+1, iv+0));
        Type f01 = Closest(vec2ui(iu+0, iv+1));
        Type f11 = Closest(vec2ui(iu+1, iv+1));
        return Bilerp(f00, f10, f01, f11, fu, fv);
    }

    using VelSlab2 = MemorySlab<vec2f, vec2ui, Float, 2>;
    enabled_for_2D
    __bidevice__ Type AdvectPoint(VelSlab2 *vf_u, VelSlab2 *vf_v, Float dt, vec2ui index){
        vec2f p = DataPoint(index);
        vec2f k1 = vec2f(vf_u->Sample(p), vf_v->Sample(p));
        vec2f p1 = p - 0.5 * dt * k1;
        vec2f k2 = vec2f(vf_u->Sample(p1), vf_v->Sample(p1));
        vec2f p2 = p - 0.75 * dt * k2;
        vec2f k3 = vec2f(vf_u->Sample(p2), vf_v->Sample(p2));
        vec2f p_prev = p - (dt / 9.0) * (2.0 * k1 + 3.0 * k2 + 4.0 * k3);
        return Sample(p_prev);
    }
};

typedef MemorySlab<vec2f, vec2ui, Float, 2> MemorySlab2D1f;

template<typename T, typename Us, typename GridData, int dims>
class MACVelocityGrid{
    public:
    GridData *_us[3];
    unsigned int _sizes[3];
    Float dx;
    bool withCopy;

    __bidevice__ MACVelocityGrid(){
        for(int i = 0; i < 3; i++){
            _us[i] = nullptr;
            _sizes[i] = 0;
        }
    }

    __bidevice__ GridData *U_ptr(){ return GetComponent(0); }
    __bidevice__ GridData *V_ptr(){ return GetComponent(1); }
    __bidevice__ GridData *W_ptr(){ return GetComponent(2); }
    __bidevice__ unsigned int U_size(){ return _sizes[0]; }
    __bidevice__ unsigned int V_size(){ return _sizes[1]; }
    __bidevice__ unsigned int W_size(){ return _sizes[2]; }

    __host__ void InitComponent(T origin, Us size, Float hx, bool withCopy,
                                int axis)
    {
        GridData *grid = cudaAllocateVx(GridData, 1);
        T off(0.f);

        size[axis] += 1;
        grid->Build(size, 0.f);
        if(withCopy)
            grid->BuildAuxiliary();

        off[(axis+1)%dims] = 0.5f;
        if(dims > 2)
            off[(axis+2)%dims] = 0.5f;

        grid->SetGeometry(origin, hx, off);
        _us[axis] = grid;
    }

    __host__ void Init(T origin, Us size, Float hx, bool withCopy){
        withCopy = withCopy;
        InitComponent(origin, size, hx, withCopy, 0);
        InitComponent(origin, size, hx, withCopy, 1);
        if(dims > 2)
            InitComponent(origin, size, hx, withCopy, 2);
    }

    __bidevice__ GridData *GetComponent(int i){
        if(!(i < 3)){
            printf("Invalid query for MAC component !(%d < 3)\n", i);
        }
        AssertA(i < 3, "Invalid query for MAC component");
        return _us[i];
    }

    enabled_for_2D
    __bidevice__ Float U(int i, int j){
        return _us[0]->At(i, j);
    }

    enabled_for_2D
    __bidevice__ Float V(int i, int j){
        return _us[1]->At(i, j);
    }

    enabled_for_2D
    __bidevice__ void SetU(int i, int j, Float val){
        _us[0]->Set(i, j, val);
    }

    enabled_for_2D
    __bidevice__ void SetV(int i, int j, Float val){
        _us[1]->Set(i, j, val);
    }

    // Things that cannot be implemented template-aware need to be explicitly specified
    enabled_for_2D
    __bidevice__ void test(){

    }

    enabled_for_3D
    __bidevice__ void test(){
        // TODO:
    }
};

typedef MACVelocityGrid<vec2f, vec2ui, GridData2f, 2> MACVelocityGrid2;

