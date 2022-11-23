/* date = October 26th 2022 21:02 */
#pragma once
#include <vgrid.h>
#include <geometry.h>

#define enabled_for_2D template<typename Up = Us,\
    typename std::enable_if<std::is_same<Up, vec2ui>::value, bool>::type* = nullptr>

#define enabled_for_3D template<typename Up = Us,\
    typename std::enable_if<std::is_same<Up, vec3ui>::value, bool>::type* = nullptr>

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

