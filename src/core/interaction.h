#pragma once

#include <geometry.h>
#include <transform.h>
#include <vector>

// Geometry interactions
class Shape2;
class Shape;

class SurfaceInteraction2{
    public:
    vec2f p, n;
    vec2f pError;
    const Shape2 *shape = nullptr;

    bb_cpu_gpu SurfaceInteraction2(){ shape = nullptr; }
    bb_cpu_gpu SurfaceInteraction2(const vec2f &p): p(p), n(vec2f(0)), pError(vec2f(0))
    { shape = nullptr; }
    bb_cpu_gpu SurfaceInteraction2(const vec2f &po, const vec2f &no,
                                     const vec2f &pErr): p(po), n(no), pError(pErr)
    { shape = nullptr; }

    bb_cpu_gpu SurfaceInteraction2(const vec2f &po, const vec2f &no,
                                     const vec2f &pErr, const Shape2 *sh)
        : p(po), n(no), pError(pErr), shape(sh){}

};

class SurfaceInteraction{
    public:
    vec3f p;
    Normal3f n;
    vec3f pError;
    const Shape *shape = nullptr;

    bb_cpu_gpu SurfaceInteraction(){ shape = nullptr; }
    bb_cpu_gpu SurfaceInteraction(const vec3f &p) : p(p), n(Normal3f(0)), pError(vec3f(0))
    { shape = nullptr; }

    bb_cpu_gpu SurfaceInteraction(const vec3f &po, const Normal3f &no,
                                    const vec3f &pErr): p(po), n(no), pError(pErr)
    { shape = nullptr; }

    bb_cpu_gpu SurfaceInteraction(const vec3f &po, const Normal3f &no,
                                    const vec3f &pErr, const Shape *sh)
        : p(po), n(no), pError(pErr), shape(sh){}
};

inline bb_cpu_gpu
Ray SpawnRayInDirection(const SurfaceInteraction &isect, const vec3f &dir){
    vec3f o = OffsetRayOrigin(isect.p, isect.pError, isect.n, dir);
    //vec3f o = isect.p + Epsilon * dir;
    return Ray(o, dir, Infinity);
}

// Physics interactions, these are meant to hold acceleration
template<typename T>
struct ConstantInteraction{
    T value;
};

template<typename T>
void SetConstantInteraction(ConstantInteraction<T> *interaction, T value){
    interaction->value = value;
}

template<typename T> inline bb_cpu_gpu
T SampleInteraction(ConstantInteraction<T> *interaction, T p){
    return interaction->value;
}

typedef ConstantInteraction<vec2f> ConstantInteraction2;
typedef ConstantInteraction<vec3f> ConstantInteraction3;

#define FUNC_2D_CALL(name) vec2f name(vec2f p, void *prv)
typedef FUNC_2D_CALL(func2d_type);

#define FUNC_3D_CALL(name) vec3f name(vec3f p, void *prv)
typedef FUNC_3D_CALL(func3d_type);

#define DeclareFunctionalInteraction2D(name, ...)\
bb_gpu vec2f name##DeviceFn(vec2f point, void *prv){ __VA_ARGS__ }\
bb_cpu vec2f name##HostFn(vec2f point, void *prv){ __VA_ARGS__ }\
bb_gpu vec2f (*name##_basePtr)(vec2f, void *) = name##DeviceFn;

#define DeclareFunctionalInteraction3D(name, ...)\
bb_gpu vec3f name##DeviceFn(vec3f point, void *prv){ __VA_ARGS__ }\
bb_cpu vec3f name##HostFn(vec3f point, void *prv){ __VA_ARGS__ }\
bb_gpu vec3f (*name##_basePtr)(vec3f, void *) = name##DeviceFn;

#define FunctionalInteractionSet(fnIntr, funcType, name, user) do{\
    cudaMemcpyFromSymbol(&(fnIntr)->devFn, name##_basePtr, sizeof(funcType*));\
    (fnIntr)->hostFn = name##HostFn;\
    (fnIntr)->userPtr = user;\
}while(0)

template<typename FnType>
struct FunctionalInteraction{
    FnType *hostFn;
    FnType *devFn;
    void *userPtr;
};

typedef FunctionalInteraction<func2d_type> FunctionalInteraction2;
typedef FunctionalInteraction<func3d_type> FunctionalInteraction3;

inline bb_cpu_gpu
vec2f SampleInteraction(FunctionalInteraction2 *interaction, vec2f p){
#if defined(__CUDA_ARCH__)
    return interaction->devFn(p, interaction->userPtr);
#else
    return interaction->hostFn(p, interaction->userPtr);
#endif
}

inline bb_cpu_gpu
vec3f SampleInteraction(FunctionalInteraction3 *interaction, vec3f p){
#if defined(__CUDA_ARCH__)
    return interaction->devFn(p, interaction->userPtr);
#else
    return interaction->hostFn(p, interaction->userPtr);
#endif
}

template<typename T, typename FnType>
class InteractionsBuilder{
    public:
    std::vector<T> cInteractionsVecs;
    FunctionalInteraction<FnType> *fInteractions = nullptr;
    int fIt = 0;

    InteractionsBuilder() = default;
    ~InteractionsBuilder() = default;

    template<typename Q>
    ConstantInteraction<Q> *MakeConstantInteractions(std::vector<Q> *ref){
        size_t n = ref->size();
        if(n < 1)
            return nullptr;

        ConstantInteraction<Q> *ptr = cudaAllocateVx(ConstantInteraction<Q>, n);
        for(size_t i = 0; i < n; i++){
            SetConstantInteraction<Q>(&ptr[i], ref->at(i));
        }

        return ptr;
    }

    ConstantInteraction<T> *MakeConstantInteractions(int &size){
        size = cInteractionsVecs.size();
        if(size == 0)
            return nullptr;
        return MakeConstantInteractions<T>(&cInteractionsVecs);
    }

    FunctionalInteraction<FnType> *MakeFunctionalInteractions(int &size){
        size = fIt;
        return fInteractions;
    }
};


template<typename T, typename FnType>
void _AddConstantInteraction(InteractionsBuilder<T, FnType> &builder, T value){
    builder.cInteractionsVecs.push_back(value);
}

#define AddFunctionalInteraction(builder, FnType, name, user)do{\
    if(builder.fInteractions == nullptr){\
        builder.fInteractions = cudaAllocateVx(FunctionalInteraction<FnType>, 16);\
        builder.fIt = 0;\
    }\
    if(builder.fIt >= 16){\
        printf("[ERROR] Too much functional interactions\n");\
        exit(0);\
    }\
    FunctionalInteractionSet(&builder.fInteractions[builder.fIt], FnType, name, user);\
    builder.fIt++;\
}while(0)

#define AddFunctionalInteraction2D(builder, name)\
    AddFunctionalInteraction(builder, func2d_type, name, nullptr)

#define AddFunctionalInteraction3D(builder, name)\
    AddFunctionalInteraction(builder, func3d_type, name, nullptr)

#define AddFunctionalInteractionUser2D(builder, name, user)\
    AddFunctionalInteraction(builder, func2d_type, name, user)

#define AddFunctionalInteractionUser3D(builder, name)\
    AddFunctionalInteraction(builder, func3d_type, name, user)

typedef InteractionsBuilder<vec2f, func2d_type> InteractionsBuilder2;
typedef InteractionsBuilder<vec3f, func3d_type> InteractionsBuilder3;

inline void AddConstantInteraction(InteractionsBuilder2 &builder, vec2f value){
    _AddConstantInteraction<vec2f, func2d_type>(builder, value);
}

inline void AddConstantInteraction(InteractionsBuilder3 &builder, vec3f value){
    _AddConstantInteraction<vec3f, func3d_type>(builder, value);
}

