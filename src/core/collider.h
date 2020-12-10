#pragma once
#include <geometry.h>
#include <shape.h>
#include <vector>

class ClosestPointQuery2{
    public:
    vec2f point;
    vec2f normal;
    Float signedDistance;
    __bidevice__ ClosestPointQuery2(){}
    __bidevice__ ClosestPointQuery2(const vec2f &p, const vec2f &n, Float d){
        point = p; normal = n; signedDistance = d;
    }
};

class ClosestPointQuery{
    public:
    vec3f point;
    Normal3f normal;
    Float signedDistance;
    __bidevice__ ClosestPointQuery(){}
    __bidevice__ ClosestPointQuery(const vec3f &p, const Normal3f &n, Float d){
        point = p; normal = n; signedDistance = d;
    }
};

class Collider2{
    public:
    Shape2 *shape;
    Float frictionCoefficient;
    
    __bidevice__ Collider2();
    __bidevice__ Collider2(Shape2 *shape);
    __bidevice__ void Initialize(Shape2 *shape, Float frictionCoef=0);
    __bidevice__ bool IsPenetrating(const ClosestPointQuery2 &colliderPoint,
                                    const vec2f &position, Float radius);
    __bidevice__ bool ResolveCollision(Float radius, Float restitutionCoefficient,
                                       vec2f *position, vec2f *velocity);
    __host__ void GenerateSDFs(void);
};

class Collider3{
    public:
    Shape *shape;
    Float frictionCoefficient;
    __bidevice__ Collider3();
    __bidevice__ Collider3(Shape *shape);
    __bidevice__ void Initialize(Shape *shape, Float frictionCoef=0);
    __bidevice__ bool IsPenetrating(const ClosestPointQuery &colliderPoint,
                                    const vec3f &position, Float radius);
    __bidevice__ bool ResolveCollision(Float radius, Float restitutionCoefficient,
                                       vec3f *position, vec3f *velocity);
    __host__ void GenerateSDFs(void);
};

class ColliderSet2{
    public:
    Collider2 **colliders;
    int nColiders;
    
    __bidevice__ ColliderSet2();
    __bidevice__ void Initialize(Collider2 **colls, int count);
    __bidevice__ bool ResolveCollision(Float radius, Float restitutionCoefficient,
                                       vec2f *position, vec2f *velocity);
    __host__ void GenerateSDFs();
};

class ColliderSet3{
    public:
    Collider3 **colliders;
    int nColiders;
    
    __bidevice__ ColliderSet3();
    __bidevice__ void Initialize(Collider3 **colls, int count);
    __bidevice__ bool ResolveCollision(Float radius, Float restitutionCoefficient,
                                       vec3f *position, vec3f *velocity);
    __host__ void GenerateSDFs(void);
};

class ColliderSetBuilder2{
    public:
    std::vector<Collider2*> colliders;
    ColliderSet2 *setReference;
    
    __host__ ColliderSetBuilder2();
    __host__ void AddCollider2(Collider2 *collider);
    __host__ void AddCollider2(Shape2 *shape, Float frictionCoef=0.);
    __host__ ColliderSet2 *GetColliderSet();
};

class ColliderSetBuilder3{
    public:
    std::vector<Collider3*> colliders;
    ColliderSet3 *setReference;
    
    __host__ ColliderSetBuilder3();
    __host__ void AddCollider3(Collider3 *collider);
    __host__ void AddCollider3(Shape *shape, Float frictionCoef=0.);
    __host__ ColliderSet3 *GetColliderSet();
};

__host__ Collider2 *MakeCollider2(Shape2 *shape, Float frictionCoef=0.);
__host__ Collider3 *MakeCollider3(Shape *shape, Float frictionCoef=0.);