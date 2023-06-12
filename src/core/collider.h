#pragma once
#include <geometry.h>
#include <shape.h>
#include <vector>

class ClosestPointQuery2{
    public:
    vec2f point;
    vec2f normal;
    Float signedDistance;
    vec2f velocity;
    bb_cpu_gpu ClosestPointQuery2(){}
    bb_cpu_gpu ClosestPointQuery2(const vec2f &p, const vec2f &n, Float d){
        point = p; normal = n; signedDistance = d; velocity = vec2f(0);
    }
    bb_cpu_gpu ClosestPointQuery2(const vec2f &p, const vec2f &n, Float d, const vec2f &v){
        point = p; normal = n; signedDistance = d; velocity = v;
    }
};

class ClosestPointQuery{
    public:
    vec3f point;
    Normal3f normal;
    Float signedDistance;
    vec3f velocity;
    bb_cpu_gpu ClosestPointQuery(){}
    bb_cpu_gpu ClosestPointQuery(const vec3f &p, const Normal3f &n, Float d){
        point = p; normal = n; signedDistance = d; velocity = vec3f(0);
    }
    bb_cpu_gpu ClosestPointQuery(const vec3f &p, const Normal3f &n, Float d, const vec3f &v){
        point = p; normal = n; signedDistance = d; velocity = v;
    }
};

class Collider2{
    public:
    Shape2 *shape;
    Float frictionCoefficient;
    bool isActive;

    bb_cpu_gpu Collider2();
    bb_cpu_gpu Collider2(Shape2 *shape);
    bb_cpu_gpu void Initialize(Shape2 *shape, Float frictionCoef=0);
    bb_cpu_gpu bool IsPenetrating(const ClosestPointQuery2 &colliderPoint,
                                    const vec2f &position, Float radius);
    bb_cpu_gpu bool ResolveCollision(Float radius, Float restitutionCoefficient,
                                       vec2f *position, vec2f *velocity);
    bb_cpu_gpu bool IsActive();
    bb_cpu_gpu bool OptmizedClosestPointCheck(const vec2f &position);
    void SetActive(bool active);
    void GenerateSDFs(void);
};

class Collider3{
    public:
    Shape *shape;
    Float frictionCoefficient;
    bool isActive;

    bb_cpu_gpu Collider3();
    bb_cpu_gpu Collider3(Shape *shape);
    bb_cpu_gpu void Initialize(Shape *shape, Float frictionCoef=0);
    bb_cpu_gpu bool IsPenetrating(const ClosestPointQuery &colliderPoint,
                                    const vec3f &position, Float radius);
    bb_cpu_gpu bool ResolveCollision(Float radius, Float restitutionCoefficient,
                                       vec3f *position, vec3f *velocity);
    bb_cpu_gpu bool IsActive();
    bb_cpu_gpu bool OptmizedClosestPointCheck(const vec3f &position);
    void SetActive(bool active);
    void GenerateSDFs(void);
};

class ColliderSet2{
    public:
    Collider2 **colliders;
    int nColiders;

    bb_cpu_gpu ColliderSet2();
    bb_cpu_gpu void Initialize(Collider2 **colls, int count);
    bb_cpu_gpu bool ResolveCollision(Float radius, Float restitutionCoefficient,
                                       vec2f *position, vec2f *velocity);
    bb_cpu_gpu bool IsActive(int which);
    void SetActive(int which, bool active);
    void GenerateSDFs();
};

class ColliderSet3{
    public:
    Collider3 **colliders;
    int nColiders;

    bb_cpu_gpu ColliderSet3();
    bb_cpu_gpu void Initialize(Collider3 **colls, int count);
    bb_cpu_gpu bool ResolveCollision(Float radius, Float restitutionCoefficient,
                                       vec3f *position, vec3f *velocity);
    bb_cpu_gpu bool IsActive(int which);
    void SetActive(int which, bool active);
    void GenerateSDFs(void);
};

class ColliderSetBuilder2{
    public:
    std::vector<Collider2*> colliders;
    ColliderSet2 *setReference;

    ColliderSetBuilder2();
    void AddCollider2(Collider2 *collider);
    void AddCollider2(Shape2 *shape, Float frictionCoef=0.);
    ColliderSet2 *GetColliderSet();
};

class ColliderSetBuilder3{
    public:
    std::vector<Collider3*> colliders;
    ColliderSet3 *setReference;

    ColliderSetBuilder3();
    void AddCollider3(Collider3 *collider);
    void AddCollider3(Shape *shape, Float frictionCoef=0.);
    ColliderSet3 *GetColliderSet();
};

Collider2 *MakeCollider2(Shape2 *shape, Float frictionCoef=0.);
Collider3 *MakeCollider3(Shape *shape, Float frictionCoef=0.);
