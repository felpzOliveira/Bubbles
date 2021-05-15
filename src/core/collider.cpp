#include <collider.h>

template<class Collider, class T, class PointQuery>
__bidevice__ bool CollisionHandle(Collider *collider, Float radius, 
                                  Float restitutionCoefficient,
                                  Float frictionCoefficient,
                                  T *position, T *velocity)
{
    bool collided = false;
    PointQuery colliderPoint;
    if(!collider->OptmizedClosestPointCheck(*position)){
        return false;
    }

    collider->shape->ClosestPoint(*position, &colliderPoint);
    if(collider->IsPenetrating(colliderPoint, *position, radius)){
        T targetNormal = T(colliderPoint.normal);
        T targetPoint  = colliderPoint.point + radius * targetNormal;
        T colliderVelocity = colliderPoint.velocity;
        T relativeVel = *velocity - colliderVelocity;
        
        Float normalDotRel = Dot(targetNormal, relativeVel);
        T relativeVelN = normalDotRel * targetNormal;
        T relativeVelT = relativeVel - relativeVelN;
        
        if(normalDotRel < 0){
            T deltaRelN = (-1.0 - restitutionCoefficient) * relativeVelN;
            relativeVelN *= -restitutionCoefficient;
            if(relativeVelT.LengthSquared() > 0){
                Float scale = deltaRelN.Length() / relativeVelT.Length();
                Float fScale = 1.0 - frictionCoefficient * scale;
                Float frictionScale = Max(0, fScale);
                relativeVelT *= frictionScale;
            }
            
            *velocity = relativeVelN + relativeVelT + colliderVelocity;
        }
        
        *position = targetPoint;
        collided = true;
    }
    
    return collided;
}

////////////////////////
// Collider2 
///////////////////////
__bidevice__ Collider2::Collider2(){ shape = nullptr; frictionCoefficient = 0; isActive = true; }
__bidevice__ Collider2::Collider2(Shape2 *shape) : shape(shape), frictionCoefficient(0), isActive(true){}

__host__ void Collider2::SetActive(bool active){
    isActive = active;
}

__bidevice__ bool Collider2::IsActive(){
    return isActive;
}

__bidevice__ void Collider2::Initialize(Shape2 *shp, Float frictionCoef){
    shape = shp;
    frictionCoefficient = frictionCoef;
    isActive = true;
}

__bidevice__ bool Collider2::IsPenetrating(const ClosestPointQuery2 &colliderPoint,
                                           const vec2f &position, Float radius)
{
    AssertA(shape, "Invalid shape pointer for Collider2");
    return shape->IsInside(position) || Absf(colliderPoint.signedDistance) < radius;
}

__bidevice__ bool Collider2::ResolveCollision(Float radius, Float restitutionCoefficient,
                                              vec2f *position, vec2f *velocity)
{
    AssertA(shape, "Invalid shape pointer for Collider2::ResolveCollision");
    if(!isActive) return false;
    return CollisionHandle<Collider2, vec2f, ClosestPointQuery2>(this, radius,
                                                                 restitutionCoefficient,
                                                                 frictionCoefficient,
                                                                 position, velocity);
}

__bidevice__ bool Collider2::OptmizedClosestPointCheck(const vec2f &position){
    return true;
}

__host__ void Collider2::GenerateSDFs(){
    AssertA(shape != nullptr, "Invalid shape pointer for SDF generation");
    GenerateShapeSDF(shape);
}

////////////////////////
// Collider 
///////////////////////
__bidevice__ Collider3::Collider3(){ shape = nullptr; frictionCoefficient = 0; isActive = true; }
__bidevice__ Collider3::Collider3(Shape *shape): shape(shape), frictionCoefficient(0), isActive(true){}

__bidevice__ void Collider3::Initialize(Shape *shp, Float frictionCoef){
    shape = shp;
    frictionCoefficient = frictionCoef;
    isActive = true;
}

__host__ void Collider3::SetActive(bool active){
    isActive = active;
}

__bidevice__ bool Collider3::IsActive(){
    return isActive;
}

__bidevice__ bool Collider3::OptmizedClosestPointCheck(const vec3f &position){
    if(shape->type == ShapeType::ShapeMesh){
        if(!Inside(position, shape->GetBounds())){
            return false;
        }
    }

    return true;
}

__bidevice__ bool Collider3::IsPenetrating(const ClosestPointQuery &colliderPoint,
                                           const vec3f &position, Float radius)
{
    AssertA(shape, "Invalid shape pointer for Collider");
    if(shape->type == ShapeType::ShapeMesh){
        if(!Inside(position, shape->GetBounds())){
            return false;
        }
    }
    return shape->IsInside(position) || Absf(colliderPoint.signedDistance) < radius;
}

__bidevice__ bool Collider3::ResolveCollision(Float radius, Float restitutionCoefficient,
                                              vec3f *position, vec3f *velocity)
{
    AssertA(shape, "Invalid shape pointer for Collider::ResolveCollision");
    if(!isActive) return false;
    return CollisionHandle<Collider3, vec3f, ClosestPointQuery>(this, radius, 
                                                                restitutionCoefficient,
                                                                frictionCoefficient,
                                                                position, velocity);
}

__host__ void Collider3::GenerateSDFs(){
    AssertA(shape != nullptr, "Invalid shape pointer for SDF generation");
    GenerateShapeSDF(shape);
}

////////////////////////
// ColliderSet2 
///////////////////////

__bidevice__ ColliderSet2::ColliderSet2(){ nColiders = 0; colliders = nullptr; }

__bidevice__ void ColliderSet2::Initialize(Collider2 **colls, int count){
    colliders = colls;
    nColiders = count;
}

__host__ void ColliderSet2::SetActive(int which, bool active){
    if(which >= 0 && which < nColiders){
        colliders[which]->SetActive(active);
    }
}

__bidevice__ bool ColliderSet2::IsActive(int which){
    if(which >= 0 && which < nColiders){
        return colliders[which]->IsActive();
    }

    return false;
}

__bidevice__ bool ColliderSet2::ResolveCollision(Float radius, Float restitutionCoefficient,
                                                 vec2f *position, vec2f *velocity)
{
    AssertA(nColiders > 0, "No Colliders present for ColliderSet2::ResolveCollision");
    int targetCollider = -1;
    Float minDistance = Infinity;
    for(int i = 0; i < nColiders; i++){
        if(colliders[i]->OptmizedClosestPointCheck(*position)){
            Float distance = Absf(colliders[i]->shape->ClosestDistance(*position));
            if(distance < minDistance){
                targetCollider = i;
                minDistance = distance;
            }
        }
    }

    // colliders were reject by optmization, this particle
    // will not collide with anything
    if(targetCollider < 0){
        return false;
    }
    
    AssertA(targetCollider >= 0 && targetCollider < nColiders, 
            "Invalid collider id for ColliderSet2::ResolveCollision");
    return colliders[targetCollider]->ResolveCollision(radius, restitutionCoefficient, 
                                                       position, velocity);
}

__host__ void ColliderSet2::GenerateSDFs(){
    AssertA(nColiders > 0, "No Colliders present for ColliderSet2::GenerateSDFs");
    for(int i = 0; i < nColiders; i++){
        if(colliders[i]->shape->grid == nullptr){
            GenerateShapeSDF(colliders[i]->shape);
        }
    }
}

////////////////////////
// ColliderSet 
///////////////////////

__bidevice__ ColliderSet3::ColliderSet3(){ nColiders = 0; colliders = nullptr; }

__bidevice__ void ColliderSet3::Initialize(Collider3 **colls, int count){
    colliders = colls;
    nColiders = count;
}

__bidevice__ bool ColliderSet3::IsActive(int which){
    if(which >= 0 && which < nColiders){
        return colliders[which]->IsActive();
    }

    return false;
}

__host__ void ColliderSet3::SetActive(int which, bool active){
    if(which >= 0 && which < nColiders){
        colliders[which]->SetActive(active);
    }
}

__bidevice__ bool ColliderSet3::ResolveCollision(Float radius, Float restitutionCoefficient,
                                                 vec3f *position, vec3f *velocity)
{
    AssertA(nColiders > 0, "No Colliders present for ColliderSet::ResolveCollision");
    int targetCollider = -1;
    Float minDistance = Infinity;
    for(int i = 0; i < nColiders; i++){
        if(colliders[i]->OptmizedClosestPointCheck(*position)){
            Float distance = Absf(colliders[i]->shape->ClosestDistance(*position));
            if(distance < minDistance){
                targetCollider = i;
                minDistance = distance;
            }
        }
    }

    // colliders were reject by optmization, this particle
    // will not collide with anything
    if(targetCollider < 0){
        return false;
    }

    AssertA(targetCollider >= 0 && targetCollider < nColiders, 
            "Invalid collider id for ColliderSet2::ResolveCollision");
    return colliders[targetCollider]->ResolveCollision(radius, restitutionCoefficient, 
                                                       position, velocity);
}

__host__ void ColliderSet3::GenerateSDFs(){
    AssertA(nColiders > 0, "No Colliders present for ColliderSet2::GenerateSDFs");
    for(int i = 0; i < nColiders; i++){
        if(colliders[i]->shape->grid == nullptr){
            GenerateShapeSDF(colliders[i]->shape);
        }
    }
}


////////////////////////
// ColliderSet2 Builder
///////////////////////

__host__ ColliderSetBuilder2::ColliderSetBuilder2(){ setReference = nullptr; }

__host__ void ColliderSetBuilder2::AddCollider2(Collider2 *collider){
    colliders.push_back(collider);
}

__host__ void ColliderSetBuilder2::AddCollider2(Shape2 *shape, Float frictionCoef){
    colliders.push_back(MakeCollider2(shape, frictionCoef));
}

__host__ ColliderSet2 *ColliderSetBuilder2::GetColliderSet(){
    AssertA(colliders.size() > 0, "No colliders given for {GetColliderSet}");
    setReference = cudaAllocateVx(ColliderSet2, 1);
    Collider2 **colliderList = cudaAllocateVx(Collider2*, colliders.size());
    for(int i = 0; i < colliders.size(); i++){
        colliderList[i] = colliders[i];
    }
    
    setReference->Initialize(colliderList, colliders.size());
    printf("Initialized collider list with #%ld shapes\n", colliders.size());
    return setReference;
}

////////////////////////
// ColliderSet Builder
///////////////////////

__host__ ColliderSetBuilder3::ColliderSetBuilder3(){ setReference = nullptr; }

__host__ void ColliderSetBuilder3::AddCollider3(Collider3 *collider){
    colliders.push_back(collider);
}

__host__ void ColliderSetBuilder3::AddCollider3(Shape *shape, Float frictionCoef){
    colliders.push_back(MakeCollider3(shape, frictionCoef));
}

__host__ ColliderSet3 *ColliderSetBuilder3::GetColliderSet(){
    AssertA(colliders.size() > 0, "No colliders given for {GetColliderSet}");
    setReference = cudaAllocateVx(ColliderSet3, 1);
    Collider3 **colliderList = cudaAllocateVx(Collider3*, colliders.size());
    for(int i = 0; i < colliders.size(); i++){
        colliderList[i] = colliders[i];
    }
    
    setReference->Initialize(colliderList, colliders.size());
    printf("Initialized collider list with #%ld shapes\n", colliders.size());
    return setReference;
}

////////////////
// Utility 
///////////////

__host__ Collider2 *MakeCollider2(Shape2 *shape, Float frictionCoef){
    Collider2 *collider = cudaAllocateVx(Collider2, 1);
    collider->Initialize(shape, frictionCoef);
    return collider;
}

__host__ Collider3 *MakeCollider3(Shape *shape, Float frictionCoef){
    Collider3 *collider = cudaAllocateVx(Collider3, 1);
    collider->Initialize(shape, frictionCoef);
    if(shape->type == ShapeMesh){
        // make sure this mesh has SDF otherwise we can't collide
        GenerateShapeSDF(shape);
    }
    return collider;
}
