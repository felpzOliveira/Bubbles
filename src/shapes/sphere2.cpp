#include <shape.h>
#include <cutil.h>
#include <collider.h>

__host__ Shape2 *MakeSphere2(const Transform2 &toWorld, Float radius,
                             bool reverseOrientation)
{
    Shape2 *shape = cudaAllocateVx(Shape2, 1);
    shape->InitSphere2(toWorld, radius, reverseOrientation);
    return shape;
}

__bidevice__ void Shape2::InitSphere2(const Transform2 &toWorld, Float rad, bool reverseOr){
    ObjectToWorld = toWorld;
    WorldToObject = Inverse(toWorld);
    reverseOrientation = reverseOr;
    radius = rad;
    type = ShapeType::ShapeSphere2;
}

__bidevice__ Bounds2f Shape2::Sphere2GetBounds(){
    return ObjectToWorld(Bounds2f(vec2f(-radius, -radius), vec2f(radius, radius)));
}

__bidevice__ bool Shape2::Sphere2Intersect(const Ray2 &ray, SurfaceInteraction2 *isect,
                                           Float *tShapeHit) const
{
    vec2f pHit;
    Ray2 r = WorldToObject(ray);
    Float ox = r.o.x, oy = r.o.y;
    Float dx = r.d.x, dy = r.d.y;
    
    Float a = dx * dx + dy * dy;
    Float b = 2 * (ox * dx + oy * dy);
    Float c = ox * ox + oy * oy - radius * radius;
    
    Float t0, t1;
    if(!Quadratic(a, b, c, &t0, &t1)) return false;
    
    if(t0 > r.tMax || t1 <= 0) return false;
    Float tHit = t0;
    if(tHit <= 0 || IsUnsafeHit(tHit)){
        tHit = t1;
        if(tHit > r.tMax) return false;
    }
    
    pHit = r(tHit);
    pHit *= radius / Distance(pHit, vec2f(0, 0));
    vec2f nHit = pHit / radius;
    vec2f pError = gamma(5) * Abs(pHit);
    *isect = ObjectToWorld(SurfaceInteraction2(pHit, nHit, pError, this));
    if(reverseOrientation) isect->n *= -1;
    *tShapeHit = tHit;
    
    return true;
}

__bidevice__ Float Shape2::Sphere2ClosestDistance(const vec2f &point) const{
    vec2f pLocal = WorldToObject.Point(point);
    return Distance(pLocal, vec2f(0, 0)) - radius;
}

__bidevice__ void Shape2::Sphere2ClosestPoint(const vec2f &point, 
                                              ClosestPointQuery2 *query) const
{
    Float d = Absf(Sphere2ClosestDistance(point));
    vec2f pLocal = WorldToObject.Point(point);
    vec2f N(0, 1);
    if(!pLocal.IsZeroVector()){
        N = Normalize(pLocal);
    }
    
    vec2f pN = N * radius;
    
    if(reverseOrientation){
        N *= -1;
    }
    
    *query = ClosestPointQuery2(ObjectToWorld.Point(pN), 
                                ObjectToWorld.Vector(N), d);
}
