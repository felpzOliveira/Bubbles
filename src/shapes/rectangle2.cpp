#include <shape.h>
#include <collider.h>
#include <interaction.h>
#include <cutil.h>

__host__ Shape2 *MakeRectangle2(const Transform2 &toWorld, vec2f extension,
                                bool reverseOrientation)
{
    Shape2 *shape = cudaAllocateVx(Shape2, 1);
    shape->InitRectangle2(toWorld, extension, reverseOrientation);
    return shape;
}

__bidevice__ void Shape2::InitRectangle2(const Transform2 &toWorld, vec2f extension,
                                         bool reverseOr)
{
    vec2f half = extension * 0.5;
    ObjectToWorld = toWorld;
    WorldToObject = Inverse(toWorld);
    reverseOrientation = reverseOr;
    type = ShapeType::ShapeRectangle2;
    rect = Bounds2f(vec2f(-half.x, -half.y), vec2f(half.x, half.y));
}

__bidevice__ Bounds2f Shape2::Rectangle2GetBounds(){ return ObjectToWorld(rect); }

__bidevice__ bool Shape2::Rectangle2Intersect(const Ray2 &ray, SurfaceInteraction2 *isect,
                                              Float *tShapeHit) const
{
    vec2f pHit;
    Ray2 r = WorldToObject(ray);
    Float t0 = 0, t1 = 0;
    int it = 0;
    Float mind = Infinity;
    if(!rect.Intersect(r, &t0, &t1)) return false;
    
    pHit = r(t0);
    
    // I'm not proud of this
    Float dd[4] = {
        Absf(rect.pMax.x - pHit.x),
        Absf(rect.pMin.x - pHit.x),
        Absf(rect.pMax.y - pHit.y),
        Absf(rect.pMin.y - pHit.y)
    };
    
    vec2f nd[4] = { vec2f(1, 0), vec2f(-1, 0), vec2f(0, 1), vec2f(0, -1) };
    
    for(int i = 0; i < 4; i++){
        if(mind > dd[i]){
            mind = dd[i];
            it = i;
        }
    }
    
    vec2f pError(0);
    vec2f nHit = nd[it];
    *isect = ObjectToWorld(SurfaceInteraction2(pHit, nHit, pError, this));
    if(reverseOrientation) isect->n *= -1;
    *tShapeHit = t0;
    
    return true;
}

__bidevice__ Float Shape2::Rectangle2ClosestDistance(const vec2f &point) const{
    vec2f pLocal = WorldToObject.Point(point);
    vec2f half = vec2f(rect.ExtentOn(0), rect.ExtentOn(1)) * 0.5;
    vec2f pRef = Abs(pLocal) - half;
    return Max(pRef, vec2f(0,0)).Length() + Min(MaxComponent(pRef), 0);
}

__bidevice__ void Shape2::Rectangle2ClosestPoint(const vec2f &point, 
                                                 ClosestPointQuery2 *query) const
{
    Float d = Absf(Rectangle2ClosestDistance(point));
    vec2f pLocal = WorldToObject.Point(point);
    vec2f half = vec2f(rect.ExtentOn(0), rect.ExtentOn(1)) * 0.5;
    vec2f closest(0,0);
    vec2f normal(0,0);
    if(Inside(pLocal, rect)){
        vec3f dd[4] = {
            vec3f(half.x - pLocal.x, 1, 0),
            vec3f(half.y - pLocal.y, 0, 1),
            vec3f(half.x + pLocal.x, -1, 0),
            vec3f(half.y + pLocal.y, 0, -1)
        };
        
        Float mind = Infinity;
        for(int i = 0; i < 4; i++){
            if(mind > dd[i].x){
                mind = dd[i].x;
                normal = vec2f(dd[i].y, dd[i].z);
                closest = pLocal + normal * mind;
            }
        }
    }else{
        vec2f q = Abs(pLocal);
        if(q.x < half.x){
            closest = vec2f(q.x, half.y);
            normal = vec2f(0, 1);
        }else if(q.y < half.y){
            closest = vec2f(half.x, q.y);
            normal = vec2f(1, 0);
        }else{
            closest = vec2f(half.x, half.y);
            normal = q.x > q.y ? vec2f(1, 0) : vec2f(0, 1);
        }
        
        if(pLocal.x < 0){
            closest.x *= -1;
            normal.x *= -1;
        }
        
        if(pLocal.y < 0){
            closest.y *= -1;
            normal.y *= -1;
        }
    }
    
    if(reverseOrientation){
        normal *= -1;
    }
    
    *query = ClosestPointQuery2(ObjectToWorld.Point(closest),
                                ObjectToWorld.Vector(normal), d);
}