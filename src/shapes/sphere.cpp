#include <shape.h>
#include <collider.h>
#include <sstream>

Shape *MakeSphere(const Transform &toWorld, Float radius, bool reverseOrientation){
    Shape *shape = cudaAllocateVx(Shape, 1);
    shape->InitSphere(toWorld, radius, reverseOrientation);
    return shape;
}

std::string Shape::SphereSerialize() const{
    std::stringstream ss;

    ss << "ShapeBegin\n";
    ss << "\t\"Type\" sphere" << std::endl;
    ss << "\t\"Radius\" " << radius << std::endl;
    ss << "\t\"Transform\" ";
    for(int i = 0; i < 4; i++){
        for(int j = 0; j < 4; j++){
            if(i == 3 && j == 3){
                ss << ObjectToWorld.m.m[i][j];
            }else{
                ss << ObjectToWorld.m.m[i][j] << " ";
            }
        }
    }

    ss << std::endl;
    ss << "ShapeEnd";
    return ss.str();
}

bb_cpu_gpu void Shape::InitSphere(const Transform &toWorld, Float rad,
                                    bool reverseOr)
{
    ObjectToWorld = toWorld;
    WorldToObject = Inverse(toWorld);
    reverseOrientation = reverseOr;
    radius = rad;
    type = ShapeType::ShapeSphere;
}

bb_cpu_gpu Bounds3f Shape::SphereGetBounds(){
    return ObjectToWorld(Bounds3f(vec3f(-radius), vec3f(radius)));
}

bb_cpu_gpu Float Shape::SphereClosestDistance(const vec3f &point) const{
    vec3f pLocal = WorldToObject.Point(point);
    return Distance(pLocal, vec3f(0)) - radius;
}

bb_cpu_gpu void Shape::SphereClosestPoint(const vec3f &point,
                                            ClosestPointQuery *query) const
{
    Float d = SphereClosestDistance(point);
    vec3f pLocal = WorldToObject.Point(point);
    Normal3f N(0,1,0);
    vec3f p;
    if(!pLocal.IsZeroVector()){
        N = Normalize(pLocal);
    }

    vec3f pN = vec3f(N.x, N.y, N.z) * radius;
    if(reverseOrientation){
        N *= -1;
    }

    p = ObjectToWorld.Point(pN);
    *query = ClosestPointQuery(p, ObjectToWorld.Normal(N), d, VelocityAt(p));
}

bb_cpu_gpu bool Shape::SphereIntersect(const Ray &ray, SurfaceInteraction *isect,
                                         Float *tShapeHit) const
{
    vec3f pHit;
    Ray r = WorldToObject(ray);
    Float ox = r.o.x, oy = r.o.y, oz = r.o.z;
    Float dx = r.d.x, dy = r.d.y, dz = r.d.z;
    Float a = dx * dx + dy * dy + dz * dz;
    Float b = 2 * (ox * dx + oy * dy + oz * dz);
    Float c = ox * ox + oy * oy + oz * oz - radius * radius;

    Float t0, t1;
    if(!Quadratic(a, b, c, &t0, &t1)) return false;

    if(t0 > r.tMax || t1 <= 0) return false;
    Float tHit = t0;
    if(tHit <= 0 || IsUnsafeHit(tHit) || IsNaN(tHit)){
        tHit = t1;
        if(tHit > r.tMax || IsNaN(tHit)) return false;
    }

    pHit = r(tHit);
    pHit *= radius / Distance(pHit, vec3f(0));
    vec3f nHit = pHit / radius;
    vec3f pError = gamma(5) * Abs(pHit);

    *isect = ObjectToWorld(SurfaceInteraction(pHit, nHit, pError, this));
    if(reverseOrientation) isect->n *= -1;
    *tShapeHit = tHit;

    return true;
}
