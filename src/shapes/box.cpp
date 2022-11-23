#include <shape.h>
#include <transform.h>
#include <collider.h>
#include <sstream>

__host__ Shape *MakeBox(const Transform &toWorld, const vec3f &size,
                        bool reverseOrientation)
{
    Shape *shape = cudaAllocateVx(Shape, 1);
    shape->InitBox(toWorld, size.x, size.y, size.z, reverseOrientation);
    return shape;
}

__bidevice__ Plane3::Plane3(){}
__bidevice__ Plane3::Plane3(const vec3f &p, const Normal3f &n){
    Set(p, n);
}

__bidevice__ void Plane3::Set(const vec3f &p, const Normal3f &n){
    point = p;
    normal = n;
}

__bidevice__ vec3f Plane3::ClosestPoint(const vec3f &p) const{
    vec3f r = p - point;
    vec3f vn = ToVec3(normal);
    return r - Dot(r, vn) * vn + point;
}


__host__ void Shape::InitBox(const Transform &toWorld, Float sx, Float sy,
                             Float sz, bool reverseOr)
{
    type = ShapeType::ShapeBox;
    ObjectToWorld = toWorld;
    WorldToObject = Inverse(toWorld);
    reverseOrientation = reverseOr;
    sizex = sx; sizey = sy; sizez = sz;
}

__host__ std::string Shape::BoxSerialize() const{
    std::stringstream ss;

    ss << "ShapeBegin\n";
    ss << "\t\"Type\" box" << std::endl;
    ss << "\t\"Length\" " << sizex << " " << sizey << " " << sizez << std::endl;
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

__bidevice__ Bounds3f Shape::BoxGetBounds(){
    Float hx = sizex/2.0;
    Float hy = sizey/2.0;
    Float hz = sizez/2.0;
    return ObjectToWorld(Bounds3f(vec3f(-hx, -hy, -hz), vec3f(hx, hy, hz)));
}

__bidevice__ bool Shape::BoxIntersect(const Ray &ray, SurfaceInteraction *isect,
                                      Float *tShapeHit) const
{
    // TODO
    printf("Warning: Called unimplemented method\n");
    return false;
}

__bidevice__ Float Shape::BoxClosestDistance(const vec3f &point) const{
    vec3f pLocal = WorldToObject.Point(point);
    Float hx = sizex/2.0;
    Float hy = sizey/2.0;
    Float hz = sizez/2.0;
    Bounds3f localBound(vec3f(-hx, -hy, -hz), vec3f(hx, hy, hz));
    vec3f closest;
    int isNeg = 0;

    if(Inside(pLocal, localBound)){
        Plane3 planes[6] = {
            Plane3(localBound.pMax, Normal3f(1, 0, 0)),
            Plane3(localBound.pMax, Normal3f(0, 1, 0)),
            Plane3(localBound.pMax, Normal3f(0, 0, 1)),
            Plane3(localBound.pMin, Normal3f(-1, 0, 0)),
            Plane3(localBound.pMin, Normal3f(0, -1, 0)),
            Plane3(localBound.pMin, Normal3f(0, 0, -1))
        };

        closest = planes[0].ClosestPoint(pLocal);
        Float distanceSquared = (closest - pLocal).LengthSquared();
        for(int i = 1; i < 6; i++){
            vec3f local = planes[i].ClosestPoint(pLocal);
            Float localDistSquared = (local - pLocal).LengthSquared();
            if(localDistSquared < distanceSquared){
                closest = local;
                distanceSquared = localDistSquared;
            }
        }
        isNeg = 1;
    }else{
        closest = Clamp(pLocal, localBound.pMin, localBound.pMax);
    }

    return isNeg ? -Distance(pLocal, closest) : Distance(pLocal, closest);
}

__bidevice__ void Shape::BoxClosestPoint(const vec3f &point,
                                         ClosestPointQuery *query) const
{
    vec3f pLocal = WorldToObject.Point(point);
    Float hx = sizex/2.0;
    Float hy = sizey/2.0;
    Float hz = sizez/2.0;
    Bounds3f localBound(vec3f(-hx, -hy, -hz), vec3f(hx, hy, hz));
    vec3f closest;
    Normal3f normal;
    int isNeg = 0;

    Plane3 planes[6] = {
        Plane3(localBound.pMax, Normal3f(1, 0, 0)),
        Plane3(localBound.pMax, Normal3f(0, 1, 0)),
        Plane3(localBound.pMax, Normal3f(0, 0, 1)),
        Plane3(localBound.pMin, Normal3f(-1, 0, 0)),
        Plane3(localBound.pMin, Normal3f(0, -1, 0)),
        Plane3(localBound.pMin, Normal3f(0, 0, -1))
    };

    normal = planes[0].normal;
    if(Inside(pLocal, localBound)){
        closest = planes[0].ClosestPoint(pLocal);
        Float distanceSquared = (closest - pLocal).LengthSquared();
        for(int i = 1; i < 6; i++){
            vec3f local = planes[i].ClosestPoint(pLocal);
            Float localDistSquared = (local - pLocal).LengthSquared();
            if(localDistSquared < distanceSquared){
                closest = local;
                normal = planes[i].normal;
                distanceSquared = localDistSquared;
            }
        }
        isNeg = 1;
    }else{
        closest = Clamp(pLocal, localBound.pMin, localBound.pMax);
        vec3f cov = pLocal - closest;
        Float maxCosine = Dot(cov, ToVec3(normal));

        for(int i = 1; i < 6; i++){
            Float cosine = Dot(ToVec3(planes[i].normal), cov);
            if(cosine > maxCosine){
                normal = planes[i].normal;
                maxCosine = cosine;
            }
        }
    }

    Float distance = Distance(closest, pLocal);

    if(isNeg){
        distance *= -1;
    }

    if(reverseOrientation){
        normal *= -1;
    }

    *query = ClosestPointQuery(ObjectToWorld.Point(closest),
                               ObjectToWorld.Normal(normal), distance);
}
