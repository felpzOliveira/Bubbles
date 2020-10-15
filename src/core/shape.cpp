#include <shape.h>
#include <interaction.h>
#include <collider.h>

/*************************************************************/
//                   2 D    S H A P E S                      //
/*************************************************************/
__bidevice__ Shape2::Shape2(const Transform2 &toWorld, bool reverseOrientation)
: ObjectToWorld(toWorld), WorldToObject(Inverse(toWorld)), 
reverseOrientation(reverseOrientation){}

__bidevice__ bool Shape2::IsInside(const vec2f &point) const{
    return reverseOrientation == !(ClosestDistance(point) < 0);
}

__bidevice__ Float Shape2::SignedDistance(const vec2f &point) const{
    Float d = ClosestDistance(point);
    if(IsInside(point)) return -Absf(d);
    return Absf(d);
}

__bidevice__ Bounds2f Shape2::GetBounds(){
    switch(type){
        case ShapeType::ShapeSphere2:{
            return Sphere2GetBounds();
        } break;
        
        case ShapeType::ShapeRectangle2:{
            return Rectangle2GetBounds();
        } break;
        
        default:{
            printf("Unknown shape for Shape2::GetBounds\n");
            return Bounds2f();
        }
    }
}

__bidevice__ bool Shape2::Intersect(const Ray2 &ray, SurfaceInteraction2 *isect,
                                    Float *tShapeHit) const
{
    switch(type){
        case ShapeType::ShapeSphere2:{
            return Sphere2Intersect(ray, isect, tShapeHit);
        } break;
        
        case ShapeType::ShapeRectangle2:{
            return Rectangle2Intersect(ray, isect, tShapeHit);
        } break;
        
        default:{
            printf("Unknown shape for Shape2::Intersect\n");
            return false;
        }
    }
}

__bidevice__ Float Shape2::ClosestDistance(const vec2f &point) const{
    switch(type){
        case ShapeType::ShapeSphere2:{
            return Sphere2ClosestDistance(point);
        } break;
        
        case ShapeType::ShapeRectangle2:{
            return Rectangle2ClosestDistance(point);
        } break;
        
        default:{
            printf("Unknown shape for Shape2::ClosestDistance\n");
            return Infinity;
        }
    }
}

__bidevice__ void Shape2::ClosestPoint(const vec2f &point, 
                                       ClosestPointQuery2 *query) const
{
    switch(type){
        case ShapeType::ShapeSphere2:{
            return Sphere2ClosestPoint(point, query);
        } break;
        
        case ShapeType::ShapeRectangle2:{
            return Rectangle2ClosestPoint(point, query);
        } break;
        
        default:{
            printf("Unknown shape for Shape2::ClosestPoint\n");
        }
    }
}


/*************************************************************/
//                   3 D    S H A P E S                      //
/*************************************************************/
__bidevice__ Shape::Shape(const Transform &toWorld, bool reverseOrientation) :
ObjectToWorld(toWorld), WorldToObject(Inverse(toWorld)),
reverseOrientation(reverseOrientation){}

__bidevice__ bool Shape::CanSolveSdf() const{
    switch(type){
        case ShapeType::ShapeSphere:{
            return true;
        } break;
        
        case ShapeType::ShapeBox:{
            return true;
        } break;
        
        case ShapeType::ShapeMesh:{
            return false;
        } break;
        
        default:{
            printf("Unknown shape for Shape::CanSolveSdf\n");
            return false;
        }
    }
}

__bidevice__ Bounds3f Shape::GetBounds(){
    switch(type){
        case ShapeType::ShapeSphere:{
            return SphereGetBounds();
        } break;
        
        case ShapeType::ShapeBox:{
            return BoxGetBounds();
        } break;
        
        case ShapeType::ShapeMesh:{
            return MeshGetBounds();
        } break;
        
        default:{
            printf("Unknown shape for Shape::GetBounds\n");
            return Bounds3f();
        }
    }
}

__bidevice__ bool Shape::Intersect(const Ray &ray, SurfaceInteraction *isect,
                                   Float *tShapeHit) const
{
    switch(type){
        case ShapeType::ShapeSphere:{
            return SphereIntersect(ray, isect, tShapeHit);
        } break;
        
        case ShapeType::ShapeBox:{
            return BoxIntersect(ray, isect, tShapeHit);
        } break;
        
        case ShapeType::ShapeMesh:{
            return MeshIntersect(ray, isect, tShapeHit);
        } break;
        
        default:{
            printf("Unknown shape for Shape::Intersect\n");
            return false;
        }
    }
}

__bidevice__ Float Shape::ClosestDistance(const vec3f &point) const{
    switch(type){
        case ShapeType::ShapeSphere:{
            return SphereClosestDistance(point);
        } break;
        
        case ShapeType::ShapeBox:{
            return BoxClosestDistance(point);
        } break;
        
        case ShapeType::ShapeMesh:{
            return MeshClosestDistance(point);
        } break;
        
        default:{
            printf("Unknown shape for Shape::ClosestDistance\n");
            return Infinity;
        }
    }
}

__bidevice__ void Shape::ClosestPoint(const vec3f &point, 
                                      ClosestPointQuery *query) const
{
    switch(type){
        case ShapeType::ShapeSphere:{
            SphereClosestPoint(point, query);
        } break;
        
        case ShapeType::ShapeBox:{
            return BoxClosestPoint(point, query);
        } break;
        
        case ShapeType::ShapeMesh:{
            MeshClosestPoint(point, query);
        } break;
        
        default:{
            printf("Unknown shape for Shape::ClosestPoint\n");
        }
    }
}

__bidevice__ bool Shape::IsInside(const vec3f &point) const{
    return reverseOrientation == !(ClosestDistance(point) < 0);
}

__bidevice__ Float Shape::SignedDistance(const vec3f &point) const{
    Float d = ClosestDistance(point);
    if(IsInside(point)) return -Absf(d);
    return Absf(d);
}
