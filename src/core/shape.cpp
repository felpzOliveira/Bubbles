#include <shape.h>
#include <interaction.h>
#include <collider.h>
#include <sstream>

/*************************************************************/
//                   2 D    S H A P E S                      //
/*************************************************************/
bb_cpu_gpu Shape2::Shape2(const Transform2 &toWorld, bool reverseOrientation)
: ObjectToWorld(toWorld), WorldToObject(Inverse(toWorld)),
reverseOrientation(reverseOrientation), linearVelocity(vec2f(0)), angularVelocity(0){}

bb_cpu_gpu bool Shape2::IsInside(const vec2f &point) const{
    if(grid){
        if(grid->Filled()){
            return (grid->Sample(point) < 0);
        }
    }
    return reverseOrientation == !(ClosestDistance(point) < 0);
}

bb_cpu_gpu Float Shape2::SignedDistance(const vec2f &point) const{
    Float d = ClosestDistance(point);
    if(IsInside(point)) return -Absf(d);
    return Absf(d);
}

bb_cpu_gpu Bounds2f Shape2::GetBounds(){
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

bb_cpu_gpu bool Shape2::Intersect(const Ray2 &ray, SurfaceInteraction2 *isect,
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

bb_cpu_gpu Float Shape2::ClosestDistance(const vec2f &point) const{
    switch(type){
        case ShapeType::ShapeSphere2:{
            return Sphere2ClosestDistance(point);
        } break;

        case ShapeType::ShapeRectangle2:{
            return Rectangle2ClosestDistance(point);
        } break;

        case ShapeType::ShapeSDF:{
            ClosestPointQuery2 query;
            ClosestPointBySDF(point, &query);
            return query.signedDistance;
        } break;

        default:{
            printf("Unknown shape for Shape2::ClosestDistance\n");
            return Infinity;
        }
    }
}

bb_cpu_gpu void Shape2::ClosestPoint(const vec2f &point,
                                       ClosestPointQuery2 *query) const
{
    if(grid){
        ClosestPointBySDF(point, query);
    }else{
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
}

bb_cpu_gpu void Shape2::SetVelocities(const vec2f &vel, const Float &angular){
    linearVelocity = vel;
    angularVelocity = angular;
}

void Shape2::Update(const Transform2 &toWorld){
    ObjectToWorld = toWorld;
    WorldToObject = Inverse(toWorld);
}

bb_cpu_gpu vec2f Shape2::VelocityAt(const vec2f &point) const{
    vec2f translation = Translation(ObjectToWorld.m);
    vec2f p = point - translation;
    vec2f angularVel = angularVelocity * vec2f(-p.y, p.x);
    return linearVelocity + angularVel;
}

/*************************************************************/
//                   3 D    S H A P E S                      //
/*************************************************************/
bb_cpu_gpu Shape::Shape(const Transform &toWorld, bool reverseOrientation) :
ObjectToWorld(toWorld), WorldToObject(Inverse(toWorld)),
reverseOrientation(reverseOrientation), linearVelocity(vec3f(0)), angularVelocity(vec3f(0)){}

bb_cpu_gpu bool Shape::CanSolveSdf() const{
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

        case ShapeType::ShapeSDF:{
            return true;
        } break;

        default:{
            printf("Unknown shape for Shape::CanSolveSdf\n");
            return false;
        }
    }
}

bb_cpu_gpu Bounds3f Shape::GetBounds(){
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

        case ShapeType::ShapeSDF:{
            return bounds;
        } break;

        default:{
            printf("Unknown shape for Shape::GetBounds\n");
            return Bounds3f();
        }
    }
}

bb_cpu_gpu bool Shape::Intersect(const Ray &ray, SurfaceInteraction *isect,
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

bb_cpu_gpu Float Shape::ClosestDistance(const vec3f &point) const{
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

        case ShapeType::ShapeSDF:{
            return grid->Sample(point);
            //ClosestPointQuery query;
            //ClosestPointBySDF(point, &query);
            //return query.signedDistance;
        } break;

        default:{
            printf("Unknown shape for Shape::ClosestDistance\n");
            return Infinity;
        }
    }
}

bb_cpu_gpu void Shape::ClosestPoint(const vec3f &point,
                                      ClosestPointQuery *query) const
{
    if(grid){
        ClosestPointBySDF(point, query);
    }else{
        switch(type){
            case ShapeType::ShapeSphere:{
                SphereClosestPoint(point, query);
            } break;

            case ShapeType::ShapeBox:{
                return BoxClosestPoint(point, query);
            } break;

            default:{
                printf("Unknown shape for Shape::ClosestPoint\n");
            }
        }
    }
}

std::string Shape::Serialize() const{
    switch(type){
        case ShapeType::ShapeSphere:{
            return SphereSerialize();
        } break;
        case ShapeType::ShapeBox:{
            return BoxSerialize();
        } break;
        case ShapeType::ShapeMesh:{
            return MeshSerialize();
        } break;

        default:{
            printf("Unsupported shape serialization\n");
            return std::string();
        }
    }
}

bb_cpu_gpu bool Shape::IsInside(const vec3f &point) const{
    if(grid){
        if(grid->Filled()){
            return (grid->Sample(point) < 0);
        }
    }
    return reverseOrientation == !(ClosestDistance(point) < 0);
}

bb_cpu_gpu Float Shape::SignedDistance(const vec3f &point) const{
    Float d = ClosestDistance(point);
    if(IsInside(point)) return -Absf(d);
    return Absf(d);
}

bb_cpu_gpu void Shape::SetVelocities(const vec3f &vel, const vec3f &angular){
    linearVelocity = vel;
    angularVelocity = angular;
}

void Shape::Update(const Transform &toWorld){
    ObjectToWorld = toWorld;
    WorldToObject = Inverse(toWorld);
}

bb_cpu_gpu vec3f Shape::VelocityAt(const vec3f &point) const{
    vec3f translation = Translation(ObjectToWorld.m);
    vec3f p = point - translation;
    vec3f angularVel = Cross(angularVelocity, p);
    return linearVelocity + angularVel;
}

std::string Shape::MeshSerialize() const{
    std::stringstream ss;
    Transform transform = mesh->transform;
    ss << "ShapeBegin\n";
    ss << "\t\"Type\" mesh" << std::endl;
    ss << "\t\"Name\" " << std::string(mesh->name) << std::endl;
    ss << "\t\"Transform\" ";
    for(int i = 0; i < 4; i++){
        for(int j = 0; j < 4; j++){
            if(i == 3 && j == 3){
                ss << transform.m.m[i][j];
            }else{
                ss << transform.m.m[i][j] << " ";
            }
        }
    }

    ss << std::endl;
    ss << "ShapeEnd";
    return ss.str();
}

/*
* Compute a ray direction from a point and a bounding box being sampled,
* all computations must be performed centered at the origin.
*/
bb_cpu_gpu vec3f GenerateMinimalRayDirection(const vec3f &origin, const Bounds3f &bound){
    vec3f dir(0);
    Float dx = Absf(bound.ExtentOn(0) * 0.5f - origin.x);
    Float dy = Absf(bound.ExtentOn(1) * 0.5f - origin.y);
    Float dz = Absf(bound.ExtentOn(2) * 0.5f - origin.z);
    vec4f ref[] = {
        vec4f(dx, 1, 0, 0),
        vec4f(dy, 0, 1, 0),
        vec4f(dz, 0, 0, 1),
        vec4f(Absf(bound.ExtentOn(0) - dx), -1, 0, 0),
        vec4f(Absf(bound.ExtentOn(1) - dy), 0, -1, 0),
        vec4f(Absf(bound.ExtentOn(2) - dz), 0, 0, -1)
    };

    Float mind = Infinity;
    for(int i = 0; i < 6; i++){
        if(mind > ref[i].x){
            dir = vec3f(ref[i].y, ref[i].z, ref[i].w);
            mind = ref[i].x;
        }
    }

    return dir;
}

bb_cpu_gpu bool MeshIsPointInside(const vec3f &point, Shape *meshShape,
                                  const Bounds3f &bounds)
{
    int hits = 0;
    bool hit_anything = false;
    vec3f direction = GenerateMinimalRayDirection(point, bounds);
    Ray ray(point, direction);

    do{
        SurfaceInteraction isect;
        Float tHit = 0;
        hit_anything = meshShape->Intersect(ray, &isect, &tHit);
        if(hit_anything){
            ray = SpawnRayInDirection(isect, ray.d);
            hits++;
        }
    }while(hit_anything);

    return (hits % 2 != 0 && hits > 0);
}

bb_cpu_gpu bool MeshIsPointInsideBounded(const vec3f &point, Shape *meshShape,
                                         const Bounds3f &bounds, Bounds3f cutBounds)
{
    int hits = 0;
    bool hit_anything = false;
    vec3f direction = GenerateMinimalRayDirection(point, bounds);
    Ray ray(point, direction);

    do{
        SurfaceInteraction isect;
        Float tHit = 0;
        hit_anything = meshShape->Intersect(ray, &isect, &tHit);
        if(hit_anything){
            if(!Inside(isect.p, cutBounds))
                hits++;

            ray = SpawnRayInDirection(isect, ray.d);
        }
    }while(hit_anything);

    return (hits % 2 != 0 && hits > 0);
}

bb_cpu_gpu void SetNodeSDFKernel(FieldGrid2f *grid, Shape2 *shape, int i){
    vec2ui u = DimensionalIndex(i, grid->resolution, 2);
    vec2f p = grid->GetDataPosition(u);
    Float d = shape->ClosestDistance(p);
    bool interior = shape->reverseOrientation == !(d < 0);
    Float psd = Max(Absf(d), 0.00001);
    grid->SetValueAt(interior ? -psd : psd, u);
}

bb_cpu_gpu void SetNodeSDFKernel(FieldGrid3f *grid, Shape *shape, int i){
    vec3ui u = DimensionalIndex(i, grid->resolution, 3);
    vec3f p = grid->GetDataPosition(u);
    Float d = shape->ClosestDistance(p);
    bool interior = false;
    /* For meshes we need to perform a full BVH query, for others we can check orientation*/
    if(shape->type == ShapeType::ShapeMesh)
        interior = MeshIsPointInside(p, shape, shape->GetBounds());
    else
        interior = shape->IsInside(p);

    Float psd = Max(Absf(d), 0.00001);
    grid->SetValueAt(interior ? -psd : psd, u);
}

bb_kernel void CreateShapeSDFGPU(Shape *shape){
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if(i < shape->grid->total){
        SetNodeSDFKernel(shape->grid, shape, i);
    }
}

bb_kernel void CreateShapeSDFGPU2D(Shape2 *shape){
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if(i < shape->grid->total){
        SetNodeSDFKernel(shape->grid, shape, i);
    }
}

bb_cpu_gpu bool MeshShapeIsPointInside(Shape *meshShape, const vec3f &p,
Float radius, Float offset)
{
    if(meshShape == nullptr) return false;
    if(meshShape->grid == nullptr) return false;
    ClosestPointQuery colliderPoint;
    meshShape->ClosestPoint(p, &colliderPoint);
    return Absf(colliderPoint.signedDistance) < radius + offset;
}

void GenerateShapeSDF(Shape2 *shape, Float dx, Float margin){
    //TODO: Update grid if already exists

    int resolution = 0;
    Bounds2f bounds = shape->GetBounds();
    vec2f scale(bounds.ExtentOn(0), bounds.ExtentOn(1));
    bounds.pMin -= margin * scale;
    bounds.pMax += margin * scale;

    Float width = bounds.ExtentOn(0);
    Float height = bounds.ExtentOn(1);

    resolution = (int)std::ceil(width / dx);
    dx = width / (Float)resolution;
    int resolutionY = (int)std::ceil(resolution * height / width);

    shape->grid = cudaAllocateVx(FieldGrid2f, 1);
    shape->grid->Build(vec2ui(resolution, resolutionY), vec2f(dx),
    bounds.pMin, VertexCentered);

    printf("Generating SDF for shape: [%d x %d] ... ",
    resolution, resolutionY);

    GPULaunch(shape->grid->total, CreateShapeSDFGPU2D, shape);
    shape->grid->MarkFilled();
    printf("OK\n");
}

void GenerateShapeSDF(Shape *shape, Float dx, Float margin){
    //TODO: Update grid if already exists
    int resolution = 0;
    //dx = 0.005;
    Bounds3f bounds = shape->GetBounds();
    vec3f scale(bounds.ExtentOn(0), bounds.ExtentOn(1), bounds.ExtentOn(2));
    bounds.pMin -= margin * scale;
    bounds.pMax += margin * scale;

    Float width = bounds.ExtentOn(0);
    Float height = bounds.ExtentOn(1);
    Float depth = bounds.ExtentOn(2);

    resolution = (int)std::ceil(width / dx);

    dx = width / (Float)resolution;
    int resolutionY = (int)std::ceil(resolution * height / width);
    int resolutionZ = (int)std::ceil(resolution * depth / width);

    shape->grid = cudaAllocateVx(FieldGrid3f, 1);
    shape->grid->Build(vec3ui(resolution, resolutionY, resolutionZ), vec3f(dx),
                              bounds.pMin, VertexCentered);

    printf("Generating SDF for shape %s: [%d x %d x %d] ... ",
            shape->mesh->name, resolution, resolutionY, resolutionZ);

    GPULaunch(shape->grid->total, CreateShapeSDFGPU, shape);
    shape->grid->MarkFilled();
    printf("OK\n");
}

