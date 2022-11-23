#include <shape.h>
#include <cutil.h>
#include <statics.h>
#include <shape.h>
#include <obj_loader.h>
#include <collider.h>

typedef struct{
    Node *nodes;
    PrimitiveHandle *handles;
    int length;
    int head;
    int maxElements;
    int handleHead;
    int maxHandles;
    int skippedSorts;
    int totalSorts;
}NodeDistribution;

template<typename T, class C> inline __bidevice__
bool QuickSort(T *arr, int elements, C compare){
#define  MAX_LEVELS  1000
    int beg[MAX_LEVELS], end[MAX_LEVELS], i=0, L, R;
    T piv;
    beg[0] = 0;
    end[0] = elements;
    while(i >= 0){
        L = beg[i];
        R = end[i]-1;
        if(L < R){
            piv = arr[L];
            if(i == MAX_LEVELS-1) return false;

            while(L < R){
                while(compare(&arr[R], &piv) && L < R) R--; if (L < R) arr[L++] = arr[R];
                while(compare(&piv, &arr[L]) && L < R) L++; if (L < R) arr[R--] = arr[L];
            }

            arr[L] = piv;
            beg[i+1] = L+1;
            end[i+1] = end[i];
            end[i++] = L;
        }else{
            i--;
        }
    }

    return true;
}

__host__ Shape *MakeMesh(ParsedMesh *mesh, const Transform &toWorld,
                         bool reverseOrientation)
{
    toWorld.Mesh(mesh);
    Shape *meshShape = cudaAllocateVx(Shape, 1);
    meshShape->InitMesh(mesh, reverseOrientation);
    return meshShape;
}

__host__ Shape *MakeMesh(const char *path, const Transform &toWorld,
                         bool reverseOrientation)
{
    ParsedMesh *mesh = LoadObj(path);
    return MakeMesh(mesh, toWorld, reverseOrientation);
}

__host__ void MakeNodeDistribution(NodeDistribution *dist, int nElements,
                                   int maxDepth)
{
    Float fh = Log2(nElements);
    int h = ceil(fh);
    h = h > maxDepth ? maxDepth : h;
    int c = std::pow(2, h+1) - 1;
    int leafs = std::pow(2, h);
    long mem = sizeof(Node) * c;
    mem /= (1024 * 1024);

    printf(" * Requesting %ld Mb for nodes ...", mem);
    dist->nodes = (Node *)DefaultAllocatorMemory(sizeof(Node) * c);
    printf("OK\n");

    mem = sizeof(PrimitiveHandle) * nElements;
    mem /= (1024 * 1024);
    printf(" * Requsting %ld Mb for handles ...", mem);
    dist->handles = (PrimitiveHandle *)DefaultAllocatorMemory(sizeof(PrimitiveHandle) * nElements);
    printf("OK\n");

    dist->length = c;
    dist->head = 0;
    dist->handleHead = 0;
    dist->maxHandles = nElements;
    dist->maxElements = 0;
    dist->totalSorts = 0;
    dist->skippedSorts = 0;
}

__bidevice__ int CompareX(PrimitiveHandle *p0, PrimitiveHandle *p1){
    return p0->bound.pMin.x >= p1->bound.pMin.x ? 1 : 0;
}

__bidevice__ int CompareY(PrimitiveHandle *p0, PrimitiveHandle *p1){
    return p0->bound.pMin.y >= p1->bound.pMin.y ? 1 : 0;
}

__bidevice__ int CompareZ(PrimitiveHandle *p0, PrimitiveHandle *p1){
    return p0->bound.pMin.z >= p1->bound.pMin.z ? 1 : 0;
}

__host__ Node *GetNode(int n, NodeDistribution *nodeDist){
    if(!(nodeDist->head < nodeDist->length)){
        printf(" ** [ERROR] : Allocated %d but requested more nodes\n", nodeDist->length);
        AssertA(0, "Too many node requirement");
    }

    Node *node = &nodeDist->nodes[nodeDist->head++];
    node->left = nullptr;
    node->right = nullptr;
    node->handles = nullptr;
    node->n = n;
    node->is_leaf = 0;
    return node;
}

__host__ void NodeSetItens(Node *node, int n, NodeDistribution *dist){
    AssertA(dist->handleHead+n <= dist->maxHandles, "Too many handles requirement");
    node->n = n;
    node->handles = &dist->handles[dist->handleHead];
    dist->handleHead += n;
}

__host__ Node *BVHBuild(PrimitiveHandle *handles,int n, int depth,
                        int max_depth, NodeDistribution *distr, int last_axis=-1)
{
    Node *node = GetNode(n, distr);
    int axis = int(3 * rand_float());

    if(axis != last_axis){
        last_axis = axis;
        distr->totalSorts ++;
        if(axis == 0){
            QuickSort(handles, n, CompareX);
        }else if(axis == 1){
            QuickSort(handles, n, CompareY);
        }else if(axis == 2){
            QuickSort(handles, n, CompareZ);
        }
    }else{
        distr->skippedSorts ++;
    }

    if(n == 1){
        NodeSetItens(node, n, distr);
        memcpy(node->handles, handles, n * sizeof(PrimitiveHandle));
        node->bound = handles[0].bound;
        node->is_leaf = 1;
        if(distr->maxElements < n) distr->maxElements = n;
        return node;
    }else if(depth >= max_depth){
        NodeSetItens(node, n, distr);
        memcpy(node->handles, handles, n*sizeof(PrimitiveHandle));
        node->bound = handles[0].bound;
        for(int i = 1; i < n; i++){
            node->bound = Union(node->bound, handles[i].bound);
        }

        node->is_leaf = 1;
        if(distr->maxElements < n) distr->maxElements = n;
        return node;
    }else{
        node->left  = BVHBuild(handles, n/2, depth+1, max_depth, distr, last_axis);
        node->right = BVHBuild(&handles[n/2], n-n/2, depth+1, max_depth, distr, last_axis);
    }

    node->bound = Union(node->left->bound, node->right->bound);
    return node;
}

__host__ Node *CreateBVH(PrimitiveHandle *handles, int n, int depth,
                         int max_depth, int *totalNodes, int *maxNodes)
{
    NodeDistribution distr;
    memset(&distr, 0x00, sizeof(NodeDistribution));
    MakeNodeDistribution(&distr, n, max_depth);

    Node *root = BVHBuild(handles, n, depth, max_depth, &distr);

    *maxNodes = distr.maxElements;
    *totalNodes = distr.head;

    Float totalSorts = (Float)distr.totalSorts + (Float)distr.skippedSorts;
    Float sortReduction = 100.0f * (((Float)distr.skippedSorts) / totalSorts);
    printf(" * Sort reduction algorihtm gain: %g%%\n", sortReduction);
    return root;
}


__bidevice__ Bounds3f BVHBoundsOf(ParsedMesh *mesh, int triId){
    int i0 = mesh->indices[3 * triId + 0].x;
    int i1 = mesh->indices[3 * triId + 1].x;
    int i2 = mesh->indices[3 * triId + 2].x;

    Point3f p0 = mesh->p[i0];
    Point3f p1 = mesh->p[i1];
    Point3f p2 = mesh->p[i2];

    Bounds3f bound(p0, p1);
    bound = Union(bound, p2);
    return bound;
}

__host__ void BVHMeshTrianglesBoundsCPU(ParsedMesh *mesh, PrimitiveHandle *handles){
    for(int i = 0; i < mesh->nTriangles; i++){
        handles[i].bound  = BVHBoundsOf(mesh, i);
        handles[i].handle = i;
    }
}

__bidevice__ Float DistanceTriangle(const vec3f &p, int triNum, ParsedMesh *mesh){
    int i0 = mesh->indices[3 * triNum + 0].x;
    int i1 = mesh->indices[3 * triNum + 1].x;
    int i2 = mesh->indices[3 * triNum + 2].x;
    vec3f a = mesh->p[i0];
    vec3f b = mesh->p[i1];
    vec3f c = mesh->p[i2];

    vec3f ba = b - a; vec3f pa = p - a;
    vec3f cb = c - b; vec3f pb = p - b;
    vec3f ac = a - c; vec3f pc = p - c;

    vec3f nor = Cross(ba, ac);
    return std::sqrt(
                     Sign(Dot(Cross(ba, nor), pa)) +
                     Sign(Dot(Cross(cb, nor), pb)) +
                     Sign(Dot(Cross(ac, nor), pc)) < 2.0 ?
                     Min(Min(
                             Dot2(ba * Clamp(Dot(ba, pa)/Dot2(ba), 0.0, 1.0) - pa),
                             Dot2(cb * Clamp(Dot(cb, pb)/Dot2(cb), 0.0, 1.0) - pb) ),
                         Dot2(ac * Clamp(Dot(ac, pc)/Dot2(ac), 0.0, 1.0) - pc) )
                     :
                     Dot(nor, pa) * Dot(nor, pa) / Dot2(nor) );
}

__bidevice__ bool IntersectTriangle(const Ray &ray, SurfaceInteraction * isect,
                                    int triNum, ParsedMesh *mesh, Float *tHit)
{
    int i0 = mesh->indices[3 * triNum + 0].x;
    int i1 = mesh->indices[3 * triNum + 1].x;
    int i2 = mesh->indices[3 * triNum + 2].x;
    Point3f p0 = mesh->p[i0];
    Point3f p1 = mesh->p[i1];
    Point3f p2 = mesh->p[i2];

    Point3f p0t = p0 - vec3f(ray.o);
    Point3f p1t = p1 - vec3f(ray.o);
    Point3f p2t = p2 - vec3f(ray.o);

    int kz = MaxDimension(Abs(ray.d));
    int kx = kz + 1;
    if(kx == 3) kx = 0;
    int ky = kx + 1;
    if(ky == 3) ky = 0;
    vec3f d = Permute(ray.d, kx, ky, kz);

    p0t = Permute(p0t, kx, ky, kz);
    p1t = Permute(p1t, kx, ky, kz);
    p2t = Permute(p2t, kx, ky, kz);

    Float Sx = -d.x / d.z;
    Float Sy = -d.y / d.z;
    Float Sz = 1.f / d.z;
    p0t.x += Sx * p0t.z;
    p0t.y += Sy * p0t.z;
    p1t.x += Sx * p1t.z;
    p1t.y += Sy * p1t.z;
    p2t.x += Sx * p2t.z;
    p2t.y += Sy * p2t.z;

    Float e0 = p1t.x * p2t.y - p1t.y * p2t.x;
    Float e1 = p2t.x * p0t.y - p2t.y * p0t.x;
    Float e2 = p0t.x * p1t.y - p0t.y * p1t.x;

    if((IsZero(e0) || IsZero(e1) || IsZero(e2))){
        double p2txp1ty = (double)p2t.x * (double)p1t.y;
        double p2typ1tx = (double)p2t.y * (double)p1t.x;
        e0 = (float)(p2typ1tx - p2txp1ty);
        double p0txp2ty = (double)p0t.x * (double)p2t.y;
        double p0typ2tx = (double)p0t.y * (double)p2t.x;
        e1 = (float)(p0typ2tx - p0txp2ty);
        double p1txp0ty = (double)p1t.x * (double)p0t.y;
        double p1typ0tx = (double)p1t.y * (double)p0t.x;
        e2 = (float)(p1typ0tx - p1txp0ty);
    }

    if((e0 < 0 || e1 < 0 || e2 < 0) && (e0 > 0 || e1 > 0 || e2 > 0))
        return false;
    Float det = e0 + e1 + e2;
    if(IsZero(det)) return false;

    p0t.z *= Sz;
    p1t.z *= Sz;
    p2t.z *= Sz;
    Float tScaled = e0 * p0t.z + e1 * p1t.z + e2 * p2t.z;
    if(det < 0 && (tScaled >= 0 || tScaled < ray.tMax * det))
        return false;
    else if(det > 0 && (tScaled <= 0 || tScaled > ray.tMax * det))
        return false;

    Float invDet = 1 / det;
    Float b0 = e0 * invDet;
    Float b1 = e1 * invDet;
    Float b2 = e2 * invDet;
    Float t = tScaled * invDet;

    Float maxZt = MaxComponent(Abs(vec3f(p0t.z, p1t.z, p2t.z)));
    Float deltaZ = gamma(3) * maxZt;

    Float maxXt = MaxComponent(Abs(vec3f(p0t.x, p1t.x, p2t.x)));
    Float maxYt = MaxComponent(Abs(vec3f(p0t.y, p1t.y, p2t.y)));
    Float deltaX = gamma(5) * (maxXt + maxZt);
    Float deltaY = gamma(5) * (maxYt + maxZt);

    Float deltaE = 2 * (gamma(2) * maxXt * maxYt + deltaY * maxXt + deltaX * maxYt);

    Float maxE = MaxComponent(Abs(vec3f(e0, e1, e2)));
    Float deltaT = 3 * (gamma(3) * maxE * maxZt + deltaE * maxZt + deltaZ * maxE) * Absf(invDet);
    if (t <= deltaT) return false;

    vec3f dpdu, dpdv;
    Point2f st[3];
    st[0] = Point2f(0, 0);
    st[1] = Point2f(1, 0);
    st[2] = Point2f(1, 1);

    vec2f dst02 = st[0] - st[2], dst12 = st[1] - st[2];
    vec3f dp02 = p0 - p2, dp12 = p1 - p2;
    Float determinant = dst02[0] * dst12[1] - dst02[1] * dst12[0];

    bool degenerateUV = Absf(determinant) < 1e-8;

    if(!degenerateUV){
        Float invdet = 1 / determinant;
        dpdu = (dst12[1] * dp02 - dst02[1] * dp12) * invdet;
        dpdv = (-dst12[0] * dp02 + dst02[0] * dp12) * invdet;
    }

    Float xAbsSum = (Absf(b0 * p0.x) + Absf(b1 * p1.x) + Absf(b2 * p2.x));
    Float yAbsSum = (Absf(b0 * p0.y) + Absf(b1 * p1.y) + Absf(b2 * p2.y));
    Float zAbsSum = (Absf(b0 * p0.z) + Absf(b1 * p1.z) + Absf(b2 * p2.z));
    vec3f pError = gamma(7) * vec3f(xAbsSum, yAbsSum, zAbsSum);

    Point3f pHit = b0 * p0 + b1 * p1 + b2 * p2;
    Point2f stHit = b0 * st[0] + b1 * st[1] + b2 * st[2];

    *isect = SurfaceInteraction(pHit, Normal3f(0), pError, nullptr);

    isect->n = Normal3f(Normalize(Cross(dp02, dp12)));

    *tHit = t;
    return true;
}

__bidevice__ bool IntersectMeshNode(Node *node, ParsedMesh *mesh, const Ray &r,
                                    SurfaceInteraction * isect, Float *tHit)
{
    Assert(node->n > 0 && node->is_leaf && node->handles);
    bool hit_anything = false;
    for(int i = 0; i < node->n; i++){

        int nTri = node->handles[i].handle;
        if(IntersectTriangle(r, isect, nTri, mesh, tHit)){
            hit_anything = true;
            r.tMax = *tHit;
        }
    }

    return hit_anything;
}

__bidevice__ Float DistanceMeshNode(const vec3f &point, ParsedMesh *mesh, Node *node,
                                    int *handle)
{
    Assert(node->n > 0 && node->is_leaf && node->handles);
    Float distance = Infinity;
    for(int i = 0; i < node->n; i++){
        int nTri = node->handles[i].handle;
        Float d = DistanceTriangle(point, nTri, mesh);
        if(d < distance){
            distance = d;
            *handle = nTri;
        }
    }

    return distance;
}

#define MAX_STACK_SIZE 256
__bidevice__ Float BVHMeshClosestDistance(const vec3f &point, int *closest,
                                          ParsedMesh *mesh, Node *bvh)
{
    NodePtr stack[MAX_STACK_SIZE];
    NodePtr *stackPtr = stack;
    Float closestDistance = SqrtInfinity;
    NodePtr node = bvh;
    int tmpClosest = -1;

    *stackPtr++ = NULL;

    while(node != nullptr){
        if(node->is_leaf){
            Float distance = DistanceMeshNode(point, mesh, node, &tmpClosest);
            if(distance < closestDistance){
                closestDistance = distance;
                *closest = tmpClosest;
            }

            node = *--stackPtr;

        }else{
            const Float closestDistance2 = closestDistance * closestDistance;
            NodePtr childL = node->left;
            NodePtr childR = node->right;
            vec3f closestLeft  = childL->bound.Clamped(point);
            vec3f closestRight = childR->bound.Clamped(point);

            Float leftDist2 = SquaredDistance(closestLeft, point);
            Float rightDist2 = SquaredDistance(closestRight, point);

            bool shouldVisitLeft  = leftDist2  < closestDistance2;
            bool shouldVisitRight = rightDist2 < closestDistance2;

            if(shouldVisitLeft && shouldVisitRight){
                NodePtr firstChild = nullptr, secondChild = nullptr;
                if(leftDist2 < rightDist2){
                    firstChild = childL;
                    secondChild = childR;
                }else{
                    firstChild = childR;
                    secondChild = childL;
                }

                *stackPtr++ = secondChild;
                node = firstChild;
            }else if(shouldVisitRight){
                node = childR;
            }else if(shouldVisitLeft){
                node = childL;
            }else{
                node = *--stackPtr;
            }
        }
    }

    return closestDistance;
}

__bidevice__ bool BVHMeshIntersect(const Ray &r, SurfaceInteraction *isect,
                                   Float *tHit, ParsedMesh *mesh, Node *bvh)
{
    NodePtr stack[MAX_STACK_SIZE];
    NodePtr *stackPtr = stack;
    *stackPtr++ = NULL;
    NodePtr node = bvh;
    bool hit_anything = false;

    bool hit_bound = node->bound.Intersect(r);
    if(!hit_bound) return false;

    while(node != nullptr){
        if(node->is_leaf){
            hit_anything |= IntersectMeshNode(node, mesh, r, isect, tHit);
            node = *--stackPtr;
        }else{
            Float tl0, tr0;
            NodePtr childL = node->left;
            NodePtr childR = node->right;
            bool shouldVisitLeft  = childL->bound.Intersect(r, &tl0);
            bool shouldVisitRight = childR->bound.Intersect(r, &tr0);

            if(shouldVisitRight && shouldVisitLeft){
                NodePtr firstChild = nullptr, secondChild = nullptr;
                if(tr0 < tl0){
                    firstChild  = childR;
                    secondChild = childL;
                }else{
                    firstChild  = childL;
                    secondChild = childR;
                }

                *stackPtr++ = secondChild;
                node = firstChild;
            }else if(shouldVisitLeft){
                node = childL;
            }else if(shouldVisitRight){
                node = childR;
            }else{
                node = *--stackPtr;
            }
        }
    }

    return hit_anything;
}

__host__ Node *MakeBVH(ParsedMesh *mesh, int maxDepth){
    TimerList timers;
    int totalNodes = 0;
    int maxNodes = 0;
    Node *bvh = nullptr;
    // TODO: GPU
    PrimitiveHandle *handles = new PrimitiveHandle[mesh->nTriangles];
    printf("[CPU] Computing triangle bounds...");
    fflush(stdout);
    timers.Start();

    // TODO: GPU
    BVHMeshTrianglesBoundsCPU(mesh, handles);

    timers.StopAndNext();
    printf("OK { %g ms }\nPacking\n", timers.GetElapsedCPU(0));
    fflush(stdout);
    bvh = CreateBVH(handles, mesh->nTriangles, 0, maxDepth, &totalNodes, &maxNodes);

    timers.Stop();
    Point3f pMin = bvh->bound.pMin;
    Point3f pMax = bvh->bound.pMax;

    printf("[ BVH with %d nodes, max: %d, bounds: " v3fA(pMin) ", " v3fA(pMax) " ]\n",
           totalNodes, maxNodes, v3aA(pMin), v3aA(pMax));
    printf("[CPU] Time { %g ms }\n", timers.GetElapsedCPU(1));
    fflush(stdout);
    timers.Reset();

    delete[] handles;

    return bvh;
}

__host__ void Shape::InitMesh(ParsedMesh *msh, bool reverseOr, int maxDepth){
    mesh = msh;
    grid = nullptr; // don't generate the sdf in case this is a emitter
    reverseOrientation = reverseOr;
    WorldToObject = Transform();
    ObjectToWorld = Transform();
    bvh = MakeBVH(msh, maxDepth);
    type = ShapeType::ShapeMesh;
}

__bidevice__ Bounds3f Shape::MeshGetBounds(){
    return bvh->bound;
}

__bidevice__ bool Shape::MeshIntersect(const Ray &ray, SurfaceInteraction *isect,
                                       Float *tShapeHit) const
{
    return BVHMeshIntersect(ray, isect, tShapeHit, mesh, bvh);
}

__bidevice__ Float Shape::MeshClosestDistance(const vec3f &point) const{
    int id = -1;
    return BVHMeshClosestDistance(point, &id, mesh, bvh);
}

#define MAX_ITERATIONS_COUNT 5
__bidevice__ void Shape::ClosestPointBySDF(const vec3f &pWorld,
                                           ClosestPointQuery *query) const
{
    vec3f targetNormal(0, 1, 0);
    vec3f point = WorldToObject.Point(pWorld);
    vec3f targetPoint = point;
    int hasGradient = 0;
    if(grid == nullptr){
        printf("Error: Query for closest point on shape without SDF\n");
        return;
    }

    for(int i = 0; i < MAX_ITERATIONS_COUNT; i++){
        Float sdf = grid->Sample(targetPoint);

        if(Absf(sdf) < 0.001){ break; }

        targetNormal = grid->Gradient(targetPoint);
        Float length = targetNormal.Length();
        if(length > 0){
            targetNormal = targetNormal / length;
        }

        targetPoint = targetPoint - sdf * targetNormal;
        AssertA(!targetPoint.HasNaN(), "NaN in ray marching");
        hasGradient = 1;
    }

    if(!hasGradient){
        targetNormal = grid->Gradient(targetPoint);
        Float length = targetNormal.Length();
        if(length > 0){
            targetNormal = targetNormal / length;
        }
    }

    Float distance = Distance(point, targetPoint);
    Normal3f normal(targetNormal.x, targetNormal.y, targetNormal.z);
    *query = ClosestPointQuery(ObjectToWorld.Point(targetPoint),
                               ObjectToWorld.Normal(normal), distance);
}

__bidevice__ void Shape2::ClosestPointBySDF(const vec2f &pWorld,
                                            ClosestPointQuery2 *query) const
{
    vec2f targetNormal(0, 1);
    vec2f point = WorldToObject.Point(pWorld);
    vec2f targetPoint = point;
    int hasGradient = 0;
    if(grid == nullptr){
        printf("Error: Query for closest point on shape2 without SDF\n");
        return;
    }

    for(int i = 0; i < MAX_ITERATIONS_COUNT; i++){
        Float sdf = grid->Sample(targetPoint);

        if(Absf(sdf) < 0.001){ break; }

        targetNormal = grid->Gradient(targetPoint);

        Float length = targetNormal.Length();
        if(length > 0){
            targetNormal = targetNormal / length;
        }

        targetPoint = targetPoint - sdf * targetNormal;
        AssertA(!targetPoint.HasNaN(), "NaN in ray marching");
        hasGradient = 1;
    }

    if(!hasGradient){
        targetNormal = grid->Gradient(targetPoint);
        Float length = targetNormal.Length();
        if(length > 0){
            targetNormal = targetNormal / length;
        }
    }

    Float distance = Distance(point, targetPoint);
    vec2f normal(targetNormal.x, targetNormal.y);
    *query = ClosestPointQuery2(ObjectToWorld.Point(targetPoint),
                                ObjectToWorld.Vector(normal), distance);
}
