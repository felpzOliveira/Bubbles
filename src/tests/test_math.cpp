#include <graphy-inl.h>
#include <point_generator.h>
#include <kernel.h>
#include <graphy.h>
#include <unistd.h>
#include <tests.h>
#include <transform.h>
#include <shape.h>
#include <collider.h>
#include <emitter.h>
#include <sph_solver.h>
#include <sampling.h>
#include <doring.h>
#include <algorithm>
#include <util.h>
#include <memory.h>
#include <marching_cubes.h>
#include <sdfs.h>

// God, float point precision is hard. Don't want to use double but accumulating
// float multiplications in matrix breaks precision on zero computations
inline bool IsLowZero(Float v){
    if(sizeof(Float) == sizeof(float)){
        return Absf(v) < 1e-2;
    }else{
        return Absf(v) < 1e-5;
    }
}

inline vec2f PointOnBox(Float length){
    return vec2f(rand_float() * length * (rand_float() > 0.5 ? 1 : -1),
                 rand_float() * length * (rand_float() > 0.5 ? 1 : -1));
}

inline vec2f PointOnSphere2(Float radius){
    vec2f p;
    do{
        p = PointOnBox(radius);
    }while(p.Length() > radius);
    return p;
}

void test_sampling_barycentric(){
    printf("===== Test Sampling Barycentric\n");
    vec3f p(5.3, 5.2, 5.4);
    vec3f o(0);
    Float spacing = 1;
    Float fx, fy, fz;
    int iSize = 11;
    int i, j, k;

    vec3f N = (p - o) / spacing;

    GetBarycentric(N.x, 0, iSize-1, &i, &fx);
    GetBarycentric(N.y, 0, iSize-1, &j, &fy);
    GetBarycentric(N.z, 0, iSize-1, &k, &fz);

    printf("(i,j,k) = (%d,%d,%d)\n", i, j, k);
    printf("(fx,fy,fz) = (%g,%g,%g)\n", fx, fy, fz);

    printf("===== OK\n");
}

void test_simple_triangle_distance(){
    printf("===== Test Triangle Distance 3D\n");

    // 2 triangles
    Float pos[] = {
        0.4, -0.5, 0.0,
        0.8, -0.5, 0.0,
        0.6, 1.0, 0.0,

        -0.4, -0.5, 0.0,
        -0.8, -0.5, 0.0,
        -0.6, 1.0, 0.0
    };

    unsigned int indices[] = {
        0, 1, 2,
        3, 4, 5
    };

    int pointCount = (sizeof(pos) / sizeof(Float)) / 3;
    int iCount = (sizeof(indices) / sizeof(unsigned int));
    int it = Max(pointCount, iCount);

    /* manually construct ParsedMesh */
    ParsedMesh mesh;
    memset(&mesh, 0x00, sizeof(ParsedMesh));
    mesh.p = new Point3f[pointCount];
    mesh.indices = new Point3i[iCount];

    for(int i = 0; i < it; i++){
        if(i < pointCount){
            int st = 3 * i;
            mesh.p[i] = Point3f(pos[st+0], pos[st+1], pos[st+2]);
        }

        if(i < iCount){
            mesh.indices[i] = Point3i(indices[i], 0, 0);
        }
    }

    mesh.nTriangles = iCount / 3;
    mesh.nVertices = pointCount;

    Node *bvh = MakeBVH(&mesh, 8);
    int tri = -1;
    Float dist = BVHMeshClosestDistance(vec3f(-0.1, 0.0, 0.0), &tri, &mesh, bvh);
    TEST_CHECK(tri == 1, "Failed to find closest triangle");
    dist = BVHMeshClosestDistance(vec3f(0.1, 0.0, 0.0), &tri, &mesh, bvh);
    TEST_CHECK(tri == 0, "Failed to find closest triangle");
    printf("===== OK\n");
}

// Theres not really a way to validate particles are indeed from boundary
// without performing a comparation with Dilts method, don't want to implement
// that right now, so look at graphy output and see if it looks like boundary.
// TODO: Implement Dilts.
void test_color_field_2D(){
    printf("===== Test Color Field 2D\n");
    Float spacing = 0.05;
    Float targetDensity = WaterDensity;
    vec2f center(0,0);
    Float r1 = 0.5;
    Float r2 = 1.0;
    vec2f pMin, pMax;
    Bounds2f containerBounds;
    ParticleSetBuilder2 builder;
    Grid2 grid;
    SphSolver2 solver;
    solver.Initialize(DefaultSphSolverData2());
    vec2ui res(10, 10);

    Shape2 sphere; sphere.InitSphere2(Translate2(center.x, center.y), r1);
    Shape2 container; container.InitSphere2(Translate2(center.x, center.y), r2, true);

    containerBounds = container.GetBounds();

    // NOTE: Need to make sure bounds are expanded to fit all particles
    pMin = containerBounds.pMin - vec2f(spacing);
    pMax = containerBounds.pMax + vec2f(spacing);

    grid.Build(res, pMin, pMax);

    VolumeParticleEmitter2 emitter(&sphere, sphere.GetBounds(), spacing);

    emitter.Emit(&builder);
    SphParticleSet2 *sphSet = SphParticleSet2FromBuilder(&builder);

    solver.Setup(targetDensity, spacing, 1.8, &grid, sphSet);

    solver.UpdateDensity();

    // Do boundary computation by hand
    ParticleSet2 *pSet = solver.solverData->sphpSet->GetParticleSet();
    Grid2 *domain = solver.solverData->domain;
    SphStdKernel2 kernel(solver.GetKernelRadius());
    int *neighbors = nullptr;
    Float mass = pSet->GetMass();

    float *bpos = new float[pSet->GetParticleCount() * 3];
    float *cols = new float[pSet->GetParticleCount() * 3];
    int currIt = 0;

    Float *lens = new Float[pSet->GetParticleCount()];
    Float maxS = 0;
    for(int i = 0; i < pSet->GetParticleCount(); i++){
        vec2f pi = pSet->GetParticlePosition(i);
        unsigned int cellId = domain->GetLinearHashedPosition(pi);

        int count = domain->GetNeighborsOf(cellId, &neighbors);
        vec2f s(0, 0);
        for(int id = 0; id < count; id++){
            Cell<Bounds2f> *cell = domain->GetCell(neighbors[id]);
            ParticleChain *pChain = cell->GetChain();
            int size = cell->GetChainLength();

            for(int j = 0; j < size; j++){
                vec2f pj = pSet->GetParticlePosition(pChain->pId);
                Float distance = Distance(pi, pj);
                Float density = pSet->GetParticleDensity(pChain->pId);
                Float volume = mass / density;
                vec2f dir(0,0);

                if(distance > 0){
                    dir = (pj - pi) / distance;
                }

                s += volume * kernel.gradW(distance, dir);
                pChain = pChain->next; // NOTE: Don't forget to move the chain node
            }
        }

        Float sLen = s.Length();
        lens[i] = sLen;
        maxS = sLen > maxS ? sLen : maxS;
    }

    for(int i = 0; i < pSet->GetParticleCount(); i++){
        vec2f pi = pSet->GetParticlePosition(i);
        Float ls = lens[i];
        bpos[currIt * 3 + 0] = pi[0];
        bpos[currIt * 3 + 1] = pi[1];
        bpos[currIt * 3 + 2] = 0;

        if(ls > 3){
            cols[currIt * 3 + 0] = 0;
            cols[currIt * 3 + 1] = 1;
            cols[currIt * 3 + 2] = 0;
        }else{
            cols[currIt * 3 + 0] = 1;
            cols[currIt * 3 + 1] = 0;
            cols[currIt * 3 + 2] = 0;
        }
        currIt ++;
    }

    graphy_render_pointsEx(bpos, cols, currIt, -2, 2, 2, -2);
    printf(" * Graphy integration\n");
    sleep(4);
    delete[] bpos;
    delete[] cols;
    delete[] lens;
    printf("===== OK\n");
}

void test_closest_point_sphere2D(){
    printf("===== Test Closest Point Sphere 2D\n");
    Float spacing = 0.09;
    vec2f center(0,0);
    Float radius = 1.0;
    Shape2 sphere; sphere.InitSphere2(Translate2(center.x, center.y), radius);
    Shape2 container; container.InitSphere2(Translate2(center.x, center.y), radius, true);

    ParticleSetBuilder2 builder;
    VolumeParticleEmitter2 emitter(&sphere, sphere.GetBounds(), spacing);
    emitter.Emit(&builder);
    SphParticleSet2 *sphSet = SphParticleSet2FromBuilder(&builder);
    ParticleSet2 *pSet = sphSet->GetParticleSet();

    Collider2 collider(&container);

    float *pos = new float[pSet->GetParticleCount() * 3 * 2];
    float *colors = new float[pSet->GetParticleCount() * 3 * 2];

    for(int i = 0; i < pSet->GetParticleCount(); i++){
        ClosestPointQuery2 query;
        vec2f p = pSet->GetParticlePosition(i);
        vec2f p1 = p;
        sphere.ClosestPoint(p, &query);
        p = query.point;
        Float distance = Distance(query.point, center);
        Float dr = Absf(distance - radius);
        bool is_zero = IsLowZero(dr);
        TEST_CHECK(is_zero, "Particle closest not on sphere");

        p = PointOnBox(radius);
        sphere.ClosestPoint(p, &query);
        p = query.point;
        distance = Distance(query.point, center);
        dr = Absf(distance - radius);
        is_zero = IsLowZero(dr);
        TEST_CHECK(is_zero, "Random point not on sphere");

        if(!is_zero){
            printf("Distance is %g {dr: %g} [%g %g]\n", distance,
                   dr, query.point.x, query.point.y);
        }

        TEST_CHECK(is_zero, "Query point not on sphere");

        pos[3 * i + 0] = p1[0]; colors[3 * i + 0] = 1;
        pos[3 * i + 1] = p1[1]; colors[3 * i + 1] = 0;
        pos[3 * i + 2] = 0; colors[3 * i + 2] = 0;
    }

    Float h = 1.5f;
    int c = pSet->GetParticleCount();

    for(int i = 0; i < 1024; i++){
        vec2f p = PointOnSphere2(radius);
        vec2f p1 = p;
        vec2f v = PointOnBox(radius);
        Float d0 = Distance(p, center);
        if(collider.ResolveCollision(spacing, 0, &p, &v)){
            Float d1 = Distance(p, center);
            if(!(d1 < d0)){
                printf("Distance before: %g, Distance after: %g\n", d0, d1);
            }

            TEST_CHECK(d1 < d0, "Collision startup is further away from collider");
            pos[3 * c + 0] = p1[0]; colors[3 * c + 0] = 0;
            pos[3 * c + 1] = p1[1]; colors[3 * c + 1] = 0;
            pos[3 * c + 2] = 0; colors[3 * c + 2] = 1;
            c ++;

            pos[3 * c + 0] = p[0]; colors[3 * c + 0] = 0;
            pos[3 * c + 1] = p[1]; colors[3 * c + 1] = 1;
            pos[3 * c + 2] = 0; colors[3 * c + 2] = 0;
            c ++;
        }
    }

    graphy_render_pointsEx(pos, colors, c, -h, h, h, -h);
    printf(" * Graphy integration");
    sleep(4);
    //getchar();

    delete[] pos;
    delete[] colors;

    printf("===== OK\n");
}

void test_ray2_intersect(){
    printf("===== Test Ray/Sphere Intersect 2D\n");
    // sample points in center axis and just shoot through there 
    for(int i = 0; i < 2048; i++){
        SurfaceInteraction2 isect;
        Float u1 = rand_float() * 38 * (rand_float() > 0.5 ? 1 : -1);
        Float u2 = rand_float() * 38 * (rand_float() > 0.5 ? 1 : -1);
        vec2f center(u1, u2);
        Float u = 2.0 * rand_float() - 1.0;
        vec2f sample = center + vec2f(0, u);
        Float e = 1.5 + rand_float() * 10.0;
        vec2f source = center - vec2f(e, 0);
        vec2f dir = Normalize(sample - source);
        vec2f bi(u1 - 1.0, u2 - 1.0);
        vec2f bo(u1 + 1.0, u2 + 1.0);

        Ray2 ray(source, dir);
        Float tHit = 0;

        Shape2 sphere; sphere.InitSphere2(Translate2(u1, u2), 1.0);

        bool hit = sphere.Intersect(ray, &isect, &tHit);
        Float d = sphere.ClosestDistance(source);
        Bounds2f b = sphere.GetBounds();
        vec2f pMin = b.pMin;
        vec2f pMax = b.pMax;
        vec2f zi = pMin - bi;
        vec2f zo = pMax - bo;
        if(!hit){
            printf("Failed at source: {%g %g} sample: {%g %g} center: {%g %g} dir: {%g %g} dist: %g [%d]\n",
                   source.x, source.y, sample.x, sample.y, center.x, center.y,
                   dir.x, dir.y, d, i);
        }

        for(int i = 0; i < 2; i++){
            Float ai = zi[i];
            Float ao = zo[i];
            TEST_CHECK(IsZero(ai) && IsZero(ao), "Invalid axis for Sphere2::GetBounds");
        }

        TEST_CHECK(IsLowZero(d - (e - 1.0)), "Failed closest distance computation");
        TEST_CHECK(hit, "Did not hit sphere");
    }

    printf("===== OK\n");
}

vec2f sample_outside(Bounds2f bound){
    vec2f p(0,0);
    Float lenx = bound.ExtentOn(0) * 1.5;
    Float leny = bound.ExtentOn(1) * 1.5;
    do{
        p.x = (2 * rand_float() - 1) * lenx;
        p.y = (2 * rand_float() - 1) * leny;
    }while(Inside(p, bound));
    return p;
}

vec2f sample_inside(Bounds2f bound){
    vec2f p(0,0);
    Float lenx = bound.ExtentOn(0) * 0.5;
    Float leny = bound.ExtentOn(1) * 0.5;
    p.x = (2 * rand_float() - 1) * lenx;
    p.y = (2 * rand_float() - 1) * leny;
    return p;
}

void test_rectangle_distance_inside(){
    printf("===== Test Inside Rectangle Distance\n");
    Shape2 rectangle;
    ClosestPointQuery2 query;
    rectangle.InitRectangle2(Translate2(0, 0), vec2f(2, 2));
    int points = 2048;

    float *pos = new float[points * 3 * 2];
    float *col = new float[points * 3 * 2];
    int it = 0;

    for(int i = 0; i < points; i++){
        vec2f p = sample_inside(rectangle.GetBounds());
        rectangle.ClosestPoint(p, &query);

        Float dmin = Min(Min(Absf(1 - p.x), Absf(1 + p.x)),
                         Min(Absf(1 - p.y), Absf(1 + p.y)));

        if(!IsZero(dmin - query.signedDistance)){
            printf("Distance: %g, Queried: %g, Point {%g %g}\n",
                   dmin, query.signedDistance, p.x, p.y);
        }

        TEST_CHECK(IsZero(dmin - query.signedDistance), "Distance is not minimal");

        vec2f n = query.normal;
        pos[3 * it + 0] = p[0]; pos[3 * it + 1] = p[1]; pos[3 * it + 2] = 0;
        col[3 * it + 0] = Absf(n.x);
        col[3 * it + 1] = 0;
        col[3 * it + 2] = Absf(n.y);
        it++;

        p = query.point;
        pos[3 * it + 0] = p[0]; pos[3 * it + 1] = p[1]; pos[3 * it + 2] = 0;
        col[3 * it + 0] = 0; col[3 * it + 1] = 1; col[3 * it + 2] = 0;
        it++;
    }

    Float h = 2;
    graphy_render_pointsEx(pos, col, 2 * points, -h, h, h, -h);
    printf(" * Graphy integration\n");
    sleep(4);
    delete[] pos;
    delete[] col;

    printf("===== OK\n");
}

void test_rectangle_distance_outside(){
    printf("===== Test Outside Rectangle Distance\n");
    Shape2 rectangle;
    ClosestPointQuery2 query;
    rectangle.InitRectangle2(Translate2(0, 0), vec2f(1,1));
    int points = 1024;

    float *pos = new float[points * 3 * 2];
    float *col = new float[points * 3 * 2];
    int it = 0;

    for(int i = 0; i < points; i++){
        vec2f p = sample_outside(rectangle.GetBounds());
        rectangle.ClosestPoint(p, &query);
        vec2f n = query.normal;
        pos[3 * it + 0] = p[0]; pos[3 * it + 1] = p[1]; pos[3 * it + 2] = 0;
        col[3 * it + 0] = Absf(n.x);
        col[3 * it + 1] = 0;
        col[3 * it + 2] = Absf(n.y);
        it++;

        p = query.point;
        pos[3 * it + 0] = p[0]; pos[3 * it + 1] = p[1]; pos[3 * it + 2] = 0;
        col[3 * it + 0] = 0; col[3 * it + 1] = 1; col[3 * it + 2] = 0;
        it++;
    }

    Float h = 2;
    graphy_render_pointsEx(pos, col, 2 * points, -h, h, h, -h);
    printf(" * Graphy integration\n");
    sleep(4);
    delete[] pos;
    delete[] col;

    printf("===== OK\n");
}

void test_rectangle_emit(){
    printf("===== Test Rectangle Emittion\n");
    Shape2 rectangle;
    Float spacing = 0.09;
    rectangle.InitRectangle2(Translate2(0,0), vec2f(1,1));

    ParticleSetBuilder2 builder;
    VolumeParticleEmitter2 emitter(&rectangle, rectangle.GetBounds(), spacing);
    emitter.Emit(&builder);

    SphParticleSet2 *sphSet = SphParticleSet2FromBuilder(&builder);
    ParticleSet2 *pSet = sphSet->GetParticleSet();

    float *pos = new float[pSet->GetParticleCount() * 3];
    for(int i = 0; i < pSet->GetParticleCount(); i++){
        vec2f p = pSet->GetParticlePosition(i);
        pos[3 * i + 0] = p[0];
        pos[3 * i + 1] = p[1];
        pos[3 * i + 2] = 0;
    }

    float col[3] = {1,0,0};

    Float h = 1;
    graphy_render_points(pos, col, pSet->GetParticleCount(), -h, h, h, -h);
    sleep(4);

    delete[] pos;
    printf("===== OK\n");
}

void test_box_distance(){
    printf("===== Test Box Distance\n");
    Shape shape;
    shape.InitBox(Transform(), 1, 1, 1);

    Bounds3f testBound(vec3f(-0.48), vec3f(0.48));

    int points = 500;

    std::vector<vec3f> targets;

    for(int i = 0; i < points; i++){
        Float x = 2 * rand_float() - 1;
        Float y = 2 * rand_float() - 1;
        Float z = 2 * rand_float() - 1;

        ClosestPointQuery query;
        shape.ClosestPoint(vec3f(x,y,z), &query);
        targets.push_back(query.point);

        TEST_CHECK(!Inside(query.point, testBound), "Point is not on box face");
    }

    float *pos = new float[3 * targets.size()];

    int it = 0;
    for(vec3f &p : targets){
        pos[3 * it + 0] = p.x;
        pos[3 * it + 1] = p.y;
        pos[3 * it + 2] = p.z;
        it ++;
    }

    vec3f origin(0,0,3);
    vec3f target(0,0,0);
    float rgb[3] = {1, 0, 0};

    graphy_vector_set(origin, target);
    graphy_render_points3(pos, rgb, targets.size(), 0.01);

    sleep(4);
    delete[] pos;
    printf("===== OK\n");
}

void test_bounds_split3(){
    printf("===== Test Bounds3 Split\n");
    vec3f A(-1), B(1);
    Bounds3f bounds(vec3f(-1), vec3f(1));
    Bounds3f splits[8];

    Float lx = (B.x - A.x) * 0.5;
    Float ly = (B.y - A.y) * 0.5;
    Float lz = (B.z - A.z) * 0.5;
    vec3f res[] = {
        vec3f(A.x, A.y, A.z),
        vec3f(A.x + lx, A.y + ly, A.z + lz),
        vec3f(A.x + lx, A.y, A.z),
        vec3f(B.x, A.y + ly, A.z + lz),
        vec3f(A.x, A.y + ly, A.z),
        vec3f(A.x + lx, B.y, A.z + lz),
        vec3f(A.x + lx, A.y + ly, A.z),
        vec3f(B.x, B.y, A.z + lz),

        vec3f(A.x, A.y, A.z + lz),
        vec3f(A.x + lx, A.y + ly, A.z + lz + lz),
        vec3f(A.x + lx, A.y, A.z + lz),
        vec3f(B.x, A.y + ly, A.z + lz + lz),
        vec3f(A.x, A.y + ly, A.z + lz),
        vec3f(A.x + lx, B.y, A.z + lz + lz),
        vec3f(A.x + lx, A.y + ly, A.z + lz),
        vec3f(B.x, B.y, A.z + lz + lz),
    };

    printf("Splitting : ");
    bounds.PrintSelf();
    printf("\n");

    int k = SplitBounds(bounds, &splits[0]);
    TEST_CHECK(k == 8, "Invalid returned bound split count");

    int it = 0;
    for(int i = 0; i < 8; i++){
        Bounds3f b = splits[i];
        vec3f pMin = b.pMin;
        vec3f pMax = b.pMax;
        TEST_CHECK((IsZero(pMin.x - res[it].x) &&
                   IsZero(pMin.y - res[it].y) &&
                   IsZero(pMax.x - res[it+1].x) &&
                   IsZero(pMax.y - res[it+1].y) &&
                   IsZero(pMax.z - res[it+1].z) &&
                   IsZero(pMax.z - res[it+1].z)),
                   "Unexpected position");
        it += 2;

        // print for easy inspection
        b.PrintSelf();
        printf("\n");

        TEST_CHECK(Inside(b, bounds), "Not inside");
    }

    Bounds3f bout(vec3f(-2), vec3f(1));
    TEST_CHECK(!Inside(bout, bounds), "Invalid computation of bounds inside");

    printf("===== OK\n");
}

void test_bounds_split2(){
    printf("===== Test Bounds2 Split\n");
    Bounds2f bounds(vec2f(-1), vec2f(1));
    Bounds2f splits[4];

    vec2f res[] = {
        vec2f(-1, -1), vec2f(0, 0),
        vec2f(-1, 0), vec2f(0, 1),
        vec2f(0, 0), vec2f(1, 1),
        vec2f(0, -1), vec2f(1, 0),
    };

    printf("Splitting : ");
    bounds.PrintSelf();
    printf("\n");

    int k = SplitBounds(bounds, &splits[0]);
    TEST_CHECK(k == 4, "Invalid returned bound split count");

    int it = 0;
    for(int i = 0; i < 4; i++){
        Bounds2f b = splits[i];
        vec2f pMin = b.pMin;
        vec2f pMax = b.pMax;
        TEST_CHECK((IsZero(pMin.x - res[it].x) &&
                   IsZero(pMin.y - res[it].y) &&
                   IsZero(pMax.x - res[it+1].x) &&
                   IsZero(pMax.y - res[it+1].y)),
                   "Unexpected position");
        it += 2;

        // print for easy inspection
        b.PrintSelf();
        printf("\n");

        TEST_CHECK(Inside(b, bounds), "Not inside");
    }

    Bounds2f bout(vec2f(-2), vec2f(1));
    TEST_CHECK(!Inside(bout, bounds), "Invalid computation of bounds inside");

    printf("===== OK\n");
}

void test_matrix_operations2(){
    printf("===== Test MatrixNxN operations\n");
    Matrix2x2 m(5, 4, 3, 2);
    m.PrintSelf();

    Matrix2x2 s = Inverse(m);
    s.PrintSelf();

    Float L = ComputeMinEigenvalue(m);
    printf("Min : %g\n", L);

    Matrix3x3 t(-2, -4, 2, -2, 1, 2, 4, 2, 5);
    L = ComputeMinEigenvalue(t);
    printf("Min : %g\n", L);

    printf("===== OK\n");
}

void test_matrix_operations(){
    printf("===== Test Matrix3x3\n");

    for(int i = 0; i < 2048; i++){
        Float u1 = rand_float() * 1.0 + 1.0;
        Float u2 = rand_float() * 1.0 + 1.0;
        Float s1 = rand_float() * 6.0 + 1.0;
        Float s2 = rand_float() * 5.0 + 1.0;

        vec2f p(u1,  u2);
        Transform2 scale = Scale2(s1, s2);
        Transform2 translate = Translate2(s1, s2);
        vec2f t = scale.Point(p);
        vec2f s = translate.Point(p);

        Matrix3x3 identity = Matrix3x3::Mul(scale.m, scale.mInv);

        TEST_CHECK(IsZero(t.x - u1 * s1), "Invalid Scale in X axis");
        TEST_CHECK(IsZero(t.y - u2 * s2), "Invalid Scale in Y axis");
        TEST_CHECK(IsZero(s.x - (u1 + s1)), "Invalid Translate in X axis");
        TEST_CHECK(IsZero(s.y - (u2 + s2)), "Invalid Translate in Y axis");

        for(int i = 0; i < 3; i++){
            for(int j = 0; j < 3; j++){
                Float a = identity.m[i][j];
                if(i == j){
                    Float da = a - 1.0;
                    if(!IsLowZero(da)){
                        printf("d: %g\n", da);
                        identity.PrintSelf();
                    }
                    TEST_CHECK(IsLowZero(da), "Not 1 at i == j on identity matrix");
                }
                else{
                    if(!IsLowZero(a)){
                        printf("a: %g\n", a);
                        identity.PrintSelf();
                    }
                    TEST_CHECK(IsLowZero(a), "Not 0 at i != j on identity matrix");
                }
            }
        }

        Float mat[3][3];
        for(int i = 0; i < 3; i++){
            for(int j = 0; j < 3; j++){
                mat[i][j] = (rand_float() * 15.0 + 0.5) * (rand_float() > 0.5 ? 1 : -1);
            }
        }

        Matrix3x3 m1(mat);
        Matrix3x3 m1Inv = Inverse(m1);
        identity = Matrix3x3::Mul(m1, m1Inv);
        for(int i = 0; i < 3; i++){
            for(int j = 0; j < 3; j++){
                Float a = identity.m[i][j];
                if(i == j){
                    Float da = a - 1.0;
                    if(!IsLowZero(da)){
                        identity.PrintSelf();
                        printf("------------\n");
                        m1.PrintSelf();
                        printf("------------\n");
                        m1Inv.PrintSelf();
                        printf("d: %g\n", da);
                    }
                    TEST_CHECK(IsLowZero(da), "Not 1 at i == j on identity matrix");
                }
                else{
                    if(!IsLowZero(a)){
                        identity.PrintSelf();
                        printf("------------\n");
                        m1.PrintSelf();
                        printf("------------\n");
                        m1Inv.PrintSelf();
                        printf("a: %g\n", a);
                    }
                    TEST_CHECK(IsLowZero(a), "Not 0 at i != j on identity matrix");
                }
            }
        }
    }

    printf("===== OK\n");
}

bb_cpu_gpu
vec2d EnrightField2D(const vec2d &pi, double t, double T){
    double sinX  = std::sin(Pi * pi.x);
    double sinY  = std::sin(Pi * pi.y);
    double sin2Y = std::sin(TwoPi * pi.y);
    double sin2X = std::sin(TwoPi * pi.x);
    double direction = std::cos(Pi * t / T);
    return vec2d( 2.0 * sinX * sinX * sin2Y * direction,
                 -2.0 * sinY * sinY * sin2X * direction );
}

bb_cpu_gpu
vec3d EnrightField3D(const vec3d &p, double t, double T){
    double sinX  = std::sin(Pi * p.x);
    double sinY  = std::sin(Pi * p.y);
    double sinZ  = std::sin(Pi * p.z);

    double sin2X = std::sin(TwoPi * p.x);
    double sin2Y = std::sin(TwoPi * p.y);
    double sin2Z = std::sin(TwoPi * p.z);

    double direction = std::cos(Pi * t / T);

    vec3d v =  vec3d(
        2.0 * (2.0 * sinX * sinX * sin2Y * sin2Z * direction),
        2.0 * (-sin2X * sinY * sinY * sin2Z * direction),
        2.0 * (-sin2X * sin2Y * sinZ * sinZ * direction)
    );

    if(v.LengthSquared() < 1e-5){
        //printf("Zero velocity {%g %g %g} [%g %g %g]\n",
               //v.x, v.y, v.z, p.x, p.y, p.z);
    }

    return v;
}

template<typename Q, typename Fn> bb_cpu_gpu
Q RK3Sample(const Q &p, double dt, double t, double T, Fn field){
    Q k1 = field(p, t, T);
    Q p1 = p + 0.5 * dt * k1;

    Q k2 = field(p1, t + 0.5 * dt, T);
    Q p2 = p + 0.75 * dt * k2;

    Q k3 = field(p2, t + 0.75 * dt, T);
    return p + (dt / 9.0) * (2.0 * k1 + 3.0 * k2 + 4.0 * k3);
}

template<typename Q, typename Fn> bb_cpu_gpu
Q RK4Sample(const Q &p, double dt, double t, double T, Fn field) {
    Q k1 = field(p, t, T);
    Q p1 = p + 0.5 * dt * k1;

    Q k2 = field(p1, t + 0.5 * dt, T);
    Q p2 = p + 0.5 * dt * k2;

    Q k3 = field(p2, t + 0.5 * dt, T);
    Q p3 = p + dt * k3;

    Q k4 = field(p3, t + dt, T);

    return p + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4);
}

bb_cpu_gpu
vec2f RK4Enright2D(const vec2f &pi, Float dt, Float t, Float T){
    vec2d d_pi(pi.x, pi.y);
    vec2d p = RK4Sample<vec2f, decltype(EnrightField2D)>
                                (d_pi, (double)dt, (double)t, (double)T,
                                EnrightField2D);

    return vec2f(p.x, p.y);
}

bb_cpu_gpu
vec3f RK4Enright3D(const vec3f &pi, Float dt, Float t, Float T){
    vec3d d_pi(pi.x, pi.y, pi.z);
    vec3d p = RK4Sample<vec3d, decltype(EnrightField3D)>
                                (d_pi, (double)dt, (double)t, (double)T,
                                 EnrightField3D);
    return vec3f(p.x, p.y, p.z);
}

bb_cpu_gpu
vec2f RK3Enright2D(const vec2f &pi, Float dt, Float t, Float T){
    vec2d d_pi(pi.x, pi.y);
    vec2d p = RK3Sample<vec2f, decltype(EnrightField2D)>
                                (d_pi, (double)dt, (double)t, (double)T,
                                EnrightField2D);

    return vec2f(p.x, p.y);
}

bb_cpu_gpu
vec3f RK3Enright3D(const vec3f &pi, Float dt, Float t, Float T){
    vec3d d_pi(pi.x, pi.y, pi.z);
    vec3d p = RK3Sample<vec3d, decltype(EnrightField3D)>
                                (d_pi, (double)dt, (double)t, (double)T,
                                 EnrightField3D);
    return vec3f(p.x, p.y, p.z);
}

void test_enright_2D(){
    printf("===== Test Enright 2D\n");
    CudaMemoryManagerStart(__FUNCTION__);

    const Float sphereRadius = 0.2f;
    const Float spacing = 0.001;
    vec2f center(0.5f, 0.75f);

    Shape2 *sphere = MakeSphere2(Transform2(), sphereRadius);
    VolumeParticleEmitterSet2 emitters;

    emitters.SetJitter(0.01);
    emitters.AddEmitter(sphere, spacing);

    ParticleSetBuilder2 builder;
    emitters.Emit(&builder);

    SphParticleSet2 *sphSet = SphParticleSet2FromBuilder(&builder);
    ParticleSet2 *set = sphSet->GetParticleSet();

    int count = set->GetParticleCount();
    printf("Total particles= %d\n", count);

    float *col = new float[count * 3];
    float *pos = new float[count * 3];
    for(int i = 0; i < count; i++){
        vec3f rgb = vec3f(1, 0, 0);
        vec2f pi = set->GetParticlePosition(i);

        pos[3 * i + 0] = pi.x;  pos[3 * i + 1] = pi.y;
        pos[3 * i + 2] = 0;     col[3 * i + 0] = rgb.x;
        col[3 * i + 1] = rgb.y; col[3 * i + 2] = rgb.z;
    }

    int *frameId = cudaAllocateVx(int, 1);
    *frameId = 1;
    Float currentTime = 0;
    do{
        Float frameTime = 0.00001f;
        currentTime = frameTime * (*frameId);
        std::cout << "\r Frame " << (*frameId) << " ( " <<
                   currentTime << " s )" << std::flush;
        GPUParallelLambda("Update", count, GPU_LAMBDA(int i){
            vec2f pi = set->GetParticlePosition(i);
            vec2f projPi = pi + center;
            Float t = frameTime * ((*frameId) - 1);
            Float T = 6.f;

            projPi = RK4Enright2D(projPi, frameTime, t, T);

            pi = projPi - center;

            set->SetParticlePosition(i, pi);
        });

        for(int s = 0; s < count; s++){
            vec2f pi = set->GetParticlePosition(s);
            pos[3 * s + 0] = pi.x;
            pos[3 * s + 1] = pi.y;
        }

        Debug_GraphyDisplaySolverParticles(set, pos, col);
        *frameId += 1;
    }while(currentTime < 12);

    CudaMemoryManagerClearCurrent();
    printf("===== OK\n");
}

void test_enright_3D(){
    printf("===== Test Enright\n");
    CudaMemoryManagerStart(__FUNCTION__);

    vec3f center (0.35f, 0.35f, 0.35f);
    //vec3f center(0.f);
    const Float sphereRadius = 0.15f;
    const int numParticles = 1000000;
    const Float spacing = 0.002;
    const Float spacingScale = 2.f;
    vec3f offset = vec3f(0.0f, 0.f, 0.f);

    SphSolver3 solver;
    SphParticleSet3 *sphpSet = nullptr;
    ParticleSet3 *set = nullptr;
    Grid3 *domain = nullptr;

    vec3f *testPs = cudaAllocateVx(vec3f, numParticles);
    int *testFlags = cudaAllocateVx(int, numParticles);
    Float *rng = cudaAllocateVx(Float, numParticles * 3);

    for(int i = 0; i < numParticles; i++){
        rng[3 * i + 0] = rand_float();
        rng[3 * i + 1] = rand_float();
        rng[3 * i + 2] = rand_float();
    }

    GPUParallelLambda("Initialize", numParticles, GPU_LAMBDA(int i){
        Float u1 = rng[3 * i + 0];
        Float u2 = rng[3 * i + 1];
        Float u3 = rng[3 * i + 2];
        Float theta = u1 * 2.0 * Pi;
        Float phi = std::acos(2.0 * u2 - 1.0);
        Float r = sphereRadius * std::pow(u3, 1.0 / 3.0);

        Float x = r * std::sin(phi) * std::cos(theta);
        Float y = r * std::sin(phi) * std::sin(theta);
        Float z = r * std::cos(phi);
        testPs[i] = vec3f(x, y, z);
        testFlags[i] = 1;
    });

    ParticleSetBuilder3 testBuilder, pBuilder;
    for(int i = 0; i < numParticles; i++){
        testBuilder.AddParticle(testPs[i]);
    }

    solver.Initialize(DefaultSphSolverData3());
    domain = UtilBuildGridForBuilder(&testBuilder, spacing, spacingScale);
    sphpSet = SphParticleSet3FromBuilder(&testBuilder);
    set = sphpSet->GetParticleSet();
    solver.Setup(WaterDensity, spacing, spacingScale, domain, sphpSet);

    UpdateGridDistributionGPU(solver.solverData);

    int N = set->GetParticleCount();
    AutoParallelFor("Initialize", N, AutoLambda(int pId){
        int *neighbors = nullptr;
        vec3f pi = set->GetParticlePosition(pId);
        unsigned int cellId = domain->GetLinearHashedPosition(pi);

        int count = domain->GetNeighborsOf(cellId, &neighbors);
        for(int i = 0; i < count; i++){
            Cell3 *cell = domain->GetCell(neighbors[i]);
            ParticleChain *pChain = cell->GetChain();
            int size = cell->GetChainLength();
            for(int j = 0; j < size; j++){
                vec3f pj = set->GetParticlePosition(pChain->pId);
                Float distance = Distance(pi, pj);
                if(distance < 0.5f * spacing && pChain->pId < pId){
                    testFlags[pId] = 0;
                    return;
                }
            }
        }
    });

    for(int i = 0; i < numParticles; i++){
        if(testFlags[i] == 1){
            vec3f pi = testPs[i];
            pBuilder.AddParticle(pi + center);
        }
    }

    sphpSet = SphParticleSet3FromBuilder(&pBuilder);
    set = sphpSet->GetParticleSet();


    int totalCount = set->GetParticleCount();
    printf("Total particles= %d\n", totalCount);

    float *col = new float[totalCount * 3];
    float *pos = new float[totalCount * 3];

    for(int i = 0; i < totalCount; i++){
        vec3f rgb = vec3f(1, 0, 0);
        vec3f pi = set->GetParticlePosition(i);

        pos[3 * i + 0] = pi.x;  pos[3 * i + 1] = pi.y;
        pos[3 * i + 2] = pi.z;  col[3 * i + 0] = rgb.x;
        col[3 * i + 1] = rgb.y; col[3 * i + 2] = rgb.z;
    }

    int *frameId = cudaAllocateVx(int, 1);
    *frameId = 1;
    Float currentTime = 0;

    //vec3f origin(1.5f, 1.5f, 1.5f);
    //vec3f target(0.5f, 0.5f, 0.5f);
    vec3f origin(0.527492, -0.0615508, -0.676749);
    vec3f target(0.532272, 0.483669, 0.483669);
    graphy_vector_set(origin, target);
    do{
        Float frameTime = 0.001f;
        currentTime = frameTime * ((*frameId)-1);
        std::cout << "\r Frame " << (*frameId) << " ( " <<
                   currentTime << " s )" << std::flush;

        GPUParallelLambda("Update", totalCount, GPU_LAMBDA(int i){
            vec3f pi = set->GetParticlePosition(i);
            vec3f projPi = pi + offset;
            Float t = frameTime * ((*frameId) - 1);
            Float T = 2.f;

            projPi = RK4Enright3D(projPi, frameTime, t, T);
            pi = projPi - offset;

            set->SetParticlePosition(i, pi);
        });

        for(int s = 0; s < totalCount; s++){
            vec3f pi = set->GetParticlePosition(s);
            pos[3 * s + 0] = pi.x;
            pos[3 * s + 1] = pi.y;
            pos[3 * s + 2] = pi.z;
        }

        graphy_render_points3(pos, col, totalCount, spacing);

        if(SerializerIsWrittable()){
            std::string respath =
                    FrameOutputPath("enright3D/output_", *frameId);
            SerializerSaveParticleSet3Legacy(set, respath.c_str(),
                                            SERIALIZER_POSITION);
        }

        *frameId += 1;
    }while(currentTime < 6);

    CudaMemoryManagerClearCurrent();
    printf("===== OK\n");
}

// NOTE: can use this routines for generating the teddies/origami scene.
void sdf_teddies(){
    HostTriangleMesh3 mesh;
    Float iso = 0.0;
    Float dx  = 0.02;
    Bounds3f bounds(vec3f(-5), vec3f(5));
    FieldGrid3f *field = CreateSDF(bounds, dx, AutoLambda(vec3f point){
        //return T_OrigamiBoat(point, -1);
        //return T_OrigamiDragon(point);
        //return T_OrigamiWhale(point, 2);
        //return Teddy_Lying(point);
        return Teddy_Sitting(point);
        //return Teddy_Standing(point);
    });

    vec3ui res = field->GetResolution();
    printf("Resolution= {%u %u %u}\n", res.x, res.y, res.z);
#if 0
    Bounds3f reducedB(vec3f(-1), vec3f(1));

    Transform transform = Scale(vec3f(5.f));
    auto sample_fn = GPU_LAMBDA(vec3f point, Shape *, int) -> Float{
        vec3f query = transform.Point(point);
        return field->Sample(query);
    };

    Shape *testShape = MakeSDFShape(reducedB, sample_fn);

    MarchingCubes(testShape->grid, &mesh, iso, false);
#else
    MarchingCubes(field, &mesh, iso, false);
#endif
    mesh.writeToDisk("test_sdf.obj", FORMAT_PLY);
    exit(0);
}

