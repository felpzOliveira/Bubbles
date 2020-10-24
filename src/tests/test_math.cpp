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
    solver.Cleanup();
    cudaFree(pSet);
    builder.Release();
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
    
    cudaFree(pSet);
    builder.Release();
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
    Float rgb[3] = {1, 0, 0};
    
    graphy_vector_set(origin, target);
    graphy_render_points3(pos, rgb, targets.size(), 0.01);
    
    sleep(4);
    delete[] pos;
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
