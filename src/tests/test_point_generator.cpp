#include <point_generator.h>
#include <kernel.h>
#include <graphy.h>
#include <unistd.h>
#include <tests.h>
#include <transform.h>
#include <shape.h>
#include <emitter.h>
#include <obj_loader.h>

void test_mesh_collision(){
    printf("===== Test Particle Mesh Collision\n");
    
    const char *whaleObj = "/home/felipe/Documents/CGStuff/models/HappyWhale.obj";
    Transform transform = Translate(-0.5, -0.5, 0) * Scale(0.05); // happy whale
    UseDefaultAllocatorFor(AllocatorType::GPU);
    
    ParsedMesh *mesh = LoadObj(whaleObj);
    Shape *shape = MakeMesh(mesh, transform, true);
    
    ColliderSetBuilder3 cBuilder;
    cBuilder.AddCollider3(shape);
    ColliderSet3 *colliders = cBuilder.GetColliderSet();
    
    vec3f p0 = shape->grid->bounds.pMin;
    vec3f p1 = shape->grid->bounds.pMax;
    Float h = 0.05;
    std::vector<vec3f> particles;
    
    int tests = 0;
    for(Float x = p0.x; x < p1.x; x += h){
        for(Float y = p0.y; y < p1.y; y += h){
            for(Float z = p0.z; z < p1.z; z += h){
                vec3f p(x, y, z);
                vec3f v(1, 0, 0);
                
                if(colliders->ResolveCollision(h, 0, &p, &v)){
                    particles.push_back(p);
                }
                
                tests++;
            }
        }
    }
    
    int size = particles.size();
    float *pos = new float[size * 3];
    float *col = new float[size * 3];
    
    int itp = 0;
    int itc = 0;
    
    printf("Got %d particles from %d tests\n", size, tests);
    
    for(vec3f &p : particles){
        pos[itp++] = p.x; pos[itp++] = p.y; pos[itp++] = p.z;
        col[itc++] = 1; col[itc++] = 0; col[itc++] = 0;
    }
    
    vec3f at(0, 1, 4);
    vec3f to(0,0,0);
    
    graphy_set_3d(at.x, at.y, at.z, to.x, to.y, to.z, 45.0, 0.1f, 100.0f);
    graphy_render_points3f(pos, col, itp/3, h/2.0);
    
    getchar();
    graphy_close_display();
    
    printf("===== OK\n");
}

void test_volume_particle_emitter3_mesh(){
    printf("===== Test Volume Particle Emitter3 Mesh\n");
    vec3f origin(1.5, 3, 5);
    vec3f target(0,0.35,0);
    
    UseDefaultAllocatorFor(AllocatorType::CPU);
    ParsedMesh *mesh = LoadObj("/home/felpz/Documents/models/sssDragonAligned.obj");
    
    Shape *shape = MakeMesh(mesh, Scale(0.025) * RotateY(-70));
    
    Float spacing = 0.02;
    Bounds3f bound = shape->GetBounds();
    Float lenc = Max(bound.ExtentOn(0), Max(bound.ExtentOn(1), bound.ExtentOn(2)));
    int reso = (int)std::floor(lenc / (spacing * 4.0));
    printf("Using grid with resolution %d x %d x %d\n", reso, reso, reso);
    
    ParticleSetBuilder3 pBuilder;
    VolumeParticleEmitter3 vEmitter(shape, bound, spacing);
    
    vEmitter.Emit(&pBuilder);
    ParticleSet3 *pSet = pBuilder.MakeParticleSet();
#if 0
    Grid3 *grid = MakeGrid(vec3ui(reso), bound.pMin, bound.pMax);
    
    printf("Distributing particles\n");
    grid->DistributeByParticle(pSet);
    
    CNMInvalidateCells(grid);
    TimerList timers;
    timers.Start();
    int size = CNMClassifyLazyGPU(grid);
    timers.Stop();
    printf("Size: %d <=> %g\n", size, timers.GetElapsedGPU(0));
    
#endif
    int count = 0;
    int pCount = pSet->GetParticleCount();
    Float *pos = new Float[3 * pCount];
    Float *col = new Float[3 * pCount];
    
    int it = 0;
    for(int i = 0; i < pSet->GetParticleCount(); i++){
        vec3f pi = pSet->GetParticlePosition(i);
        pos[3 * it + 0] = pi.x; pos[3 * it + 1] = pi.y;
        pos[3 * it + 2] = pi.z;
        //col[3 * it + 0] = color[0]; col[3 * it + 1] = color[1];
        //col[3 * it + 2] = color[2];
        col[3 * it + 0] = 1; col[3 * it + 1] = 0;
        col[3  *it + 1] = 0;
        it ++;
    }
    
    count = pCount;
    //count = set_poscol_cnm(pos, col, pSet, grid, accept_call);
    
    printf(" * Graphy interation\n");
    graphy_vector_set(origin, target);
    graphy_render_points3f(pos, col, count, 0.01);
    
    getchar();
    
    delete[] pos;
    delete[] col;
    printf("===== OK\n");
}

void test_volume_particle_emitter3(){
    printf("===== Test Volume Particle Emitter3\n");
    
    vec3f origin(3.0, 0.0, 0.0);
    vec3f target(0,0,0);
    
    Bounds3f bound(vec3f(-1), vec3f(1));
    Float spacing = 0.02;
    Shape *sphere = MakeSphere(Transform(), 1);
    
    Float lenc = 2.0;
    int reso = (int)std::floor(lenc / (spacing * 1.8));
    printf("Using grid with resolution %d x %d x %d\n", reso, reso, reso);
    ParticleSetBuilder3 pBuilder;
    VolumeParticleEmitter3 vEmitter(sphere, bound, spacing);
    
    vEmitter.Emit(&pBuilder);
    
    Grid3 *grid = MakeGrid(vec3ui(reso), vec3f(-1), vec3f(1));
    ParticleSet3 *pSet = pBuilder.MakeParticleSet();
    
    int pCount = pSet->GetParticleCount();
    
    printf("Distributing particles\n");
    grid->DistributeByParticle(pSet);
    
    CNMInvalidateCells(grid);
    
    TimerList timers;
    timers.Start();
    int size = CNMClassifyLazyGPU(grid);
    timers.Stop();
    printf("Size: %d <=> %g\n", size, timers.GetElapsedGPU(0));
    
    int count = 0;
    Float *pos = new Float[3 * pCount];
    Float *col = new Float[3 * pCount];
    
    for(int i = 0; i < pCount; i++){
        vec3f point = pSet->GetParticlePosition(i);
        Float distance = Distance(point, vec3f(0));
        TEST_CHECK(distance <= 1, "Invalid particle center");
    }
    
    auto accept_call = [&](const vec3f &pi) -> bool{
        if(pi.x < 0) return true;
        return false;
    };
    
    count = set_poscol_cnm(pos, col, pSet, grid, accept_call);
    
    printf(" * Graphy interation\n");
    graphy_vector_set(origin, target);
    graphy_render_points3f(pos, col, count, 0.01);
    
    sleep(4);
    delete[] pos;
    delete[] col;
    printf("===== OK\n");
}

void test_bcclattice_point_generator(){
    printf("===== Test BCC Lattice Point Generator\n");
    BccLatticePointGenerator pGenerator;
    
    vec3f origin(3.0, 3.0, 0.0);
    vec3f target(0,0,0);
    
    Bounds3f bound(vec3f(-1), vec3f(1));
    Float spacing = 0.1;
    std::vector<vec3f> points;
    
    pGenerator.Generate(bound, spacing, &points);
    
    printf(" * Generated %ld points\n", points.size());
    printf(" * Graphy interation\n");
    
    Float *p = new Float[3 * points.size()];
    Float rgb[3] = {1,0,0};
    
    for(int i = 0; i < points.size(); i++){
        vec3f point = points[i];
        p[3 * i + 0] = point.x;
        p[3 * i + 1] = point.y;
        p[3 * i + 2] = point.z;
    }
    
    graphy_vector_set(origin, target);
    graphy_render_points3(p, rgb, points.size(), 0.01);
    
    sleep(4);
    delete[] p;
    points.clear();
    printf("===== OK\n");
}

void test_triangle_point_generator(){
    printf("===== Test Triangle Point Generator\n");
    TrianglePointGenerator pGenerastor;
    Float scale = 1.8;
    Float spacing = 0.1;
    Float radius = spacing * scale;
    SphStdKernel2 kernel(radius);
    Bounds2f bounds(vec2f(-1.5 * radius), vec2f(1.5 * radius));
    
    std::vector<vec2f> points;
    pGenerastor.Generate(bounds, spacing, &points);
    
    float *po = new float[3 * points.size()];
    float *co = new float[3 * points.size()];
    Float w0 = kernel.W(0);
    printf(" * Kernel W(0) = %g\n", w0);
    int it = 0;
    for(vec2f &p : points){
        po[it + 0] = p.x; co[it + 0] = 1;
        po[it + 1] = p.y; co[it + 1] = 0;
        po[it + 2] = 0;   co[it + 2] = 0;
        it += 3;
    }
    
    graphy_render_pointsEx(po, co, points.size(), -0.5, 0.5, 0.5, -0.5);
    delete[] po;
    delete[] co;
    sleep(2);
    
    Float n = 0;
    for(int i = 0; i < points.size(); i++){
        vec2f pi = points[i];
        Float sum = 0.f;
        for(int j = 0; j < points.size(); j++){
            vec2f pj = points[j];
            sum += kernel.W((pi - pj).Length());
        }
        
        n = Max(n, sum);
    }
    
    TEST_CHECK(!IsZero(n), "Zero number density");
    Float mass = 1000.0 / n;
    printf(" * Generated mass is %g\n", mass);
    
    printf("===== OK\n");
}