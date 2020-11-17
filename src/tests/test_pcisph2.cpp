#include <pcisph_solver.h>
#include <emitter.h>
#include <tests.h>
#include <grid.h>
#include <util.h>
#include <memory.h>
#include <marching_squares.h>

void test_pcisph2_marching_squares(){
    printf("===== SPH Solver 2D -- Marching Squares\n");
    Float spacing = 0.01;
    Float targetDensity = WaterDensity;
    vec2f center(0,0);
    Float r1 = 0.5;
    Float r2 = 1.0;
    vec2f pMin, pMax;
    Bounds2f containerBounds;
    ParticleSetBuilder2 builder;
    ColliderSetBuilder2 colliderBuilder;
    PciSphSolver2 solver;
    
    CudaMemoryManagerStart(__FUNCTION__);
    
    int reso = (int)std::floor(2 * r2 / (spacing * 2.0));
    printf("Using grid with resolution %d x %d\n", reso, reso);
    vec2ui res(reso, reso);
    
    solver.Initialize(DefaultSphSolverData2());
    Shape2 *sphere = MakeSphere2(Translate2(center.x, center.y+r1/2.f), r1);
    Shape2 *container = MakeSphere2(Translate2(center.x, center.y), r2, true);
    
    containerBounds = container->GetBounds();
    pMin = containerBounds.pMin - vec2f(spacing);
    pMax = containerBounds.pMax + vec2f(spacing);
    
    Grid2 *grid = MakeGrid(res, pMin, pMax);
    
    VolumeParticleEmitter2 emitter(sphere, sphere->GetBounds(), spacing);
    
    emitter.Emit(&builder);
    SphParticleSet2 *sphSet = SphParticleSet2FromBuilder(&builder);
    ParticleSet2 *set2 = sphSet->GetParticleSet();
    int count = set2->GetParticleCount();
    
    colliderBuilder.AddCollider2(container);
    ColliderSet2 *collider = colliderBuilder.GetColliderSet();
    
    solver.Setup(targetDensity, spacing, 2.0, grid, sphSet);
    solver.SetColliders(collider);
    
    Float targetInterval = 1.0 / 248.0;
    float *pos = new float[count * 3];
    float *col = new float[count * 3];
    
    SphSolverData2 *data = solver.GetSphSolverData();
    
    for(int i = 0; i < 125; i++){
        solver.Advance(targetInterval);
        set_colors_pressure(col, data);
        Debug_GraphyDisplaySolverParticles(sphSet->GetParticleSet(), pos, col);
    }
    
    FieldGrid2f *field = cudaAllocateVx(FieldGrid2f, 1);
    
    UtilSphDataToFieldGrid2f(data, field);
    
    std::vector<vec3f> triangles;
    MarchingSquares(field, 0, &triangles);
    int totalLines = triangles.size() * 3;
    float *poss = new float[totalLines * 3];
    
    int it = 0;
    for(int i = 0; i < triangles.size()/3; i++){
        vec3f p0 = triangles[3 * i + 0];
        vec3f p1 = triangles[3 * i + 1];
        vec3f p2 = triangles[3 * i + 2];
        
        poss[it++] = p0.x; poss[it++] = p0.y; poss[it++] = p0.z;
        poss[it++] = p1.x; poss[it++] = p1.y; poss[it++] = p1.z;
        poss[it++] = p1.x; poss[it++] = p1.y; poss[it++] = p1.z;
        poss[it++] = p2.x; poss[it++] = p2.y; poss[it++] = p2.z;
        poss[it++] = p2.x; poss[it++] = p2.y; poss[it++] = p2.z;
        poss[it++] = p0.x; poss[it++] = p0.y; poss[it++] = p0.z;
    }
    
    float rgb[3] = {1, 0, 0};
    Float os = 1.0;
    graphy_set_orthographic(-os, os, os, -os);
    graphy_render_lines(poss, rgb, totalLines);
    getchar();
    graphy_close_display();
    delete[] poss;
    
    delete[] pos;
    delete[] col;
    CudaMemoryManagerClearCurrent();
    printf("===== OK\n");
}

void test_pcisph2_water_block(){
    printf("===== SPH Solver 2D -- Water Block\n");
    Float spacing = 0.015;
    Float targetDensity = WaterDensity;
    vec2f center(0,0);
    Float lenc = 2;
    
    vec2f pMin, pMax;
    Bounds2f containerBounds;
    ParticleSetBuilder2 builder;
    ColliderSetBuilder2 colliderBuilder;
    
    PciSphSolver2 solver;
    
    int reso = (int)std::floor(lenc / (spacing * 2.0));
    printf("Using grid with resolution %d x %d\n", reso, reso);
    vec2ui res(reso, reso);
    
    solver.Initialize(DefaultSphSolverData2());
    Shape2 *rect = MakeRectangle2(Translate2(center.x, center.y+0.45), vec2f(1));
    Shape2 *block = MakeSphere2(Translate2(center.x, center.y-0.3), 0.2);
    Shape2 *container = MakeRectangle2(Translate2(center.x, center.y), vec2f(lenc), true);
    
    containerBounds = container->GetBounds();
    pMin = containerBounds.pMin - vec2f(spacing);
    pMax = containerBounds.pMax + vec2f(spacing);
    
    Grid2 *grid = MakeGrid(res, pMin, pMax);
    
    VolumeParticleEmitter2 emitter(rect, rect->GetBounds(), spacing);
    
    emitter.Emit(&builder);
    SphParticleSet2 *sphSet = SphParticleSet2FromBuilder(&builder);
    ParticleSet2 *set2 = sphSet->GetParticleSet();
    int count = set2->GetParticleCount();
    
    colliderBuilder.AddCollider2(block);
    colliderBuilder.AddCollider2(container);
    ColliderSet2 *collider = colliderBuilder.GetColliderSet();
    
    solver.Setup(targetDensity, spacing, 2.0, grid, sphSet);
    solver.SetColliders(collider);
    
    Float targetInterval = 1.0 / 240.0;
    float *pos = new float[count * 3];
    float *col = new float[count * 3];
    
    SphSolverData2 *data = solver.GetSphSolverData();
    //set_colors_cnm(col, data);
    set_colors_pressure(col, data);
    
    for(int i = 0; i < 20 * 26; i++){
        solver.Advance(targetInterval);
        set_colors_pressure(col, data);
        Debug_GraphyDisplaySolverParticles(sphSet->GetParticleSet(), pos, col);
    }
    
    delete[] pos;
    delete[] col;
    printf("===== OK\n");
}


void test_pcisph2_water_block_cnm(){
    printf("===== SPH Solver 2D -- Water Block CNM\n");
    Float spacing = 0.01;
    Float targetDensity = WaterDensity;
    vec2f center(0,0);
    Float lenc = 2;
    
    vec2f pMin, pMax;
    Bounds2f containerBounds;
    ParticleSetBuilder2 builder;
    ColliderSetBuilder2 colliderBuilder;
    
    PciSphSolver2 solver;
    
    CudaMemoryManagerStart(__FUNCTION__);
    
    int reso = (int)std::floor(lenc / (spacing * 2.0));
    printf("Using grid with resolution %d x %d\n", reso, reso);
    vec2ui res(reso, reso);
    
    solver.Initialize(DefaultSphSolverData2());
    Shape2 *rect = MakeRectangle2(Translate2(center.x, center.y+0.45), vec2f(1));
    Shape2 *block = MakeSphere2(Translate2(center.x, center.y-0.3), 0.2);
    Shape2 *container = MakeRectangle2(Translate2(center.x, center.y), vec2f(lenc), true);
    
    containerBounds = container->GetBounds();
    pMin = containerBounds.pMin - vec2f(spacing);
    pMax = containerBounds.pMax + vec2f(spacing);
    
    Grid2 *grid = MakeGrid(res, pMin, pMax);
    
    VolumeParticleEmitter2 emitter(rect, rect->GetBounds(), spacing);
    
    emitter.Emit(&builder);
    SphParticleSet2 *sphSet = SphParticleSet2FromBuilder(&builder);
    ParticleSet2 *set2 = sphSet->GetParticleSet();
    int count = set2->GetParticleCount();
    
    colliderBuilder.AddCollider2(block);
    colliderBuilder.AddCollider2(container);
    ColliderSet2 *collider = colliderBuilder.GetColliderSet();
    
    solver.Setup(targetDensity, spacing, 2.0, grid, sphSet);
    solver.SetColliders(collider);
    
    float *pos = new float[count * 3];
    float *col = new float[count * 3];
    
    SphSolverData2 *data = solver.GetSphSolverData();
    //set_colors_cnm(col, data);
    //set_colors_pressure(col, data);
    Float targetInterval = 1.0 / 240.0;
    for(int i = 0; i < 200 * 10; i++){
        solver.Advance(targetInterval);
        set_colors_cnm(col, data, 0, 0);
        //set_colors_pressure(col, data);
        Debug_GraphyDisplaySolverParticles(sphSet->GetParticleSet(), pos, col);
    }
    
    getchar();
    
    CudaMemoryManagerClearCurrent();
    delete[] pos;
    delete[] col;
    printf("===== OK\n");
}


void test_pcisph2_water_drop(){
    printf("===== PCISPH Solver 2D -- Water Drop\n");
    Float spacing = 0.015;
    Float targetDensity = WaterDensity;
    vec2f center(0,0);
    Float lenc = 2;
    Float r = 0.2;
    
    vec2f pMin, pMax;
    Bounds2f containerBounds;
    ParticleSetBuilder2 builder;
    ColliderSetBuilder2 colliderBuilder;
    
    PciSphSolver2 solver;
    
    int reso = (int)std::floor(lenc / (spacing * 2.0));
    printf("Using grid with resolution %d x %d\n", reso, reso);
    vec2ui res(reso, reso);
    
    Float boxLenx = lenc - 2 * spacing;
    Float boxLeny = 0.3;
    vec2f boxDim(boxLenx, boxLeny);
    
    solver.Initialize(DefaultSphSolverData2());
    Shape2 *rect = MakeRectangle2(Translate2(0, -lenc/2.f + boxLeny/2.f + spacing), boxDim);
    Shape2 *sphere = MakeSphere2(Translate2(0, lenc/2.f - r - spacing), r);
    Shape2 *container = MakeRectangle2(Translate2(center.x, center.y), vec2f(lenc), true);
    
    containerBounds = container->GetBounds();
    pMin = containerBounds.pMin - vec2f(spacing);
    pMax = containerBounds.pMax + vec2f(spacing);
    
    Grid2 *grid = MakeGrid(res, pMin, pMax);
    
    VolumeParticleEmitterSet2 emitterSet;
    
    emitterSet.AddEmitter(sphere, sphere->GetBounds(), spacing);
    emitterSet.AddEmitter(rect, rect->GetBounds(), spacing);
    
    emitterSet.Emit(&builder);
    
    SphParticleSet2 *sphSet = SphParticleSet2FromBuilder(&builder);
    ParticleSet2 *set2 = sphSet->GetParticleSet();
    int count = set2->GetParticleCount();
    
    colliderBuilder.AddCollider2(container);
    ColliderSet2 *collider = colliderBuilder.GetColliderSet();
    
    solver.Setup(targetDensity, spacing, 2.0, grid, sphSet);
    solver.SetColliders(collider);
    
    Float targetInterval = 1.0 / 240.0;
    float *pos = new float[count * 3];
    float *col = new float[count * 3];
    
    SphSolverData2 *data = solver.GetSphSolverData();
    
    set_colors_pressure(col, data);
    Debug_GraphyDisplaySolverParticles(sphSet->GetParticleSet(), pos, col);
    for(int i = 0; i < 20 * 26; i++){
        solver.Advance(targetInterval);
        set_colors_pressure(col, data);
        Debug_GraphyDisplaySolverParticles(sphSet->GetParticleSet(), pos, col);
    }
    
    delete[] pos;
    delete[] col;
    printf("===== OK\n");
}

void test_pcisph2_double_dam_break(){
    printf("===== PCISPH Solver 2D -- Double Dam Break\n");
    Float spacing = 0.015;
    Float targetDensity = WaterDensity;
    vec2f center(0,0);
    Float lenc = 2;
    
    vec2f pMin, pMax;
    Bounds2f containerBounds;
    ParticleSetBuilder2 builder;
    ColliderSetBuilder2 colliderBuilder;
    
    PciSphSolver2 solver;
    
    int reso = (int)std::floor(lenc / (spacing * 2.0));
    printf("Using grid with resolution %d x %d\n", reso, reso);
    vec2ui res(reso, reso);
    
    Float boxLenx = 0.4;
    Float boxLeny = 1.0;
    vec2f boxDim(boxLenx, boxLeny);
    
    solver.Initialize(DefaultSphSolverData2());
    Shape2 *rect = MakeRectangle2(Translate2(-(lenc - boxLenx)/2.f + spacing, 
                                             -(boxLeny/2.f - spacing)), boxDim);
    
    Shape2 *rect2 = MakeRectangle2(Translate2((lenc - boxLenx)/2.f - spacing, 
                                              -(boxLeny/2.f-spacing)), boxDim);
    
    Shape2 *container = MakeRectangle2(Translate2(center.x, center.y), vec2f(lenc), true);
    
    containerBounds = container->GetBounds();
    pMin = containerBounds.pMin - vec2f(spacing);
    pMax = containerBounds.pMax + vec2f(spacing);
    
    Grid2 *grid = MakeGrid(res, pMin, pMax);
    
    VolumeParticleEmitterSet2 emitterSet;
    
    emitterSet.AddEmitter(rect, rect->GetBounds(), spacing);
    emitterSet.AddEmitter(rect2, rect2->GetBounds(), spacing);
    
    emitterSet.Emit(&builder);
    
    SphParticleSet2 *sphSet = SphParticleSet2FromBuilder(&builder);
    ParticleSet2 *set2 = sphSet->GetParticleSet();
    int count = set2->GetParticleCount();
    
    colliderBuilder.AddCollider2(container);
    ColliderSet2 *collider = colliderBuilder.GetColliderSet();
    
    solver.Setup(targetDensity, spacing, 2.0, grid, sphSet);
    solver.SetColliders(collider);
    
    Float targetInterval = 1.0 / 240.0;
    float *pos = new float[count * 3];
    float *col = new float[count * 3];
    
    memset(col, 0x00, sizeof(float) * 3 * count);
    SphSolverData2 *data = solver.GetSphSolverData();
    
    for(int i = 0; i < 20 * 26; i++){
        solver.Advance(targetInterval);
        set_colors_pressure(col, data);
        Debug_GraphyDisplaySolverParticles(sphSet->GetParticleSet(), pos, col);
    }
    
    delete[] pos;
    delete[] col;
    printf("===== OK\n");
}

void test_pcisph2_water_sphere(){
    printf("===== PCISPH Solver 2D -- Water in Ball\n");
    Float spacing = 0.01;
    Float targetDensity = WaterDensity;
    vec2f center(0,0);
    Float r1 = 0.5;
    Float r2 = 1.0;
    vec2f pMin, pMax;
    Bounds2f containerBounds;
    ParticleSetBuilder2 builder;
    ColliderSetBuilder2 colliderBuilder;
    PciSphSolver2 solver;
    
    int reso = (int)std::floor(2 * r2 / (spacing * 2.0));
    printf("Using grid with resolution %d x %d\n", reso, reso);
    vec2ui res(reso, reso);
    
    solver.Initialize(DefaultSphSolverData2());
    Shape2 *sphere = MakeSphere2(Translate2(center.x-0.4, center.y+r1/2.f), r1);
    //Shape2 *sphere = MakeSphere2(Translate2(center.x, center.y+0.4), r1);
    Shape2 *container = MakeSphere2(Translate2(center.x, center.y), r2, true);
    
    containerBounds = container->GetBounds();
    pMin = containerBounds.pMin - vec2f(spacing);
    pMax = containerBounds.pMax + vec2f(spacing);
    
    Grid2 *grid = MakeGrid(res, pMin, pMax);
    
    VolumeParticleEmitter2 emitter(sphere, sphere->GetBounds(), spacing);
    
    emitter.Emit(&builder);
    SphParticleSet2 *sphSet = SphParticleSet2FromBuilder(&builder);
    ParticleSet2 *set2 = sphSet->GetParticleSet();
    int count = set2->GetParticleCount();
    
    colliderBuilder.AddCollider2(container);
    ColliderSet2 *collider = colliderBuilder.GetColliderSet();
    
    solver.Setup(targetDensity, spacing, 2.0, grid, sphSet);
    solver.SetColliders(collider);
    
    Float targetInterval = 1.0 / 248.0;
    float *pos = new float[count * 3];
    float *col = new float[count * 3];
    
    SphSolverData2 *data = solver.GetSphSolverData();
    
    for(int i = 0; i < 20 * 26 * 20; i++){
        solver.Advance(targetInterval);
        set_colors_pressure(col, data);
        Debug_GraphyDisplaySolverParticles(sphSet->GetParticleSet(), pos, col);
    }
    
    delete[] pos;
    delete[] col;
    printf("===== OK\n");
}
