#include <pcisph_solver.h>
#include <emitter.h>
#include <tests.h>
#include <grid.h>
#include <util.h>
#include <memory.h>
#include <marching_squares.h>

void test_pcisph2_marching_squares(){
    printf("===== PCISPH Solver 2D -- Marching Squares\n");
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
    printf("===== PCISPH Solver 2D -- Water Block\n");
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
    //set_colors_lnm(col, data);
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

void test_pcisph2_continuous_emitter(){
    printf("===== PCISPH Solver 2D -- Continuous Emitter\n");
    PciSphSolver2 solver;
    Float spacing = 0.01;
    Float spacingScale = 2.0;
    Float targetDensity = WaterDensity;
    Float lenc = 1.2;
    
    CudaMemoryManagerStart(__FUNCTION__);
    Shape2 *rect   = MakeRectangle2(Transform2(), vec2f(lenc), true);
    Shape2 *block  = MakeRectangle2(Translate2(-0.5, 0.5), 0.05);
    Shape2 *block2 = MakeRectangle2(Translate2(0.3, 0.2), 0.05);
    
    Grid2 *grid = UtilBuildGridForDomain(rect->GetBounds(), spacing, spacingScale);
    
    ContinuousParticleSetBuilder2 builder(50000);
    builder.SetKernelRadius(spacing * spacingScale);
    
    auto velocityField = [&](const vec2f &p) -> vec2f{
        Float u1 = rand_float() * 10.0;
        vec2f v(u1, 0);
        if(p.x > 0) v *= -1.0;
        return v;
    };
    
    VolumeParticleEmitterSet2 emitter;
    emitter.AddEmitter(block, block->GetBounds(), spacing);
    emitter.AddEmitter(block2, block2->GetBounds(), spacing);
    
    emitter.Emit(&builder, velocityField);
    
    SphParticleSet2 *sphSet = SphParticleSet2FromContinuousBuilder(&builder);
    ParticleSet2 *set2 = sphSet->GetParticleSet();
    
    ColliderSetBuilder2 colliderBuilder;
    colliderBuilder.AddCollider2(rect);
    ColliderSet2 *collider = colliderBuilder.GetColliderSet();
    
    solver.Initialize(DefaultSphSolverData2());
    solver.SetViscosityCoefficient(0.04);
    solver.Setup(targetDensity, spacing, spacingScale, grid, sphSet);
    solver.SetColliders(collider);
    
    SphSolverData2 *data = solver.GetSphSolverData();
    Float targetInterval = 1.0 / 240.0;
    builder.MapGrid(grid);
    
    int maxframes = 1000;
    
    auto onStepUpdate = [&](int step) -> int{
        if(step == 0) return 1;
        if(step < maxframes){
            builder.MapGridEmit(velocityField, spacing);
        }
        return step > maxframes ? 0 : 1;
    };
    
    auto colorFunction = [&](float *colors, int pCount) -> void{
        (void)pCount;
        set_colors_lnm(colors, data, 0, 0);
    };
    
    UtilRunSimulation2<PciSphSolver2, ParticleSet2>(&solver, set2, spacing,
                                                    vec2f(-1, -1), vec2f(1, 1),
                                                    targetInterval, onStepUpdate,
                                                    colorFunction);
    CudaMemoryManagerClearCurrent();
    printf("===== OK\n");
}

void test_pcisph2_water_block_lnm(){
    printf("===== PCISPH Solver 2D -- Water Block LNM\n");
    Float spacing = 0.01;
    Float spacingScale = 2.0;
    Float targetDensity = WaterDensity;
    vec2f center(0,0);
    Float lenc = 2;
    
    Bounds2f containerBounds;
    ParticleSetBuilder2 builder;
    ColliderSetBuilder2 colliderBuilder;
    
    PciSphSolver2 solver;
    
    CudaMemoryManagerStart(__FUNCTION__);
    
    Shape2 *rect = MakeRectangle2(Translate2(center.x, center.y+0.45), vec2f(1));
    Shape2 *block = MakeSphere2(Translate2(center.x, center.y-0.3), 0.2);
    Shape2 *container = MakeRectangle2(Translate2(center.x, center.y), vec2f(lenc), true);
    
    containerBounds = container->GetBounds();
    
    solver.Initialize(DefaultSphSolverData2());
    
    Grid2 *grid = UtilBuildGridForDomain(containerBounds, spacing, spacingScale);
    
    VolumeParticleEmitter2 emitter(rect, rect->GetBounds(), spacing);
    
    emitter.Emit(&builder);
    SphParticleSet2 *sphSet = SphParticleSet2FromBuilder(&builder);
    ParticleSet2 *set2 = sphSet->GetParticleSet();
    
    colliderBuilder.AddCollider2(block);
    colliderBuilder.AddCollider2(container);
    ColliderSet2 *collider = colliderBuilder.GetColliderSet();
    
    solver.Setup(targetDensity, spacing, spacingScale, grid, sphSet);
    solver.SetColliders(collider);
    
    SphSolverData2 *data = solver.GetSphSolverData();
    Float targetInterval = 1.0 / 240.0;
    
    auto colorFunction = [&](float *colors, int pCount) -> void{
        (void)pCount;
        set_colors_lnm(colors, data, 0, 0);
    };
    
    UtilRunSimulation2<PciSphSolver2, ParticleSet2>(&solver, set2, spacing,
                                                    vec2f(-1, -1), vec2f(1, 1),
                                                    targetInterval, EmptyCallback,
                                                    colorFunction);
    
    CudaMemoryManagerClearCurrent();
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
    
    solver.SetViscosityCoefficient(0.02);
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
    
    Float targetInterval = 1.0 / 240.0;
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
