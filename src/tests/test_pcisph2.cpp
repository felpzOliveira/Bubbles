#include <pcisph_solver.h>
#include <emitter.h>
#include <tests.h>
#include <grid.h>
#include <util.h>
#include <memory.h>
#include <marching_squares.h>
#include <interval.h>
#include <boundary.h>

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

    CudaMemoryManagerStart(__FUNCTION__);

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
    pMin = containerBounds.pMin - 4.0 * vec2f(spacing);
    pMax = containerBounds.pMax + 4.0 * vec2f(spacing);

    VolumeParticleEmitter2 emitter(rect, rect->GetBounds(), spacing);

    emitter.Emit(&builder);
    SphParticleSet2 *sphSet = SphParticleSet2FromBuilder(&builder);
    ParticleSet2 *set2 = sphSet->GetParticleSet();
    int count = set2->GetParticleCount();

    Grid2 *grid = MakeGrid(res, pMin, pMax);

    colliderBuilder.AddCollider2(block);
    colliderBuilder.AddCollider2(container);
    ColliderSet2 *collider = colliderBuilder.GetColliderSet();

    solver.Setup(targetDensity, spacing, 2.0, grid, sphSet);
    solver.SetColliders(collider);

    Float targetInterval = 1.0 / 240.0;
    float *pos = new float[count * 3];
    float *col = new float[count * 3];

    SphSolverData2 *data = solver.GetSphSolverData();
    set_colors_lnm(col, data, 0, 0);
    //set_colors_pressure(col, data);

    SandimWorkQueue2 *vpWorkQ = cudaAllocateVx(SandimWorkQueue2, 1);
    vpWorkQ->SetSlots(grid->GetCellCount());
    Float sphRadius = data->sphpSet->GetKernelRadius();
    WorkQueue<vec4f> *marroneWorkQ = cudaAllocateVx(WorkQueue<vec4f>, 1);
    marroneWorkQ->SetSlots(set2->GetParticleCount());

    for(int i = 0; i < 20 * 26; i++){
        solver.Advance(targetInterval);
        //set_colors_pressure(col, data);

        for(int k = 0; k < set2->GetParticleCount(); k++){
            set2->SetParticleV0(k, 0);
        }

        vpWorkQ->Reset();
        marroneWorkQ->Reset();
        ComputeNormalGPU(data);

        //IntervalBoundary(set2, grid, sphRadius);
        //MarroneBoundary(set2, grid, sphRadius);
        MarroneAdaptBoundary(set2, grid, sphRadius, marroneWorkQ);
        //DiltsSpokeBoundary(set2, grid);
        //CFBoundary(set2, grid, spacing);
        //XiaoweiBoundary(set2, grid, spacing);
        //SandimBoundary(set2, grid, vpWorkQ);
        //LNMBoundary(set2, grid, spacing);
        //LNMBoundarySingle(set2, grid, spacing);
        //RandlesDoringBoundary(set2, grid, spacing);

        set_colors_lnm(col, data, 0, 0);
        Debug_GraphyDisplaySolverParticles(sphSet->GetParticleSet(), pos, col);
        if(i == 180) getchar();
        //getchar();
    }

    delete[] pos;
    delete[] col;
    CudaMemoryManagerClearCurrent();

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

void test_pcisph2_water_rotating_obstacles(){
    printf("===== PCISPH Solver 2D -- Water Block with rotating obstacles\n");
    Float spacing = 0.02;
    Float spacingScale = 2.0;
    Float targetDensity = WaterDensity;
    vec2f positions[] = {
        vec2f(-0.6, 0.0), vec2f(0.5, 0.1),
        vec2f(-0.1, -0.2), vec2f(0.3, -0.6),
        vec2f(-0.4, -0.73)
    };

    int pCount = sizeof(positions) / sizeof(positions[0]);
    Float *angles = new Float[pCount];
    for(int i = 0; i < pCount; i++) angles[i] = 0;

    Float domainLen = 2.3;
    Bounds2f containerBounds;
    ParticleSetBuilder2 builder;
    ColliderSetBuilder2 colliderBuilder;
    std::vector<Shape2 *> shapes;

    PciSphSolver2 solver;

    CudaMemoryManagerStart(__FUNCTION__);

    Shape2 *domainShape = MakeRectangle2(Translate2(0, 0), vec2f(domainLen), true);
    Shape2 *waterShape  = MakeRectangle2(Translate2(0, 0.55), vec2f(1.5, 0.7));

    containerBounds = Bounds2f(vec2f(-5,-5), vec2f(5, 5));
    solver.Initialize(DefaultSphSolverData2());

    Grid2 *grid = UtilBuildGridForDomain(containerBounds, spacing, spacingScale);

    VolumeParticleEmitter2 emitter(waterShape, waterShape->GetBounds(), spacing);
    emitter.Emit(&builder);

    SphParticleSet2 *sphSet = SphParticleSet2FromBuilder(&builder);
    ParticleSet2 *set = sphSet->GetParticleSet();

    colliderBuilder.AddCollider2(domainShape);

    for(int i = 0; i < pCount; i++){
        vec2f p = positions[i];
        Shape2 *b = MakeRectangle2(Translate2(p.x, p.y), vec2f(0.2));
        colliderBuilder.AddCollider2(b);
        shapes.push_back(b);
    }

    ColliderSet2 *colliders = colliderBuilder.GetColliderSet();

    solver.Setup(targetDensity, spacing, spacingScale, grid, sphSet);
    solver.SetColliders(colliders);

    SphSolverData2 *data = solver.GetSphSolverData();
    Float targetInterval = 1.0 / 240.0;

    auto colorFunction = [&](float *colors, int pCount) -> void{
        (void)pCount;
        set_colors_lnm(colors, data, 0, 0);
    };
    Float a = 0;

    auto updateCallback = [&](int frame) -> int{
        for(int i = 0; i < pCount; i++){
            Shape2 *shape = shapes[i];
            Float w = angles[i];
            vec2f p = positions[i];
            Float dw = 0.1 + rand_float() * 0.1;
            if(i % 2 != 0) dw = -dw;

            w += dw;
            Transform2 transform = Translate2(p.x, p.y) * Rotate2(w);

            if(!(w > -2.0 * Pi && w < 2.0 * Pi)) w = 0;
            angles[i] = w;

            shape->Update(transform);
            //shape->SetVelocities(vec2f(0,0), dw / targetInterval);
        }

        a += 0.1;
        if(a > 2.0 * Pi) a = 0;
        Transform2 t = Rotate2(a);
        domainShape->Update(t);
        //domainShape->SetVelocities(vec2f(0), a / targetInterval);

        return 1;
    };

    int c = 160 + 100;
    auto filler = [&](float *pos, float *col) -> int{
        int f = c / pCount;
        int n = 0;
        for(int i = 0; i < pCount; i++){
            Shape2 *shape = shapes[i];
            if(colliders->IsActive(i+1)){
                n += UtilGenerateSquarePoints(&pos[3 * n], &col[3 * n], vec3f(1,1,0),
                                              shape->ObjectToWorld, vec2f(0.2), f);
            }
        }

        n += UtilGenerateSquarePoints(&pos[3 * n], &col[3 * n], vec3f(1,1,0),
                                      domainShape->ObjectToWorld, vec2f(domainLen), 100);
        return n;
    };

    Float v = 1.7;
    UtilRunDynamicSimulation2<PciSphSolver2, ParticleSet2>(&solver, set, spacing,
                                                           vec2f(-v, -v), vec2f(v, v),
                                                           targetInterval, 2 * c,
                                                           updateCallback, colorFunction,
                                                           filler);

    CudaMemoryManagerClearCurrent();
    printf("===== OK\n");
}

void test_pcisph2_water_square_dynamic(){
    printf("===== PCISPH Solver 2D -- Water Block with movable square\n");
    Float spacing = 0.01;
    Float spacingScale = 2.0;
    Float targetDensity = WaterDensity;
    vec2f center(0,0);
    vec2f direction(1.0);
    Float circSphereRadius = 1.0 * std::sqrt(3) * 0.5;

    Bounds2f containerBounds;
    ParticleSetBuilder2 builder;
    ColliderSetBuilder2 colliderBuilder;

    PciSphSolver2 solver;

    CudaMemoryManagerStart(__FUNCTION__);

    Shape2 *rect = MakeRectangle2(Translate2(center.x, center.y), vec2f(0.75));
    Shape2 *block = MakeRectangle2(Translate2(center.x, center.y), vec2f(1.0), true);
    //Shape2 *container = MakeRectangle2(Translate2(center.x, center.y), vec2f(lenc), true);
    Shape2 *container = MakeSphere2(Translate2(center.x, center.y), circSphereRadius, true);

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
    Float alpha = 0.0;

    auto colorFunction = [&](float *colors, int pCount) -> void{
        (void)pCount;
        set_colors_lnm(colors, data, 0, 0);
    };

    auto updateCallback = [&](int frame) -> int{
        if(collider->IsActive(0)){
            Float off = spacing;
            Float y = center.y + direction.y * off;
            Float x = center.x + direction.x * off;
            Float v = off / targetInterval;
            if(Absf(y) > 0.5){
                direction.y = -direction.y;
            }

            if(Absf(x) > 0.5){
                direction.x = -direction.x;
            }

            vec2f vel = direction * vec2f(v, v);
            Transform2 transform = Translate2(center.x, center.y) * Rotate2(alpha);
            alpha += 0.1;
            if(alpha > 2.0 * Pi) alpha = 0;

            block->Update(transform);
            if(frame > 200){
                collider->SetActive(0, false);
            }
        }
        return 1;
    };

    int c = 160;
    auto filler = [&](float *pos, float *col) -> int{
        int n = 0;
        if(collider->IsActive(0)){
            n = UtilGenerateSquarePoints(pos, col, vec3f(1,1,0),
                                         block->ObjectToWorld, vec2f(1.0), c);
        }

        n += UtilGenerateCirclePoints(&pos[3 * n], &col[3 * n], vec3f(1,1,0),
                                      center, circSphereRadius, c);
        return n;
    };

    UtilRunDynamicSimulation2<PciSphSolver2, ParticleSet2>(&solver, set2, spacing,
                                                           vec2f(-1, -1), vec2f(1, 1),
                                                           targetInterval, 2 * c,
                                                           updateCallback, colorFunction,
                                                           filler);

    CudaMemoryManagerClearCurrent();
    printf("===== OK\n");
}

void test_pcisph2_water_sphere_dynamic(){
    printf("===== PCISPH Solver 2D -- Water Block with movable container\n");
    Float spacing = 0.008;
    Float spacingScale = 2.0;
    Float targetDensity = WaterDensity;
    vec2f center(0,0);
    Float lenc = 2;
    vec2f direction(1.0);

    Bounds2f containerBounds;
    ParticleSetBuilder2 builder;
    ColliderSetBuilder2 colliderBuilder;

    PciSphSolver2 solver;

    CudaMemoryManagerStart(__FUNCTION__);

    Shape2 *rect = MakeRectangle2(Translate2(center.x, center.y+0.35), vec2f(0.4));
    Shape2 *block = MakeSphere2(Translate2(center.x, center.y+0.2), 0.5, true);
    Shape2 *container = MakeRectangle2(Translate2(center.x, center.y), vec2f(lenc), true);

    center = vec2f(center.x, center.y+0.2);

    containerBounds = Bounds2f(vec2f(-5,-5), vec2f(5, 5));

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

    auto updateCallback = [&](int frame) -> int{
        Float off = spacing;
        Float y = center.y + direction.y * off;
        Float x = center.x + direction.x * off;
        Float v = off / targetInterval;
        if(Absf(y) > 0.5){
            direction.y = -direction.y;
        }

        if(Absf(x) > 0.5){
            direction.x = -direction.x;
        }

        center += direction * off;
        vec2f vel = direction * vec2f(v, v);
        block->Update(Translate2(center.x, center.y));
        if(direction.y > 0){
            block->SetVelocities(vel, -20.0);
        }else{
            block->SetVelocities(vel, 20.0);
        }

        return 1;
    };

    int c = 120;
    auto filler = [&](float *pos, float *col) -> int{
        vec2f pCenter = block->ObjectToWorld.Point(vec2f(0));
        return UtilGenerateCirclePoints(pos, col, vec3f(1,1,0), pCenter, block->radius, c);
    };

    UtilRunDynamicSimulation2<PciSphSolver2, ParticleSet2>(&solver, set2, spacing,
                                                           vec2f(-1, -1), vec2f(1, 1),
                                                           targetInterval, c,
                                                           updateCallback, colorFunction,
                                                           filler);

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
    vec2f direction(1.0);

    Bounds2f containerBounds;
    ParticleSetBuilder2 builder;
    ColliderSetBuilder2 colliderBuilder;

    PciSphSolver2 solver;

    CudaMemoryManagerStart(__FUNCTION__);

    Shape2 *rect = MakeRectangle2(Translate2(center.x, center.y+0.45), vec2f(1));
    Shape2 *block = MakeSphere2(Translate2(center.x, center.y-0.3), 0.2);
    Shape2 *container = MakeRectangle2(Translate2(center.x, center.y), vec2f(lenc), true);

    // Copy the sphere center
    center = vec2f(center.x, center.y - 0.3);

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

    auto updateCallback = [&](int frame) -> int{
        Float off = spacing;
        Float y = center.y + direction.y * off;
        Float x = center.x + direction.x * off;
        Float v = off / targetInterval;
        if(Absf(y) > 0.5){
            direction.y = -direction.y;
        }

        if(Absf(x) > 0.5){
            direction.x = -direction.x;
        }

        center += direction * off;
        vec2f vel = direction * vec2f(v, v);
        block->Update(Translate2(center.x, center.y));
        if(direction.y > 0){
            block->SetVelocities(vel, -20.0);
        }else{
            block->SetVelocities(vel, 20.0);
        }

        return 1;
    };

    int c = 120;
    auto filler = [&](float *pos, float *col) -> int{
        vec2f pCenter = block->ObjectToWorld.Point(vec2f(0));
        UtilGenerateCirclePoints(pos, col, vec3f(1,1,0), pCenter, block->radius, c);
        return c;
    };

    //SetCPUThreads(1);
    //SetSystemUseCPU();

    UtilRunDynamicSimulation2<PciSphSolver2, ParticleSet2>(&solver, set2, spacing,
                                                           vec2f(-1, -1), vec2f(1, 1),
                                                           targetInterval, c,
                                                           updateCallback, colorFunction,
                                                           filler);

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

    ProfilerInitKernel(count);

    colliderBuilder.AddCollider2(container);
    ColliderSet2 *collider = colliderBuilder.GetColliderSet();

    solver.SetViscosityCoefficient(0.01);
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
        ProfilerReport();
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

    collider->GenerateSDFs();

    solver.Setup(targetDensity, spacing, 2.0, grid, sphSet);
    solver.SetColliders(collider);

    Float targetInterval = 1.0 / 240.0;
    float *pos = new float[count * 3];
    float *col = new float[count * 3];

    SphSolverData2 *data = solver.GetSphSolverData();

    for(int i = 0; i < count * 3; i++){
        col[i] = 0.6;
    }

    for(int j = 0; j < 20 * 26 * 20; j++){
        solver.Advance(targetInterval);
        //set_colors_pressure(col, data);
        //set_colors_lnm(col, data, 1, 1);
        Debug_GraphyDisplaySolverParticles(sphSet->GetParticleSet(), pos, col);
        if(j == 200){
#if 0
            std::vector<int> bounds;
            for(int s = 0; s < set2->GetParticleCount(); s++){
                vec2f p = set2->GetParticlePosition(s);
                unsigned int id = data->domain->GetLinearHashedPosition(p);
                Cell2 *cell = data->domain->GetCell(id);
                int level = cell->GetLevel();
                bounds.push_back(level);
            }

            UtilEraseFile("test_out2d.txt");
            SerializerSaveSphDataSet2Legacy(data, "test_out2d.txt",
                     SERIALIZER_POSITION | SERIALIZER_BOUNDARY, &bounds);

            SerializerSaveDomain(data, "test_domain2d.txt");
            getchar();
#endif
        }
    }

    delete[] pos;
    delete[] col;
    printf("===== OK\n");
}
