#include <pcisph_solver.h>
#include <emitter.h>
#include <tests.h>
#include <grid.h>
#include <graphy.h>
#include <serializer.h>
#include <unistd.h>
#include <util.h>
#include <memory.h>
#include <boundary.h>

void set_particle_color(float *pos, float *col, ParticleSet3 *pSet);
void simple_color(float *pos, float *col, ParticleSet3 *pSet);

DeclareFunctionalInteraction3D(HelixGravity3D,
{
    //Float radius = 1.25f;
    vec2f centerXZ = vec2f(0);
    vec2f projPXZ = vec2f(point.x, point.z);

    vec2f normal = Normalize(projPXZ - centerXZ);
    vec2f tang = vec2f(-normal.y, normal.x);
    Float radial = 0.2;
    Float tangential = 2.0;
    vec2f radial_acc = -radial * normal;
    vec2f tang_acc = tangential * tang;

    Float len = 1.8f;
    vec2f centrip = (radial_acc + tang_acc) * len;
    return vec3f(centrip.x, 0.f, centrip.y);
})

void test_pcisph3_helix(){
    printf("===== PCISPH Solver 3D -- Helix\n");
    CudaMemoryManagerStart(__FUNCTION__);

    Float baseContainerSize = 2.0f;
    Float spacing = 0.02;
    Float spacingScale = 1.8;
    vec3f origin(-6.0, 3.0, 0.0);
    vec3f target(0.0f);
    vec3f emitterVelocity(0.0f, 0.0f, 0.0f);

    Float ballRadius = baseContainerSize / 16.0f;
    vec3f containerSize(baseContainerSize * 2.f);
    vec3f sphereCenter(0.f, -containerSize.y * 0.48 + ballRadius,
                       -1.25f);

    Shape *container = MakeBox(Transform(), containerSize, true);
    Shape *sphereEmitter = MakeSphere(Translate(sphereCenter), ballRadius);

    printf("Ball radius= %g\n", ballRadius);
    printf("Ball center= {%g %g %g}\n", sphereCenter.x, sphereCenter.y, sphereCenter.z);
    std::cout << "Container bounds: " << container->GetBounds() << std::endl;

    // Colliders
    ColliderSetBuilder3 cBuilder;
    cBuilder.AddCollider3(container);

    ColliderSet3 *colliders = cBuilder.GetColliderSet();

    ContinuousParticleSetBuilder3 pBuilder;
    VolumeParticleEmitterSet3 emitterSet;
    emitterSet.AddEmitter(sphereEmitter, spacing);
    pBuilder.SetKernelRadius(spacing * spacingScale);

    Assure(UtilIsEmitterOverlapping(&emitterSet, colliders) == 0);

    emitterSet.Emit(&pBuilder);
    SphParticleSet3 *sphSet = SphParticleSet3FromContinuousBuilder(&pBuilder);
    ParticleSet3 *pSet = sphSet->GetParticleSet();
    Grid3 *domainGrid = UtilBuildGridForDomain(container->GetBounds(),
                                               spacing, spacingScale);

    PciSphSolver3 solver;
    SphSolverData3 *solverData = DefaultSphSolverData3(false);

    InteractionsBuilder3 intrBuilder;
    AddFunctionalInteraction3D(intrBuilder, HelixGravity3D);

    solverData->fInteractions =
        intrBuilder.MakeFunctionalInteractions(solverData->fInteractionsCount);

    solver.Initialize(solverData);
    solver.Setup(WaterDensity, spacing, spacingScale, domainGrid, sphSet);
    solver.SetColliders(colliders);

    pBuilder.MapGrid(domainGrid);
    Float targetInterval =  1.0 / 240.0;
    int extraParts = 24 * 10;
    auto filler = [&](float *pos, float *col) -> int{
        int f = UtilGenerateBoxPoints(pos, col, vec3f(1,1,0), containerSize,
                                      extraParts, container->ObjectToWorld);
        return f;
    };

    auto colorFunction = [&](float *colors, int pCount) -> void{
        for(int i = 0; i < pCount; i++){
            colors[3 * i + 0] = 1; colors[3 * i + 1] = 0;
            colors[3 * i + 2] = 1;
        }
    };

    auto velocityField = [&](const vec3f &p) -> vec3f{ return emitterVelocity; };
    auto onStepUpdate = [&](int step) -> int{
        if(step == 0) return 1;
        if(pSet->GetParticleCount() < 300000){
            pBuilder.MapGridEmit(velocityField, spacing);
        }

        UtilPrintStepStandard(&solver, step-1);

        std::string path = FrameOutputPath("helix/out_", step-1);
        SerializerSaveSphDataSet3Legacy(solver.GetSphSolverData(), path.c_str(),
                                        SERIALIZER_POSITION);

        return 1;
    };

    UtilRunDynamicSimulation3<PciSphSolver3, ParticleSet3>(&solver, pSet, spacing, origin,
                                                           target, targetInterval, extraParts,
                                                           {}, onStepUpdate, colorFunction,
                                                           filler);

    CudaMemoryManagerClearCurrent();
    printf("===== OK\n");
}

DeclareFunctionalInteraction2D(TestHelix2D,
{
    Float centerY = 0.5f;
    Float difY = point.y - centerY;
    Float yAcc = -difY * 140.f;
    return vec2f(1.0f, yAcc);
})

DeclareFunctionalInteraction2D(TestGravity2D,
{
    vec2f center(0.f, -0.3f);
    vec2f normal = Normalize(point - center);
    vec2f tang = vec2f(-normal.y, normal.x);
    Float radial = 1.0;
    Float tangential = 0.1;
    vec2f radial_acc = -radial * normal;
    vec2f tang_acc = tangential * tang;
    Float len = 9.8f;

    return (radial_acc + tang_acc) * len;
})

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

    SphSolverData2 *solverData = DefaultSphSolverData2(false);

    solver.Initialize(solverData);
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

    InteractionsBuilder2 intrBuilder;
    AddFunctionalInteraction2D(intrBuilder, TestGravity2D);

    data->fInteractions = intrBuilder.MakeFunctionalInteractions(data->fInteractionsCount);

    SandimWorkQueue2 *vpWorkQ = cudaAllocateVx(SandimWorkQueue2, 1);
    vpWorkQ->SetSlots(grid->GetCellCount());
    Float sphRadius = data->sphpSet->GetKernelRadius();
    WorkQueue<vec4f> *marroneWorkQ = cudaAllocateVx(WorkQueue<vec4f>, 1);
    marroneWorkQ->SetSlots(set2->GetParticleCount());

    while(1){
        solver.Advance(targetInterval);
        //set_colors_pressure(col, data);

        for(int k = 0; k < set2->GetParticleCount(); k++){
            set2->SetParticleV0(k, 0);
        }

        vpWorkQ->Reset();
        marroneWorkQ->Reset();
        ComputeNormalGPU(data);

        IntervalBoundary(set2, grid, sphRadius);
        //MarroneBoundary(set2, grid, sphRadius);
        //MarroneAdaptBoundary(set2, grid, sphRadius, marroneWorkQ);
        //DiltsSpokeBoundary(set2, grid);
        //CFBoundary(set2, grid, spacing);
        //XiaoweiBoundary(set2, grid, spacing);
        //SandimBoundary(set2, grid, vpWorkQ);
        //LNMBoundary(set2, grid, spacing);
        //LNMBoundarySingle(set2, grid, spacing);
        //RandlesDoringBoundary(set2, grid, spacing);

        set_colors_lnm(col, data, 0, 0);
        Debug_GraphyDisplaySolverParticles(sphSet->GetParticleSet(), pos, col);
        //if(i == 180) getchar();
        //getchar();
    }

    delete[] pos;
    delete[] col;
    CudaMemoryManagerClearCurrent();

    printf("===== OK\n");
}

void test_pcisph2_helix(){
    printf("===== PCISPH Solver 2D -- Helix\n");
    Float spacing = 0.008;
    Float spacingScale = 1.8;
    Float targetDensity = WaterDensity;
    PciSphSolver2 solver;
    CudaMemoryManagerStart(__FUNCTION__);

    Float ballRadius = 0.05f;
    Float len = 2.0f;
    vec2f left(-len, -len);
    vec2f right(len, len);
    vec2f origin(0.f, 0.f);
    vec2f containerSize = right - left;
    vec2f emitterVelocity(1.0f, 0.0f);

    Float xPos = origin.x - 0.48 * containerSize.x + ballRadius;
    Float yPos = origin.y - ballRadius;

    Shape2 *container = MakeRectangle2(Translate2(origin.x, origin.y), containerSize, true);
    Shape2 *sphere = MakeSphere2(Translate2(xPos, yPos), ballRadius);

    Grid2 *grid = UtilBuildGridForDomain(container->GetBounds(), spacing, spacingScale);

    ContinuousParticleSetBuilder2 builder(50000);
    builder.SetKernelRadius(spacing * spacingScale);

    auto velocityField = [&](const vec2f &p) -> vec2f{ return emitterVelocity; };

    VolumeParticleEmitterSet2 emitter;
    emitter.AddEmitter(sphere, sphere->GetBounds(), spacing);
    emitter.Emit(&builder, velocityField);

    SphParticleSet2 *sphSet = SphParticleSet2FromContinuousBuilder(&builder);
    ParticleSet2 *set2 = sphSet->GetParticleSet();

    ColliderSetBuilder2 colliderBuilder;
    colliderBuilder.AddCollider2(container);
    ColliderSet2 *collider = colliderBuilder.GetColliderSet();

    SphSolverData2 *data = DefaultSphSolverData2(false);
    solver.Initialize(data);

    solver.Setup(targetDensity, spacing, spacingScale, grid, sphSet);
    solver.SetColliders(collider);

    Float targetInterval = 1.0 / 240.0;
    builder.MapGrid(grid);

    InteractionsBuilder2 intrBuilder;

    AddFunctionalInteraction2D(intrBuilder, TestHelix2D);
    data->fInteractions = intrBuilder.MakeFunctionalInteractions(data->fInteractionsCount);

    int maxframes = 1000;
    auto onStepUpdate = [&](int step) -> int{
        if(step == 0) return 1;
        if(step < maxframes){
            builder.MapGridEmit(velocityField, spacing);
        }
        return 1;
    };

    auto colorFunction = [&](float *colors, int pCount) -> void{
        (void)pCount;
        set_colors_lnm(colors, data, 0, 0);
    };

    UtilRunSimulation2<PciSphSolver2, ParticleSet2>(&solver, set2, spacing,
                                                    left, right, targetInterval,
                                                    onStepUpdate, colorFunction);
    CudaMemoryManagerClearCurrent();
    printf("===== OK\n");
}

void test_pcisph3_dam_break(){
    printf("===== PCISPH Solver 3D -- Dam Break\n");
    Float domainScaling = 2.5f;
    vec3f origin(3 * domainScaling, 1 * domainScaling, 3 * domainScaling);
    vec3f target(0,0,0);

    Float spacing = 0.02f;
    Float spacingScale = 1.8f;
    Float boxFluidLen = 0.5 * domainScaling;
    Float boxFluidYLen = 0.9 * domainScaling;
    Float boxLen = 1.3 * domainScaling;
    Float boxYLen = 1.2 * domainScaling;

    vec3f containerSize = vec3f(boxLen, boxYLen, boxLen);
    Float xof = (containerSize.x - boxFluidLen)/2.0; xof -= spacing;
    Float zof = (containerSize.z - boxFluidLen)/2.0; zof -= spacing;
    Float yof = (containerSize.y - boxFluidYLen)/2.0; yof -= spacing;

    vec3f boxSize = vec3f(boxFluidLen, boxFluidYLen, boxFluidLen);

    Shape *container = MakeBox(Transform(), containerSize, true);
    Shape *boxp = MakeBox(Translate(xof, -yof, zof), boxSize);

    ParticleSetBuilder3 pBuilder;
    VolumeParticleEmitterSet3 emitterSet;
    VolumeParticleEmitter3 emitterp(boxp, boxp->GetBounds(), spacing, vec3f(0,-6,0));

    emitterSet.AddEmitter(&emitterp);
    emitterSet.SetJitter(0.001);
    emitterSet.Emit(&pBuilder);

    Grid3 *domainGrid = UtilBuildGridForDomain(container->GetBounds(), spacing,
                                               spacingScale);

    ColliderSetBuilder3 cBuilder;
    cBuilder.AddCollider3(container);

    ColliderSet3 *colliders = cBuilder.GetColliderSet();

    PciSphSolver3 solver;
    SphParticleSet3 *sphSet = SphParticleSet3FromBuilder(&pBuilder);
    sphSet->SetRelativeKernelRadius(spacingScale);

    solver.Initialize(DefaultSphSolverData3());
    solver.Setup(WaterDensity, spacing, spacingScale, domainGrid, sphSet);
    solver.SetColliders(colliders);

    ParticleSet3 *pSet = sphSet->GetParticleSet();

    auto callback = [&](int step) -> int{
        if(step == 0)
            return 1;
        UtilPrintStepStandard(&solver, step-1);
        ProfilerReport();

        std::string path = FrameOutputPath("dam/out_", step-1);
        SerializerSaveSphDataSet3(solver.GetSphSolverData(), path.c_str(),
                                  SERIALIZER_POSITION);

        return step > 600 ? 0 : 1;
    };

    Float targetInterval =  1.0 / 240.0;
    PciSphRunSimulation3(&solver, spacing, origin, target,
                         targetInterval, {}, callback);

    printf("===== OK\n");
}


vec3f rotateY(vec3f v, Float rads){
    Float x = v.z*sin(rads) + v.x*cos(rads);
    Float y = v.y;
    Float z = v.z*cos(rads) - v.x*sin(rads);
    return vec3f(x, y, z);
}

/*
* NOTE: The idea for the following simulation comes from:
*    https://github.com/rlguy/GridFluidSim3D
* I made some slight changes. The distances don't really match
* but the overall result is interesting enough.
*/
DeclareFunctionalInteraction3D(GravityField3D,
{
    vec3f center(0.f, 0.f, 0.f);
    Float ming = 1.0f;
    Float maxg = 25.0f;
    Float mindsq = 1.0f;
    Float maxdsq = 8.0 * 8.0;
    vec3f v = center - point;

    Float distsq = v.LengthSquared();
    if(distsq < 1e-6)
        return vec3f(0.f, 0.f, 0.f);

    Float distfactor = 1.0 - (distsq - mindsq) / (maxdsq - mindsq);
    Float gstrength = ming + distfactor * (maxg - ming);
    return Normalize(v) * gstrength;
})

void test_pcisph3_gravity_field(){
    printf("===== PCISPH Solver 3D -- Gravity Field\n");
    CudaMemoryManagerStart(__FUNCTION__);
    Float baseContainerSize = 2.0f;
    Float spacing = 0.02;
    Float spacingScale = 1.8;
    int frame_index = 1;

    vec3f origin(4.0, 0.0, 0.0);
    vec3f target(0.0f);
    vec3f emitterVelocity(8.0f, 0.0f, 0.0f);

    vec3f currentEmitterVelocity = emitterVelocity;
    Float half_size = 0.5 * baseContainerSize;
    Float sphereRadius = baseContainerSize / 16.0f;
    vec3f containerSize(baseContainerSize * 2.f);
    vec3f sphereCenter(0.f, 7.0f * sphereRadius, 0.f);
    Shape *container = MakeBox(Transform(), containerSize, true);
    Shape *sphereCollider = MakeSphere(Translate(vec3f(0, 0, 0)), 2.0f * sphereRadius);
    Shape *sphereEmitter = MakeSphere(Translate(sphereCenter), sphereRadius);

    printf("Collider radius= %g\n", sphereCollider->radius);
    printf("Emitter radius= %g\n", sphereEmitter->radius);
    std::cout << "Container= " << container->GetBounds() << std::endl;

    // Colliders
    ColliderSetBuilder3 cBuilder;
    // TODO: Add the central sphere?
    cBuilder.AddCollider3(sphereCollider);
    cBuilder.AddCollider3(container);
    ColliderSet3 *colliders = cBuilder.GetColliderSet();

    // Emitter
    ContinuousParticleSetBuilder3 pBuilder;
    VolumeParticleEmitterSet3 emitterSet;
    emitterSet.AddEmitter(sphereEmitter, spacing);
    pBuilder.SetKernelRadius(spacing * spacingScale);

    Assure(UtilIsEmitterOverlapping(&emitterSet, colliders) == 0);

    // Emit into the continuous builder
    emitterSet.Emit(&pBuilder);
    // Get the results and build domain
    SphParticleSet3 *sphSet = SphParticleSet3FromContinuousBuilder(&pBuilder);
    ParticleSet3 *pSet = sphSet->GetParticleSet();
    Grid3 *domainGrid = UtilBuildGridForDomain(container->GetBounds(),
                                               spacing, spacingScale);


    // Build Solver
    PciSphSolver3 solver;
    SphSolverData3 *solverData = DefaultSphSolverData3(false);
    InteractionsBuilder3 intrBuilder;
    AddFunctionalInteraction3D(intrBuilder, GravityField3D);

    solverData->fInteractions =
        intrBuilder.MakeFunctionalInteractions(solverData->fInteractionsCount);

    solver.Initialize(solverData);
    solver.Setup(WaterDensity, spacing, spacingScale, domainGrid, sphSet);
    solver.SetColliders(colliders);

    // Map particles to domain for easy continuous emition
    pBuilder.MapGrid(domainGrid);

    // Visualization
    Float targetInterval =  1.0 / 240.0;
    int extraParts = 24 * 10;
    auto filler = [&](float *pos, float *col) -> int{
        int f = UtilGenerateBoxPoints(pos, col, vec3f(1,1,0), containerSize,
                                      extraParts, container->ObjectToWorld);
        return f;
    };

    auto colorFunction = [&](float *colors, int pCount) -> void{
        for(int i = 0; i < pCount; i++){
            colors[3 * i + 0] = 1; colors[3 * i + 1] = 0;
            colors[3 * i + 2] = 1;
        }
    };

    auto update_velocity = [&](){
        Float simulationTime = frame_index * targetInterval;
        Float minAngle = -0.25f * Pi;
        Float maxAngle = 0.25f * Pi;
        Float rotationSpeed = 0.35f * Pi;
        Float rotationFactor = sin(rotationSpeed * simulationTime);
        Float rads = minAngle + rotationFactor * (maxAngle - minAngle);
        currentEmitterVelocity = rotateY(emitterVelocity, rads);
    };

    auto velocityField = [&](const vec3f &p) -> vec3f{
        return currentEmitterVelocity;
    };

    auto onStepUpdate = [&](int step) -> int{
        update_velocity();
        if(step == 0) return 1;
        if(pSet->GetParticleCount() < 300000){
            pBuilder.MapGridEmit(velocityField, spacing);
        }

        UtilPrintStepStandard(&solver, step-1);

        std::string path = FrameOutputPath("gravity/out_", step-1);
        SerializerSaveSphDataSet3Legacy(solver.GetSphSolverData(), path.c_str(),
                                        SERIALIZER_POSITION);
        frame_index += 1;
        return 1;
    };

    UtilRunDynamicSimulation3<PciSphSolver3, ParticleSet3>(&solver, pSet, spacing, origin,
                                                           target, targetInterval, extraParts,
                                                           {}, onStepUpdate, colorFunction,
                                                           filler);

    CudaMemoryManagerClearCurrent();
    printf("===== OK\n");
}

void test_pcisph3_box_drop(){
    printf("===== PCISPH Solver 3D -- Box Drop\n");
    CudaMemoryManagerStart(__FUNCTION__);

    vec3f origin(4.0, 0, 0);
    vec3f target(0.0f);
    vec3f containerSize(2.0);
    vec3f boxEmitSize(0.4, 0.1, 0.4);
    vec3f boxSize0(0.4);
    Float spacing = 0.02;
    Float spacingScale = 1.8;

    Shape *container = MakeBox(Transform(), containerSize, true);

    Float y0of = (containerSize.y - boxSize0.y) * 0.5; y0of -= spacing;
    Shape *box0 = MakeBox(RotateY(45) * RotateX(45.0), boxSize0);

    Float yEof = (containerSize.y - boxEmitSize.y) * 0.5; yEof -= spacing;
    Shape *boxEmitter = MakeBox(Translate(0, yEof, 0), boxEmitSize);

    ColliderSetBuilder3 cBuilder;
    cBuilder.AddCollider3(box0);
    cBuilder.AddCollider3(container);

    ContinuousParticleSetBuilder3 pBuilder;
    VolumeParticleEmitterSet3 emitterSet;
    emitterSet.AddEmitter(boxEmitter, spacing);

    pBuilder.SetKernelRadius(spacing * spacingScale);
    ColliderSet3 *colliders = cBuilder.GetColliderSet();
    Assure(UtilIsEmitterOverlapping(&emitterSet, colliders) == 0);

    emitterSet.Emit(&pBuilder);
    SphParticleSet3 *sphSet = SphParticleSet3FromContinuousBuilder(&pBuilder);
    ParticleSet3 *pSet = sphSet->GetParticleSet();
    Grid3 *domainGrid = UtilBuildGridForDomain(container->GetBounds(),
                                               spacing, spacingScale);

    PciSphSolver3 solver;
    SphSolverData3 *solverData = DefaultSphSolverData3();

    solver.Initialize(solverData);
    solver.Setup(WaterDensity, spacing, spacingScale, domainGrid, sphSet);
    solver.SetColliders(colliders);

    Float targetInterval =  1.0 / 240.0;
    pBuilder.MapGrid(domainGrid);
    int extraParts = 24 * 10;

    auto velocityField = [&](const vec3f &p) -> vec3f{
        Float u1 = rand_float();
        Float u2 = rand_float();
        Float sign1 = rand_float() < 0.5 ? 1 : -1;
        Float sign2 = rand_float() < 0.5 ? 1 : -1;
        return vec3f(u1 * sign1 * 5.f, -10, u2 * sign2 * 5.f);
    };

    auto filler = [&](float *pos, float *col) -> int{
        int f = UtilGenerateBoxPoints(pos, col, vec3f(1,1,0), containerSize,
                                      extraParts * 0.8, container->ObjectToWorld);
        f += UtilGenerateBoxPoints(&pos[3 * f], &col[3 * f], vec3f(1,1,0), boxSize0,
                                   extraParts * 0.2, box0->ObjectToWorld);
        return f;
    };

    auto colorFunction = [&](float *colors, int pCount) -> void{
        for(int i = 0; i < pCount; i++){
            colors[3 * i + 0] = 1; colors[3 * i + 1] = 0;
            colors[3 * i + 2] = 1;
        }
    };

    auto onStepUpdate = [&](int step) -> int{
        if(step == 0) return 1;
        if(pSet->GetParticleCount() < 300000){
            pBuilder.MapGridEmit(velocityField, spacing);
        }

        UtilPrintStepStandard(&solver, step-1);

        std::string path = FrameOutputPath("box/out_", step-1);
        SerializerSaveSphDataSet3Legacy(solver.GetSphSolverData(), path.c_str(),
                                        SERIALIZER_POSITION);

        return 1;
    };

    UtilRunDynamicSimulation3<PciSphSolver3, ParticleSet3>(&solver, pSet, spacing, origin,
                                                           target, targetInterval, extraParts,
                                                           {}, onStepUpdate, colorFunction,
                                                           filler);

    CudaMemoryManagerClearCurrent();

    printf("===== OK\n");
}

