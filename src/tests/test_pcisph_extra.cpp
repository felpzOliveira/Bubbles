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

vec3f rotateY(vec3f v, Float rads){
    Float x = v.z*sin(rads) + v.x*cos(rads);
    Float y = v.y;
    Float z = v.z*cos(rads) - v.x*sin(rads);
    return vec3f(x, y, z);
}

/*
* NOTE: The idea for the following simulation comes from:
*    https://github.com/rlguy/GridFluidSim3D
* I made some slight changes.
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
    vec3f sphereCenter(0.f, half_size - 0.1 * baseContainerSize, 0.f);
    Shape *container = MakeBox(Transform(), containerSize, true);
    Shape *sphereCollider = MakeSphere(Translate(vec3f(0, 0, 0)), 0.3f);
    Shape *sphereEmitter = MakeSphere(Translate(sphereCenter), sphereRadius);

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

        std::string path("/home/felpz/Documents/Bubbles/simulations/gravity/out_");
        path += std::to_string(step-1);
        path += ".txt";
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
#if 0
        std::string path("/media/felipe/FluidStuff/box/out_");
        path += std::to_string(step-1);
        path += ".txt";
        SerializerSaveSphDataSet3Legacy(solver.GetSphSolverData(), path.c_str(),
                                        SERIALIZER_POSITION);
#endif
        return 1;
    };

    UtilRunDynamicSimulation3<PciSphSolver3, ParticleSet3>(&solver, pSet, spacing, origin,
                                                           target, targetInterval, extraParts,
                                                           {}, onStepUpdate, colorFunction,
                                                           filler);

    CudaMemoryManagerClearCurrent();

    printf("===== OK\n");
}

