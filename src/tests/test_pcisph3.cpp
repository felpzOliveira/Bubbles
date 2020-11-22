#include <pcisph_solver.h>
#include <emitter.h>
#include <tests.h>
#include <grid.h>
#include <graphy.h>
#include <serializer.h>
#include <string>
#include <unistd.h>
#include <obj_loader.h>
#include <util.h>
#include <memory.h>

//NOTE: Using ContinuousParticleSetBuilder3 shows a issue??
void test_pcisph3_water_drop(){
    printf("===== PCISPH Solver 3D -- Water Drop\n");
    vec3f origin(3);
    vec3f target(0);
    vec3f boxSize(3.0, 2.0, 3.0);
    Float spacing = 0.02;
    Float spacingScale = 2.0;
    Float sphereRadius = 0.3;
    vec3f waterBox(3.0-spacing, 0.3, 3.0-spacing);
    
    CudaMemoryManagerStart(__FUNCTION__);
    
    Float sphereY = boxSize.y * 0.5 - sphereRadius; sphereY -= spacing;
    Float yof = (boxSize.y - waterBox.y) * 0.5; yof -= spacing;
    Shape *container = MakeBox(Transform(), boxSize, true);
    Shape *baseWaterShape = MakeBox(Translate(0, -yof, 0), waterBox);
    Shape *sphere = MakeSphere(Translate(0, sphereY, 0), sphereRadius);
    
    VolumeParticleEmitterSet3 emitters;
    emitters.AddEmitter(baseWaterShape, spacing);
    emitters.AddEmitter(sphere, spacing);
    
    ColliderSetBuilder3 cBuilder;
    cBuilder.AddCollider3(container);
    ColliderSet3 *colliders = cBuilder.GetColliderSet();
    
    Assure(UtilIsEmitterOverlapping(&emitters, colliders) == 0);
    
    ParticleSetBuilder3 pBuilder;
    Float intensity = 10.0;
    auto velocityField = [&](const vec3f &p) -> vec3f{
        vec3f v(0);
        if(p.y > 0){
            Float u1 = rand_float() * 0.5 + 0.5;
            v = vec3f(0, -1, 0) * intensity * u1;
        }
        
        return v;
    };
    
    emitters.Emit(&pBuilder, velocityField);
    
    SphParticleSet3 *sphSet = SphParticleSet3FromBuilder(&pBuilder);
    Grid3 *domainGrid = UtilBuildGridForDomain(container->GetBounds(), 
                                               spacing, spacingScale);
    
    PciSphSolver3 solver;
    solver.Initialize(DefaultSphSolverData3());
    solver.Setup(WaterDensity, spacing, spacingScale, domainGrid, sphSet);
    solver.SetColliders(colliders);
    
    Float targetInterval =  1.0 / 240.0;
#if 0
    auto onStepUpdate = [&](int step) -> int{
        if(step == 0) return 1;
        std::string respath("/home/felipe/Documents/Bubbles/simulations/water_drop/output_");
        respath += std::to_string(step-1);
        respath += ".txt";
        int flags = (SERIALIZER_POSITION | SERIALIZER_DENSITY | SERIALIZER_MASS);
        SerializerSaveSphDataSet3(solver.GetSphSolverData(), respath.c_str(), flags);
        
        UtilPrintStepStandard(&solver, step-1);
        return step > 500 ? 0 : 1;
    };
#endif
    
    auto onStepUpdate = [&](int step) -> int{
        if(step == 0) return 1;
        UtilPrintStepStandard(&solver, step-1, {0, 16, 31, 74, 151, 235, 
                                  256, 278, 361, 420});
        ProfilerReport();
        return step > 450 ? 0 : 1;
    };
    
    PciSphRunSimulation3(&solver, spacing, origin, target, 
                         targetInterval, {}, onStepUpdate);
    
    CudaMemoryManagerClearCurrent();
    printf("===== OK\n");
}

void test_pcisph3_box_many_emission(){
    printf("===== PCISPH Solver 3D -- Box Many Emission\n");
    vec3f origin(2);
    vec3f target(0);
    vec3f boxSize(2.0, 1.0, 1.0);
    vec2i pointRes(10, 5); //xz
    Float spacingScale = 2.0;
    Float spacing = 0.02;
    int maxframes = 500;
    Float pointHeight = boxSize.y * 0.5 - 0.1 - spacing;
    CudaMemoryManagerStart(__FUNCTION__);
    
    Shape *container = MakeBox(Transform(), boxSize, true);
    
    Float dx = boxSize.x / Float(pointRes.x+1);
    Float dz = boxSize.z / Float(pointRes.y+1);
    Float r = Min(dx, dz) / 6.0;
    
    VolumeParticleEmitterSet3 emitterSet;
    Float fx0 = -boxSize.x * 0.5 + dx;
    Float fz0 = -boxSize.z * 0.5 + dz;
    for(int x = 0; x < pointRes.x; x++){
        for(int z = 0; z < pointRes.y; z++){
            Float fx = fx0 + dx * (x);
            Float fz = fz0 + dz * (z);
            vec3f p(fx, pointHeight, fz);
            emitterSet.AddEmitter(MakeSphere(Translate(p), r), spacing);
        }
    }
    
    emitterSet.SetJitter(0.01);
    
    ContinuousParticleSetBuilder3 pBuilder;
    
    ColliderSetBuilder3 cBuilder;
    cBuilder.AddCollider3(container);
    ColliderSet3 *colliders = cBuilder.GetColliderSet();
    
    Assure(UtilIsEmitterOverlapping(&emitterSet, colliders) == 0);
    
    Float intensity = 10.0;
    auto velocityField = [&](const vec3f &p) -> vec3f{
        Float u1 = rand_float() * 0.5 + 0.5;
        vec3f vel(0, -1, 0);
        return vel * u1 * intensity;
    };
    
    emitterSet.Emit(&pBuilder, velocityField);
    SphParticleSet3 *sphSet = SphParticleSet3FromContinuousBuilder(&pBuilder);
    Grid3 *domainGrid = UtilBuildGridForDomain(container->GetBounds(), 
                                               spacing, spacingScale);
    PciSphSolver3 solver;
    solver.Initialize(DefaultSphSolverData3());
    solver.Setup(WaterDensity, spacing, spacingScale, domainGrid, sphSet);
    solver.SetColliders(colliders);
    
    Float targetInterval =  1.0 / 240.0;
    pBuilder.MapGrid(domainGrid);
    
    auto onStepUpdate = [&](int step) -> int{
        if(step == 0) return 1;
        if(step < maxframes){
            pBuilder.MapGridEmit(velocityField, spacing);
        }
        UtilPrintStepStandard(&solver, step-1);
        UtilSaveSph3Frame("box_shower/output_", step-1, solver.GetSphSolverData());
        return step > maxframes ? 0 : 1;
    };
    
    PciSphRunSimulation3(&solver, spacing, origin, target, 
                         targetInterval, {}, onStepUpdate);
    
    CudaMemoryManagerClearCurrent();
    printf("\n===== OK\n");
}

void test_pcisph3_ball_many_emission(){
    printf("===== PCISPH Solver 3D -- Ball Many Emission\n");
    Float ballRadius = 1.0;
    Float baseRadius = 0.02;
    int nPoints = 6;
    vec3f ballCenter(0);
    vec3f origin(0,0,3);
    vec3f target(0);
    Float spacingScale = 2.0;
    Float spacing = 0.02;
    CudaMemoryManagerStart(__FUNCTION__);
    
    Shape *ball = MakeSphere(Translate(ballCenter), ballRadius, true);
    
    ColliderSetBuilder3 cBuilder;
    cBuilder.AddCollider3(ball);
    
    Float alpha = 90 / (nPoints+1);
    VolumeParticleEmitterSet3 emitterSet;
    for(int i = 0; i < nPoints; i++){
        Float beta = i * alpha;
        Float x = ballRadius * std::cos(Radians(beta));
        Float y = ballRadius * std::sin(Radians(beta));
        vec3f basePoint(x, y, 0);
        vec3f mirrorPoint(-x, y, 0);
        
        vec3f normal = -basePoint;
        basePoint = basePoint + spacing * normal;
        mirrorPoint = mirrorPoint - spacing * normal;
        basePoint += ballCenter;
        mirrorPoint += ballCenter;
        
        Shape *shape = MakeSphere(Translate(basePoint), baseRadius);
        Shape *mShape = MakeSphere(Translate(mirrorPoint), baseRadius);
        emitterSet.AddEmitter(shape, spacing);
        emitterSet.AddEmitter(mShape, spacing);
    }
    
    Float height = ballCenter.y + ballRadius;
    vec3f ballTop(0, height - spacing, 0);
    Shape *upShape = MakeSphere(Translate(ballTop), baseRadius);
    emitterSet.AddEmitter(upShape, spacing);
    
    ContinuousParticleSetBuilder3 pBuilder;
    
    ColliderSet3 *colliders = cBuilder.GetColliderSet();
    Assure(UtilIsEmitterOverlapping(&emitterSet, colliders) == 0);
    
    Float intensity = 10.0;
    auto velocityField = [&](const vec3f &p) -> vec3f{
        Float u1 = rand_float() * 0.5 + 0.5;
        vec3f normal = ballCenter - p;
        Float len = normal.Length();
        vec3f vel(0);
        if(!IsZero(len) && len > 0){
            //vel = normal * u1 * intensity / len;
            vel = vec3f(0, -1, 0) * u1 * intensity;
        }
        return vel;
    };
    
    emitterSet.Emit(&pBuilder, velocityField);
    
    SphParticleSet3 *sphSet = SphParticleSet3FromContinuousBuilder(&pBuilder);
    Grid3 *domainGrid = UtilBuildGridForDomain(ball->GetBounds(), 
                                               spacing, spacingScale);
    PciSphSolver3 solver;
    solver.Initialize(DefaultSphSolverData3());
    solver.Setup(WaterDensity, spacing, spacingScale, domainGrid, sphSet);
    solver.SetColliders(colliders);
    
    Float targetInterval =  1.0 / 240.0;
    pBuilder.MapGrid(domainGrid);
    
    auto onStepUpdate = [&](int step) -> int{
        if(step == 0) return 1;
        if(step < 400){
            pBuilder.MapGridEmit(velocityField, spacing);
        }
        UtilPrintStepStandard(&solver, step-1);
        return 1;
    };
    
    PciSphRunSimulation3(&solver, spacing, origin, target, 
                         targetInterval, {}, onStepUpdate);
    
    CudaMemoryManagerClearCurrent();
    printf("===== OK\n");
}


void test_pcisph3_dragon_pool(){
    printf("===== PCISPH Solver 3D -- Dragon Pool\n");
    vec3f origin(2, 1, -4);
    vec3f target(0, 0, 0);
    Float spacingScale = 2.0;
    Float spacing = 0.02;
    vec3f boxSize(2.0, 3.0, 2.0);
    Float xIntensity = 6.0;
    Float emitRadius = 0.15;
    Float targetScale = 0.0f;
    CudaMemoryManagerStart(__FUNCTION__);
    
    const char *dragonPath = "/home/felipe/Documents/CGStuff/models/stanfordDragon.obj";
    ParsedMesh *mesh = LoadObj(dragonPath);
    Transform dragonScale = UtilComputeFitTransform(mesh, boxSize.x, &targetScale);
    Bounds3f bounds = UtilComputeBoundsAfter(mesh, dragonScale);
    Float yof = (boxSize.y - bounds.ExtentOn(1)) * 0.5; yof += 2.2 * spacing;
    
    Shape *dragonShape = MakeMesh(mesh, Translate(0, -yof, 0) * RotateY(-90) * Scale(0.9) * dragonScale);
    Shape *container = MakeBox(Transform(), boxSize, true);
    Shape *shower = MakeSphere(Translate(0.3, 1, 0), emitRadius);
    Shape *shower2 = MakeSphere(Translate(-0.3, 1, 0), emitRadius);
    
    Assure(UtilIsDomainContaining(container->GetBounds(), 
                                  {dragonShape->GetBounds()}) == 1);
    
    vec3f waterBox(2.0-spacing, 0.3, 2.0-spacing);
    yof = (boxSize.y - waterBox.y) * 0.5; yof -= spacing;
    Shape *baseWaterShape = MakeBox(Translate(0, -yof, 0), waterBox);
    
    ContinuousParticleSetBuilder3 pBuilder;
    VolumeParticleEmitterSet3 emitter;
    VolumeParticleEmitter3 boxEmitter(baseWaterShape, spacing);
    emitter.AddEmitter(shower, spacing);
    emitter.AddEmitter(shower2, spacing);
    emitter.SetJitter(0.01);
    
    ColliderSetBuilder3 cBuilder;
    cBuilder.AddCollider3(container);
    cBuilder.AddCollider3(dragonShape);
    ColliderSet3 *colliders = cBuilder.GetColliderSet();
    
    Assure(UtilIsEmitterOverlapping(&emitter, colliders) == 0);
    
    vec3f pInit(xIntensity, -2.0, 0.0);
    vec3f nInit(-xIntensity, -2.0, 0.0);
    auto velocityField = [&](const vec3f &p) -> vec3f{
        Float u1 = rand_float() * 2;
        Float u2 = rand_float() * 0.5 + 0.5;
        if(p.x < 0){
            return pInit * vec3f(u2, u1, 1);
        }else{
            return nInit * vec3f(u2, u1, 1);
        }
    };
    
    emitter.Emit(&pBuilder, velocityField);
    
    Grid3 *domainGrid = UtilBuildGridForDomain(container->GetBounds(), 
                                               spacing, spacingScale);
    
    SphParticleSet3 *sphSet = SphParticleSet3FromContinuousBuilder(&pBuilder);
    ResetAndDistribute(domainGrid, sphSet->GetParticleSet());
    
    pBuilder.MapGrid(domainGrid);
    
    ParticleSet3 *pSet = sphSet->GetParticleSet();
    bounds = container->GetBounds();
    bounds.Reduce(spacing);
    bounds.PrintSelf();
    printf("\n");
    auto validator = [&](const vec3f &p) -> int{
        if(Inside(p, bounds)){
            return MeshShapeIsPointInside(dragonShape, p, 
                                          pSet->GetRadius(), spacing) ? 0 : 1;
        }
        
        return 0;
    };
    
    boxEmitter.SetValidator(validator);
    boxEmitter.Emit(&pBuilder);
    
    PciSphSolver3 solver;
    solver.Initialize(DefaultSphSolverData3());
    solver.Setup(WaterDensity, spacing, spacingScale, domainGrid, sphSet);
    solver.SetColliders(colliders);
    
    Float targetInterval =  1.0 / 240.0;
    
    std::vector<int> boundaries;
    std::vector<int> boundarySamples;
    boundaries.resize(pSet->GetParticleCount());
    std::vector<int> targetSteps = {0, 16, 31, 74, 151, 235, 256, 278, 361};
    
    const char *targetOutput =
        "/home/felipe/Documents/Bubbles/simulations/dragon_pool/output_";
    
    auto callback = [&](int step) -> int{
        if(step == 0) return 1;
        std::string respath(targetOutput);
        respath += std::to_string(step-1);
        respath += ".txt";
        
        int bCount = UtilFillBoundaryParticles(pSet, &boundaries);
        boundarySamples.push_back(bCount);
        
        int flags = SERIALIZER_POSITION | SERIALIZER_DENSITY | 
            SERIALIZER_MASS | SERIALIZER_BOUNDARY;
        SerializerSaveSphDataSet3(solver.GetSphSolverData(), respath.c_str(), 
                                  flags, &boundaries);
        
        UtilPrintStepStandard(&solver, step-1, targetSteps);
        
        for(int i = 0; i < targetSteps.size(); i++){
            if(step == targetSteps[i]){
                Float average = UtilComputeMedium(boundarySamples.data(), 
                                                  boundarySamples.size());
                printf("Boundary Average: %g\n", average);
            }
        }
        
        if(step < 400){
            pBuilder.MapGridEmit(velocityField, spacing);
        }
        return step < 800 ? 1 : 0;
    };
    
    PciSphRunSimulation3(&solver, spacing, origin, target, 
                         targetInterval, {dragonShape}, callback);
    Float medium = UtilComputeMedium(boundarySamples.data(), 
                                     boundarySamples.size());
    printf("Final average: %g\n", medium);
    
    CudaMemoryManagerClearCurrent();
    printf("===== OK\n");
}


void test_pcisph3_dragon_shower(){
    printf("===== PCISPH Solver 3D -- Dragon Shower\n");
    vec3f origin(2, 1, -4);
    vec3f target(0, 0, 0);
    Float spacingScale = 2.0;
    Float spacing = 0.02;
    vec3f boxSize(2.0, 3.0, 2.0);
    Float xIntensity = 6.0;
    Float emitRadius = 0.15;
    CudaMemoryManagerStart(__FUNCTION__);
    
    const char *dragonPath = "/home/felipe/Documents/CGStuff/models/stanfordDragon.obj";
    ParsedMesh *mesh = LoadObj(dragonPath);
    Transform dragonScale = UtilComputeFitTransform(mesh, boxSize.x);
    Bounds3f bounds = UtilComputeBoundsAfter(mesh, dragonScale);
    Float yof = (boxSize.y - bounds.ExtentOn(1)) * 0.5; yof += 2.2 * spacing;
    yof += 0.08;
    
    Shape *dragonShape = MakeMesh(mesh, Translate(0, -yof, -0.08) * RotateY(-90) * dragonScale);
    Shape *container = MakeBox(Transform(), boxSize, true);
    Shape *shower = MakeSphere(Translate(0.3, 1, 0), emitRadius);
    Shape *shower2 = MakeSphere(Translate(-0.3, 1, 0), emitRadius);
    
    ContinuousParticleSetBuilder3 pBuilder;
    VolumeParticleEmitterSet3 emitter;
    emitter.AddEmitter(shower, spacing);
    emitter.AddEmitter(shower2, spacing);
    emitter.SetJitter(0.01);
    
    ColliderSetBuilder3 cBuilder;
    cBuilder.AddCollider3(container);
    cBuilder.AddCollider3(dragonShape);
    ColliderSet3 *colliders = cBuilder.GetColliderSet();
    
    Assure(UtilIsEmitterOverlapping(&emitter, colliders) == 0);
    
    vec3f pInit(xIntensity, -2.0, 0.0);
    vec3f nInit(-xIntensity, -2.0, 0.0);
    auto velocityField = [&](const vec3f &p) -> vec3f{
        Float u1 = rand_float() * 2;
        Float u2 = rand_float() * 0.5 + 0.5;
        if(p.x < 0){
            return pInit * vec3f(u2, u1, 1);
        }else{
            return nInit * vec3f(u2, u1, 1);
        }
    };
    
    emitter.Emit(&pBuilder, velocityField);
    
    SphParticleSet3 *sphSet = SphParticleSet3FromContinuousBuilder(&pBuilder);
    Grid3 *domainGrid = UtilBuildGridForDomain(container->GetBounds(), 
                                               spacing, spacingScale);
    PciSphSolver3 solver;
    solver.Initialize(DefaultSphSolverData3());
    solver.Setup(WaterDensity, spacing, spacingScale, domainGrid, sphSet);
    solver.SetColliders(colliders);
    
    Float targetInterval =  1.0 / 240.0;
    pBuilder.MapGrid(domainGrid);
    
    auto callback = [&](int step) -> int{
        if(step == 0) return 1;
        //std::string respath("shower/output_");
        //respath += std::to_string(step-1);
        //respath += ".txt";
        //SerializerSaveSphDataSet3(solver.GetSphSolverData(), respath.c_str(),
        //SERIALIZER_POSITION);
        
        UtilPrintStepStandard(&solver, step-1, {0, 16, 31, 74, 151, 235, 256, 278, 361});
        
        if(step < 400){
            pBuilder.MapGridEmit(velocityField, spacing);
        }
        
        return 1;
    };
    
    PciSphRunSimulation3(&solver, spacing, origin, target, 
                         targetInterval, {dragonShape}, callback);
    
    CudaMemoryManagerClearCurrent();
    printf("===== OK\n");
}

void test_pcisph3_dragon(){
    printf("===== PCISPH Solver 3D -- Dragon in Ball\n");
    vec3f origin;
    vec3f center;
    Float radius;
    vec3f target(0);
    Float spacingScale = 2.0;
    Float spacing = 0.02;
    CudaMemoryManagerStart(__FUNCTION__);
    
    ParticleSetBuilder3 builder;
    Bounds3f bounds = UtilParticleSetBuilder3FromBB("resources/cuteDragon", &builder);
    
    bounds.BoundingSphere(&center, &radius);
    radius *= 1.3;
    
    origin = bounds.Center() + radius * vec3f(0, 1, 2.5);
    target = bounds.Center();
    
    Shape *container = MakeSphere(Translate(center), radius, true);
    ColliderSetBuilder3 cBuilder;
    cBuilder.AddCollider3(container);
    ColliderSet3 *colliders = cBuilder.GetColliderSet();
    
    SphParticleSet3 *sphSet = SphParticleSet3FromBuilder(&builder);
    Grid3 *domainGrid = UtilBuildGridForDomain(container->GetBounds(), 
                                               spacing, spacingScale);
    
    PciSphSolver3 solver;
    solver.Initialize(DefaultSphSolverData3());
    solver.Setup(WaterDensity, spacing, spacingScale, domainGrid, sphSet);
    solver.SetColliders(colliders);
    
    Float targetInterval = 1.0 / 240.0;
    PciSphRunSimulation3(&solver, spacing, origin, target, targetInterval);
    
    CudaMemoryManagerClearCurrent();
    printf("===== OK\n");
}

void test_pcisph3_multiple_emission(){
    printf("===== PCISPH Solver 3D -- Multiple Emission\n");
    Float spacing = 0.04;
    Float spacingScale = 2.0;
    vec3f origin(3, 0, 0);
    vec3f target(0,0,0);
    CudaMemoryManagerStart(__FUNCTION__);
    
    Shape *shape = MakeSphere(Transform(), 0.2);
    Shape *container = MakeSphere(Transform(), 1.0, true);
    
    ContinuousParticleSetBuilder3 pBuilder(50000);
    VolumeParticleEmitterSet3 emitter;
    emitter.AddEmitter(shape, spacing);
    
    ColliderSetBuilder3 cBuilder;
    cBuilder.AddCollider3(container);
    ColliderSet3 *colliders = cBuilder.GetColliderSet();
    
    Assure(UtilIsEmitterOverlapping(&emitter, colliders) == 0);
    
    emitter.Emit(&pBuilder);
    
    SphParticleSet3 *sphSet = SphParticleSet3FromContinuousBuilder(&pBuilder);
    Grid3 *domainGrid = UtilBuildGridForDomain(container->GetBounds(), 
                                               spacing, spacingScale);
    PciSphSolver3 solver;
    solver.Initialize(DefaultSphSolverData3());
    solver.Setup(WaterDensity, spacing, spacingScale, domainGrid, sphSet);
    solver.SetColliders(colliders);
    
    Float targetInterval = 1.0 / 240.0;
    
    auto callback = [&](int step) -> int{
        if(step % 80 == 0 && step > 0){
            emitter.Emit(&pBuilder);
        }
        
        return 1;
    };
    
    PciSphRunSimulation3(&solver, spacing, origin, target, 
                         targetInterval, {}, callback);
    
    CudaMemoryManagerClearCurrent();
    printf("===== OK\n");
}

void test_pcisph3_rock_dam(){
    printf("===== PCISPH Solver 3D -- Rock Dam\n");
    Float spacing = 0.04;
    Float spacingScale = 2.0;
    vec3f origin(8, 0, 0);
    vec3f target(0,-1,0);
    const char *objPath = "/home/felipe/Documents/CGStuff/models/rock.obj";
    
    vec3f targetPos(0, -1.1, 1.0);
    Float rockMaxSize = 2.4;
    vec3f containerSize(3.08, 3.30324, 4.46138);
    vec3f waterBlockSize(3.0, 3.0, 0.5);
    
    CudaMemoryManagerStart("test_pcisph3_rock_dam");
    
    ParsedMesh *mesh = LoadObj(objPath);
    
    Transform meshScale = UtilComputeFitTransform(mesh, rockMaxSize);
    Shape *rock = MakeMesh(mesh, Translate(targetPos) * RotateY(90) * meshScale);
    
    vec3f of = (containerSize - waterBlockSize) * 0.5; of -= vec3f(spacing);
    Shape *waterBox = MakeBox(Translate(vec3f(of.x, -of.y, -of.z)), waterBlockSize);
    Shape *container = MakeBox(Transform(), containerSize, true);
    
    VolumeParticleEmitterSet3 emitter;
    emitter.AddEmitter(waterBox, spacing);
    
    ColliderSetBuilder3 cBuilder;
    cBuilder.AddCollider3(container);
    cBuilder.AddCollider3(rock);
    ColliderSet3 *colliders = cBuilder.GetColliderSet();
    
    Assure(UtilIsEmitterOverlapping(&emitter, colliders) == 0);
    
    ParticleSetBuilder3 pBuilder;
    emitter.Emit(&pBuilder);
    pBuilder.SetVelocityForAll(vec3f(0, -10, 0));
    
    SphParticleSet3 *sphSet = SphParticleSet3FromBuilder(&pBuilder);
    
    Grid3 *domainGrid = UtilBuildGridForDomain(container->GetBounds(), 
                                               spacing, spacingScale);
    PciSphSolver3 solver;
    solver.Initialize(DefaultSphSolverData3());
    solver.Setup(WaterDensity, spacing, spacingScale, domainGrid, sphSet);
    solver.SetColliders(colliders);
    
    Float targetInterval = 1.0 / 240.0;
    
    PciSphRunSimulation3(&solver, spacing, origin, target, targetInterval, {rock});
    
    CudaMemoryManagerClearCurrent();
    printf("===== OK\n");
}

void test_cnm_happy_whale(){
    printf("===== LNM 3D -- Happy Whale\n");
    const char *pFile = "output.txt";
    Bounds3f meshBounds;
    Float spacingScale = 2.0;
    Float spacing = 0.02;
    int count = 0;
    CudaMemoryManagerStart(__FUNCTION__);
    
    /* Load particles, must have been previously generated, use MeshToParticles */
    ParticleSetBuilder3 builder;
    std::vector<vec3f> points;
    SerializerLoadPoints3(&points, pFile, SERIALIZER_POSITION);
    
    /* Get mesh bounds and define a view point */
    count = points.size();
    for(int i = 0; i < count; i++){
        vec3f pi = points[i];
        meshBounds = Union(meshBounds, pi);
    }
    
    /* Move to center, emit particles manually and recompute bounds */
    vec3f offset = meshBounds.Center();
    meshBounds = Bounds3f(vec3f(0));
    for(int i = 0; i < count; i++){
        vec3f pi = points[i] - offset;
        builder.AddParticle(pi);
        
        // I like the bounds let it consider the erroneous particle
        meshBounds = Union(meshBounds, pi);
    }
    
    vec3f lower = meshBounds.pMin + vec3f(0, -0.1, 0);
    meshBounds = Union(meshBounds, lower);
    
    meshBounds.PrintSelf();
    printf("\n");
    
    Float scale = 1.1;
    /* The mesh is now centered, can safely create a box at origin */
    vec3f size(meshBounds.ExtentOn(0), meshBounds.ExtentOn(1), meshBounds.ExtentOn(2));
    size *= scale;
    
    // For spheres
    vec3f center;
    Float radius;
    meshBounds.BoundingSphere(&center, &radius);
    Float refLen = 2.0f * radius * 0.8;
    radius *= 0.9;
    Shape *container = MakeSphere(Translate(center), radius, true);
    printf("Sphere center: {%g %g %g}, radius: %g\n", center.x, center.y, center.z, radius);
    
    // expand by spacing for safe hash
    vec3f pMin = -vec3f(radius + 2 * spacing);
    vec3f pMax =  vec3f(radius + 2 * spacing);
    
    // grid resolution for better performance
    int resolution = (int)std::floor(refLen / (spacing * spacingScale));
    // make grid
    Grid3 *grid = MakeGrid(vec3ui(resolution), pMin, pMax);
    SphParticleSet3 *sphSet = SphParticleSet3FromBuilder(&builder);
    ParticleSet3 *pSet = sphSet->GetParticleSet();
    
    for(int i = 0; i < grid->GetCellCount(); i++){
        grid->DistributeResetCell(i);
    }
    
    grid->DistributeByParticle(pSet);
    grid->UpdateQueryState();
    
    TimerList timers;
    timers.Start();
    LNMBoundary(pSet, grid, spacing, 0);
    timers.Stop();
    
    std::cout << "Time taken " << timers.GetElapsedGPU(0) << " ms" << std::endl;
    
    CudaMemoryManagerClearCurrent();
    printf("===== OK\n");
}

void test_pcisph3_happy_whale(){
    printf("===== PCISPH Solver 3D -- Happy Whale\n");
    vec3f origin;
    vec3f target(0.f, 0.f, 0.f);
    Bounds3f meshBounds;
    Float spacingScale = 2.0;
    Float spacing = 0.02;
    int count = 0;
    const char *pFile = "output.txt";
    
    CudaMemoryManagerStart(__FUNCTION__);
    
    /* Load particles, must have been previously generated, use MeshToParticles */
    ParticleSetBuilder3 builder;
    std::vector<vec3f> points;
    SerializerLoadPoints3(&points, pFile, SERIALIZER_POSITION);
    
    /* Get mesh bounds and define a view point */
    count = points.size();
    for(int i = 0; i < count; i++){
        vec3f pi = points[i];
        meshBounds = Union(meshBounds, pi);
    }
    
    /* Move to center, emit particles manually and recompute bounds */
    vec3f offset = meshBounds.Center();
    meshBounds = Bounds3f(vec3f(0));
    for(int i = 0; i < count; i++){
        vec3f pi = points[i] - offset;
        builder.AddParticle(pi);
        
        // I like the bounds let it consider the erroneous particle
        meshBounds = Union(meshBounds, pi);
    }
    
    vec3f lower = meshBounds.pMin + vec3f(0, -0.1, 0);
    meshBounds = Union(meshBounds, lower);
    // set view, target now is zero
    origin = meshBounds.pMax + 0.5 * meshBounds.ExtentOn(meshBounds.MaximumExtent());
    //origin.x *= -1;
    
    meshBounds.PrintSelf();
    printf("\n");
    
    Float scale = 1.1;
    /* The mesh is now centered, can safely create a box at origin */
    vec3f size(meshBounds.ExtentOn(0), meshBounds.ExtentOn(1), meshBounds.ExtentOn(2));
    size *= scale;
    
    // For spheres
    vec3f center;
    Float radius;
    meshBounds.BoundingSphere(&center, &radius);
    Float refLen = 2.0f * radius * 0.8;
    radius *= 0.9;
    Shape *container = MakeSphere(Translate(center), radius, true);
    printf("Sphere center: {%g %g %g}, radius: %g\n", center.x, center.y, center.z, radius);
    
    // expand by spacing for safe hash
    vec3f pMin = -vec3f(radius + 2 * spacing);
    vec3f pMax =  vec3f(radius + 2 * spacing);
    
    // grid resolution for better performance
    int resolution = (int)std::floor(refLen / (spacing * spacingScale));
    // make grid
    Grid3 *grid = MakeGrid(vec3ui(resolution), pMin, pMax);
    
    // make collider
    ColliderSetBuilder3 cBuilder;
    cBuilder.AddCollider3(container);
    ColliderSet3 *colliders = cBuilder.GetColliderSet();
    
    /* Setup solver */
    PciSphSolver3 solver;
    SphParticleSet3 *sphSet = SphParticleSet3FromBuilder(&builder);
    
    solver.Initialize(DefaultSphSolverData3());
    solver.Setup(WaterDensity, spacing, spacingScale, grid, sphSet);
    solver.SetColliders(colliders);
    
    ParticleSet3 *pSet = sphSet->GetParticleSet();
    Float targetInterval = 1.0 / 240.0;
    
    auto callback = [&](int step) -> int{
        if(step == 0) return 1;
        UtilPrintStepStandard(&solver, step-1, {0, 16, 31, 74, 151, 235, 
                                  256, 278, 361, 420});
        return step > 450 ? 0 : 1;
    };
    
    PciSphRunSimulation3(&solver, spacing, origin, target, 
                         targetInterval, {}, callback);
    
    
    CudaMemoryManagerClearCurrent();
    printf("===== OK\n");
}


void test_pcisph3_double_dam_break(){
    printf("===== PCISPH Solver 3D -- Double Dam Break\n");
    vec3f origin(0, 1, -3);
    vec3f target(0,0,0);
    
    Float spacing = 0.02;
    Float boxLen = 1.5;
    Float boxFluidLen = 0.5;
    Float boxFluidYLen = 0.9;
    Float spacingScale = 2.0;
    
    /* Build shapes */
    Float xof = (boxLen - boxFluidLen)/2.0; xof -= spacing;
    Float zof = (boxLen - boxFluidLen)/2.0; zof -= spacing;
    Float yof = (boxLen - boxFluidYLen)/2.0; yof -= spacing;
    
    vec3f boxSize = vec3f(boxFluidLen, boxFluidYLen, boxFluidLen);
    
    Shape *container = MakeBox(Transform(), vec3f(boxLen), true);
    Shape *boxp = MakeBox(Translate(xof, -yof, zof), boxSize);
    Shape *boxn = MakeBox(Translate(-xof, -yof, -zof), boxSize);
    
    /* Emit particles */
    ParticleSetBuilder3 pBuilder;
    VolumeParticleEmitterSet3 emitterSet;
    
    VolumeParticleEmitter3 emitterp(boxp, boxp->GetBounds(), spacing);
    VolumeParticleEmitter3 emittern(boxn, boxn->GetBounds(), spacing);
    
    emitterSet.AddEmitter(&emitterp);
    emitterSet.AddEmitter(&emittern);
    //emitterSet.SetJitter(0.001);
    emitterSet.Emit(&pBuilder);
    
    /* Build domain and colliders */
    Bounds3f containerBounds = container->GetBounds();
    vec3f pMin = containerBounds.pMin - vec3f(spacing);
    vec3f pMax = containerBounds.pMax + vec3f(spacing);
    
    int resolution = (int)std::floor(boxLen / (spacing * spacingScale));
    Grid3 *grid = MakeGrid(vec3ui(resolution), pMin, pMax);
    
    ColliderSetBuilder3 cBuilder;
    cBuilder.AddCollider3(container);
    
    ColliderSet3 *colliders = cBuilder.GetColliderSet();
    
    /* Setup solver */
    PciSphSolver3 solver;
    SphParticleSet3 *sphSet = SphParticleSet3FromBuilder(&pBuilder);
    
    solver.Initialize(DefaultSphSolverData3());
    solver.Setup(WaterDensity, spacing, spacingScale, grid, sphSet);
    solver.SetColliders(colliders);
    
    /* Set timestep and view stuff */
    sphSet->SetRelativeKernelRadius(spacingScale);
    ParticleSet3 *pSet = sphSet->GetParticleSet();
    
    int count = pSet->GetParticleCount();
    Float targetInterval = 1.0 / 240.0;
    float *pos = new float[count * 3];
    float *col = new float[count * 3];
    
    memset(col, 0, sizeof(float) * 3 * count);
    graphy_vector_set(origin, target);
    
    //simple_color(pos, col, pSet);
    //graphy_render_points3f(pos, col, count, spacing/2.0);
    
    vec3ui res = grid->GetIndexCount();
    Bounds3f bound = grid->GetBounds();
    pMin = bound.pMin;
    pMax = bound.pMax;
    vec3f c = grid->GetCellSize();
    
    
    TimerList timers;
    timers.Start();
    
    LNMBoundary(pSet, grid, spacing, 0);
    
    timers.Stop();
    Float val = timers.GetElapsedGPU(0);
    std::cout << "GPU Time > " << val << std::endl;
    
    int ii = set_poscol_cnm(col, pos, solver.GetSphSolverData(), 0, 0);
    printf("Size > %d\n", ii);
    graphy_render_points3f(pos, col, ii, spacing/2.0);
    getchar();
    
    for(int i = 0; i < 20 * 26 * 20; i++){
        solver.Advance(targetInterval);
        simple_color(pos, col, pSet);
        graphy_render_points3f(pos, col, count, spacing/2.0);
        printf("Step: %d            \n", i+1);
    }
    
    
    delete[] pos;
    delete[] col;
    printf("===== OK\n");
}

void test_pcisph3_quadruple_dam(){
    printf("===== PCISPH Solver 3D -- Quadruple Dam Break\n");
    Float spacing = 0.02;
    Float spacingScale = 2.0;
    vec3f boxLen(4.0, 2.0, 4.0);
    vec3f waterBoxLen(0.6, 1.8, 0.6);
    vec3f initialVelocity(0.0, -3.0, 0);
    vec3f origin(0, 2, -5);
    vec3f target(0);
    
    CudaMemoryManagerStart(__FUNCTION__);
    
    Float xof = (boxLen.x - waterBoxLen.x) * 0.5; xof -= spacing;
    Float yof = (boxLen.y - waterBoxLen.y) * 0.5; yof -= spacing;
    Float zof = (boxLen.z - waterBoxLen.z) * 0.5; zof -= spacing;
    
    Shape *container = MakeBox(Transform(), boxLen, true);
    Shape *box0 = MakeBox(Translate(0, -yof, -zof), waterBoxLen);
    Shape *box1 = MakeBox(Translate(0, -yof, +zof), waterBoxLen);
    Shape *box2 = MakeBox(Translate(+xof, -yof, 0), waterBoxLen);
    Shape *box3 = MakeBox(Translate(-xof, -yof, 0), waterBoxLen);
    
    ParticleSetBuilder3 pBuilder;
    VolumeParticleEmitterSet3 emitterSet;
    emitterSet.AddEmitter(box0, spacing, initialVelocity);
    emitterSet.AddEmitter(box1, spacing, initialVelocity);
    emitterSet.AddEmitter(box2, spacing, initialVelocity);
    emitterSet.AddEmitter(box3, spacing, initialVelocity);
    //emitterSet.SetJitter(0.001);
    
    ColliderSetBuilder3 cBuilder;
    cBuilder.AddCollider3(container);
    ColliderSet3 *colliders = cBuilder.GetColliderSet();
    
    Assure(UtilIsEmitterOverlapping(&emitterSet, colliders) == 0);
    
    emitterSet.Emit(&pBuilder);
    SphParticleSet3 *sphSet = SphParticleSet3FromBuilder(&pBuilder);
    
    Grid3 *domainGrid = UtilBuildGridForDomain(container->GetBounds(), spacing,
                                               spacingScale);
    
    PciSphSolver3 solver;
    solver.Initialize(DefaultSphSolverData3());
    solver.Setup(WaterDensity, spacing, spacingScale, domainGrid, sphSet);
    solver.SetColliders(colliders);
    
#if 0
    auto callback = [&](int step) -> int{
        if(step <= 0) return 1;
        std::string respath("/media/felipe/Novo volume/quadruple_out/sim_data/out_");
        respath += std::to_string(step-1);
        respath += ".txt";
        SerializerSaveSphDataSet3(solver.GetSphSolverData(), respath.c_str(),
                                  SERIALIZER_POSITION);
        UtilPrintStepStandard(&solver, step-1);
        return 1;
    };
#endif
    
    auto callback = [&](int step) -> int{
        if(step == 0) return 1;
        UtilPrintStepStandard(&solver, step-1, {0, 16, 31, 74, 151, 235, 
                                  256, 278, 361, 420});
        return step > 450 ? 0 : 1;
    };
    
    Float targetInterval =  1.0 / 240.0;
    PciSphRunSimulation3(&solver, spacing, origin, target, 
                         targetInterval, {}, callback);
    
    CudaMemoryManagerClearCurrent();
    printf("===== OK\n");
}

void test_pcisph3_lucy_ball(){
    printf("===== PCISPH Solver 3D -- Lucy Ball\n");
    Float spacing = 0.02;
    Float spacingScale = 2.0;
    Float sphereRadius = 1.5;
    Float waterRadius = 0.5;
    
    const char *lucyPath = "/home/felipe/Documents/CGStuff/models/lucy.obj";
    ParsedMesh *lucyMesh = LoadObj(lucyPath);
    Transform lucyScale = UtilComputeFitTransform(lucyMesh, sphereRadius);
    Shape *lucyShape = MakeMesh(lucyMesh, Translate(0, -sphereRadius * 0.5, 0) * 
                                RotateY(-90) * lucyScale);
    
    Shape *containerShape = MakeSphere(Transform(), sphereRadius, true);
    Shape *waterShape = MakeSphere(Translate(0, 0.8, 0), waterRadius);
    
    vec3f toLucy = lucyShape->GetBounds().Center() - waterShape->GetBounds().Center();
    Float length = toLucy.Length();
    toLucy = toLucy / length;
    
    ParticleSetBuilder3 pBuilder;
    VolumeParticleEmitterSet3 emitterSet;
    emitterSet.AddEmitter(waterShape, waterShape->GetBounds(), spacing,
                          toLucy * 10);
    
    ColliderSetBuilder3 colliderSetBuilder;
    colliderSetBuilder.AddCollider3(lucyShape);
    colliderSetBuilder.AddCollider3(containerShape);
    ColliderSet3 *colliders = colliderSetBuilder.GetColliderSet();
    
    Assure(UtilIsEmitterOverlapping(&emitterSet, colliders) == 0);
    
    emitterSet.Emit(&pBuilder);
    SphParticleSet3 *sphSet = SphParticleSet3FromBuilder(&pBuilder);
    
    Grid3 *domainGrid = UtilBuildGridForDomain(containerShape->GetBounds(), spacing,
                                               spacingScale);
    
    PciSphSolver3 solver;
    solver.Initialize(DefaultSphSolverData3());
    solver.Setup(WaterDensity, spacing, spacingScale, domainGrid, sphSet);
    solver.SetColliders(colliders);
    
    std::vector<vec3f> particles;
    UtilGetSDFParticles(lucyShape->grid, &particles, 0, spacing);
    ParticleSet3 *pSet = sphSet->GetParticleSet();
    
    int total = particles.size() + pSet->GetParticleCount();
    float *pos = new float[total * 3];
    float *col = new float[total * 3];
    
    vec3f origin(0, 0, -5);
    vec3f target(0);
    memset(col, 0, sizeof(float) * 3 * total);
    graphy_vector_set(origin, target);
    
    simple_color(pos, col, pSet);
    int it = 0;
    for(int i = pSet->GetParticleCount(); i < total; i++){
        vec3f pi = particles[it++];
        pos[3 * i + 0] = pi.x; pos[3 * i + 1] = pi.y;
        pos[3 * i + 2] = pi.z; col[3 * i + 2] = 1;
    }
    
    graphy_render_points3f(pos, col, total, spacing/2.0);
    Float targetInterval = 1.0 / 240.0;
    for(int i = 0; i < 20 * 26 * 20; i++){
        std::string outputName("lucy/output_");
        solver.Advance(targetInterval);
        simple_color(pos, col, pSet);
        graphy_render_points3f(pos, col, total, spacing/2.0);
        outputName += std::to_string(i);
        outputName += ".txt";
        SerializerSaveSphDataSet3(solver.GetSphSolverData(), outputName.c_str(),
                                  SERIALIZER_POSITION);
        //printf("Step: %d            \n", i+1);
    }
    
    
    delete[] pos;
    delete[] col;
    printf("===== OK\n");
}

void test_pcisph3_lucy_dam(){
    printf("===== PCISPH Solver 3D -- Lucy Dam\n");
    vec3f origin(0, 1, -3);
    vec3f target(0,0,0);
    
    Float spacing = 0.02;
    Float boxLen = 1.5;
    Float boxFluidLen = 0.5;
    Float boxFluidYLen = 0.9;
    Float spacingScale = 2.0;
    
    /* Build shapes */
    Float xof = (boxLen - boxFluidLen)/2.0; xof -= spacing;
    Float zof = (boxLen - boxFluidLen)/2.0; zof -= spacing;
    Float yof = (boxLen - boxFluidYLen)/2.0; yof -= spacing;
    
    vec3f boxSize = vec3f(boxFluidLen, boxFluidYLen, boxFluidLen);
    
    const char *lucyPath = "/home/felipe/Documents/CGStuff/models/lucy.obj";
    
    Shape *container = MakeBox(Transform(), vec3f(boxLen), true);
    Shape *boxp = MakeBox(Translate(xof, -yof, zof), boxSize);
    Shape *lucyShape = MakeMesh(lucyPath, Translate(0,-0.3,0) * Scale(0.015));
    
    
    /* Emit particles */
    ParticleSetBuilder3 pBuilder;
    VolumeParticleEmitterSet3 emitterSet;
    
    VolumeParticleEmitter3 emitterp(boxp, boxp->GetBounds(), spacing);
    
    emitterSet.AddEmitter(&emitterp);
    //emitterSet.SetJitter(0.001);
    emitterSet.Emit(&pBuilder);
    
    /* Build domain and colliders */
    Bounds3f containerBounds = container->GetBounds();
    vec3f pMin = containerBounds.pMin - vec3f(spacing);
    vec3f pMax = containerBounds.pMax + vec3f(spacing);
    
    int resolution = (int)std::floor(boxLen / (spacing * spacingScale));
    Grid3 *grid = MakeGrid(vec3ui(resolution), pMin, pMax);
    
    ColliderSetBuilder3 cBuilder;
    cBuilder.AddCollider3(container);
    cBuilder.AddCollider3(lucyShape);
    
    ColliderSet3 *colliders = cBuilder.GetColliderSet();
    
    /* Setup solver */
    PciSphSolver3 solver;
    SphParticleSet3 *sphSet = SphParticleSet3FromBuilder(&pBuilder);
    
    solver.Initialize(DefaultSphSolverData3());
    solver.Setup(WaterDensity, spacing, spacingScale, grid, sphSet);
    solver.SetColliders(colliders);
    
    /* Set timestep and view stuff */
    std::vector<vec3f> particles;
    UtilGetSDFParticles(lucyShape->grid, &particles, 0, spacing/2.0);
    
    ParticleSet3 *pSet = sphSet->GetParticleSet();
    int count = pSet->GetParticleCount() + particles.size();
    Float targetInterval = 1.0 / 240.0;
    float *pos = new float[count * 3];
    float *col = new float[count * 3];
    
    memset(col, 0, sizeof(float) * 3 * count);
    graphy_vector_set(origin, target);
    
    simple_color(pos, col, pSet);
    int it = 0;
    for(int i = pSet->GetParticleCount(); i < count; i++){
        vec3f pi = particles[it++];
        pos[3 * i + 0] = pi.x; pos[3 * i + 1] = pi.y;
        pos[3 * i + 2] = pi.z; col[3 * i + 2] = 1;
    }
    
    graphy_render_points3f(pos, col, count, spacing/2.0);
    
    for(int i = 0; i < 20 * 26 * 20; i++){
        std::string outputName("lucy/output_");
        solver.Advance(targetInterval);
        simple_color(pos, col, pSet);
        graphy_render_points3f(pos, col, count, spacing/2.0);
        outputName += std::to_string(i);
        outputName += ".txt";
        SerializerSaveSphDataSet3(solver.GetSphSolverData(), outputName.c_str(),
                                  SERIALIZER_POSITION);
        //printf("Step: %d            \n", i+1);
    }
    
    
    delete[] pos;
    delete[] col;
    printf("===== OK\n");
}

void test_pcisph3_whale_obstacle(){
    printf("===== PCISPH Solver 3D -- Whale Obstacle\n");
    Float spacing = 0.03;
    Float boxLen = 1.5;
    Float boxFluidLen = 1.3;
    Float boxFluidYLen = 0.2;
    Float spacingScale = 2.0;
    
    /* Build shapes */
    Float xof = (boxLen - boxFluidLen)/2.0; xof -= spacing;
    Float zof = (boxLen - boxFluidLen)/2.0; zof -= spacing;
    Float yof = (boxLen * 0.4 - boxFluidYLen); yof -= spacing;
    vec3f boxSize = vec3f(boxFluidLen, boxFluidYLen, boxFluidLen);
    
    const char *whaleObj = "/home/felipe/Documents/CGStuff/models/HappyWhale.obj";
    Transform transform = Translate(0, -0.8, 0) * Scale(0.1); // happy whale
    
    Shape *container = MakeBox(Transform(), vec3f(boxLen), true);
    Shape *waterBox = MakeSphere(Translate(xof, yof, zof), 0.2);
    ParsedMesh *mesh = LoadObj(whaleObj);
    Shape *meshShape = MakeMesh(mesh, transform);
    
    /* Emit particles */
    ParticleSetBuilder3 pBuilder;
    VolumeParticleEmitterSet3 emitterSet;
    
    VolumeParticleEmitter3 emitterp(waterBox, waterBox->GetBounds(), spacing);
    emitterSet.AddEmitter(&emitterp);
    
    emitterSet.Emit(&pBuilder);
    
    /* Build domain and colliders */
    Bounds3f containerBounds = container->GetBounds();
    vec3f pMin = containerBounds.pMin - vec3f(spacing);
    vec3f pMax = containerBounds.pMax + vec3f(spacing);
    
    int resolution = (int)std::floor(boxLen / (spacing * spacingScale));
    Grid3 *grid = MakeGrid(vec3ui(resolution), pMin, pMax);
    
    ColliderSetBuilder3 cBuilder;
    cBuilder.AddCollider3(container);
    cBuilder.AddCollider3(meshShape);
    
    ColliderSet3 *colliders = cBuilder.GetColliderSet();
    
    std::vector<vec3f> particles;
    UtilGetSDFParticles(meshShape->grid, &particles, 0, 0.01);
    
    /* Setup solver */
    PciSphSolver3 solver;
    SphParticleSet3 *sphSet = SphParticleSet3FromBuilder(&pBuilder);
    
    solver.Initialize(DefaultSphSolverData3());
    solver.Setup(WaterDensity, spacing, spacingScale, grid, sphSet);
    solver.SetColliders(colliders);
    
    /* Set timestep and view stuff */
    ParticleSet3 *pSet = sphSet->GetParticleSet();
    int count = pSet->GetParticleCount() + particles.size();
    Float targetInterval = 1.0 / 240.0;
    float *pos = new float[count * 3];
    float *col = new float[count * 3];
    
    vec3f origin(1.5, 1, 0);
    vec3f target(0,-0.2,0);
    
    memset(col, 0, sizeof(float) * 3 * count);
    graphy_vector_set(origin, target);
    
    int itp = 0, itc = 0;
    for(int id = 0; id < pSet->GetParticleCount(); id++){
        vec3f pi = pSet->GetParticlePosition(id);
        pos[itp++] = pi.x; pos[itp++] = pi.y; pos[itp++] = pi.z;
        col[itc++] = 1; col[itc++] = 0; col[itc++] = 0;
    }
    
    for(vec3f &pi : particles){
        pos[itp++] = pi.x; pos[itp++] = pi.y; pos[itp++] = pi.z;
        col[itc++] = 0; col[itc++] = 0; col[itc++] = 1;
    }
    
    graphy_render_points3f(pos, col, count, spacing/2.0);
    for(int i = 0; i < 20 * 26 * 20; i++){
        solver.Advance(targetInterval);
        itp = 0;
        itc = 0;
        for(int id = 0; id < pSet->GetParticleCount(); id++){
            vec3f pi = pSet->GetParticlePosition(id);
            pos[itp++] = pi.x; pos[itp++] = pi.y; pos[itp++] = pi.z;
            col[itc++] = 1; col[itc++] = 0; col[itc++] = 0;
        }
        
        for(vec3f &pi : particles){
            pos[itp++] = pi.x; pos[itp++] = pi.y; pos[itp++] = pi.z;
            col[itc++] = 0; col[itc++] = 0; col[itc++] = 1;
        }
        
        //simple_color(pos, col, pSet);
        graphy_render_points3f(pos, col, count, spacing/2.0);
        printf("Step: %d            \n", i+1);
    }
    
    
    delete[] pos;
    delete[] col;
    printf("===== OK\n");
}

void test_pcisph3_water_sphere(){
    printf("===== PCISPH Solver 3D -- Water in Ball\n");
    Float spacing = 0.03;
    Float targetDensity = WaterDensity;
    vec3f center(0);
    Float r1 = 0.5;
    Float r2 = 1.0;
    vec3f pMin, pMax;
    Bounds3f containerBounds;
    ParticleSetBuilder3 builder;
    ColliderSetBuilder3 colliderBuilder;
    PciSphSolver3 solver;
    
    int reso = (int)std::floor(2 * r2 / (spacing * 2.0));
    printf("Using grid with resolution %d x %d x %d\n", reso, reso, reso);
    vec3ui res(reso);
    
    solver.Initialize(DefaultSphSolverData3());
    Shape *sphere = MakeSphere(Translate(center.x-0.4, center.y+r1/2.f, 0), r1);
    Shape *container = MakeSphere(Translate(center.x, center.y, 0), r2, true);
    
    containerBounds = container->GetBounds();
    pMin = containerBounds.pMin - vec3f(spacing);
    pMax = containerBounds.pMax + vec3f(spacing);
    
    Grid3 *grid = MakeGrid(res, pMin, pMax);
    
    VolumeParticleEmitter3 emitter(sphere, sphere->GetBounds(), spacing);
    
    emitter.Emit(&builder);
    SphParticleSet3 *sphSet = SphParticleSet3FromBuilder(&builder);
    ParticleSet3 *set2 = sphSet->GetParticleSet();
    int count = set2->GetParticleCount();
    
    colliderBuilder.AddCollider3(container);
    ColliderSet3 *collider = colliderBuilder.GetColliderSet();
    
    solver.Setup(targetDensity, spacing, 2.0, grid, sphSet);
    solver.SetColliders(collider);
    
    Float targetInterval = 1.0 / 248.0;
    float *pos = new float[count * 3];
    float *col = new float[count * 3];
    
    SphSolverData3 *data = solver.GetSphSolverData();
    simple_color(pos, col, set2);
    
    vec3f origin(0,0,3);
    vec3f target(0,0,0);
    graphy_vector_set(origin, target);
    graphy_render_points3f(pos, col, count, 0.01);
    
    for(int i = 0; i < 20 * 26 * 20; i++){
        solver.Advance(targetInterval);
        simple_color(pos, col, set2);
        graphy_render_points3f(pos, col, count, 0.01);
        printf("Step: %d          \n", i+1);
    }
    
    delete[] pos;
    delete[] col;
    printf("===== OK\n");
}
