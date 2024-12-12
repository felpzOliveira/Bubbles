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
#include <transform_sequence.h>
#include <obj_loader.h>
#include <algorithm>
#include <random>
#include <sdfs.h>

void set_particle_color(float *pos, float *col, ParticleSet3 *pSet);
void simple_color(float *pos, float *col, ParticleSet3 *pSet);

void test_pcisph3_water_drop2(){
    printf("===== PCISPH Solver 3D -- Water Drop 2\n");
    vec3f origin(4);
    vec3f target(0);
    Float spacing = 0.02;
    Float spacingScale = 2.f;
    vec3f boxSize = vec3f(2.f, 4.f, 2.f);

    vec3f waterBaseSize = boxSize * vec3f(1.f, 0.25f, 1.f) -
                                    vec3f(4.f * spacing, 0.f, 4.f * spacing);
    Float sphereRadius = 0.15 * boxSize.x;
    Float yOff = (boxSize.y - waterBaseSize.y) * 0.5;
    Float sOff = (boxSize.y - sphereRadius * 2.f) * 0.5 - spacing;
    Shape *container = MakeBox(Transform(), boxSize, true);
    Shape *baseWater = MakeBox(Translate(0.f, -yOff, 0.f), waterBaseSize);
    Shape *waterSphere = MakeSphere(Translate(0.f, sOff, 0.f), sphereRadius * 0.5f);

    VolumeParticleEmitterSet3 emitters;
    emitters.AddEmitter(waterSphere, spacing);
    emitters.AddEmitter(baseWater, spacing);

    ColliderSetBuilder3 cBuilder;
    cBuilder.AddCollider3(container);
    ColliderSet3 *colliders = cBuilder.GetColliderSet();

    Assure(UtilIsEmitterOverlapping(&emitters, colliders) == 0);

    ParticleSetBuilder3 pBuilder;
    auto velocityField = [&](const vec3f &p) -> vec3f{
        if(p.y > 1.3){
            return vec3f(0.f, -40.f, 0.f);
        }
        return vec3f(0);
    };

    emitters.Emit(&pBuilder, velocityField);

    SphParticleSet3 *sphSet = SphParticleSet3FromBuilder(&pBuilder);
    Grid3 *domainGrid = UtilBuildGridForDomain(container->GetBounds(),
                                               spacing, spacingScale);
    vec3ui res = domainGrid->GetIndexCount();

    ParticleSet3 *pSet = sphSet->GetParticleSet();

    PciSphSolver3 solver;
    solver.Initialize(DefaultSphSolverData3());
    solver.SetViscosityCoefficient(0.f);
    solver.Setup(WaterDensity, spacing, spacingScale, domainGrid, sphSet);
    solver.SetColliders(colliders);

    Float targetInterval =  1.0 / 240.0;

    //ProfilerInitKernel(pSet->GetParticleCount());
    int extraParts = 12 * 10 + 12;
    auto colorFunction = [&](float *colors, int pCount) -> void{
        for(int i = 0; i < pCount; i++){
            colors[3 * i + 0] = 1; colors[3 * i + 1] = 0;
            colors[3 * i + 2] = 1;
        }
    };

    std::vector<int> boundaries;
    auto onStepUpdate = [&](int step) -> int{
        if(step == 0) return 1;
        std::string respath = FrameOutputPath("water_drop/output_", step-1);

        int flags = (SERIALIZER_POSITION);
        SerializerSaveSphDataSet3(solver.GetSphSolverData(), respath.c_str(), flags);

        UtilPrintStepStandard(&solver, step-1);
        return step > 500 ? 0 : 1;
    };


    auto filler = [&](float *pos, float *col) -> int{
        return UtilGenerateBoxPoints(pos, col, vec3f(1,1,0), boxSize,
                                     extraParts-12, container->ObjectToWorld);
    };

    UtilRunDynamicSimulation3<PciSphSolver3, ParticleSet3>(&solver, pSet, spacing, origin,
                                                           target, targetInterval, extraParts,
                                                           {}, onStepUpdate, colorFunction,
                                                           filler);

    CudaMemoryManagerClearCurrent();

    printf("===== OK\n");
}

void test_pcisph3_tank_dam(){
    printf("===== PCISPH Solver 3D -- Tank Dam\n");
    CudaMemoryManagerStart(__FUNCTION__);

    vec3f origin(0, 4, 15);
    vec3f target(0);

    Float spacing = 0.2;
    //Float spacing = 0.04;
    Float spacingScale = 2.0;
    Float targetInterval =  1.0 / 240.0;

    //srand(time(0));

    ///////////////// CONTAINER
    vec2f xRange = vec2f(-10, 5);
    vec2f yRange = vec2f(-3, 3);
    vec2f zRange = vec2f(-5, 5);

    vec3f containerSize = vec3f(xRange.y - xRange.x,
                                yRange.y - yRange.x,
                                zRange.y - zRange.x);

    Shape *containerShape = MakeBox(Transform(), containerSize, true);

    std::cout << "Domain: " << containerShape->GetBounds() << std::endl;
    printf("Container Size= {%g %g %g}\n", containerSize.x,
    containerSize.y, containerSize.z);
    ////////////////////////////

    ///////////////// EMITTER
    VolumeParticleEmitterSet3 emitterSet;
    Float delta = spacing * spacingScale;
    Float xScaleSize = 0.067;
    Float yScaleSize = 0.80f;
    Float zScaleSize = 0.98f;
    Float zExt  = (zRange.y - zRange.x);
    Float xSize = (xRange.y - xRange.x) * xScaleSize;
    Float ySize = (yRange.y - yRange.x) * yScaleSize;
    Float zSize = zExt * zScaleSize;

    Float emitterXOff = (xRange.y - xRange.x) * 0.5f - xSize * 0.5f - delta;
    Float emitterYOff = (yRange.y - yRange.x) * 0.5f - ySize * 0.5f - delta;
    zSize = (zSize + 2 * delta) > zExt ? zExt - 2 * delta : zSize;

    Transform emitterTransform = Translate(
                vec3f(emitterXOff, -emitterYOff, 0)
    );

    Shape *emitterShape = MakeBox(emitterTransform, vec3f(xSize, ySize, zSize));
    emitterSet.AddEmitter(emitterShape, spacing);

    std::cout << emitterShape->GetBounds() << std::endl;
    printf("Emitter size= {%g %g %g}\n", xSize, ySize, zSize);
    printf("Emitter Offset= {%g %g}\n", emitterXOff, emitterYOff);
    /////////////////////////

    ///////////////////// COLLIDERS
    std::vector<Shape *> sdfs;
    ColliderSetBuilder3 cBuilder;

    vec2f xColExt = vec2f(0.50f, 0.90f);
    vec2f zColExt = vec2f(0.01f, 0.98f);
    int extra_parts = 150;

    Float _x0 = xRange.x + xColExt.x * (xRange.y - xRange.x);
    Float _x1 = xRange.x + xColExt.y * (xRange.y - xRange.x);

    Float _z0 = zRange.x + zColExt.x * (zRange.y - zRange.x);
    Float _z1 = zRange.x + zColExt.y * (zRange.y - zRange.x);

    int max_per_side = 10;
    Float box_w = std::min((_z1 - _z0) / (Float)max_per_side,
                           (_x1 - _x0) / (Float)max_per_side);

    int boxes_per_side = 4;
    Float y0 = yRange.x + box_w * 0.5f;

    printf("Boxes width= %g\n", box_w);
    printf("Boxes domain= { %g %g } x { %g %g }\n", _x0, _z0, _x1, _z1);

    Float thetaZ = (_z1 - _z0) / (Float)boxes_per_side - box_w;
    Float thetaX = (_x1 - _x0) / (Float)boxes_per_side - box_w;

    thetaZ *= 0.5f;

    printf("θ_z = %g\n", thetaZ);
    printf("θ_x = %g\n", thetaX);

    int total_parts = extra_parts + 2;
    std::vector<Shape *> boxList;
    std::string litStr;

    std::vector<int> ids;
    struct MeshPack{
        ParsedMesh *mesh;
        Transform scale;
    };

    std::vector<MeshPack> meshPack;
    int expected = boxes_per_side * boxes_per_side;
    for(int s = 0; s < expected; s++){
        ids.push_back(-1);
    }

#if 0

    std::vector<std::string> meshesPaths = {
        "budda_0_1.obj",
        "bunny.obj",
        "HappyWhale.obj",
        "lucy_2.obj",
        "killeroo.obj",
        "stanfordDragon.obj",
        "pyramid.obj",
        "pyramid.obj",
        "cylinder.obj",
        "cylinder.obj",
        "bunny.obj",
        "stanfordDragon.obj",
        "killeroo.obj",
    };

    std::vector<Float> meshesScales = {
        box_w * 4.f, // max axis is Y so it is ok
        box_w * 2.f,
        box_w * 4.f,
        box_w * 4.f,
        box_w * 4.f,
        box_w * 2.f,
        box_w * 4.f,
        box_w * 2.f,
        box_w * 2.f,
        box_w * 1.5f,
        box_w * 2.f,
        box_w * 3.f,
        box_w * 6.f,
    };

    for(int i = 0; i < meshesPaths.size(); i++){
        Float targetScale = 1;
        std::string path = meshesPaths[i];
        Float maxLen = meshesScales[i];

        std::string actualPath = ModelPath(path.c_str());
        ParsedMesh *mesh = LoadObj(actualPath.c_str());

        Transform scale = UtilComputeFitTransform(mesh, maxLen, &targetScale);
        if(i == 3){
            scale = RotateX(90) * scale;
        }

        meshPack.push_back({mesh, scale});
    }

    std::vector<int> list;
    for(int s = 0; s < expected; s++){
        list.push_back(s);
    }

    unsigned seed = 1022;
    std::mt19937 g(seed);
    std::shuffle(list.begin(), list.end(), g);

    for(int i = 0; i < meshPack.size(); i++){
        ids[list[i]] = i;
    }


#else
    std::vector<std::string> meshesPaths;
#endif

    int counter = 0;
    for(int i = 0; i < boxes_per_side; i++){
        Float baseZ = _z0 + 1.5 * thetaZ;

        for(int j = 0; j < boxes_per_side; j++){
            Shape *shape = nullptr;
            Float alpha = rand_float() * 360.f;
            Float x0 = _x0 + thetaX + i * box_w + (i-1) * thetaX;
            Float x1 = x0 + box_w;
            Float z0 = baseZ;
            Float z1 = z0 + box_w;
            //Float y1 = y0 + rand_float() * 0.05f * (yRange.y - yRange.x);
            Float y1 = y0 + box_w + 0.2f * (yRange.y - yRange.x);

            baseZ += box_w + rand_float() * thetaZ;

            Float x = (x0 + x1) * 0.5f;
            Float z = (z0 + z1) * 0.5f;
            Float dy = (y1 - y0);
            Float boxYOff = (yRange.y - yRange.x) * 0.5f - dy * 0.5f;
            vec3f translationVector = vec3f(x, -boxYOff, z);

            Transform box_transform = Translate(
                    translationVector
            ) * RotateY(alpha);

            int index = ids[counter];

            if(index >= 0){
                MeshPack *pack = &meshPack[index];
                Bounds3f bounds = UtilComputeBoundsAfter(pack->mesh, pack->scale);
                Float ySize = bounds.ExtentOn(1);
                Float meshYOff = (yRange.y - yRange.x) * 0.5f - ySize * 0.5f;

                Transform meshTranslate = Translate(
                    vec3f(translationVector.x, -meshYOff, translationVector.z)
                );

                Transform meshTransform = meshTranslate * RotateY(alpha) * pack->scale;

                bounds = UtilComputeBoundsAfter(pack->mesh, meshTransform);

                Float tmpYOff = (bounds.pMin.y - yRange.x);
                meshTransform = Translate(0, -tmpYOff, 0) * meshTranslate *
                                RotateY(alpha) * pack->scale;

                shape = MakeMesh(pack->mesh, meshTransform);
                sdfs.push_back(shape);

                litStr += "Shape{ type[mesh] mat[mesh_mat] geometry[models/" + meshesPaths[index];
                litStr += "] transform[";
                for(int i = 0; i < 4; i++){
                    for(int j = 0; j < 4; j++){
                        if(i == 3 && j == 3){
                            litStr += std::to_string(meshTransform.m.m[i][j]);
                        }else{
                            litStr += std::to_string(meshTransform.m.m[i][j]) + " ";
                        }
                    }
                }
                litStr += "] }\n";
            }else if(rand_float() > 0.5){
                if(true){
                    shape = MakeBox(box_transform, vec3f(box_w, dy, box_w));
                    total_parts += extra_parts;

                    boxList.push_back(shape);

                    litStr += "Shape{ type[box] translate[ " + std::to_string(x);
                    litStr += " " + std::to_string(-boxYOff) + " " + std::to_string(z);
                    litStr += " ] scale [ " + std::to_string(box_w) + " ";
                    litStr += std::to_string(dy) + " " + std::to_string(box_w);
                    litStr += " ] rotate [ " + std::to_string(alpha) +
                              " 0 1 0 ] mat [ box_mat ] }\n";
                }
            }

            Float dz = 0.9 * thetaZ;
            if(shape){
                cBuilder.AddCollider3(shape);
                std::cout << shape->GetBounds() << std::endl;
                dz = shape->GetBounds().ExtentOn(2);
            }

            counter += 1;

            baseZ += dz + 0.1f * thetaZ;
        }
    }
#if 0
    std::string out_path = FrameOutputPath("tank_break/box_lit", 0);
    std::ofstream ofs(out_path.c_str());
    if(ofs.is_open())
        ofs << litStr;
    ofs.close();
#endif
    auto filler = [&](float *pos, float *col) -> int{
        int f = UtilGenerateBoxPoints(pos, col, vec3f(1,1,0), containerSize,
                                      extra_parts, containerShape->ObjectToWorld);
        for(Shape *shape : boxList){
            Bounds3f bounds = shape->GetBounds();
            vec3f size = vec3f(bounds.ExtentOn(0),
                               bounds.ExtentOn(1),
                               bounds.ExtentOn(2));
            f += UtilGenerateBoxPoints(&pos[3 * f], &col[3 * f], vec3f(1,0,0),
                                       size, extra_parts, shape->ObjectToWorld);
        }
        return f;
    };

    cBuilder.AddCollider3(containerShape);
    ///////////////////////////////

    ////////////////////////// SETUP
    ParticleSetBuilder3 pBuilder;
    PciSphSolver3 solver;

    ColliderSet3 *colliders = cBuilder.GetColliderSet();
    Assure(UtilIsEmitterOverlapping(&emitterSet, colliders) == 0);

    emitterSet.SetJitter(0.001);
    emitterSet.Emit(&pBuilder);

    Grid3 *domainGrid = UtilBuildGridForDomain(containerShape->GetBounds(), spacing,
    spacingScale);

    SphParticleSet3 *sphSet = SphParticleSet3FromBuilder(&pBuilder);
    sphSet->SetRelativeKernelRadius(spacingScale);

    solver.Initialize(DefaultSphSolverData3());
    solver.Setup(WaterDensity, spacing, spacingScale, domainGrid, sphSet);
    solver.SetColliders(colliders);
    ///////////////////////////////

    //////////////////// SIM
    ParticleSet3 *pSet = sphSet->GetParticleSet();
    auto onStepUpdate = [&](int step) -> int{
        if(step == 0)
        return 1;

        std::string path = FrameOutputPath("tank_break/out_", step-1);
        int flags = SERIALIZER_POSITION;
        UtilSaveSimulation3(&solver, pSet, path.c_str(), flags);
        //SerializerSaveSphDataSet3(solver.GetSphSolverData(), path.c_str(),
                                  //SERIALIZER_POSITION);

        UtilPrintStepStandard(&solver, step-1);
        return step < 1000;
    };

    auto colorFunction = [&](float *colors, int pCount) -> void{
        for(int i = 0; i < pCount; i++){
            colors[3 * i + 0] = 1; colors[3 * i + 1] = 0;
            colors[3 * i + 2] = 1;
        }
    };

    UtilRunDynamicSimulation3<PciSphSolver3, ParticleSet3>
                        (&solver, pSet, spacing, origin,
                         target, targetInterval, total_parts,
                         sdfs, onStepUpdate, colorFunction, filler);
    ////////////////////////

    CudaMemoryManagerClearCurrent();
    printf("\n===== OK\n");
}

void test_pcisph3_tank(){
    printf("===== PCISPH Solver 3D -- Tank\n");
    CudaMemoryManagerStart(__FUNCTION__);
    /*
        The main tank should be large in X so:
        (-5, 5) x (-2, 2) x (-2, 2),
        however the domain needs to have some spaces in Y so i
        allows us to drop water in it so make it:
        (-5, 5) x (-2, 4) x (-2, 2)

        particles should spawn in the right side (+X) and on top (+Y)
        considering the domain has has z (-2, 2) a sphere with radius 1
        should be enough. Position needs to be extrem might be fine at (4-δ, 3-δ, 0)
    */
    vec3f origin(0, 4, 15);
    vec3f target(0);

    // Domain definitions
    vec2f xRange = vec2f(-5, 5);
    vec2f yRange = vec2f(-2, 4);
    vec2f zRange = vec2f(-2, 2);
    vec3f boxSize = vec3f(0.5f, 0.5f, 1.9f);
    Float radius = 0.5f;

    vec3f tankSize = vec3f(xRange.y - xRange.x,
                           yRange.y - yRange.x,
                           zRange.y - zRange.x);

    // make sure it centers in origin
    Float xOff = 0.f;
    Float yOff = yRange.x - (-(yRange.y + yRange.x) * 0.5f);
    Float zOff = 0.f;

    Float boxYOffset = yRange.x * 0.8f + boxSize.y * 0.5f;

    printf("Offset = {%g %g %g} { %g }\n", xOff, yOff, zOff, boxYOffset);

    vec3f boxPos = vec3f(2.f, boxYOffset, 0.f);
    Transform boxModel = Translate(boxPos);
    Shape *box = MakeBox(boxModel, boxSize);
    Shape *container = MakeBox(Translate(-xOff, -yOff, -zOff), tankSize, true);

    auto rotation_angle = [](int currStep) -> Float{
        int steps = 100;
        Float f = ((Float)(currStep % steps)) / (Float)steps;
        return Lerp(0.f, 360.f, f);
    };
#if 0
    for(int i = 0; i < 796; i++){
        Float angle = rotation_angle(i);
        std::string name = "../simulations/tank/output_box_";
        name += std::to_string(i) + ".txt";
        std::ofstream ofs(name.c_str());

        if(!ofs.is_open()){
            printf("Could not open file { %d }\n", i);
            exit(0);
        }

        ofs << "Shape{ type[box] mat[dragon_mat] translate[2 -1.35 0]\n";
        ofs << "       rotate[" << std::to_string(angle);
        ofs << " 0 0 1] scale[0.5 0.5 1.9] }\n";
        ofs.close();
    }
#endif
    //std::cout << "Container: " << container->GetBounds() << std::endl;
    //std::cout << "Box: " << box->GetBounds() << ", p: " << boxPos << std::endl;
    //std::cout << "Angle: " << rotation_angle(797) << std::endl;
    //exit(0);
    // Solver definitions
    Float spacing = 0.08;
    Float spacingScale = 2.0;
    Float targetInterval =  1.0 / 240.0;

    // Particle and obstacle setup
    Float delta = 2.0 * spacing * spacingScale;
    vec3f emiPos = vec3f(xRange.y - radius - delta,
                         yRange.y * 0.7 - radius - delta + yOff, 0.f);

    printf("Emitter pos= {%g %g %g}\n", emiPos.x, emiPos.y, emiPos.z);
    std::cout << container->GetBounds() << std::endl;

    std::string dragpath = ModelPath("sssdragon.obj");
    ParsedMesh *mesh = LoadObj(dragpath.c_str());

    Float targetScale = 0;
    (void)UtilComputeFitTransform(mesh, tankSize.z, &targetScale);
    Transform baseScale = Scale(targetScale * 0.8);
    Bounds3f bounds = UtilComputeBoundsAfter(mesh, baseScale);
    Float df = (tankSize.y - bounds.ExtentOn(1)) * 0.5 + yOff;

    Shape *dragon = MakeMesh(mesh,
                Translate(-1.0, -df - 0.05, 0) *
                //RotateY(14) * baseScale);
                RotateY(-76) * baseScale);

    VolumeParticleEmitterSet3 emitterSet;
    ContinuousParticleSetBuilder3 pBuilder(2500000);
    pBuilder.SetKernelRadius(spacing * spacingScale);

    Shape *emitterBall = MakeSphere(Translate(emiPos), radius);
    emitterSet.AddEmitter(emitterBall, spacing);
    //emitterSet.SetJitter(0.01);

    // NOTE: Always remenber to add container as the last object
    //       (makes it easier for serialization to find it)
    ColliderSetBuilder3 cBuilder;
    cBuilder.AddCollider3(box);
    cBuilder.AddCollider3(dragon);
    cBuilder.AddCollider3(container);

    ColliderSet3 *colliders = cBuilder.GetColliderSet();
    Assure(UtilIsEmitterOverlapping(&emitterSet, colliders) == 0);

    Grid3 *domainGrid = UtilBuildGridForDomain(container->GetBounds(),
                                               spacing, spacingScale);

    int extraParts = 12 * 10 + 12;
    std::vector<Shape *> sdfs;
    sdfs.push_back(dragon);

    auto colorFunction = [&](float *colors, int pCount) -> void{
        for(int i = 0; i < pCount; i++){
            colors[3 * i + 0] = 1; colors[3 * i + 1] = 0;
            colors[3 * i + 2] = 0;
        }
    };

    auto filler = [&](float *pos, float *col) -> int{
        return UtilGenerateBoxPoints(pos, col, vec3f(1,1,0), boxSize,
                                     extraParts-12, box->ObjectToWorld);
    };

    // TODO: Adjust the velocity field as needed
    auto velocityField = [&](const vec3f &p) -> vec3f{
        Float intensity = 2.f;
        vec3f dir = boxPos - emiPos;
        return dir * intensity;
    };

    emitterSet.Emit(&pBuilder, velocityField);

    // Setup the continuous spawn procedure
    SphParticleSet3 *sphSet = SphParticleSet3FromContinuousBuilder(&pBuilder);
    ResetAndDistribute(domainGrid, sphSet->GetParticleSet());

    pBuilder.MapGrid(domainGrid);

    ParticleSet3 *pSet = sphSet->GetParticleSet();
    PciSphSolver3 solver;
    solver.Initialize(DefaultSphSolverData3());
    solver.Setup(WaterDensity, spacing, spacingScale, domainGrid, sphSet);
    solver.SetColliders(colliders);

    Assure(UtilIsDistributionConsistent(pSet, domainGrid) == 1);
    auto callback = [&](int step) -> int{
        if(step == 0) return 1;

        if(step < 700)
            pBuilder.MapGridEmit(velocityField, spacing);

        box->Update(boxModel * RotateZ(rotation_angle(step)));

        std::string respath = FrameOutputPath("tank/output_", step-1);
        int flags = SERIALIZER_POSITION;
        UtilSaveSimulation3(&solver, pSet, respath.c_str(), flags);
        UtilPrintStepStandard(&solver, step-1);

        return step < 1100 ? 1 : 0;
    };

    UtilRunDynamicSimulation3<PciSphSolver3, ParticleSet3>(&solver, pSet, spacing, origin,
                                                           target, targetInterval, extraParts,
                                                           sdfs, callback, colorFunction,
                                                           filler);

    CudaMemoryManagerClearCurrent();
    printf("===== OK\n");
}


void test_pcisph3_tank_dam_paper(){
    printf("===== PCISPH Solver 3D -- Tank Dam\n");
    CudaMemoryManagerStart(__FUNCTION__);

    vec3f origin(0, 4, 15);
    vec3f target(0);

    //Float spacing = 0.2;
    Float spacing = 0.04;
    Float spacingScale = 2.0;
    Float targetInterval =  1.0 / 240.0;

    ///////////////// CONTAINER
    vec2f xRange = vec2f(-10, 5);
    vec2f yRange = vec2f(-3, 3);
    vec2f zRange = vec2f(-5, 5);

    vec3f containerSize = vec3f(xRange.y - xRange.x,
                                yRange.y - yRange.x,
                                zRange.y - zRange.x);

    Shape *containerShape = MakeBox(Transform(), containerSize, true);

    std::cout << "Domain: " << containerShape->GetBounds() << std::endl;
    printf("Container Size= {%g %g %g}\n", containerSize.x,
            containerSize.y, containerSize.z);
    ////////////////////////////

    ///////////////// EMITTER
    VolumeParticleEmitterSet3 emitterSet;
    Float delta = spacing * spacingScale;
    Float xScaleSize = 0.067;
    Float yScaleSize = 0.80f;
    Float zScaleSize = 0.98f;
    Float zExt  = (zRange.y - zRange.x);
    Float xSize = (xRange.y - xRange.x) * xScaleSize;
    Float ySize = (yRange.y - yRange.x) * yScaleSize;
    Float zSize = zExt * zScaleSize;

    Float emitterXOff = (xRange.y - xRange.x) * 0.5f - xSize * 0.5f - delta;
    Float emitterYOff = (yRange.y - yRange.x) * 0.5f - ySize * 0.5f - delta;
    zSize = (zSize + 2 * delta) > zExt ? zExt - 2 * delta : zSize;

    Transform emitterTransform = Translate(
                vec3f(emitterXOff, -emitterYOff, 0)
    );

    Shape *emitterShape = MakeBox(emitterTransform, vec3f(xSize, ySize, zSize));
    emitterSet.AddEmitter(emitterShape, spacing);

    std::cout << emitterShape->GetBounds() << std::endl;
    printf("Emitter size= {%g %g %g}\n", xSize, ySize, zSize);
    printf("Emitter Offset= {%g %g}\n", emitterXOff, emitterYOff);
    /////////////////////////

    ///////////////////// COLLIDERS
    std::vector<Shape *> boxList;
    ColliderSetBuilder3 cBuilder;
    int extra_parts = 150;
    int total_parts = extra_parts + 2;

    const vec3f translations[] = {
        vec3f(-0.700000, -2.100000, -3.231250),
        vec3f(0.800000, -2.100000, 1.071131),
        vec3f(2.300000, -2.100000, -0.831093),
        vec3f(2.300000, -2.100000, 3.341714),
    };

    const Float rotations[] = {
        131.322418,
        184.655670,
        277.688782,
        126.885010,
    };

    const vec3f scale = vec3f(0.600000, 1.800000, 0.600000);

    for(int i = 0; i < 4; i++){
        Transform box_transform = Translate(translations[i]) * RotateY(rotations[i]);
        Shape *box = MakeBox(box_transform, scale);
        boxList.push_back(box);

        cBuilder.AddCollider3(box);
        total_parts += extra_parts;
    }

    auto filler = [&](float *pos, float *col) -> int{
        int f = UtilGenerateBoxPoints(pos, col, vec3f(1,1,0), containerSize,
                                      extra_parts, containerShape->ObjectToWorld);
        for(Shape *shape : boxList){
            Bounds3f bounds = shape->GetBounds();
            vec3f size = vec3f(bounds.ExtentOn(0),
                               bounds.ExtentOn(1),
                               bounds.ExtentOn(2));
            f += UtilGenerateBoxPoints(&pos[3 * f], &col[3 * f], vec3f(1,0,0),
                                       size, extra_parts, shape->ObjectToWorld);
        }
        return f;
    };

    cBuilder.AddCollider3(containerShape);
    ///////////////////////////////

    ////////////////////////// SETUP
    ParticleSetBuilder3 pBuilder;
    PciSphSolver3 solver;

    ColliderSet3 *colliders = cBuilder.GetColliderSet();
    Assure(UtilIsEmitterOverlapping(&emitterSet, colliders) == 0);

    emitterSet.SetJitter(0.001);
    emitterSet.Emit(&pBuilder);

    Grid3 *domainGrid = UtilBuildGridForDomain(containerShape->GetBounds(), spacing,
                                               spacingScale);

    SphParticleSet3 *sphSet = SphParticleSet3FromBuilder(&pBuilder);
    sphSet->SetRelativeKernelRadius(spacingScale);

    solver.Initialize(DefaultSphSolverData3());
    solver.Setup(WaterDensity, spacing, spacingScale, domainGrid, sphSet);
    solver.SetColliders(colliders);
    ///////////////////////////////

    //////////////////// SIM
    ParticleSet3 *pSet = sphSet->GetParticleSet();
    auto onStepUpdate = [&](int step) -> int{
        if(step == 0)
        return 1;

        std::string path = FrameOutputPath("tank_break/out_", step-1);
        int flags = SERIALIZER_POSITION;
        UtilSaveSimulation3(&solver, pSet, path.c_str(), flags);

        UtilPrintStepStandard(&solver, step-1);
        return step < 1000;
    };

    auto colorFunction = [&](float *colors, int pCount) -> void{
        for(int i = 0; i < pCount; i++){
            colors[3 * i + 0] = 1; colors[3 * i + 1] = 0;
            colors[3 * i + 2] = 1;
        }
    };

    UtilRunDynamicSimulation3<PciSphSolver3, ParticleSet3>
                        (&solver, pSet, spacing, origin,
                         target, targetInterval, total_parts,
                         {}, onStepUpdate, colorFunction, filler);
    ////////////////////////

    CudaMemoryManagerClearCurrent();
    printf("\n===== OK\n");
}


DeclareFunctionalInteraction3D(HelixGravity3D,
{
    //Float radius = 1.25f;
    vec2f centerXZ = vec2f(0);
    vec2f projPXZ = vec2f(point.x, point.z);

    vec2f normal = Normalize(projPXZ - centerXZ);
    vec2f tang = vec2f(-normal.y, normal.x);
    Float radial = 3.0;
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
    Float spacingScale = 2.0;
    vec3f origin(-6.0, 3.0, 0.0);
    vec3f target(0.0f);
    vec3f emitterVelocity(0.0f, 0.0f, 0.0f);

    Float ballRadius = baseContainerSize / 16.0f;
    vec3f containerSize(baseContainerSize * 2.f,
                        baseContainerSize * 2.f,
                        baseContainerSize * 2.f);
    vec3f sphereCenter(0.f, -containerSize.y * 0.48 + ballRadius,
                       -1.5f);

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


DeclareFunctionalInteraction3D(GravityField_3D_1,
{
    vec3f center(0.f, 0.f, -1.f);
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

DeclareFunctionalInteraction3D(GravityField_3D_2,
{
    vec3f center(0.f, 0.f, 1.f);
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

void test_pcisph3_gravity_field2(){
    printf("===== PCISPH Solver 3D -- Gravity Field 2\n");
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
    Shape *sphereCollider0 = MakeSphere(Translate(vec3f(0, 0, 1)), 2.0f * sphereRadius);
    Shape *sphereCollider1 = MakeSphere(Translate(vec3f(0, 0, -1)), 2.0f * sphereRadius);
    Shape *sphereEmitter = MakeSphere(Translate(sphereCenter), sphereRadius);

    printf("Collider radius= %g\n", sphereCollider0->radius);
    printf("Emitter radius= %g\n", sphereEmitter->radius);
    std::cout << "Container= " << container->GetBounds() << std::endl;

    // Colliders
    ColliderSetBuilder3 cBuilder;
    // TODO: Add the central sphere?
    cBuilder.AddCollider3(sphereCollider0);
    cBuilder.AddCollider3(sphereCollider1);
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
    AddFunctionalInteraction3D(intrBuilder, GravityField_3D_1);
    AddFunctionalInteraction3D(intrBuilder, GravityField_3D_2);

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

        std::string path = FrameOutputPath("gravity2/out_", step-1);
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

