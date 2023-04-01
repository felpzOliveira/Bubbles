#include <args.h>
#include <cutil.h>
#include <transform.h>
#include <particle.h>
#include <emitter.h>
#include <obj_loader.h>
#include <serializer.h>
#include <iostream>
#include <graphy.h>
#include <memory.h>
#include <util.h>

// To generate whale mesh use:
//    ./bbtool convert -in HappyWhale.obj -scale 0.3 -spacing 0.02

typedef struct{
    std::string input;
    std::string output;
    Float spacing;
    Float xrot, yrot, zrot;
    Float scale;
    Transform transform;
    int preview;
    int gen_data;
    std::string dataArgs;
    std::string outformArgs;
    std::string informArgs;
    vec3f origin;
    vec3f target;
    int to_legacy;
}config_opts;

static config_opts g_opts;

void default_opts(config_opts *opts){
    opts->spacing = 0;
    opts->transform = Transform();
    opts->output = "output.txt";
    opts->dataArgs = "pn";
    opts->informArgs = "p";
    opts->outformArgs = "p";
    opts->spacing = 0.02;
    opts->gen_data = 0;
    opts->scale = 1;
    opts->xrot = 0;
    opts->yrot = 0;
    opts->zrot = 0;
    opts->preview = 0;
    opts->origin = vec3f(3);
    opts->target = vec3f(0);
    opts->to_legacy = 0;
}

void print_configs(config_opts *opts){
    std::cout << "Configs: " << std::endl;
    std::cout << "    * Target file : " << opts->input << std::endl;
    std::cout << "    * Target output : " << opts->output << std::endl;
    std::cout << "    * Spacing : " << opts->spacing << std::endl;
    std::cout << "    * Transforms : " << std::endl;
    std::cout << "        - Rotate X : " << opts->xrot << std::endl;
    std::cout << "        - Rotate Y : " << opts->yrot << std::endl;
    std::cout << "        - Rotate Z : " << opts->zrot << std::endl;
    std::cout << "        - Scale : " << opts->scale << std::endl;
    std::cout << "    * Preview : " << opts->preview << std::endl;
    if(opts->preview){
        vec3f o = opts->origin;
        vec3f t = opts->target;
        std::cout << "        - Origin : " << o.x << " " << o.y << " " << o.z << std::endl;
        std::cout << "        - Target : " << t.x << " " << t.y << " " << t.z << std::endl;
    }
    std::cout << "    * Data Generation : " << opts->gen_data << std::endl;
    if(opts->gen_data){
        std::cout << "        - Inform : " << opts->informArgs << std::endl;
        std::cout << "        - Outform : " << opts->outformArgs << std::endl;
        std::cout << "        - Data : " << opts->dataArgs << std::endl;
    }
    std::cout << "    * Legacy : " << opts->to_legacy << std::endl;
}

ARGUMENT_PROCESS(legacy_arg){
    config_opts *opts = (config_opts *)config;
    opts->to_legacy = 1;
    return 0;
}

ARGUMENT_PROCESS(rotate_x_arg){
    config_opts *opts = (config_opts *)config;
    Float angle = ParseNextFloat(argc, argv, i, "-rotateX");
    opts->xrot = angle;
    opts->transform = RotateX(angle) * opts->transform;
    return 0;
}

ARGUMENT_PROCESS(rotate_y_arg){
    config_opts *opts = (config_opts *)config;
    Float angle = ParseNextFloat(argc, argv, i, "-rotateY");
    opts->yrot = angle;
    opts->transform = RotateY(angle) * opts->transform;
    return 0;
}

ARGUMENT_PROCESS(rotate_z_arg){
    config_opts *opts = (config_opts *)config;
    Float angle = ParseNextFloat(argc, argv, i, "-rotateZ");
    opts->zrot = angle;
    opts->transform = RotateZ(angle) * opts->transform;
    return 0;
}

ARGUMENT_PROCESS(scale_arg){
    config_opts *opts = (config_opts *)config;
    Float scale = ParseNextFloat(argc, argv, i, "-scale");
    opts->scale = scale;
    opts->transform = Scale(scale) * opts->transform;
    return 0;
}

ARGUMENT_PROCESS(input_arg){
    config_opts *opts = (config_opts *)config;
    opts->input = ParseNext(argc, argv, i, "-in");
    return 0;
}

ARGUMENT_PROCESS(output_arg){
    config_opts *opts = (config_opts *)config;
    opts->output = ParseNext(argc, argv, i, "-out");
    return 0;
}

ARGUMENT_PROCESS(spacing_arg){
    config_opts *opts = (config_opts *)config;
    Float spacing = ParseNextFloat(argc, argv, i, "-spacing");
    opts->spacing = spacing;
    return 0;
}

ARGUMENT_PROCESS(preview_origin_arg){
    config_opts *opts = (config_opts *)config;
    std::string strdist = ParseNext(argc, argv, i, "-origin", 3);
    const char *ptr = strdist.c_str();
    ParseV3(&opts->origin, &ptr);
    return 0;
}

ARGUMENT_PROCESS(preview_target_arg){
    config_opts *opts = (config_opts *)config;
    std::string strdist = ParseNext(argc, argv, i, "-target", 3);
    const char *ptr = strdist.c_str();
    ParseV3(&opts->target, &ptr);
    return 0;
}

ARGUMENT_PROCESS(preview_arg){
    config_opts *opts = (config_opts *)config;
    opts->preview = 1;
    return 0;
}

ARGUMENT_PROCESS(gen_data_arg){
    config_opts *opts = (config_opts *)config;
    opts->dataArgs = ParseNext(argc, argv, i, "-gen-data", 1);
    opts->gen_data = 1;
    return 0;
}

ARGUMENT_PROCESS(convert_file_arg){
    config_opts *opts = (config_opts *)config;
    opts->gen_data = 2;
    return 0;
}

ARGUMENT_PROCESS(parse_outform_arg){
    config_opts *opts = (config_opts *)config;
    opts->outformArgs = ParseNext(argc, argv, i, "-outform", 1);
    return 0;
}

ARGUMENT_PROCESS(parse_inform_arg){
    config_opts *opts = (config_opts *)config;
    opts->informArgs = ParseNext(argc, argv, i, "-inform", 1);
    return 0;
}

std::map<const char *, arg_desc> argument_map = {
    {"-gen-data",
        {.processor = gen_data_arg,
            .help = "Given a bubbles output compute extra information not previously obtained."
        }
    },
    {"-cvt",
        {.processor = convert_file_arg,
            .help = "Given a bubbles output convert it to different format."
        }
    },
    {"-inform",
        {.processor = parse_inform_arg,
            .help = "When reading simulation file use specific format."
        }
    },
    {"-outform",
        {.processor = parse_outform_arg,
            .help = "When writting output use specific format."
        }
    },
    {"-origin",
        { .processor = preview_origin_arg,
            .help = "Sets the origin point for previewing. (requires: -preview)"
        }
    },
    {"-target",
        { .processor = preview_target_arg,
            .help = "Sets the target point for previewing. (requires: -preview)"
        }
    },
    {"-rotateX",
        { .processor = rotate_x_arg,
            .help = "Rotates the input in the X-direction. (degrees)"
        }
    },
    {"-rotateY",
        { .processor = rotate_y_arg,
            .help = "Rotates the input in the Y-direction. (degrees)"
        }
    },
    {"-rotateZ",
        { .processor = rotate_z_arg,
            .help = "Rotates the input in the Z-direction. (degrees)"
        }
    },
    {"-scale",
        { .processor = scale_arg,
            .help = "Scales the input uninformly."
        }
    },
    {"-in",
        { .processor = input_arg,
            .help = "Where to read input file."
        }
    },
    {"-out",
        { .processor = output_arg,
            .help = "Where to write output."
        }
    },
    {"-spacing",
        { .processor = spacing_arg,
            .help = "Spacing to use when generating particle cloud."
        }
    },
    {"-preview",
        { .processor = preview_arg,
            .help = "Use Graphy to preview the point cloud."
        }
    },
    {"-legacy",
        { .processor = legacy_arg,
            .help = "Use legacy file format for outputing fluid."
        }
    },
};


void PreviewParticles(SphParticleSet3 *sphSet, config_opts *opts){
    vec3f at = opts->origin;
    vec3f to = opts->target;
    ParticleSet3 *pSet = sphSet->GetParticleSet();
    int pCount = pSet->GetParticleCount();
    int it = 0;
    float *pos = new float[3 * pCount];
    float *col = new float[3 * pCount];

    memset(col, 0x00, 3 * pCount * sizeof(float));
    for(int i = 0; i < pCount; i++){
        vec3f pi = pSet->GetParticlePosition(i);
        pos[it++] = pi.x; pos[it++] = pi.y;
        pos[it++] = pi.z; col[3 * i + 0] = 1;
    }

    graphy_set_3d(at.x, at.y, at.z, to.x, to.y, to.z, 45.0, 0.1f, 100.0f);
    graphy_render_points3f(pos, col, pCount, 0.012);

    std::cout << "Press anything ... " << std::flush;
    getchar();

    graphy_close_display();

    delete[] pos;
    delete[] col;
}

void GenerateData(config_opts *opts){
    if(opts->gen_data == 1){
        printf("===== Generating Data\n");
    }else if(opts->gen_data == 2){
        printf("===== Converting Data\n");
    }
    CudaMemoryManagerStart(__FUNCTION__);
    // For this case input is a Bubbles simulation file
    ParticleSetBuilder3 builder;
    std::vector<SerializedShape> shapes;
    Float h = 0.02;
    //int chosenBoundaryAlgorithm = 0; // TODO
    std::vector<int> boundary;
    int ok = 0;
    int flagsIn  = SerializerFlagsFromString(opts->informArgs.c_str());
    int flagsOut = SerializerFlagsFromString(opts->outformArgs.c_str());
    int toGen    = SerializerFlagsFromString(opts->dataArgs.c_str());
    if(!(flagsIn == -1 || flagsOut == -1 || toGen == -1)){
        if(opts->to_legacy){
            std::vector<vec3f> points;
            SerializerLoadLegacySystem3(&points, opts->input.c_str(),
                                        flagsIn, &boundary);
            for(vec3f &p : points){
                builder.AddParticle(p);
            }
        }else{
            SerializerLoadSystem3(&builder, &shapes, opts->input.c_str(),
                                  flagsIn, &boundary);
        }

        SphParticleSet3 *sphpSet = SphParticleSet3FromBuilder(&builder);
        ParticleSet3 *pSet = sphpSet->GetParticleSet();
        Bounds3f bounds = UtilComputeParticleSetBounds(pSet);
        Grid3 *grid = UtilBuildGridForDomain(bounds, h, 2.0);

        SphSolver3 solver;
        solver.Initialize(DefaultSphSolverData3());
        solver.Setup(WaterDensity, h, 2.0, grid, sphpSet);

        UpdateGridDistributionGPU(solver.solverData);
        if((toGen & SERIALIZER_NORMAL) || (toGen & SERIALIZER_DENSITY)){
            printf("Generating Density ... "); fflush(stdout);
            ComputeDensityGPU(solver.solverData);
            printf("OK\n");
        }

        if(toGen & SERIALIZER_NORMAL){
            printf("Generating Normals ... "); fflush(stdout);
            ComputeNormalGPU(solver.solverData);
            printf("OK\n");
        }

        if((toGen & SERIALIZER_LAYERS) || (toGen & SERIALIZER_BOUNDARY)){
            grid->UpdateQueryState();
            LNMInvalidateCells(grid);
            pSet->ClearDataBuffer(&pSet->v0s);

            if(toGen & SERIALIZER_LAYERS){
                printf("Classifying layers ..."); fflush(stdout);
                int max_level = 8;

                LNMBoundaryExtended(pSet, grid, h, max_level, 0);

                printf(" %d\n", grid->GetLNMMaxLevel());
                //(void)chosenBoundaryAlgorithm; //TODO
            }else{
                printf("Classifying boundary particles ... "); fflush(stdout);
                LNMBoundary(pSet, grid, h, 0);
                printf("OK\n");
            }

            boundary.clear();
            UtilGetBoundaryState(pSet, &boundary);
        }

        printf("Outputing to %s ... ", opts->output.c_str()); fflush(stdout);
        UtilEraseFile(opts->output.c_str());

        if(opts->to_legacy){
            SerializerSaveSphDataSet3Legacy(solver.solverData, opts->output.c_str(),
                                            flagsOut, &boundary);
        }else{
            SerializerWriteShapes(&shapes, opts->output.c_str());

            SerializerSaveSphDataSet3(solver.solverData, opts->output.c_str(),
                                      flagsOut, &boundary);
        }

        printf("OK\n");
        ok = 1;
    }

    CudaMemoryManagerClearCurrent();

    if(ok){
        printf("===== OK\n");
    }else{
        printf("===== FAILED\n");
    }
}

void MeshToParticles(const char *name, const Transform &transform,
                     Float spacing, SphSolverData3 **data)
{
    SphParticleSet3 *sphSet = nullptr;
    printf("===== Emitting particles from mesh\n");

    ParsedMesh *mesh = LoadObj(name);

    Shape *shape = MakeMesh(mesh, transform);
    ParticleSetBuilder3 builder;
    VolumeParticleEmitter3 emitter(shape, shape->GetBounds(), spacing);

    emitter.Emit(&builder);

    sphSet = SphParticleSet3FromBuilder(&builder);
    sphSet->SetTargetSpacing(spacing);

    *data = DefaultSphSolverData3();
    (*data)->sphpSet = sphSet;
    (*data)->domain = nullptr;

    printf("===== OK\n");
}

void convert_command(int argc, char **argv){
    SphSolverData3 *data = nullptr;
    default_opts(&g_opts);
    argument_process(argument_map, argc, argv, "convert", &g_opts);
    print_configs(&g_opts);

    if(g_opts.gen_data == 0){
        MeshToParticles(g_opts.input.c_str(), g_opts.transform,
                        g_opts.spacing, &data);
        if(g_opts.preview){
            PreviewParticles(data->sphpSet, &g_opts);
        }

        if(g_opts.to_legacy){
            SerializerSaveSphDataSet3Legacy(data, g_opts.output.c_str(),
                                            SERIALIZER_POSITION, nullptr);
        }else{
            SerializerSaveSphDataSet3(data, g_opts.output.c_str(), SERIALIZER_POSITION);
        }
    }else if(g_opts.gen_data == 1){
        GenerateData(&g_opts);
    }else{ // == 2
        GenerateData(&g_opts);
    }
}
