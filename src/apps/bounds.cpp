#include <args.h>
#include <graphy.h>
#include <serializer.h>
#include <util.h>
#include <string>
#include <boundary.h>
#include <memory.h>

typedef struct{
    std::string input;
    std::string output;
    Float spacing;
    Float spacingScale;
    int inflags;
    int outflags;
    int legacy;
    BoundaryMethod method;
}boundary_opts;

void default_boundary_opts(boundary_opts *opts){
    opts->output = "output_bound.txt";
    opts->method = BOUNDARY_NONE;
    opts->inflags  = SERIALIZER_POSITION;
    opts->outflags = SERIALIZER_POSITION | SERIALIZER_BOUNDARY;
    opts->spacing = 0.02;
    opts->spacingScale = 2.0;
    opts->legacy = 0;
}

void print_boundary_configs(boundary_opts *opts){
    std::cout << "Configs: " << std::endl;
    std::cout << "    * Input : " << opts->input << std::endl;
    std::cout << "    * Output : " << opts->output << std::endl;
    std::cout << "    * Method : " << GetBoundaryMethodName(opts->method) << std::endl;
    std::cout << "    * Spacing : " << opts->spacing << std::endl;
    std::cout << "    * Spacing Scale : " << opts->spacingScale << std::endl;
}

ARGUMENT_PROCESS(boundary_in_args){
    boundary_opts *opts = (boundary_opts *)config;
    opts->input = ParseNext(argc, argv, i, "-in", 1);
    return 0;
}

ARGUMENT_PROCESS(boundary_spacing_args){
    boundary_opts *opts = (boundary_opts *)config;
    opts->spacing = ParseNextFloat(argc, argv, i, "-spacing");
    if(opts->spacing < 0.001){
        return -1;
    }

    return 0;
}

ARGUMENT_PROCESS(boundary_spacing_scale_args){
    boundary_opts *opts = (boundary_opts *)config;
    opts->spacingScale = ParseNextFloat(argc, argv, i, "-spacing-scale");
    if(opts->spacingScale < 1){
        return -1;
    }

    return 0;
}

ARGUMENT_PROCESS(boundary_legacy_args){
    boundary_opts *opts = (boundary_opts *)config;
    opts->legacy = 1;
    return 0;
}

ARGUMENT_PROCESS(boundary_out_args){
    boundary_opts *opts = (boundary_opts *)config;
    opts->output = ParseNext(argc, argv, i, "-out", 1);
    return 0;
}

ARGUMENT_PROCESS(boundary_method_args){
    boundary_opts *opts = (boundary_opts *)config;
    std::string method = ParseNext(argc, argv, i, "-out", 1);
    opts->method = GetBoundaryMethod(method);
    return 0;
}

ARGUMENT_PROCESS(boundary_inflags_args){
    boundary_opts *opts = (boundary_opts *)config;
    std::string flags = ParseNext(argc, argv, i, "-inflags", 1);
    opts->inflags = SerializerFlagsFromString(flags.c_str());
    if(opts->inflags < 0) return -1;
    return 0;
}

ARGUMENT_PROCESS(boundary_outflags_args){
    boundary_opts *opts = (boundary_opts *)config;
    std::string flags = ParseNext(argc, argv, i, "-outflags", 1);
    opts->outflags = SerializerFlagsFromString(flags.c_str());
    if(opts->outflags < 0) return -1;
    return 0;
}

std::map<const char *, arg_desc> bounds_arg_map = {
    {"-in",
        {
            .processor = boundary_in_args,
            .help = "Sets the input file for boundary computation."
        }
    },
    {"-out",
        {
            .processor = boundary_out_args,
            .help = "Sets the output file."
        }
    },
    {"-spacing",
        {
            .processor = boundary_spacing_args,
            .help = "Sets the spacing of the domain ( Default : 0.02 )."
        }
    },
    {"-spacingScale",
        {
            .processor = boundary_spacing_scale_args,
            .help = "Sets the spacing scale ( Default : 2.0 )."
        }
    },
    {"-method",
        {
            .processor = boundary_method_args,
            .help = "Sets the method to be executed."
        }
    },
    {"-inform",
        {
            .processor = boundary_inflags_args,
            .help = "Set the input format."
        }
    },
    {"-outform",
        {
            .processor = boundary_outflags_args,
            .help = "Sets the output format."
        }
    },
    {"-legacy",
        {
            .processor = boundary_legacy_args,
            .help = "Sets the loader to use legacy format for reading input."
        }
    }
};

void process_boundary_request(boundary_opts *opts){
    ParticleSetBuilder3 builder;
    std::vector<SerializedShape> shapes;
    std::vector<int> boundary;

    CudaMemoryManagerStart(__FUNCTION__);
    if(opts->input.size() == 0){
        printf("No input file\n");
        return;
    }

    if(opts->output.size() == 0){
        printf("No output path\n");
        return;
    }

    if(!(opts->method >= BOUNDARY_LNM && opts->method < BOUNDARY_NONE)){
        printf("No method specified\n");
        return;
    }

    if(opts->legacy){
        std::vector<vec3f> points;
        SerializerLoadLegacySystem3(&points, opts->input.c_str(),
                                    opts->inflags, nullptr);
        for(vec3f &v : points){
            builder.AddParticle(v);
        }
    }else{
        SerializerLoadSystem3(&builder, &shapes, opts->input.c_str(),
                              opts->inflags, nullptr);
    }

    Grid3 *grid = UtilBuildGridForBuilder(&builder, opts->spacing,
                                          opts->spacingScale);
    SphSolver3 solver;
    SphParticleSet3 *sphpSet = SphParticleSet3FromBuilder(&builder);
    ParticleSet3 *pSet = sphpSet->GetParticleSet();
    solver.Initialize(DefaultSphSolverData3());
    solver.Setup(WaterDensity, opts->spacing, opts->spacingScale, grid, sphpSet);

    UpdateGridDistributionGPU(solver.solverData);
    ComputeDensityGPU(solver.solverData);

    grid->UpdateQueryState();
    LNMInvalidateCells(grid);
    pSet->ClearDataBuffer(&pSet->v0s);

    TimerList timer;
    timer.Start();
    if(opts->method == BOUNDARY_LNM){
        //LNMBoundaryExtended(pSet, grid, opts->spacing, 5, 0);
        LNMBoundary(pSet, grid, opts->spacing, 0);
    }else if(opts->method == BOUNDARY_DILTS){
        DiltsSpokeBoundary(pSet, grid);
        Float rad = DiltsGetParticleRadius(pSet->GetRadius());
        printf("Dilts radius %g\n", rad);
    }else if(opts->method == BOUNDARY_MULLER){
        MullerBoundary(pSet, grid, opts->spacing);
    }else if(opts->method == BOUNDARY_XIAOWEI){
        XiaoweiBoundary(pSet, grid, opts->spacing);
    }else if(opts->method == BOUNDARY_INTERVAL){
        IntervalBoundary(pSet, grid, opts->spacing);
    }
    else{
        // TODO: 3D Interval and other methods
    }

    timer.Stop();
    Float interval = timer.GetElapsedGPU(0);

    boundary.clear();
    int n = UtilGetBoundaryState(pSet, &boundary);
    printf("Got %d / %d - %g ms\n", n, (int)boundary.size(), interval);
    printf("Outputing to %s ... ", opts->output.c_str()); fflush(stdout);
    UtilEraseFile(opts->output.c_str());

    opts->outflags |= SERIALIZER_BOUNDARY;

    if(opts->legacy){
        SerializerSaveSphDataSet3Legacy(solver.solverData, opts->output.c_str(),
                                        opts->outflags, &boundary);
    }else{
        SerializerWriteShapes(&shapes, opts->output.c_str());
        SerializerSaveSphDataSet3(solver.solverData, opts->output.c_str(),
                                  opts->outflags, &boundary);
    }

    printf("OK\n");
    CudaMemoryManagerClearCurrent();
}

void boundary_command(int argc, char **argv){
    boundary_opts opts;
    default_boundary_opts(&opts);
    arg_desc desc = bounds_arg_map["-method"];
    std::vector<std::string> methods;
    GetBoundaryNames(methods);

    std::string value("Sets the method to be executed ( Choices: ");
    for(int i = 0; i < methods.size(); i++){
        value += methods[i];
        if(i < methods.size() - 1){
            value += ",";
        }
        value += " ";
    }
    value += ").";
    desc.help = value;
    bounds_arg_map["-method"] = desc;

    argument_process(bounds_arg_map, argc, argv, "boundary", &opts);
    print_boundary_configs(&opts);

    process_boundary_request(&opts);
}
