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
    int lnmalgo;
    int use_cpu;
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
    opts->lnmalgo = 2;
    opts->use_cpu = 0;
}

void print_boundary_configs(boundary_opts *opts){
    std::cout << "Configs: " << std::endl;
    std::cout << "    * Input : " << opts->input << std::endl;
    std::cout << "    * Output : " << opts->output << std::endl;
    std::cout << "    * Method : " << GetBoundaryMethodName(opts->method) << std::endl;
    std::cout << "    * Spacing : " << opts->spacing << std::endl;
    std::cout << "    * Spacing Scale : " << opts->spacingScale << std::endl;
    if(opts->method == BOUNDARY_LNM){
        std::cout << "    * LNM Algo: " << opts->lnmalgo << std::endl;
    }
}

ARGUMENT_PROCESS(boundary_in_args){
    boundary_opts *opts = (boundary_opts *)config;
    opts->input = ParseNext(argc, argv, i, "-in", 1);
    if(!FileExists(opts->input.c_str())){
        printf("Input file does not exist\n");
        return -1;
    }
    return 0;
}

ARGUMENT_PROCESS(boundary_lnm_algo_args){
    boundary_opts *opts = (boundary_opts *)config;
    opts->lnmalgo = ParseNextFloat(argc, argv, i, "-lnm-algo");
    if(opts->lnmalgo != 0 && opts->lnmalgo != 1 &&
       opts->lnmalgo != 2 && opts->lnmalgo != 5)
    {
        printf("Unknown algorithm value\n");
        return -1;
    }
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

ARGUMENT_PROCESS(boundary_use_cpu_args){
    boundary_opts *opts = (boundary_opts *)config;
    auto fn = [&](std::string val) -> int{
        char *endptr[1];
        char *nptr = (char *)val.c_str();
        int n = strtol(nptr, endptr, 10);
        if(*endptr == nptr) return 0;
        return n;
    };
    int nt = ParseNextOrNone(argc, argv, i, "-cpu", fn);
    opts->use_cpu = nt > 0 ? nt : GetConcurrency();
    return 0;
}

std::map<const char *, arg_desc> bounds_arg_map = {
    {"-in",
        {
            .processor = boundary_in_args,
            .help = "Sets the input file for boundary computation."
        }
    },
    {"-lnmalgo",
        {
            .processor = boundary_lnm_algo_args,
            .help = "Sets the LNM algorithm to use ( Applies to LNM method only )."
        }
    },
    {"-out",
        {
            .processor = boundary_out_args,
            .help = "Sets the output file."
        }
    },
    {"-cpu",
        {
            .processor = boundary_use_cpu_args,
            .help = "Sets the computation to run on the CPU instead."
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
        printf("No valid method specified\n");
        return;
    }

    if(opts->use_cpu){
        SetCPUThreads(opts->use_cpu);
        SetSystemUseCPU();
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

    SphSolver3 solver;
    SphParticleSet3 *sphpSet = nullptr;
    ParticleSet3 *pSet = nullptr;

    solver.Initialize(DefaultSphSolverData3());
    Grid3 *grid = nullptr;
    if(opts->method != BOUNDARY_SANDIM){
        grid = UtilBuildGridForBuilder(&builder, opts->spacing,
                                       opts->spacingScale);
        sphpSet = SphParticleSet3FromBuilder(&builder);
        pSet = sphpSet->GetParticleSet();
    }else{
        sphpSet = SphParticleSet3FromBuilder(&builder);
        pSet = sphpSet->GetParticleSet();
        grid = SandimComputeCompatibleGrid(pSet, opts->spacing);
    }

    Bounds3f dBounds = grid->GetBounds();
    vec3f p0 = dBounds.pMin;
    vec3f p1 = dBounds.pMax;
    printf("Built domain with extension:\n"
            "    [%g %g %g]  x  [%g %g %g]\n",
           p0.x, p0.y, p0.z, p1.x, p1.y, p1.z);

    
    solver.Setup(WaterDensity, opts->spacing, opts->spacingScale, grid, sphpSet);

    UpdateGridDistributionGPU(solver.solverData);
    ComputeDensityGPU(solver.solverData);

    grid->UpdateQueryState();
    LNMInvalidateCells(grid);
    pSet->ClearDataBuffer(&pSet->v0s);

    TimerList timer;
    boundary.clear();
    if(opts->method == BOUNDARY_LNM){
        if(opts->lnmalgo == 5){
            timer.Start();
            LNMBoundaryExtended(pSet, grid, opts->spacing, 5, 0);
            timer.Stop();
        }else{
            int npart = pSet->GetParticleCount();
            LNMWorkQueue *workQ = nullptr;
            if(opts->lnmalgo >= 2){
                workQ = cudaAllocateVx(LNMWorkQueue, 1);
                workQ->SetSlots(npart);
            }
            timer.Start();
            LNMBoundary(pSet, grid, opts->spacing, opts->lnmalgo, workQ);
            timer.Stop();
            if(workQ){
                int workQueueParts = workQ->size;
                Float bounds = 0;
                int *localId = new int[npart];
                CUCHECK(cudaMemcpy(localId, workQ->ids, sizeof(int) * npart,
                                                    cudaMemcpyDeviceToHost));

                for(int i = 0; i < workQueueParts; i++){
                    if(localId[i] > 0){
                        bounds++;
                    }
                }

                Float evals = (Float)workQ->size;
                Float ratio = 100.0 * bounds / evals;
                printf("WorkQueue ratio: %g%%\n", ratio);

                delete[] localId;
            }
        }
    }else if(opts->method == BOUNDARY_DILTS){
        timer.Start();
        DiltsSpokeBoundary(pSet, grid);
        timer.Stop();
        Float rad = DiltsGetParticleRadius(pSet->GetRadius());
        printf("Dilts radius %g\n", rad);
    }else if(opts->method == BOUNDARY_MULLER){
        timer.Start();
        MullerBoundary(pSet, grid, opts->spacing);
        timer.Stop();
    }else if(opts->method == BOUNDARY_XIAOWEI){
        timer.Start();
        XiaoweiBoundary(pSet, grid, opts->spacing);
        timer.Stop();
    }else if(opts->method == BOUNDARY_SANDIM){
        timer.Start();
        SandimWorkQueue3 *vpWorkQ = cudaAllocateVx(SandimWorkQueue3, 1);
        vpWorkQ->SetSlots(grid->GetCellCount());
        SandimBoundary(pSet, grid, vpWorkQ);
        timer.Stop();
    }else if(opts->method == BOUNDARY_INTERVAL){
        timer.Start();
        IntervalBoundary(pSet, grid, opts->spacing);
        timer.Stop();
    }
    else{
        // TODO: 3D Interval seems dubious and others methods
    }

    Float interval = timer.GetElapsedGPU(0);

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