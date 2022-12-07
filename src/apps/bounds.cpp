#include <args.h>
#include <graphy.h>
#include <serializer.h>
#include <util.h>
#include <string>
#include <boundary.h>
#include <memory.h>
#include <fstream>
#include <sstream>
#include <map>
#include <narrowband.h>
#include <delaunay.h>

typedef struct{
    int count;
    Float accepted;
}synchronous_stats;

typedef struct{
    Float partl2ratio;
    Float partl2acceptedratio;
    int partl2count;
    int partl2accepted;
    std::vector<int> partcount, partaccepted;
    std::vector<Float> partacceptedRatio;
}work_queue_stats;

typedef struct{
    std::string input;
    std::string output;
    Float doringMu;
    Float spacing;
    Float spacingScale;
    Float nbRho;
    int countstart;
    int countend;
    int inner_cmd;
    int inflags;
    int outflags;
    int legacy_in, legacy_out;
    int lnmalgo;
    int use_cpu;
    int write_domain;
    int noout;
    SubdivisionMethod interval_sub_method;
    BoundaryMethod method;
    bool unbounded;
    bool narrowband;
}boundary_opts;

void default_boundary_opts(boundary_opts *opts){
    opts->output = "output_bound.txt";
    opts->method = BOUNDARY_NONE;
    opts->inflags  = SERIALIZER_POSITION;
    opts->outflags = SERIALIZER_POSITION | SERIALIZER_BOUNDARY;
    opts->spacing = 0.02;
    opts->spacingScale = 2.0;
    opts->legacy_in = 0;
    opts->legacy_out = 0;
    opts->lnmalgo = 2;
    opts->use_cpu = 0;
    opts->countstart = 0;
    opts->countend = 0;
    opts->write_domain = 0;
    opts->inner_cmd = 0;
    opts->noout = 0;
    opts->nbRho = 0.02;
    opts->doringMu = RDM_MU_3D;
    opts->unbounded = false;
    opts->narrowband = false;
    opts->interval_sub_method = PolygonSubdivision;
}

void print_boundary_configs(boundary_opts *opts){
    std::cout << "Configs: " << std::endl;
    std::cout << "    * Input : " << opts->input << std::endl;
    if(opts->inner_cmd == 0){
        std::cout << "    * Output : " << opts->output << std::endl;
        std::cout << "    * Method : " << GetBoundaryMethodName(opts->method) << std::endl;
        std::cout << "    * Spacing : " << opts->spacing << std::endl;
        std::cout << "    * Spacing Scale : " << opts->spacingScale << std::endl;
        std::cout << "    * Write Domain : " << opts->write_domain << std::endl;
        std::cout << "    * Noout : " << opts->noout << std::endl;
        std::cout << "    * Unbounded : " << opts->unbounded << std::endl;
        std::cout << "    * Narrow-Band : " << opts->narrowband << std::endl;
        if(opts->method == BOUNDARY_LNM){
            std::cout << "    * LNM Algo : " << opts->lnmalgo << std::endl;
        }

        if(opts->method == BOUNDARY_DORING){
            std::cout << "    * Doring μ : " << opts->doringMu << std::endl;
        }

        if(opts->narrowband){
            std::cout << "    * Narrow-Band ρ : " << opts->nbRho << std::endl;
        }
    }else if(opts->inner_cmd == 1){
        std::cout << "    * Statistics Run" << std::endl;
    }else{
        std::cout << "    * Reference Run" << std::endl;
    }
}

ARGUMENT_PROCESS(boundary_stats_arg){
    boundary_opts *opts = (boundary_opts *)config;
    opts->inner_cmd = 1;
    return 0;
}

ARGUMENT_PROCESS(boundary_nb_arg){
    boundary_opts *opts = (boundary_opts *)config;
    opts->narrowband = true;
    return 0;
}

ARGUMENT_PROCESS(boundary_nb_rho_arg){
    boundary_opts *opts = (boundary_opts *)config;
    opts->nbRho = ParseNextFloat(argc, argv, i, "-nbrho");
    if(opts->nbRho <= 0){
        printf("Invalid ρ value for NB\n");
        return -1;
    }

    return 0;
}

ARGUMENT_PROCESS(boundary_reference_arg){
    boundary_opts *opts = (boundary_opts *)config;
    opts->inner_cmd = 2;
    return 0;
}

ARGUMENT_PROCESS(boundary_noout_arg){
    boundary_opts *opts = (boundary_opts *)config;
    opts->noout = 1;
    return 0;
}

ARGUMENT_PROCESS(boundary_count_arg){
    boundary_opts *opts = (boundary_opts *)config;
    int i0 = ParseNextFloat(argc, argv, i, "-count");
    int i1 = ParseNextFloat(argc, argv, i, "-count");
    opts->countstart = i0;
    opts->countend   = i1;
    return 0;
}

ARGUMENT_PROCESS(boundary_in_args){
    boundary_opts *opts = (boundary_opts *)config;
    opts->input = ParseNext(argc, argv, i, "-in", 1);
    if(!FileExists(opts->input.c_str())){
        printf("Input file '%s' does not exist\n", opts->input.c_str());
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

ARGUMENT_PROCESS(boundary_interval_method_args){
    boundary_opts *opts = (boundary_opts *)config;
    std::string val = ParseNext(argc, argv, i, "-imethod", 1);
    if(val == "bb" || val == "BB")
        opts->interval_sub_method = BoundingBoxSubdivision;
    else if(val == "poly" || val == "POLY")
        opts->interval_sub_method = PolygonSubdivision;
    else{
        printf("Unknown value\n");
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

ARGUMENT_PROCESS(boundary_doring_mu_arg){
    boundary_opts *opts = (boundary_opts *)config;
    opts->doringMu = ParseNextFloat(argc, argv, i, "-mu");
    if(opts->doringMu > 1 || opts->doringMu < 0){
        printf("Doring μ must be: 0 < μ < 1\n");
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
    opts->legacy_in = 1;
    opts->legacy_out = 1;
    return 0;
}

ARGUMENT_PROCESS(boundary_legacy_in_args){
    boundary_opts *opts = (boundary_opts *)config;
    opts->legacy_in = 1;
    return 0;
}

ARGUMENT_PROCESS(boundary_legacy_out_args){
    boundary_opts *opts = (boundary_opts *)config;
    opts->legacy_out = 1;
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

ARGUMENT_PROCESS(boundary_write_domain_args){
    boundary_opts *opts = (boundary_opts *)config;
    opts->write_domain = 1;
    return 0;
}

ARGUMENT_PROCESS(boundary_lnm_unbounded_args){
    boundary_opts *opts = (boundary_opts *)config;
    opts->unbounded = true;
    return 0;
}

std::map<const char *, arg_desc> bounds_arg_map = {
    {"-in",
        {
            .processor = boundary_in_args,
            .help = "Sets the input file for boundary computation when used with methods."
                    " Sets the base path for statistics ( Requires -stats )."
        }
    },
    {"-nbrho",
        {
            .processor = boundary_nb_rho_arg,
            .help = "Sets the ρ value for the narrow-band extraction ( Requires -narrow-band )."
        }
    },
    {"-narrow-band",
        {
            .processor = boundary_nb_arg,
            .help = "Extends the boundary with ρ-bounded narrow-band over the boundary."
        }
    },
    {"-ref",
        {
            .processor = boundary_reference_arg,
            .help = "Sets solver to compute cell level references."
        }
    },
    {"-noout",
        {
            .processor = boundary_noout_arg,
            .help = "Sets the boundary computation to not output any files."
        }
    },
    {"-lnmalgo",
        {
            .processor = boundary_lnm_algo_args,
            .help = "Sets the LNM algorithm to use ( Applies to LNM method only )."
        }
    },
    {"-unbounded",
        {
            .processor = boundary_lnm_unbounded_args,
            .help = "Sets the LNM solver to use unbounded L2 computation."
        }
    },
    {"-count",
        {
            .processor = boundary_count_arg,
            .help = "Sets the range of files to be automatically loaded for analysis ( Requires -stats )."
        }
    },
    {"-stats",
        {
            .processor = boundary_stats_arg,
            .help = "Triggers computation of particle boundary statistics."
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
    {"-mu",
        {
            .processor = boundary_doring_mu_arg,
            .help = "Sets the μ value for the Randles-Doring method ( Default : 0.75 )."
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
            .help = "Sets the loader to use legacy format for input and output."
        }
    },
    {"-legacy_in",
        {
            .processor = boundary_legacy_in_args,
            .help = "Sets the loader to use legacy format for input."
        }
    },
    {"-legacy_out",
        {
            .processor = boundary_legacy_out_args,
            .help = "Sets the loader to use legacy format for output."
        }
    },
    {"-imethod",
        {
            .processor = boundary_interval_method_args,
            .help = "Sets the method to use for space subdivision when "
                    "running Sandim's Interval Method ( Choices: bb, poly (default) )."
        }
    },
    {"-write-domain",
        {
            .processor = boundary_write_domain_args,
            .help = "Writes the domain of the boundary computation (grid) in particle format."
        }
    }
};

void process_stats_procedure(boundary_opts *opts){
    long unsigned int total  = 0;
    long unsigned int totalb = 0;
    if(!(opts->inflags & SERIALIZER_BOUNDARY)){
        printf("\n {ERROR} This procedure requires boundary files\n");
        return;
    }

    int count = opts->countend - opts->countstart;
    if(count <= 0){
        printf("Invalid range given for -stats\n");
        return;
    }

    int s = 0;
    double worst_case_fraction = 0, best_case_fraction = 100;
    long unsigned int worst_case_b = 0, best_case_p = 0;
    long unsigned int worst_case_p = 0, best_case_b = 0;
    long unsigned int max_b = 0, min_b = 999999999;
    for(int i = opts->countstart; i < opts->countend; i++){
        CudaMemoryManagerStart(__FUNCTION__);
        ParticleSetBuilder3 builder;
        std::vector<SerializedShape> shapes;
        std::vector<int> boundary;

        std::stringstream ss;
        ss << opts->input;
        ss << i << ".txt";

        std::string str = ss.str();
        if(opts->legacy_in){
            std::vector<vec3f> points;
            SerializerLoadLegacySystem3(&points, str.c_str(),
                                        opts->inflags, &boundary);
        }else{
            SerializerLoadSystem3(&builder, &shapes, str.c_str(),
                                  opts->inflags, &boundary);
        }

        long unsigned int bb = 0, pp = 0;
        for(int &b : boundary){
            pp += 1;
            bb += b > 0 ? 1 : 0;
        }

        totalb += bb;
        total  += pp;
        max_b = Max(max_b, bb);
        min_b = Min(min_b, bb);

        double f = (double)bb / (double)pp; f *= 100.0;
        if(f > worst_case_fraction){
            worst_case_fraction = f;
            worst_case_p = pp;
            worst_case_b = bb;
        }

        if(f < best_case_fraction){
            best_case_fraction = f;
            best_case_p = pp;
            best_case_b = bb;
        }


        CudaMemoryManagerClearCurrent();
        printf("\r %d / %d", s+1, count); fflush(stdout);
        s++;
    }

    double average_p = (double) total / (double) count;
    double average_b = (double)totalb / (double)count;
    double fraction = (double)totalb / (double)total;
    printf("\n Average : P = %g, B = %g ( %g %% )\n", average_p, average_b, fraction * 100.0);
    printf(" Worst case: P = %lu, B = %lu ( %g %% )\n",
            worst_case_p, worst_case_b, worst_case_fraction);
    printf(" Best case: P = %lu, B = %lu ( %g %% )\n",
            best_case_p, best_case_b, best_case_fraction);
    printf(" Approx parts: %ld\n", (long int)(fraction * average_p));
    printf(" Max boundary: %lu\n", max_b);
    printf(" Min boundary: %lu\n", min_b);
}

template<typename T>
__bidevice__ void atomic_increase_stats(T *val){
#if defined(__CUDA_ARCH__)
    (void)atomicAdd(val, T(1));
#else
    (void)__atomic_fetch_add(val, T(1), __ATOMIC_SEQ_CST);
#endif
}

void advance_boundary_references(work_queue_stats *workQstats, int targetLevel,
                                 SphSolverData3 *data, int bcount)
{
    int accepted = 0;
    Float acceptedRatio = 0;
    ParticleSet3 *pSet = data->sphpSet->GetParticleSet();
    Grid3 *grid = data->domain;
    std::string kernel("Reference Kernel #");
    kernel += std::to_string(targetLevel);

    synchronous_stats *stats = cudaAllocateUnregisterVx(synchronous_stats, 1);
    stats->count = 0;
    stats->accepted = 0;

    GPUParallelLambda(kernel.c_str(), pSet->GetParticleCount(), GPU_LAMBDA(int i){
        vec3f pi = pSet->GetParticlePosition(i);
        unsigned int cellId = grid->GetLinearHashedPosition(pi);
        int level = grid->GetCellLevel(cellId);
        if(level == targetLevel){
            int bid = pSet->GetParticleV0(i);
            atomic_increase_stats(&stats->count);
            if(bid > 0){
                atomic_increase_stats(&stats->accepted);
            }
        }
    });

    workQstats->partcount.push_back(stats->count);
    if(stats->count > 0){
        acceptedRatio = stats->accepted / Float(bcount);
        accepted = (int)stats->accepted;
    }

    workQstats->partacceptedRatio.push_back(acceptedRatio);
    workQstats->partaccepted.push_back(accepted);

    printf("*******************************\n");
    printf(" * Level : %d\n", targetLevel);
    printf(" * Count : %d\n", stats->count);
    printf(" * Accepted : %d\n", accepted);
    printf(" * Ratio : %g%%\n", acceptedRatio * 100.0);
    printf("*******************************\n");

    cudaFree(stats);
}

__bidevice__
vec2i compute_part2_minimum_neighboors(ParticleSet3 *pSet, Grid3 *grid, int i){
    vec2i res(0, 0);
    vec3f pi = pSet->GetParticlePosition(i);
    int bi = pSet->GetParticleV0(i);
    int minCount = -1;
    unsigned int cellId = grid->GetLinearHashedPosition(pi);
    int level = grid->GetCellLevel(cellId);
    if(level == 2 && bi != 2){
        res.y = 1;
    }else if(bi == 2){
        if(level != 2){
            printf("Ooops\n");
        }

        grid->ForAllNeighborsOf(cellId, 1, [&](Cell3 *cell, vec3ui cid, int lid) -> int{
            if(i == lid) return 0;
            if(cell->GetChainLength() > 0){
                if(minCount < 0) minCount = cell->GetChainLength();

                minCount = Min(minCount, cell->GetChainLength());
            }
            return 0;
        });
    }

    res.x = minCount;

    return res;
}

vec2i *compute_particle_l2_afinnity(ParticleSet3 *pSet, Grid3 *grid,
                                    Float h, int *cellsAff)
{
    vec2i *minimums = cudaAllocateVx(vec2i, pSet->GetParticleCount());
    int *cAf = cudaAllocateUnregisterVx(int, 1);
    Float rho = h;

    *cAf = 0;
    GPUParallelLambda("Affinity", pSet->GetParticleCount(), GPU_LAMBDA(int i){
        vec2i minCount = compute_part2_minimum_neighboors(pSet, grid, i);
        minimums[i] = minCount;
    });

    GPUParallelLambda("Affinity", grid->GetActiveCellCount(), GPU_LAMBDA(int id){
        int i = grid->GetActiveCellId(id);
        Cell3 *self = grid->GetCell(i);
        Float delta = LNMComputeDelta(grid, rho);
        int level = self->GetLevel();
        bool is_l2 = false;

        if(level == 2){
            grid->ForAllNeighborsOf(i, 1, [&](Cell3 *cell, vec3ui cid, int lid)->int{
                if(i == lid) return 0;

                if(cell->GetLevel() == 1){
                    int n = cell->GetChainLength();
                    if(n <= delta){
                        is_l2 = true;
                        return 1;
                    }
                }

                return 0;
            });
        }

        if(is_l2){
            atomic_increase_stats(cAf);
        }
    });

    *cellsAff = *cAf;
    cudaFree(cAf);
    return minimums;
}

void process_boundary_request(boundary_opts *opts, work_queue_stats *workQstats=nullptr,
                              bool dump_to_file=true)
{
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

    if(opts->legacy_in){
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
    vec3ui res = grid->GetIndexCount();
    vec3f p0 = dBounds.pMin;
    vec3f p1 = dBounds.pMax;
    printf("Built domain with extension:\n"
            "    [%u x %u x %u]\n"
            "    [%g %g %g]  x  [%g %g %g]\n",
           res.x, res.y, res.z, p0.x, p0.y, p0.z, p1.x, p1.y, p1.z);


    solver.Setup(WaterDensity, opts->spacing, opts->spacingScale, grid, sphpSet);

    UpdateGridDistributionGPU(solver.solverData);
    /* compute density and normal just in case algorithm picked needs it */
    ComputeDensityGPU(solver.solverData);
    ComputeNormalGPU(solver.solverData);

    grid->UpdateQueryState();
    LNMInvalidateCells(grid);
    pSet->ClearDataBuffer(&pSet->v0s);

    TimerList timer;
    boundary.clear();
    if(opts->method == BOUNDARY_LNM){
        if(opts->lnmalgo == 5){
            timer.Start();
            LNMBoundaryExtended(pSet, grid, opts->spacing, 5, 0, opts->unbounded);
            timer.Stop();
        }else if(opts->lnmalgo == 0){
            timer.Start();
            LNMBoundarySingle(pSet, grid, opts->spacing, opts->unbounded);
            timer.Stop();
        }else{
            int npart = pSet->GetParticleCount();
            LNMWorkQueue *workQ = cudaAllocateVx(LNMWorkQueue, 1);
            workQ->SetSlots(npart);
            timer.Start();
            LNMBoundary(pSet, grid, opts->spacing, opts->lnmalgo,
                        workQ, opts->unbounded);
            timer.Stop();
            Float evals = (Float)workQ->size;
            Float counter = (Float)workQ->counter;
            Float pl2aratio = 100.0 * counter / evals;
            Float pl2ratio  = 100.0 * evals / (Float)npart;
            if(workQstats){
                workQstats->partl2acceptedratio = pl2aratio;
                workQstats->partl2ratio = pl2ratio;
                workQstats->partl2count = workQ->size;
                workQstats->partl2accepted = counter;
            }

            printf("WorkQueue Stats: \n"
                   "  * L2 ratio: %g%%\n"
                   "  * Acceptance ratio: %g%%\n"
                   "  * L2 particles: %d\n"
                   "  * Accepted particles: %d\n",
                   pl2ratio, pl2aratio, workQ->size, workQ->counter);
        }

        int cAf = 0;
        vec2i *aff = compute_particle_l2_afinnity(pSet, grid, opts->spacing, &cAf);
        Float before = 0, after = 0;

        std::map<int, int> id_map;
        for(int i = 0; i < pSet->GetParticleCount(); i++){
            int mi = aff[i].x;
            if(mi > -1){
                if(id_map.find(mi) == id_map.end()){
                    id_map[mi] = 1;
                }else{
                    int v = id_map[mi];
                    id_map[mi] = v+1;
                }
            }

            if(aff[i].y == 0){
                before += 1;
            }else{
                after += 1;
            }
        }

        std::ofstream ofs("samples_af.txt");
        int delta = (int)LNMComputeDelta(grid, opts->spacing);
        unsigned int preDelta = 0, postDelta = 0;

        for(auto it = id_map.begin(); it != id_map.end(); it++){
            if(it->first <= delta){
                preDelta += it->second;
            }else{
                postDelta += it->second;
            }
            ofs << it->first << "," << it->second << std::endl;
        }

        Float totall2 = 0, partl2 = Float(cAf);
        for(int i = 0; i < grid->GetActiveCellCount(); i++){
            unsigned int id = grid->GetActiveCellId(i);
            Cell3 *cell = grid->GetCell(id);
            if(cell->GetLevel() == 2){
                totall2 += 1;
            }
        }

        printf("Pre : %u, Pos: %u ( %g%% )\n", preDelta, postDelta,
                double(preDelta * 100.0) / double(preDelta + postDelta));

        printf("PreL2: %g, PosL2: %g ( %g%% )\n", before, after,
                (before * 100.0) / (before + after));

        printf("L2: %g, PartL2: %g ( %g%% )\n", totall2, partl2,
                (partl2 * 100.0) / totall2);

        ofs.close();
    }else if(opts->method == BOUNDARY_DILTS){
        timer.Start();
        DiltsSpokeBoundary(pSet, grid);
        timer.Stop();
        Float rad = DiltsGetParticleRadius(pSet->GetRadius());
        printf("Dilts radius %g\n", rad);
    }else if(opts->method == BOUNDARY_COLOR_FIELD){
        timer.Start();
        CFBoundary(pSet, grid, opts->spacing);
        timer.Stop();
    }else if(opts->method == BOUNDARY_XIAOWEI){
        timer.Start();
        XiaoweiBoundary(pSet, grid, opts->spacing);
        timer.Stop();
    }else if(opts->method == BOUNDARY_SANDIM){
        SandimWorkQueue3 *vpWorkQ = cudaAllocateVx(SandimWorkQueue3, 1);
        vpWorkQ->SetSlots(grid->GetCellCount());
        timer.Start();
        SandimBoundary(pSet, grid, vpWorkQ);
        timer.Stop();
    }else if(opts->method == BOUNDARY_INTERVAL){
        timer.Start();
        IntervalBoundary(pSet, grid, opts->spacing, opts->interval_sub_method);
        timer.Stop();
    }else if(opts->method == BOUNDARY_MARRONE){
        timer.Start();
        MarroneBoundary(pSet, grid, opts->spacing);
        timer.Stop();
    }else if(opts->method == BOUNDARY_MARRONE_ALT){
        WorkQueue<vec4f> *marroneWorkQ = cudaAllocateVx(WorkQueue<vec4f>, 1);
        marroneWorkQ->SetSlots(pSet->GetParticleCount());
        timer.Start();
        MarroneAdaptBoundary(pSet, grid, opts->spacing, marroneWorkQ);
        timer.Stop();
    }
    /*
        The following methods are implemented correctly
        **to the best of my knowledge** but they are unstable, i.e.:
        only working under some frames. I don't know if I'm simply stupid
        and don't understand their papers or if there are missing information
        in the presentation, use at your own risk.
     */
    else if(opts->method == BOUNDARY_DORING){
        timer.Start();
        RandlesDoringBoundary(pSet, grid, opts->spacing, opts->doringMu);
        timer.Stop();
    }
    else{

    }

    Float interval = timer.GetElapsedGPU(0);

    int n = UtilGetBoundaryState(pSet, &boundary);
    printf("Got %d / %d - %g ms\n", n, (int)boundary.size(), interval);

#if 0
    DelaunayTriangulation triangulation;
    printf("Computing raw triangulation ... "); fflush(stdout);
    DelaunayTriangulate(triangulation, sphpSet, grid);

    printf("Done.\nFiltering ... "); fflush(stdout);

    DelaunayShrink(triangulation, sphpSet, boundary);

    DelaunayWritePly(triangulation, "delaunay.ply");

    printf("Done.\n");
#endif

    if(opts->narrowband){
        boundary.clear();
        printf("Extracting narrow-band with ρ = %g\n", opts->nbRho);
        GeometricalNarrowBand(pSet, grid, opts->nbRho);
        int s = UtilGetBoundaryState(pSet, &boundary);
        printf("Extended by %d particles\n", s - n);
    }

    if(workQstats){
        LNMInvalidateCells(grid);
        LNMClassifyLazyGPU(grid);
        int maxLevel = grid->GetLNMMaxLevel();
        int bcount = 0;
        for(int i = 0; i < boundary.size(); i++){
            bcount += boundary[i] > 0 ? 1 : 0;
        }

        printf(" * Max level class %d\n", maxLevel);
        SphSolverData3 *data = solver.GetSphSolverData();
        for(int i = 1; i < maxLevel; i++){
            advance_boundary_references(workQstats, i, data, bcount);
        }
    }

    if(dump_to_file){
        printf("Outputing to %s ... ", opts->output.c_str()); fflush(stdout);
        UtilEraseFile(opts->output.c_str());

        opts->outflags |= SERIALIZER_BOUNDARY;

        if(opts->legacy_out){
            SerializerSaveSphDataSet3Legacy(solver.solverData, opts->output.c_str(),
                                            opts->outflags, &boundary);
        }else{
            SerializerWriteShapes(&shapes, opts->output.c_str());
            SerializerSaveSphDataSet3(solver.solverData, opts->output.c_str(),
                                      opts->outflags, &boundary);
        }

        printf("OK\n");

        if(opts->write_domain){
            printf("Writting domain grid to 'domain.txt'... "); fflush(stdout);
            SerializerSaveDomain(solver.solverData, "domain.txt");
            printf("OK\n");
        }
    }

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
    if(opts.inner_cmd == 1){
        process_stats_procedure(&opts);
    }else if(opts.inner_cmd == 0){
        int start = opts.countstart;
        int end   = opts.countend;
        if(start != end && end > 0){
            std::string base = opts.input;
            std::string baseout = opts.output;
            std::vector<work_queue_stats> vals;
            for(int i = start; i < end; i++){
                work_queue_stats ratios = {0, 0, 0, 0};
                std::string path = base + std::to_string(i);
                path += ".txt";
                printf(" ***** %s *****\n", path.c_str());
                opts.input = path;
                process_boundary_request(&opts, &ratios);
                vals.push_back(ratios);
            }

            std::ofstream ifs("ratio.txt");
            if(ifs.is_open()){
                std::stringstream ss;
                for(int i = 0; i < vals.size(); i++){
                    work_queue_stats stats = vals[i];
                    ifs << (i+1) << "," << stats.partl2ratio <<
                    "," << stats.partl2acceptedratio << "," <<
                    stats.partl2accepted << "," << stats.partl2count << "\n";
                }

                ifs << ss.str();
                ifs.close();
            }
        }else{
            process_boundary_request(&opts);
        }
    }else if(opts.inner_cmd == 2){
        work_queue_stats ratio = {0, 0, 0, 0};
        process_boundary_request(&opts, &ratio, opts.noout == 0);
    }
}
