#include <args.h>
#include <marching_cubes.h>
#include <emitter.h>
#include <grid.h>
#include <util.h>
#include <memory.h>

typedef struct{
    Float kernel;
    Float spacing;
    Float marchingCubeSpacing;
    std::string input;
    std::string output;
    int inflags;
    int legacy;
}surface_opts;

void default_surface_opts(surface_opts *opts){
    opts->output = "surface.obj";
    opts->legacy = 0;
    opts->inflags = SERIALIZER_POSITION;
    opts->kernel = 0.04;
    opts->spacing = 0.02;
    opts->marchingCubeSpacing = 0.01;
}

void print_surface_configs(surface_opts *opts){
    std::cout << "Configs: " << std::endl;
    std::cout << "    * Input : " << opts->input << std::endl;
    std::cout << "    * Output : " << opts->output << std::endl;
    std::cout << "    * Spacing : " << opts->spacing << std::endl;
    std::cout << "    * Kernel : " << opts->kernel << std::endl;
    std::cout << "    * Mesh Spacing : " << opts->marchingCubeSpacing << std::endl;
    std::cout << "    * Legacy : " << opts->legacy << std::endl;
}

ARGUMENT_PROCESS(surface_in_args){
    surface_opts *opts = (surface_opts *)config;
    opts->input = ParseNext(argc, argv, i, "-in", 1);
    if(!FileExists(opts->input.c_str())){
        printf("Input file does not exist\n");
        return -1;
    }
    return 0;
}

ARGUMENT_PROCESS(surface_inflags_args){
    surface_opts *opts = (surface_opts *)config;
    std::string flags = ParseNext(argc, argv, i, "-inflags", 1);
    opts->inflags = SerializerFlagsFromString(flags.c_str());
    if(opts->inflags < 0) return -1;
    return 0;
}

ARGUMENT_PROCESS(surface_out_args){
    surface_opts *opts = (surface_opts *)config;
    opts->output = ParseNext(argc, argv, i, "-out", 1);
    return 0;
}

ARGUMENT_PROCESS(surface_spacing_args){
    surface_opts *opts = (surface_opts *)config;
    opts->spacing = ParseNextFloat(argc, argv, i, "-spacing");
    if(opts->spacing < 0.001){
        printf("Invalid spacing\n");
        return -1;
    }

    return 0;
}

ARGUMENT_PROCESS(surface_kernel_args){
    surface_opts *opts = (surface_opts *)config;
    opts->kernel = ParseNextFloat(argc, argv, i, "-kernel");
    if(opts->kernel < 0.001){
        printf("Invalid kernel\n");
        return -1;
    }

    return 0;
}

ARGUMENT_PROCESS(surface_cubes_spacing_args){
    surface_opts *opts = (surface_opts *)config;
    opts->marchingCubeSpacing = ParseNextFloat(argc, argv, i, "-resolution");
    if(opts->marchingCubeSpacing < 0.001){
        printf("Invalid mesh resolution\n");
        return -1;
    }

    return 0;
}

ARGUMENT_PROCESS(surface_legacy_args){
    surface_opts *opts = (surface_opts *)config;
    opts->legacy = 1;
    return 0;
}

std::map<const char *, arg_desc> surface_arg_map = {
    {"-in",
        {
            .processor = surface_in_args,
            .help = "Sets the input file for mesh computation."
        }
    },
    {"-out",
        {
            .processor = surface_out_args,
            .help = "Sets the output filename for the generated mesh."
        }
    },
    {"-inform",
        {
            .processor = surface_inflags_args,
            .help = "Sets the input flags to use for loading a simulation, for legacy."
        }
    },
    {"-legacy",
        {
            .processor = surface_legacy_args,
            .help = "Sets the loader to use legacy format."
        }
    },
    {"-spacing",
        {
            .processor = surface_spacing_args,
            .help = "Configures spacing used during simulation."
        }
    },
    {"-kernel",
        {
            .processor = surface_kernel_args,
            .help = "Sets the kernel radius to use."
        }
    },
    {"-sspacing",
        {
            .processor = surface_cubes_spacing_args,
            .help = "Sets the spacing to use for surface reconstruction (invert resolution)."
        }
    }
};

template <typename T>
__bidevice__ T cubic(T x){ return x * x * x; }
__bidevice__ double k(double s) { return Max(0.0, cubic(1.0 - s * s)); }

__bidevice__ Float ZhuBridsonSDF(vec3f p, Grid3 *grid, ParticleSet3 *pSet,
                                 Float kernelRadius, Float threshold)
{
    int *neighbors = nullptr;
    Float inf = grid->GetBounds().Diagonal().Length();
    if(!grid->Contains(p)) return inf;

    unsigned int cellId = grid->GetLinearHashedPosition(p);

    int count = grid->GetNeighborsOf(cellId, &neighbors);
    Float wSum = 0.0;
    vec3f xAvg(0);

    for(int i = 0; i < count; i++){
        Cell3 *cell = grid->GetCell(neighbors[i]);
        ParticleChain *pChain = cell->GetChain();
        int size = cell->GetChainLength();
        for(int j = 0; j < size; j++){
            vec3f xi = pSet->GetParticlePosition(pChain->pId);
            Float dist = Distance(p, xi);
            if(dist < kernelRadius){
                const Float wi = k((p - xi).Length() / kernelRadius);
                wSum += wi;
                xAvg += wi * xi;
            }

            pChain = pChain->next;
        }
    }

    if(wSum > 0.0){
        // In case wSum is too low, i.e.: < 1e-8, I'm not sure
        // we should continue computation or simply assume it is 0
        // but anyways we need to invert computation to avoid the
        // error dividing by ε < 1e-8
        Float inv = 1.0 / wSum;
        xAvg = xAvg * inv;
        return (p - xAvg).Length() - kernelRadius * threshold;
    }else{
        return inf;
    }
}

FieldGrid3f *ParticlesToSDF(ParticleSetBuilder3 *pBuilder, Float spacing,
                            Float sdfSpacing, Float radius)
{
    Float spacingScale = 1.5;
    Float threshold = 0.25;
    Float kernelRadius = radius;
    Float queryRadius = radius;
    vec3f sp(sdfSpacing);

    Bounds3f bounds;
    for(vec3f &p : pBuilder->positions){
        bounds = Union(bounds, p);
    }

    int minAxis = bounds.MinimumExtent();
    Float minAxisSize = bounds.ExtentOn(minAxis);
    for(int i = 0; i < 3; i++){
        if(i != minAxis){
            Float axisSize = bounds.ExtentOn(i);
            sp[i] = sdfSpacing * (axisSize / minAxisSize);
        }
    }

    SphParticleSet3 *sphSet = SphParticleSet3FromBuilder(pBuilder);
    ParticleSet3 *pSet = sphSet->GetParticleSet();
    Bounds3f sdfBounds = bounds;

    sdfBounds.Expand(5.0 * spacing);
    bounds.Expand(10.0 * spacing);

    Grid3 *domainGrid = UtilBuildGridForDomain(bounds, queryRadius, spacingScale);
    vec3f cellSize = domainGrid->GetCellSize();
    // Cell size must be able to fetch particles up to 'kernelRadius'
    AssureA(MinComponent(cellSize) >= queryRadius,
            "Cell size must be larger than smoothing radius");

    SphSolver3 solver;

    solver.Initialize(DefaultSphSolverData3());
    solver.Setup(WaterDensity, queryRadius, spacingScale, domainGrid, sphSet);

    std::cout << "Distributing particles... " << std::flush;
    UpdateGridDistributionGPU(solver.solverData);
    domainGrid->UpdateQueryState();
    std::cout << "OK" << std::endl;

    std::cout << "Generating SDF... " << std::flush;
    FieldGrid3f *sdfGrid = CreateSDF(sdfBounds, sp, GPU_LAMBDA(vec3f p){
        return ZhuBridsonSDF(p, domainGrid, pSet, kernelRadius, threshold);
    });

    vec3ui res = sdfGrid->GetResolution();
    std::cout << "OK [ Resolution: " << res.x << " x " << res.y << " x " << res.z
              << " ]" << std::endl;

    return sdfGrid;
}

void fill_pct(double val, char str[7]){
    if(val >= 100){
        sprintf(str, "100.00");
        return;
    }

    if(val < 10){
        sprintf(str, "00%.2f", val);
    }else if(val < 100){
        sprintf(str, "0%.2f", val);
    }

    str[6] = 0;
}

void fill_number(unsigned int val, char str[4]){
    int at = 0;
    if(val > 1000){
        str[0] = 'N'; str[1] = 'a'; str[2] = 'N';
        str[3] = 0;
        return;
    }

    if(val < 10){
        str[0] = '0'; str[1] = '0'; at = 2;
    }else if(val < 100){
        str[0] = '0'; at = 1;
    }

    sprintf(&str[at], "%u", val);
    str[4] = 0;
}

void surface_command(int argc, char **argv){
    surface_opts opts;
    ParticleSetBuilder3 builder;

    default_surface_opts(&opts);

    argument_process(surface_arg_map, argc, argv, "surface", &opts);
    print_surface_configs(&opts);

    if(opts.input.size() == 0){
        printf("No input file\n");
        return;
    }

    if(opts.output.size() == 0){
        printf("No output path\n");
        return;
    }

    CudaMemoryManagerStart(__FUNCTION__);
    std::cout << "**********************************" << std::endl;
    std::cout << "Loading simulation file... " << std::flush;
    if(opts.legacy){
        std::vector<vec3f> points;
        SerializerLoadLegacySystem3(&points, opts.input.c_str(),
                                    opts.inflags, nullptr);
        for(vec3f &v : points){
            builder.AddParticle(v);
        }
    }else{
        std::vector<SerializedShape> shapes;
        SerializerLoadSystem3(&builder, &shapes, opts.input.c_str(),
                              opts.inflags, nullptr);
    }

    std::cout << "OK [ Particles: " << builder.positions.size() << " ]" << std::endl;

    Float spacing = opts.spacing;
    Float kernel = opts.kernel;
    Float mcSpacing = opts.marchingCubeSpacing;
    FieldGrid3f *grid = ParticlesToSDF(&builder, spacing, mcSpacing, kernel);

    HostTriangleMesh3 mesh;
    vec3f sp = grid->GetSpacing();

    printf("Running Marching Cubes\n");

    vec3ui res = grid->GetResolution();
    double processed = 0;
    double total = res.x * res.y * res.z;
    auto reporter = [&](vec3ui u) -> void{
        unsigned int i = u.x, j = u.y, k = u.z;
        processed += 1.0;
        if(i > res.x){
            std::cout << "\r" << res.x << " / " << res.y << " / " << res.z;
            std::cout << " ( 100.00% )" << std::flush;
        }else{
            char a[4], b[4], c[4], d[7];
            fill_number(i, a); fill_number(j, b); fill_number(k, c);
            double frac = processed * 100.0 / total;
            fill_pct(frac, d);
            std::cout << "\r" << a << " / " << b << " / " << c;
            std::cout << " ( " << d << "% )" << std::flush;
        }
    };

    MarchingCubes(grid, sp, grid->minPoint, &mesh, 0.0, reporter);
    std::cout << std::endl;

    mesh.writeToDisk(opts.output.c_str());

    printf("Finished, triangle count: %ld\n", mesh.numberOfTriangles());
    std::cout << "**********************************" << std::endl;
    CudaMemoryManagerClearCurrent();
}
