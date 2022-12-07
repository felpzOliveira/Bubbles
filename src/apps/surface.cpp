#include <args.h>
#include <marching_cubes.h>
#include <emitter.h>
#include <grid.h>
#include <util.h>
#include <memory.h>
#include <host_mesh.h>
#include <counting.h>

typedef enum{
    SDF_ZHU_BRIDSON,
    SDF_PARTICLE_COUNT,
    SDF_NONE,
}SurfaceSDFMethod;

typedef struct{
    Float kernel;
    Float spacing;
    Float marchingCubeSpacing;
    std::string input;
    std::string output;
    SurfaceSDFMethod method;
    int inflags;
    int legacy;
    TriangleMeshFormat outformat;
}surface_opts;

SurfaceSDFMethod SDFMethodFromString(std::string method){
    if(method == "zhu")
        return SDF_ZHU_BRIDSON;
    else if(method == "pcount")
        return SDF_PARTICLE_COUNT;
    return SDF_NONE;
}

std::string StringFromSDFMethod(SurfaceSDFMethod method){
    switch(method){
        case SDF_ZHU_BRIDSON: return "zhu";
        case SDF_PARTICLE_COUNT: return "pcount";
        default:{
            return "none";
        }
    }
}

void default_surface_opts(surface_opts *opts){
    opts->output = "surface.out";
    opts->method = SDF_ZHU_BRIDSON;
    opts->legacy = 0;
    opts->inflags = SERIALIZER_POSITION;
    opts->kernel = 0.04;
    opts->spacing = 0.02;
    opts->marchingCubeSpacing = 0.01;
    opts->outformat = FORMAT_OBJ;
}

void print_surface_configs(surface_opts *opts){
    std::cout << "Configs: " << std::endl;
    std::cout << "    * Input : " << opts->input << std::endl;
    std::cout << "    * Output : " << opts->output << std::endl;
    std::cout << "    * Spacing : " << opts->spacing << std::endl;
    std::cout << "    * SDF Method : " << StringFromSDFMethod(opts->method) << std::endl;
    std::cout << "    * Kernel : " << opts->kernel << std::endl;
    std::cout << "    * Mesh Spacing : " << opts->marchingCubeSpacing << std::endl;
    std::cout << "    * Legacy : " << opts->legacy << std::endl;
    std::cout << "    * Output Format: " << FormatString(opts->outformat) << std::endl;
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

ARGUMENT_PROCESS(surface_outform_args){
    surface_opts *opts = (surface_opts *)config;
    std::string fmt = ParseNext(argc, argv, i, "-outformat", 1);
    opts->outformat = FormatFromString(fmt);
    if(opts->outformat == FORMAT_NONE){
        printf("Invalid or unsupported output format\n");
        return -1;
    }
    return 0;
}

ARGUMENT_PROCESS(surface_legacy_args){
    surface_opts *opts = (surface_opts *)config;
    opts->legacy = 1;
    return 0;
}

ARGUMENT_PROCESS(surface_method){
    surface_opts *opts = (surface_opts *)config;
    std::string method = ParseNext(argc, argv, i, "-method", 1);
    opts->method = SDFMethodFromString(method);
    if(opts->method == SDF_NONE){
        printf("Invalid or unsupported SDF extraction method\n");
        return -1;
    }
    return 0;
}

std::map<const char *, arg_desc> surface_arg_map = {
    {"-in",
        {
            .processor = surface_in_args,
            .help = "Sets the input file for mesh computation."
        }
    },
    {"-method",
        {
            .processor = surface_method,
            .help = "Sets the method to use for reconstruction (Options: zhu (default), count)."
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
    {"-outform",
        {
            .processor = surface_outform_args,
            .help = "Sets format of the output (Options: ply, obj (default))."
        }
    },
    {"-cspacing",
        {
            .processor = surface_cubes_spacing_args,
            .help = "Sets the spacing to use for surface reconstruction (invert resolution)."
        }
    }
};

template <typename T>
__bidevice__ T cubic(T x){ return x * x * x; }
__bidevice__ double k_cub(double s) { return Max(0.0, cubic(1.0 - s * s)); }

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
                const Float wi = k_cub((p - xi).Length() / kernelRadius);
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

FieldGrid3f *ParticlesToSDF_Pcount(ParticleSetBuilder3 *pBuilder, Float sdfSpacing){
    CountingGrid3D countSdf;
    Float spacingScale = 2.0;
    countSdf.Build(pBuilder, sdfSpacing * spacingScale);

    std::cout << "Generating SDF... " << std::flush;
    countSdf.Build(countSdf.grid);

    FieldGrid3f *sdfGrid = countSdf.Solve();
    vec3ui res = sdfGrid->GetResolution();
    std::cout << "OK [ Resolution: " << res.x << " x " << res.y << " x " << res.z
              << " ]" << std::endl;

    return sdfGrid;
}

FieldGrid3f *ParticlesToSDF_Zhu(ParticleSetBuilder3 *pBuilder, Float spacing,
                                Float sdfSpacing, Float radius)
{
    Float spacingScale = 1.5;
    Float threshold = 0.25;
    Float kernelRadius = radius;
    Float queryRadius = radius;
    vec3f sp(sdfSpacing);

    vec3f p0 = pBuilder->positions[0];
    Bounds3f bounds(p0, p0);
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

    //std::cout << "Domain Bounds: " << bounds << std::endl;
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

    FieldGrid3f *grid;
    Float spacing = opts.spacing;
    Float kernel = opts.kernel;
    Float mcSpacing = opts.marchingCubeSpacing;
    if(opts.method == SDF_ZHU_BRIDSON)
        grid = ParticlesToSDF_Zhu(&builder, spacing, mcSpacing, kernel);
    else if(opts.method == SDF_PARTICLE_COUNT)
        grid = ParticlesToSDF_Pcount(&builder, mcSpacing);
    else{
        printf("Invalid method selected\n");
        CudaMemoryManagerClearCurrent();
        return;
    }

    HostTriangleMesh3 mesh;

    printf("Running Marching Cubes\n");

    vec3ui res = grid->GetResolution();
    double processed = 0;
    double total = res.x * res.y * res.z;
    auto reporter = [&](vec3ui u) -> void{
        unsigned int i = u.x, j = u.y, k = u.z;
        processed += 1.0;
        if(i > res.x){
            std::cout << "\r" << res.x << " / " << res.y << " / " << res.z;
            std::cout << " ( 100.00% )     " << std::flush;
        }else{
            char a[4], b[4], c[4], d[7];
            fill_number(i, a); fill_number(j, b); fill_number(k, c);
            double frac = processed * 100.0 / total;
            fill_pct(frac, d);
            std::cout << "\r" << a << " / " << b << " / " << c;
            std::cout << " ( " << d << "% )    " << std::flush;
        }
    };

    if(opts.method == SDF_PARTICLE_COUNT)
        MarchingCubes(grid, &mesh, 0.25, reporter, kDirectionAll,
                      kDirectionNone, false);
    else
        MarchingCubes(grid, &mesh, 0.0, reporter);

    std::cout << std::endl;

    mesh.writeToDisk(opts.output.c_str(), opts.outformat);

    printf("Finished, triangle count: %ld\n", mesh.numberOfTriangles());
    std::cout << "**********************************" << std::endl;
    CudaMemoryManagerClearCurrent();
}
