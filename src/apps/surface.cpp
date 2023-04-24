#include <args.h>
#include <marching_cubes.h>
#include <emitter.h>
#include <grid.h>
#include <util.h>
#include <memory.h>
#include <host_mesh.h>
#include <counting.h>
#include <delaunay.h>

typedef enum{
    SDF_ZHU_BRIDSON = 0,
    SDF_SPH,
    SDF_PARTICLE_COUNT,
    SDF_DELAUNAY,
    SDF_NONE,
}SurfaceMethod;

typedef struct{
    Float kernel;
    Float spacing;
    Float marchingCubeSpacing;
    Float delaunayMu;
    std::string input;
    std::string output;
    SurfaceMethod method;
    int inflags;
    int legacy;
    bool byResolution;
    vec3ui resolution;
    TriangleMeshFormat outformat;
}surface_opts;

SurfaceMethod SDFMethodFromString(std::string method){
    if(method == "zhu")
        return SDF_ZHU_BRIDSON;
    else if(method == "sph")
        return SDF_SPH;
    else if(method == "pcount")
        return SDF_PARTICLE_COUNT;
    else if(method == "delaunay")
        return SDF_DELAUNAY;
    return SDF_NONE;
}

std::string StringFromSDFMethod(SurfaceMethod method){
    switch(method){
        case SDF_ZHU_BRIDSON: return "zhu";
        case SDF_PARTICLE_COUNT: return "pcount";
        case SDF_SPH: return "sph";
        case SDF_DELAUNAY: return "delaunay";
        default:{
            return "none";
        }
    }
}

void GetMethodNames(std::vector<std::string> &names){
    int sdf_0 = SDF_ZHU_BRIDSON;
    int sdf_n = SDF_NONE;
    for(int s = sdf_0; s < sdf_n; s++){
        SurfaceMethod method = (SurfaceMethod)s;
        names.push_back(StringFromSDFMethod(method));
    }
}

void default_surface_opts(surface_opts *opts){
    opts->output = "surface.out";
    opts->method = SDF_ZHU_BRIDSON;
    opts->legacy = 0;
    opts->inflags = SERIALIZER_POSITION;
    opts->kernel = 0.04;
    opts->spacing = 0.02;
    opts->delaunayMu = 1.1;
    opts->marchingCubeSpacing = 0.01;
    opts->outformat = FORMAT_OBJ;
    opts->byResolution = false;
    opts->resolution = vec3ui(0);
}

void print_surface_configs(surface_opts *opts){
    std::cout << "Configs: " << std::endl;
    std::cout << "    * Input : " << opts->input << std::endl;
    std::cout << "    * Output : " << opts->output << std::endl;
    std::cout << "    * Spacing : " << opts->spacing << std::endl;
    std::cout << "    * Delaunay μ : " << opts->delaunayMu << std::endl;
    std::cout << "    * Reconstruction method : " <<
                        StringFromSDFMethod(opts->method) << std::endl;
    std::cout << "    * Kernel : " << opts->kernel << std::endl;
    std::cout << "    * Mesh Spacing : " << opts->marchingCubeSpacing << std::endl;
    std::cout << "    * Legacy : " << opts->legacy << std::endl;
    std::cout << "    * Output Format: " << FormatString(opts->outformat) << std::endl;
    if(opts->byResolution){
        std::cout << "    * Forcing Resolution : " << opts->resolution.x
                   << " x " << opts->resolution.y << " x " << opts->resolution.z << std::endl;
    }
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

ARGUMENT_PROCESS(surface_delaunay_mu_args){
    surface_opts *opts = (surface_opts *)config;
    opts->delaunayMu = ParseNextFloat(argc, argv, i, "-delaunay-mu");
    if(opts->delaunayMu < 0.001){
        printf("Invalid μ\n");
        return -1;
    }

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
    opts->marchingCubeSpacing = ParseNextFloat(argc, argv, i, "-cspacing");
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

ARGUMENT_PROCESS(surface_resolution_args){
    surface_opts *opts = (surface_opts *)config;
    unsigned int nx = ParseNextFloat(argc, argv, i, "-res");
    unsigned int ny = ParseNextFloat(argc, argv, i, "-res");
    unsigned int nz = ParseNextFloat(argc, argv, i, "-res");
    if(nx * ny * nz == 0){
        printf("Invalid resolution given\n");
        return -1;
    }

    if(nx < 2 || ny < 2 || nz < 2){
        printf("Resolution must be > 1\n");
        return -1;
    }

    opts->byResolution = true;
    opts->resolution = vec3ui(nx, ny, nz);
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
            .help = "Sets the method to be executed."
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
    {"-delaunay-mu",
        {
            .processor = surface_delaunay_mu_args,
            .help = "Sets the value of the delaunay μ term. (default: 1.1)"
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
    },
    {"-res",
        {
            .processor = surface_resolution_args,
            .help = "Sets the resolution to be used (overwrites spacing properties)."
        }
    }
};

template <typename T>
__bidevice__ T cubic(T x){ return x * x * x; }
__bidevice__ double k_cub(double s) { return Max(0.0, cubic(1.0 - s * s)); }

__bidevice__ Float SphSDF(vec3f p, Grid3 *grid, ParticleSet3 *pSet, Float kernelRadius){
    const Float cutOff = 0.5; // ??
    int *neighbors = nullptr;
    Float inf = grid->GetBounds().Diagonal().Length();
    if(!grid->Contains(p)){
        return inf;
    }

    unsigned int cellId = grid->GetLinearHashedPosition(p);

    int count = grid->GetNeighborsOf(cellId, &neighbors);
    Float wSum = 0.0;
    SphStdKernel3 kernel(kernelRadius);
    Float mass = pSet->GetMass();

    for(int i = 0; i < count; i++){
        Cell3 *cell = grid->GetCell(neighbors[i]);
        ParticleChain *pChain = cell->GetChain();
        int size = cell->GetChainLength();
        for(int j = 0; j < size; j++){
            vec3f xi = pSet->GetParticlePosition(pChain->pId);
            Float di = pSet->GetParticleDensity(pChain->pId);
            Float dist = Distance(p, xi);

            wSum += (mass / di) * kernel.W(dist);

            pChain = pChain->next;
        }
    }

    return cutOff - wSum;
}

__bidevice__ Float ZhuBridsonSDF(vec3f p, Grid3 *grid, ParticleSet3 *pSet,
                                 Float kernelRadius, Float threshold)
{
    int *neighbors = nullptr;
    Float inf = grid->GetBounds().Diagonal().Length();
    if(!grid->Contains(p)){
        return inf;
    }

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
        // error dividing by ε < 1e-8. As long as it is not a NaN
        // we can continue as everything will simply go to Infinity
        // and the point 'p' is outside the fluid, however having
        // Infinity here can become an issue as the marching cubes algorithm
        // find an error, it would be best to swap this value for inf (?)
        Float inv = 1.0 / wSum;
        xAvg = xAvg * inv;
        return (p - xAvg).Length() - kernelRadius * threshold;
    }else{
        return inf;
    }
}

void ParticlesToDelaunay_Surface(ParticleSetBuilder3 *pBuilder, surface_opts *opts,
                                 HostTriangleMesh3 *mesh)
{
    DelaunayTriangulation triangulation;
    // TODO: We need the solver simply to distribute particles and update the buckets
    //       it would be nice if we could do that without having to build a solver
    SphSolver3 solver;
    solver.Initialize(DefaultSphSolverData3());

    // TODO: Even this spacingScale is irrelevant it would be best if it were
    //       to represent the original simulation domain, i.e.: add flags to cmd
    Float domainSpacing = opts->spacing * 0.5f;
    std::cout << "Building domain... " << std::flush;
    Grid3 *grid = UtilBuildGridForBuilder(pBuilder, domainSpacing, 2.0);
    SphParticleSet3 *sphpSet = SphParticleSet3FromBuilder(pBuilder);

    solver.Setup(WaterDensity, domainSpacing, 2.0, grid, sphpSet);

    std::cout << "Distributing particles..." << std::flush;
    UpdateGridDistributionGPU(solver.solverData);
    grid->UpdateQueryState();

    std::cout << "Done\nComputing delaunay surface... " << std::endl;
    DelaunaySurface(triangulation, sphpSet, opts->spacing, opts->delaunayMu, grid);

    std::cout << "Writing delaunay boundary... " << std::flush;
    DelaunayWriteBoundary(triangulation, sphpSet, "bound.txt");

    std::cout << "Done\nFetching geometry... " << std::endl;
    DelaunayGetTriangleMesh(triangulation, mesh);
}

FieldGrid3f *ParticlesToSDF_Pcount(ParticleSetBuilder3 *pBuilder, surface_opts *opts){
    CountingGrid3D countSdf;
    if(!opts->byResolution){
        Float spacingScale = 2.0;
        Float sdfSpacing = opts->marchingCubeSpacing;
        countSdf.BuildBySpacing(pBuilder, sdfSpacing * spacingScale);
    }else{
        countSdf.BuildByResolution(pBuilder, opts->resolution);
    }

    std::cout << "Generating SDF... " << std::flush;

    FieldGrid3f *sdfGrid = countSdf.Solve();
    vec3ui res = sdfGrid->GetResolution();
    std::cout << "OK [ Resolution: " << res.x << " x " << res.y << " x " << res.z
              << " ]" << std::endl;

    return sdfGrid;
}

FieldGrid3f *ParticlesToSDF_Surface(ParticleSetBuilder3 *pBuilder, surface_opts *opts){
    /* Build a domain for the particles */
    Float spacingScale = 2.0;
    Float threshold = 0.25;
    Float spacing = opts->spacing;
    Float radius = opts->kernel;
    Float sdfSpacing = opts->marchingCubeSpacing;
    SurfaceMethod method = opts->method;

    Grid3 *grid = UtilBuildGridForBuilder(pBuilder, spacing, spacingScale);
    Bounds3f dBounds = grid->GetBounds();
    vec3ui res = grid->GetIndexCount();
    vec3f p0 = dBounds.pMin;
    vec3f p1 = dBounds.pMax;
    printf("Built domain with extension:\n"
           "    [%u x %u x %u]\n"
           "    [%g %g %g]  x  [%g %g %g]\n",
           res.x, res.y, res.z, p0.x, p0.y, p0.z, p1.x, p1.y, p1.z);

    SphSolver3 solver;
    SphParticleSet3 *sphpSet = SphParticleSet3FromBuilder(pBuilder);
    ParticleSet3 *pSet = sphpSet->GetParticleSet();

    solver.Initialize(DefaultSphSolverData3());
    solver.Setup(WaterDensity, spacing, spacingScale, grid, sphpSet);

    /* distribute particles */
    UpdateGridDistributionGPU(solver.solverData);

    /* check if density is needed */
    if(method == SDF_SPH)
    ComputeDensityGPU(solver.solverData);

    grid->UpdateQueryState();

    /* build sdf grid */
    FieldGrid3f *sdfGrid = cudaAllocateVx(FieldGrid3f, 1);
    if(!opts->byResolution){
        sdfSpacing *= spacingScale;
        p0 = dBounds.pMin + 0.5 * vec3f(sdfSpacing);

        int nx = std::ceil(dBounds.ExtentOn(0) / sdfSpacing);
        int ny = std::ceil(dBounds.ExtentOn(1) / sdfSpacing);
        int nz = std::ceil(dBounds.ExtentOn(2) / sdfSpacing);
        sdfGrid->Build(vec3ui(nx, ny, nz), sdfSpacing, p0, VertexCentered);
    }else{
        // since we are doing vertex centered we need a minus 1 here
        opts->resolution = opts->resolution - vec3ui(1);
        Float dx = dBounds.ExtentOn(0) / (Float)opts->resolution.x;
        Float dy = dBounds.ExtentOn(1) / (Float)opts->resolution.y;
        Float dz = dBounds.ExtentOn(2) / (Float)opts->resolution.z;
        Float ds = Max(dx, Max(dy, dz));
        vec3f p0 = dBounds.pMin + 0.5 * vec3f(ds);
        sdfGrid->Build(opts->resolution, ds, p0, VertexCentered);
    }

    Float inf = grid->GetBounds().Diagonal().Length();
    std::cout << "Generating SDF... " << std::flush;
    /* compute sdf */
    AutoParallelFor("SDF", sdfGrid->total, AutoLambda(int i){
        vec3ui u = DimensionalIndex(i, sdfGrid->resolution, 3);
        vec3f p = sdfGrid->GetDataPosition(u);
        Float estimate = inf;
        if(method == SDF_ZHU_BRIDSON)
            estimate = ZhuBridsonSDF(p, grid, pSet, radius, threshold);
        else if(method == SDF_SPH)
            estimate = SphSDF(p, grid, pSet, radius);
        sdfGrid->SetValueAt(estimate, u);
    });

    sdfGrid->MarkFilled();
    res = sdfGrid->GetResolution();
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
    HostTriangleMesh3 mesh;
    bool requires_mc = true;

    default_surface_opts(&opts);
    arg_desc desc = surface_arg_map["-method"];
    std::vector<std::string> methods;
    GetMethodNames(methods);

    std::string value("Sets the method to be executed ( Choices: ");
    for(int i = 0; i < methods.size(); i++){
        value += methods[i];
        if(i < methods.size()-1)
            value += ",";
        value += " ";
    }
    value += ").";
    desc.help = value;
    surface_arg_map["-method"] = desc;

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
    if(opts.method == SDF_PARTICLE_COUNT)
        grid = ParticlesToSDF_Pcount(&builder, &opts);
    else if(opts.method < SDF_PARTICLE_COUNT)
        grid = ParticlesToSDF_Surface(&builder, &opts);
    else if(opts.method == SDF_DELAUNAY){
        ParticlesToDelaunay_Surface(&builder, &opts, &mesh);
        requires_mc = false;
    }else{
        printf("Invalid method selected\n");
        CudaMemoryManagerClearCurrent();
        return;
    }

    if(requires_mc){
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

        /*
        * NOTE: I'm not sure why, but the triangles generated from the particle
        * sdf with Pcount method  need to be rotated otherwise normals get flipped.
        * I don't quite get why, so maybe add a TODO here to better investigate this method
        * but for now it seems that it literally is as simple as rotating the triangles.
        */
        if(opts.method == SDF_PARTICLE_COUNT)
            MarchingCubes(grid, &mesh, 0.25, reporter, true);
        else
            MarchingCubes(grid, &mesh, 0.0, reporter, false);

        std::cout << std::endl;
    }

    mesh.writeToDisk(opts.output.c_str(), opts.outformat);

    printf("Finished, triangle count: %ld\n", mesh.numberOfTriangles());
    std::cout << "**********************************" << std::endl;
    CudaMemoryManagerClearCurrent();
}
