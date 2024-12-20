#include <args.h>
#include <marching_cubes.h>
#include <emitter.h>
#include <grid.h>
#include <util.h>
#include <memory.h>
#include <host_mesh.h>
#include <counting.h>
#include <interval.h>
#include <nicolas.h>

typedef enum{
    SDF_ZHU_BRIDSON = 0,
    SDF_SPH,
    SDF_PARTICLE_ASYMETRY,
    SDF_PARTICLE_COUNT,
    SDF_NONE,
}SurfaceMethod;

typedef struct{
    Float kernel;
    Float spacing;
    Float marchingCubeSpacing;
    Float iso;
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
    return SDF_NONE;
}

std::string StringFromSDFMethod(SurfaceMethod method){
    switch(method){
        case SDF_ZHU_BRIDSON: return "zhu";
        case SDF_PARTICLE_COUNT: return "pcount";
        case SDF_SPH: return "sph";
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
    opts->method = SDF_ZHU_BRIDSON;
    opts->legacy = 0;
    opts->inflags = SERIALIZER_POSITION;
    opts->kernel = 0.04;
    opts->spacing = 0.02;
    opts->marchingCubeSpacing = 0.01;
    opts->outformat = FORMAT_OBJ;
    opts->byResolution = false;
    opts->resolution = vec3ui(0);
    opts->iso = Infinity;
}

void print_surface_configs(surface_opts *opts){
    std::cout << "Configs: " << std::endl;
    std::cout << "    * Input : " << opts->input << std::endl;
    std::cout << "    * Output : " << opts->output << std::endl;
    std::cout << "    * Spacing : " << opts->spacing << std::endl;
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
    opts->legacy = 1;
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

ARGUMENT_PROCESS(surface_isolevel_args){
    surface_opts *opts = (surface_opts *)config;
    opts->iso = ParseNextFloat(argc, argv, i, "-iso");
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
    },
    {"-iso",
        {
            .processor = surface_isolevel_args,
            .help = "Sets the value to use as iso-value for surface reconstruction."
        }
    }
};

template <typename T>
bb_cpu_gpu T cubic(T x){ return x * x * x; }
bb_cpu_gpu Float k_cub(Float s) { return Max(0.0, cubic(1.0 - s * s)); }

bb_cpu_gpu Float SphSDF(vec3f p, Grid3 *grid, ParticleSet3 *pSet, Float kernelRadius){
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

bb_cpu_gpu Float ZhuBridsonSDF(vec3f p, Grid3 *grid, ParticleSet3 *pSet,
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

FieldGrid3f *ParticlesToSDF_Pcount(ParticleSetBuilder3 *pBuilder, surface_opts *opts,
                                   TimerList &timer, SurfaceMethod method)
{
    CountingGrid3D countSdf;
    Float spacingScale = 2.0;
    Float spacing = opts->spacing / spacingScale;
    bool asymetry = method == SDF_PARTICLE_ASYMETRY;
    std::cout << "Generating SDF... " << std::flush;
    timer.Start("Domain build");
    if(!opts->byResolution){
        Float sdfSpacing = opts->marchingCubeSpacing;
        countSdf.BuildBySpacing(pBuilder, sdfSpacing * spacingScale, spacing);
    }else{
        countSdf.BuildByResolution(pBuilder, opts->resolution, spacing);
    }
    timer.StopAndNext("Build SDF");

    FieldGrid3f *sdfGrid = countSdf.Solve(asymetry, 1.2);
    timer.Stop();
    vec3ui res = sdfGrid->GetResolution();
    std::cout << "OK [ Resolution: " << res.x << " x " << res.y << " x " << res.z
              << " ]" << std::endl;

    return sdfGrid;
}

FieldGrid3f *ParticlesToSDF_Surface(ParticleSetBuilder3 *pBuilder, surface_opts *opts,
                                    TimerList &timer)
{
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
    timer.Start("Hash Particles");
    UpdateGridDistributionGPU(solver.solverData);

    /* check if density is needed */
    timer.StopAndNext("Density Computation");
    ComputeDensityGPU(solver.solverData);

    grid->UpdateQueryState();
    timer.Stop();

    /* build sdf grid */
    FieldGrid3f *sdfGrid = cudaAllocateVx(FieldGrid3f, 1);
    timer.Start("Build SDF");
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
    timer.Stop();
    res = sdfGrid->GetResolution();
    std::cout << "OK [ Resolution: " << res.x << " x " << res.y << " x " << res.z
              << " ]" << std::endl;
    return sdfGrid;
}

void compute_weizenbock_values(HostTriangleMesh3 *mesh){
    struct WeizenbockBucket{
        uint64_t counter;
        WeizenbockBucket(){ counter = 0; }
    };

    WeizenbockBucket buckets[10];
    const Float four_sqrt3 = 6.928203230275509;

    for(vec3ui &index : mesh->pointIndices){
        vec3f p0 = mesh->points[index.x];
        vec3f p1 = mesh->points[index.y];
        vec3f p2 = mesh->points[index.z];

        double a = Distance(p0, p1);
        double b = Distance(p0, p2);
        double c = Distance(p1, p2);

        double s = (a + b + c) / 2.f;
        double area2 = s * (s - a) * (s - b) * (s - c);
        double term0 = (a * a + b * b + c * c);

        if(area2 < 0){
            if(!IsZero(area2)){
                printf("Negative area [%g] != 0 ( bug? )\n", area2);
            }
            area2 = 0;
        }

        double term1 = four_sqrt3 * sqrt(area2);
        double R = term1 / term0;
    #if 0 // TODO: ????
        int idx = std::floor(Clamp(R, 0, 0.9999) * 10.0);
    #else
        int idx = 0;
        if(R > 0.10001 && R < 0.20001) idx = 1;
        if(R > 0.20001 && R < 0.30001) idx = 2;
        if(R > 0.30001 && R < 0.40001) idx = 3;
        if(R > 0.40001 && R < 0.50001) idx = 4;
        if(R > 0.50001 && R < 0.60001) idx = 5;
        if(R > 0.60001 && R < 0.70001) idx = 6;
        if(R > 0.70001 && R < 0.80001) idx = 7;
        if(R > 0.80001 && R < 0.90001) idx = 8;
        if(R > 0.90001) idx = 9;
    #endif
        if(idx < 0 || idx > 9)
            printf("IDX = %d, R = %g, (%g / %g) ( %g )\n", idx, R, term1, term0, area2);
        buckets[idx].counter += 1;
    }

    FILE *fp = fopen("weizenbock.txt",  "w");

    std::string arr;
    std::cout << "************************" << std::endl;
    std::cout << "Weizenbock buckets:" << std::endl;
    for(int i = 0; i < 10; i++){
        std::cout << " - " << i << ":" << buckets[i].counter << std::endl;
        std::string val = std::to_string(buckets[i].counter);
        if(fp){
            fprintf(fp, "%s\n", val.c_str());
        }

        arr += val;
        if(i < 9)
             arr += ",";
    }
    if(fp)
        fclose(fp);
    std::cout << "values = [" << arr << "]" << std::endl;
    std::cout << "************************" << std::endl;
}

void surface_command(int argc, char **argv){
    //test_particle_pos();
    //return;
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
    if(opts.output.size() == 0){
        if(opts.outformat == FORMAT_OBJ)
            opts.output = "surface.obj";
        else
            opts.output = "surface.ply";
    }
    print_surface_configs(&opts);

    if(opts.input.size() == 0){
        printf("No input file\n");
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
    TimerList timer;
    if(opts.method == SDF_PARTICLE_COUNT)
        grid = ParticlesToSDF_Pcount(&builder, &opts, timer, opts.method);
    else if(opts.method < SDF_PARTICLE_COUNT)
        grid = ParticlesToSDF_Surface(&builder, &opts, timer);
    else{
        printf("Invalid method selected\n");
        CudaMemoryManagerClearCurrent();
        return;
    }

    if(requires_mc){
        Float iso = opts.iso;
        if(iso == Infinity){
            if(opts.method == SDF_PARTICLE_COUNT)
                iso = 0.25;
            else
                iso = 0.0;
        }

        printf("Running Marching Cubes [iso-value= %g]\n", iso);
        /*
        * NOTE: I'm not sure why, but the triangles generated from the particle
        * sdf with Pcount method  need to be rotated otherwise normals get flipped.
        * I don't quite get why, so maybe add a TODO here to better investigate this method
        * but for now it seems that it literally is as simple as rotating the triangles.
        */
        timer.Start("Marching Cubes");
        if(opts.method == SDF_PARTICLE_COUNT)
            MarchingCubes(grid, &mesh, iso, true);
        else
            MarchingCubes(grid, &mesh, iso, false);
        timer.Stop();
    }

    timer.Start("Mesh output");
    mesh.writeToDisk(opts.output.c_str(), opts.outformat);
    timer.Stop();

    printf("Finished, triangle count: %ld\n", mesh.numberOfTriangles());
    timer.PrintEvents();
    std::cout << "**********************************" << std::endl;
    compute_weizenbock_values(&mesh);
    CudaMemoryManagerClearCurrent();
}
