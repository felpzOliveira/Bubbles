#include <args.h>
#include <graphy.h>
#include <obj_loader.h>
#include <serializer.h>
#include <util.h>

typedef struct{
    std::string basename;
    std::string meshObj;
    int start;
    int end;
    int loop;
    Float radius;
    int origin_configured;
    int target_configured;
    Transform transform;
    vec3f origin;
    vec3f target;
}view_opts;

void default_view_opts(view_opts *opts){
    opts->start = 0;
    opts->end = 1;
    opts->loop = 1;
    opts->origin_configured = 0;
    opts->target_configured = 0;
    opts->radius = 0.012;
}

void print_view_configs(view_opts *opts){
    std::cout << "Configs: " << std::endl;
    std::cout << "    * Basename : " << opts->basename << std::endl;
    std::cout << "    * Radius : " << opts->radius << std::endl;
    std::cout << "    * Loop : " << opts->loop << std::endl;
    std::cout << "    * Start : " << opts->start << std::endl;
    std::cout << "    * End : " << opts->end << std::endl;
    std::cout << "    * Origin : " << opts->origin.x << " " <<
        opts->origin.y << " " << opts->origin.z << std::endl;
    std::cout << "    * Target : " << opts->target.x << " " <<
        opts->target.y << " " << opts->target.z << std::endl;
    if(opts->meshObj.size() > 0){
        std::cout << "    * With SDF : 1" << std::endl;
    }
}

ARGUMENT_PROCESS(view_basename_arg){
    view_opts *opts = (view_opts *)config;
    opts->basename = ParseNext(argc, argv, i, "--basename");
    return 0;
}

ARGUMENT_PROCESS(view_start_arg){
    view_opts *opts = (view_opts *)config;
    opts->start = (int)ParseNextFloat(argc, argv, i, "--start");
    return 0;
}

ARGUMENT_PROCESS(view_end_arg){
    view_opts *opts = (view_opts *)config;
    opts->end = (int)ParseNextFloat(argc, argv, i, "--end");
    return 0;
}

ARGUMENT_PROCESS(view_noloop_arg){
    view_opts *opts = (view_opts *)config;
    opts->loop = 0;
    return 0;
}

ARGUMENT_PROCESS(view_radius_arg){
    view_opts *opts = (view_opts *)config;
    opts->radius = ParseNextFloat(argc, argv, i, "--radius");
    return 0;
}

ARGUMENT_PROCESS(view_origin_arg){
    view_opts *opts = (view_opts *)config;
    std::string strdist = ParseNext(argc, argv, i, "--origin", 3);
    const char *ptr = strdist.c_str();
    ParseV3(&opts->origin, &ptr);
    opts->origin_configured = 1;
    return 0;
}

ARGUMENT_PROCESS(view_target_arg){
    view_opts *opts = (view_opts *)config;
    std::string strdist = ParseNext(argc, argv, i, "--target", 3);
    const char *ptr = strdist.c_str();
    ParseV3(&opts->target, &ptr);
    opts->target_configured = 1;
    return 0;
}

ARGUMENT_PROCESS(view_with_sdf_arg){
    view_opts *opts = (view_opts *)config;
    opts->meshObj = ParseNext(argc, argv, i, "--with-sdf");
    return 0;
}

ARGUMENT_PROCESS(view_sdf_translate_arg){
    view_opts *opts = (view_opts *)config;
    vec3f delta;
    std::string strdist = ParseNext(argc, argv, i, "--sdf-translate", 3);
    const char *ptr = strdist.c_str();
    ParseV3(&delta, &ptr);
    opts->transform = Translate(delta) * opts->transform;
    return 0;
}

ARGUMENT_PROCESS(view_sdf_scale_arg){
    view_opts *opts = (view_opts *)config;
    Float scale = ParseNextFloat(argc, argv, i, "--sdf-scale");
    opts->transform = Scale(scale) * opts->transform;
    return 0;
}


std::map<const char *, arg_desc> view_arg_map = {
    {"--basename",
        {
            .processor = view_basename_arg,
            .help = "Configures the basename of the simulation to load."
        }
    },
    {"--radius",
        {
            .processor = view_radius_arg,
            .help = "Configures the radius value to use. (default: 0.012)"
        }
    },
    {"--noloop",
        {
            .processor = view_noloop_arg,
            .help = "Configures to not loop simulation."
        }
    },
    {"--origin",
        {
            .processor = view_origin_arg,
            .help = "Sets the origin of the view point."
        }
    },
    {"--target",
        {
            .processor = view_target_arg,
            .help = "Sets the target of the view point."
        }
    },
    {"--start",
        {
            .processor = view_start_arg,
            .help = "Sets the initial index to load the simulation."
        }
    },
    {"--end",
        {
            .processor = view_end_arg,
            .help = "Sets the final index to load the simulation."
        }
    },
    {"--with-sdf",
        {
            .processor = view_with_sdf_arg,
            .help = "Adds a obj file as SDF for visualization."
        }
    },
    {"--sdf-translate",
        {
            .processor = view_sdf_translate_arg,
            .help = "Translates the SDF. (requires: --with-sdf)"
        }
    },
    {"--sdf-scale",
        {
            .processor = view_sdf_scale_arg,
            .help = "Scales the SDF. (requires: --with-sdf)"
        }
    }
};

int set_position(float *pos, std::vector<vec3f> *vpos){
    int it = 0;
    for(int i = 0; i < vpos->size(); i++){
        vec3f p = vpos->at(i);
        pos[3 * it + 0] = p.x;
        pos[3 * it + 1] = p.y;
        pos[3 * it + 2] = p.z;
        it ++;
    }
    
    return it;
}

void ViewDisplaySimulation(view_opts *opts){
    std::vector<vec3f> **frames = nullptr;
    Shape *sdfShape = nullptr;
    std::vector<vec3f> sdfParticles;
    int partCount = SerializerLoadMany3(&frames, opts->basename.c_str(),
                                        SERIALIZER_POSITION, opts->start, opts->end);
    vec3f origin = opts->origin;
    vec3f target = opts->target;
    
    if(opts->meshObj.size() > 0){ // with-sdf
        sdfShape = MakeMesh(opts->meshObj.c_str(), opts->transform);
        GenerateMeshShapeSDF(sdfShape, opts->radius/2.0);
        UtilGetSDFParticles(sdfShape->grid, &sdfParticles, 0, opts->radius/2.0);
    }
    
    int bufferSize = partCount + sdfParticles.size();
    float *pos = new float[3 * bufferSize];
    float *col = new float[3 * bufferSize];
    
    int chosenId = (opts->start + opts->end)/2;
    
    memset(col, 0x0, 3 * bufferSize * sizeof(float));
    std::vector<vec3f> *pp = frames[chosenId];
    
    vec3f p0 = pp->at(0);
    Bounds3f bounds(p0, p0);
    col[0] = 1;
    for(int i = 1; i < partCount; i++){
        vec3f pi = pp->at(i);
        bounds = Union(bounds, pi);
        col[3 * i + 0] = 1;
    }
    
    for(int i = partCount; i < bufferSize; i++){
        vec3f pi = sdfParticles[i-partCount];
        pos[3 * i + 0] = pi.x; pos[3 * i + 1] = pi.y;
        pos[3 * i + 2] = pi.z; col[3 * i + 2] = 1;
    }
    
    if(opts->target_configured != 1){
        target = bounds.Center();
    }
    
    if(opts->origin_configured != 1){
        int axis = bounds.MaximumExtent();
        Float len = bounds.ExtentOn(axis);
        origin = target + 0.75 * vec3f(len);
    }
    
    graphy_set_3d(origin.x, origin.y, origin.z, target.x, target.y, target.z,
                  45.0, 0.1f, 100.0f);
    
    int count = opts->end - opts->start;
    do{
        for(int i = 0; i < count; i++){
            int v = set_position(pos, frames[i]);
            graphy_render_points3f(pos, col, bufferSize, opts->radius);
        }
    }while(opts->loop);
    
    printf("Press anything ... ");
    fflush(stdout);
    getchar();
    
    graphy_close_display();
    
    delete[] frames;
    delete[] pos;
    delete[] col;
}

void view_command(int argc, char **argv){
    view_opts opts;
    default_view_opts(&opts);
    argument_process(view_arg_map, argc, argv, &opts);
    print_view_configs(&opts);
    
    if(opts.basename.size() == 0){
        printf("Missing input basename\n");
        return;
    }
    
    if(opts.end <= opts.start){
        printf("Invalid initial/final index\n");
        return;
    }
    
    if(IsZero(opts.radius)){
        printf("Zero radius\n");
        return;
    }
    
    ViewDisplaySimulation(&opts);
}