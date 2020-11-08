#include <args.h>
#include <cutil.h>
#include <transform.h>
#include <obj_loader.h>
#include <graphy.h>
#include <shape.h>
#include <util.h>

typedef struct{
    int view;
    Float scale;
    std::string input;
    Transform transform;
    vec3f camEye;
    vec3f camAt;
}sdf_opts;

static sdf_opts g_opts;

void default_opts(sdf_opts *opts){
    opts->input = "input.obj";
    opts->view = 1;
    opts->scale = 1;
    opts->camAt = vec3f(0,-0.2,0);
    opts->camEye = vec3f(3.0, 1.0, 0.0);
}

ARGUMENT_PROCESS(sdf_view_arg){
    sdf_opts *opts = (sdf_opts *)config;
    opts->view = 1;
    return 0;
}

ARGUMENT_PROCESS(sdf_in_arg){
    sdf_opts *opts = (sdf_opts *)config;
    opts->input = ParseNext(argc, argv, i, "-in");
    return 0;
}

ARGUMENT_PROCESS(sdf_scale_arg){
    sdf_opts *opts = (sdf_opts *)config;
    Float scale = ParseNextFloat(argc, argv, i, "-scale");
    opts->scale = scale;
    opts->transform = Scale(scale) * opts->transform;
    return 0;
}

ARGUMENT_PROCESS(sdf_origin_arg){
    sdf_opts *opts = (sdf_opts *)config;
    std::string strdist = ParseNext(argc, argv, i, "-origin", 3);
    const char *ptr = strdist.c_str();
    ParseV3(&opts->camEye, &ptr);
    return 0;
}

ARGUMENT_PROCESS(sdf_target_arg){
    sdf_opts *opts = (sdf_opts *)config;
    std::string strdist = ParseNext(argc, argv, i, "-target", 3);
    const char *ptr = strdist.c_str();
    ParseV3(&opts->camAt, &ptr);
    return 0;
}

std::map<const char *, arg_desc> sdf_arg_map = {
    {"-view",
        {
            .processor = sdf_view_arg,
            .help = "Use Graphy to view the generated SDF."
        }
    },
    {"-in",
        {
            .processor = sdf_in_arg,
            .help = "Where to read input geometry."
        }
    },
    {"-scale",
        {
            .processor = sdf_scale_arg,
            .help = "Scale the input geometry uniformly."
        }
    },
    {"-origin",
        {
            .processor = sdf_origin_arg,
            .help = "When viewing, set the origin of the view point."
        }
    },
    {"-target",
        {
            .processor = sdf_target_arg,
            .help = "When viewing, set the target of the view point."
        }
    }
};


void sdf_print_configs(sdf_opts *opts){
    std::cout << "Configs: " << std::endl;
    std::cout << "    * Target file : " << opts->input << std::endl;
    std::cout << "    * Transforms : " << std::endl;
    std::cout << "        - Scale : " << opts->scale << std::endl;
    std::cout << "    * View : " << opts->view << std::endl;
    if(opts->view){
        std::cout << "        - Origin : " << opts->camEye.x << " " << 
            opts->camEye.y << " " << opts->camEye.z << std::endl;
        std::cout << "        - Target : " << opts->camAt.x << " " << 
            opts->camAt.y << " " << opts->camAt.z << std::endl;
    }
}


Shape *GenerateMeshSDF(sdf_opts *opts){
    ParsedMesh *mesh = LoadObj(opts->input.c_str());
    Shape *meshShape = MakeMesh(mesh, opts->transform);
    GenerateMeshShapeSDF(meshShape);
    return meshShape;
}

void SDFView(Shape *shape, sdf_opts *opts){
    Float spacing = 0.01;
    std::vector<vec3f> particles;
    UtilGetSDFParticles(shape->grid, &particles, 0, spacing);
    
    int count = particles.size();
    float *pos = new float[count * 3];
    float *col = new float[count * 3];
    
    vec3f origin = opts->camEye;
    vec3f target = opts->camAt;
    
    graphy_set_3d(origin.x, origin.y, origin.z, target.x, target.y, target.z,
                  45.0, 0.1f, 100.0f);
    int itp = 0, itc = 0;
    for(vec3f &pi : particles){
        pos[itp++] = pi.x; pos[itp++] = pi.y; pos[itp++] = pi.z;
        col[itc++] = 0; col[itc++] = 0; col[itc++] = 1;
    }
    
    graphy_render_points3f(pos, col, itp/3, spacing/2.0);
    printf("Press anything ... ");
    fflush(stdout);
    getchar();
    graphy_close_display();
    
    delete[] pos;
    delete[] col;
}

void sdf_command(int argc, char **argv){
    default_opts(&g_opts);
    argument_process(sdf_arg_map, argc, argv, &g_opts);
    sdf_print_configs(&g_opts);
    Shape *shape = GenerateMeshSDF(&g_opts);
    
    if(g_opts.view){
        SDFView(shape, &g_opts);
    }
}