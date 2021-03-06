#include <args.h>
#include <serializer.h>
#include <sstream>
#include <fstream>
#include <obj_loader.h> // get parser utilities
#include <transform.h>

// Two dragons scene: -clip-plane 4 3.6 2 -layered -mat diffuse -mat-value 0.549 0.647 0.643
#define MESH_FOLDER "/home/felipe/Documents/CGStuff/models"

#define __to_stringf __to_string<Float>
#define __to_stringi __to_string<int>
#define GetStringColor(n) RGBStringFromHex(Colors[n])

typedef enum{
    LAYERED=0, FILTERED, LEVEL, ALL
}RenderMode;

typedef struct{
    Float cutSource;
    Float cutDistance;
    int cutAxis;
    std::string input;
    std::string output;
    RenderMode mode;
    Float radius;
    Transform transform;
    std::string pickedMat;
    int level;
    int flags;
    int hasClipArgs;
    vec3f mat_value;
    std::map<std::string, std::string> meshMap;
    int warn_ply_mesh;
    std::string meshFolder;
}pbrt_opts;

typedef struct{
    std::string name;
    std::string mat;
    int built;
}pbrt_mat;

const std::string ColorsString[] = {
    //"0.866 0 0.301",
    //"0.670 0.368 0.286",
    //"0.666 0.815 0.588",
    //"0.00 0.89 1.0",
    //"0.019 0.682 1.0",
    "0.78 0.78 0.78",
    "0.749 0.023 0.247",
    "0.737 0.564 0.274",
    "0.760 0.756 0.494",
    "0.505 0.698 0.901",
    "0.270 0.533 0.890",
};

const unsigned int Colors[] = {
    0xffcccccc,
    0xffc60043,
    //0xffc60043,
    0xffe3b256,
    0xffe7dd96,
    0xff81c1f4,
    0xff4a94f4,
};

pbrt_mat Materials[] = {
    {.name = "glass-BK7", .mat = "\"dielectric\" \"spectrum eta\" \"glass-BK7\"", .built=1},
    {.name = "glass-thin", .mat = "\"dielectric\" \"float eta\" [ 1.1 ]", .built=1},
    {.name = "diffuse", .mat = "\"diffuse\" \"rgb reflectance\"", .built=0}
};

std::string RGBStringFromHex(int hex){
    std::stringstream ss;
    unsigned int ur = (hex & 0x00ff0000) >> 16;
    unsigned int ug = (hex & 0x0000ff00) >> 8;
    unsigned int ub = (hex & 0x000000ff);

    Float r = ((Float)(ur)) / 255.0;
    Float g = ((Float)(ug)) / 255.0;
    Float b = ((Float)(ub)) / 255.0;

    ss << r << " " << g << " " << b;
    return ss.str();
}

std::string find_material(std::string name, int *ok){
    int count = sizeof(Materials) / sizeof(Materials[0]);
    *ok = -1;
    for(int i = 0; i < count; i++){
        if(name == Materials[i].name){
            *ok = i;
            return Materials[i].mat;
        }
    }

    return std::string();
}

std::string mesh_base_name(std::string name){
    int at = -1;
    size_t size = name.size();
    if(size < 2) return name;
    for(int i = (int)size - 1; i >= 0; i--){
        if(name[i] == '.'){
            at = i;
            break;
        }
    }

    if(at < 0){
        return name;
    }

    return name.substr(0, at);
}

std::string mesh_name_to_pbrt(std::string name, pbrt_opts *opts){
    std::string ext;
    std::string pbrt_name;
    int at = -1;
    size_t size = name.size();
    if(size < 2) return name;
    for(int i = (int)size - 1; i >= 0; i--){
        if(name[i] == '.'){
            at = i;
            break;
        }
    }

    if(at == -1){
        printf("Warning: No extension found for \'%s\'\n", name.c_str());
        return name;
    }

    ext = name.substr(at);
    pbrt_name = name;
    if(ext != ".ply"){
        opts->warn_ply_mesh = 1;
        pbrt_name = name.substr(0, at);
        pbrt_name += ".ply";
    }

    return pbrt_name;
}

ARGUMENT_PROCESS(pbrt_material_arg){
    int ok = -1;
    pbrt_opts *opts = (pbrt_opts *)config;
    opts->pickedMat = ParseNext(argc, argv, i, "-mat", 1);
    std::string m = find_material(opts->pickedMat, &ok);
    if(ok < 0){
        printf("Failed to find material \'%s\'\n", opts->pickedMat.c_str());
        return -1;
    }

    return 0;
}

ARGUMENT_PROCESS(pbrt_material_value){
    pbrt_opts *opts = (pbrt_opts *)config;
    std::string value = ParseNext(argc, argv, i, "-mat-value", 3);
    const char *token = value.c_str();
    ParseV3(&opts->mat_value, &token);
    return 0;
}

ARGUMENT_PROCESS(pbrt_list_mats){
    int count = sizeof(Materials) / sizeof(Materials[0]);
    printf("Available default materials:\n");
    for(int i = 0; i < count; i++){
        pbrt_mat mat = Materials[i];
        printf("  * %s :  %s\n", mat.name.c_str(), mat.mat.c_str());
    }

    exit(0);
    return 0;
}

ARGUMENT_PROCESS(pbrt_input_arg){
    pbrt_opts *opts = (pbrt_opts *)config;
    opts->input = ParseNext(argc, argv, i, "-in");
    return 0;
}

ARGUMENT_PROCESS(pbrt_output_arg){
    pbrt_opts *opts = (pbrt_opts *)config;
    opts->output = ParseNext(argc, argv, i, "-out");
    return 0;
}

ARGUMENT_PROCESS(pbrt_radius_arg){
    pbrt_opts *opts = (pbrt_opts *)config;
    Float radius = ParseNextFloat(argc, argv, i, "-radius");
    opts->radius = radius;
    return 0;
}

ARGUMENT_PROCESS(pbrt_serializer_inform_arg){
    pbrt_opts *opts = (pbrt_opts *)config;
    std::string format = ParseNext(argc, argv, i, "-inform", 1);
    opts->flags = SerializerFlagsFromString(format.c_str());
    return opts->flags < 0 ? -1 : 0;
}

ARGUMENT_PROCESS(pbrt_layered_arg){
    pbrt_opts *opts = (pbrt_opts *)config;
    opts->mode = RenderMode::LAYERED;
    return 0;
}

ARGUMENT_PROCESS(pbrt_filtered_arg){
    pbrt_opts *opts = (pbrt_opts *)config;
    opts->mode = RenderMode::FILTERED;
    return 0;
}

ARGUMENT_PROCESS(pbrt_level_arg){
    pbrt_opts *opts = (pbrt_opts *)config;
    std::string value = ParseNext(argc, argv, i, "-level");
    const char *token = value.c_str();
    opts->level = (int)ParseFloat(&token);
    opts->mode = RenderMode::LEVEL;
    return 0;
}

ARGUMENT_PROCESS(pbrt_rotate_y_arg){
    pbrt_opts *opts = (pbrt_opts *)config;
    std::string value = ParseNext(argc, argv, i, "-rotateY");
    const char *token = value.c_str();
    Float rotate = ParseFloat(&token);
    opts->transform = RotateY(rotate) * opts->transform;
    return 0;
}

ARGUMENT_PROCESS(pbrt_rotate_z_arg){
    pbrt_opts *opts = (pbrt_opts *)config;
    std::string value = ParseNext(argc, argv, i, "-rotateZ");
    const char *token = value.c_str();
    Float rotate = ParseFloat(&token);
    opts->transform = RotateZ(rotate) * opts->transform;
    return 0;
}

ARGUMENT_PROCESS(pbrt_rotate_x_arg){
    pbrt_opts *opts = (pbrt_opts *)config;
    std::string value = ParseNext(argc, argv, i, "-rotateX");
    const char *token = value.c_str();
    Float rotate = ParseFloat(&token);
    opts->transform = RotateX(rotate) * opts->transform;
    return 0;
}

ARGUMENT_PROCESS(pbrt_translate_arg){
    pbrt_opts *opts = (pbrt_opts *)config;
    vec3f delta;
    std::string value = ParseNext(argc, argv, i, "-translate", 3);
    const char *token = value.c_str();
    ParseV3(&delta, &token);
    opts->transform = Translate(delta) * opts->transform;
    return 0;
}

ARGUMENT_PROCESS(pbrt_clip_arg){
    vec3f data;
    pbrt_opts *opts = (pbrt_opts *)config;
    std::string value = ParseNext(argc, argv, i, "-clip-plane", 3);
    const char *token = value.c_str();
    ParseV3(&data, &token);
    opts->cutSource = data[0];
    opts->cutDistance = data[1];
    opts->cutAxis = data[2];
    opts->hasClipArgs = 1;
    return 0;
}

std::map<const char *, arg_desc> pbrt_argument_map = {
    {"-in",
        { .processor = pbrt_input_arg, 
            .help = "Where to read input file." 
        }
    },
    {"-out", 
        { .processor = pbrt_output_arg, 
            .help = "Where to write output." 
        }
    },
    {"-radius", 
        { .processor = pbrt_radius_arg, 
            .help = "Radius to use for particle cloud. (default: 0.012)" 
        }
    },
    {"-inform", 
        { .processor = pbrt_serializer_inform_arg, 
            .help = "Input data format. (<p><n><m><b><d><o>)" 
        }
    },
    {"-layered", 
        { .processor = pbrt_layered_arg, 
            .help = "Generates geometry containing only LNM-based layered boundary particles." 
        }
    },
    {"-filtered", 
        { .processor = pbrt_filtered_arg, 
            .help = "Generates geometry containing LNM-based layer and interior particles as gray." 
        }
    },
    {"-level", 
        { .processor = pbrt_level_arg, 
            .help = "Generates geometry containing only a specific level of LNM classification." 
        }
    },
    {"-mat",
        { .processor = pbrt_material_arg,
            .help = "Sets a default material to be used instead of prompting for objects."
        }
    },
    {"-list-mats",
        { .processor = pbrt_list_mats,
            .help = "List available default materials and stop execution."
        }
    },
    {"-mat-value",
        { .processor = pbrt_material_value,
            .help = "Sets a 3 dimensional vector as argument for the chosen material."
        }
    },
    {"-rotateY", 
        { .processor = pbrt_rotate_y_arg, 
            .help = "Rotate input in the Y direction. (degrees)" 
        }
    },
    {"-rotateZ", 
        { .processor = pbrt_rotate_z_arg, 
            .help = "Rotate input in the Z direction. (degrees)" 
        }
    },
    {"-rotateX", 
        { .processor = pbrt_rotate_x_arg, 
            .help = "Rotate input in the X direction. (degrees)" 
        }
    },
    {"-translate", 
        { .processor = pbrt_translate_arg, 
            .help = "Translate input set. (degrees)" 
        }
    },
    {"-clip-plane", 
        { .processor = pbrt_clip_arg, 
            .help = "Specify parameters to perform plane clip, <source> <distance> <axis>." 
        }
    },
};

const char *render_mode_string(RenderMode mode){
    switch(mode){
        case RenderMode::LAYERED: return "Layered";
        case RenderMode::LEVEL: return "Level";
        case RenderMode::FILTERED: return "Filtered";
        case RenderMode::ALL: return "All";
        default: return "None";
    }
}

template<typename T> std::string __to_string(T value){
    std::stringstream ss;
    ss << value;
    return ss.str();
}

/*
* L1 and L2 = red;
* any other is gray
*/
static std::string GetMaterialStringFiltered(int layer){
    //if(layer != 1 && layer != 2) return ColorsString[0];
    //return ColorsString[1];
    if(layer != 1 && layer != 2) return GetStringColor(0);
    return GetStringColor(1);
}

/*
* Follow color map
*/
static std::string GetMaterialStringAll(int layer){
    int max_layer = sizeof(Colors) / sizeof(Colors[0]);
    if(layer > max_layer-1){
        layer = max_layer-1;
    }
    //return ColorsString[layer];
    return GetStringColor(layer);
}

/*
* L1 = red; L2 = yellow;
* anything other should not get here but return gray
*/
static std::string GetMaterialStringLayered(int layer){
    int max_layer = sizeof(ColorsString) / sizeof(ColorsString[0]);
    if(layer > max_layer-1){
        layer = max_layer-1;
    }

    if(layer != 1 && layer != 2) return GetStringColor(0);
    return GetStringColor(layer);

    //if(layer != 1 && layer != 2) return ColorsString[0];
    //return ColorsString[layer];
    //return ColorsString[1];
}

std::string GetMaterialString(int layer, pbrt_opts *opts){
    std::string data("Material \"coateddiffuse\"\n\t\"rgb reflectance\" [ ");
    switch(opts->mode){
        case RenderMode::ALL:
        case RenderMode::LEVEL: data += GetMaterialStringAll(layer); break;
        case RenderMode::LAYERED: data += GetMaterialStringLayered(layer); break;
        case RenderMode::FILTERED: data += GetMaterialStringFiltered(layer); break;
        default:{
            printf("Unknown render mode\n");
            exit(0);
        }
    }
    data += " ]\n\t\"float roughness\" [ 0 ]\n";
    return data;
}

void default_pbrt_opts(pbrt_opts *opts){
    opts->output = "geometry.pbrt";
    opts->flags = SERIALIZER_POSITION;
    opts->radius = 0.012;
    opts->mode = RenderMode::ALL;
    opts->transform = Transform();
    opts->level = -1;
    opts->cutSource = 0;
    opts->cutAxis = -1;
    opts->cutDistance = FLT_MAX;
    opts->hasClipArgs = 0;
    opts->warn_ply_mesh = 0;
}

void print_configs(pbrt_opts *opts){
    std::cout << "Configs: " << std::endl;
    std::cout << "    * Target file : " << opts->input << std::endl;
    std::cout << "    * Target output : " << opts->output << std::endl;
    std::cout << "    * Render radius : " << opts->radius << std::endl;
    std::cout << "    * Render mode : " << render_mode_string(opts->mode) << std::endl;
    if(opts->mode == RenderMode::LEVEL){
        std::cout << "    * Level : " << opts->level << std::endl;
    }
    
    if(opts->hasClipArgs){
        std::cout << "    * Clip source : " << opts->cutSource << std::endl;
        std::cout << "    * Clip distance : " << opts->cutDistance << std::endl;
        std::cout << "    * Clip axis : " << opts->cutAxis << std::endl;
    }

    if(opts->pickedMat.size() > 0){
        std::cout << "    * Material : " << opts->pickedMat << std::endl;
    }
}

int SplitByLayer(std::vector<SerializedParticle> **renderflags, 
                 std::vector<SerializedParticle> *pSet, int pCount,
                 pbrt_opts *opts)
{
    int mCount = 0;
    int max_layer = sizeof(ColorsString) / sizeof(ColorsString[0]);
    std::vector<SerializedParticle> *groups;
    
    for(int i = 0; i < pCount; i++){
        SerializedParticle p = pSet->at(i);
        if((p.boundary + 1) > mCount) mCount = p.boundary + 1;
    }
    
    if(mCount > max_layer){
        printf("Warning: Found too many layers (%d > %d), result will be flattened\n",
                mCount, max_layer);
    }else{
        printf("Found %d layers\n", mCount);
    }
    
    groups = new std::vector<SerializedParticle>[mCount];
    for(int i = 0; i < pCount; i++){
        SerializedParticle p = pSet->at(i);
        p.position = opts->transform.Point(p.position);
        groups[p.boundary].push_back(p);
    }
    
    *renderflags = groups;
    return mCount;
}

int AcceptLayer(int layer, pbrt_opts *opts){
    if(opts->mode == RenderMode::ALL || opts->mode == RenderMode::FILTERED) return 1;
    if(opts->mode == RenderMode::LAYERED) return (layer > 0 && layer < 3 ? 1 : 0);
    if(opts->mode == RenderMode::LEVEL) return (layer == opts->level ? 1 : 0);
    printf("[Accept] Unsupported render mode\n");
    exit(0);
}

void pbrt_insert_box(SerializedShape *shape, std::string &data, int depth, Float of=0){
    char t[30];
    vec3f points[8];
    Transform transform;

    std::string tab;
    std::stringstream ss;
    Float nx = -0.5 - of, px = 0.5 + of;
    Float ny = -0.5 - of, py = 0.5 + of;
    Float nz = -0.5 - of, pz = 0.5 + of;

    memset(t, 0x00, sizeof(t));
    memset(t, '\t', depth);

    tab = std::string(t);

    if(shape->numParameters.find("Length") != shape->numParameters.end()){
        vec4f len = shape->numParameters["Length"];
        vec4f hlen = len * 0.5;
        nx = -hlen.x - of; px = hlen.x + of;
        ny = -hlen.y - of; py = hlen.y + of;
        nz = -hlen.z - of; pz = hlen.z + of;
    }

    if(shape->transfParameters.find("Transform") != shape->transfParameters.end()){
        transform = shape->transfParameters["Transform"];
    }

    points[0] = vec3f(nx, ny, pz);
    points[1] = vec3f(px, ny, pz);
    points[2] = vec3f(px, py, pz);
    points[3] = vec3f(nx, py, pz);

    points[4] = vec3f(nx, ny, nz);
    points[5] = vec3f(px, ny, nz);
    points[6] = vec3f(px, py, nz);
    points[7] = vec3f(nx, py, nz);

    data += tab + "AttributeBegin\n";
    data += tab + "\tShape \"trianglemesh\"\n";
    data += tab + "\t\t\"point3 P\" [\n";

    // NOTE: I think PBRT transform computation are a bit different from ours,
    // lets transform the points here and give the geometry without transformations
    for(int i = 0; i < 8; i++){
        vec3f p = transform.Point(points[i]);
        ss << tab << "\t\t\t" << p.x << " " << p.y << " " << p.z << "\n";
    }

    ss << tab << "\t\t]\n";

    data += ss.str();
    ss.str(std::string());

    data += tab + "\t\t\"integer indices\" [\n";
    ss << tab << "\t\t\t" << "0 1 2\n";
    ss << tab << "\t\t\t" << "2 3 0\n";
    ss << tab << "\t\t\t" << "1 5 6\n";
    ss << tab << "\t\t\t" << "6 2 1\n";
    ss << tab << "\t\t\t" << "7 6 5\n";
    ss << tab << "\t\t\t" << "5 4 7\n";
    ss << tab << "\t\t\t" << "4 0 3\n";
    ss << tab << "\t\t\t" << "3 7 4\n";
    ss << tab << "\t\t\t" << "4 5 1\n";
    ss << tab << "\t\t\t" << "1 0 4\n";
    ss << tab << "\t\t\t" << "3 2 6\n";
    ss << tab << "\t\t\t" << "6 7 3\n";
    ss << tab << "\t\t]\n";

    data += ss.str();

    data += tab + "AttributeEnd\n";
}

std::string pbrt_build_material(pbrt_mat *mat, pbrt_opts *opts){
    std::stringstream ss;
    vec3f v = opts->mat_value;
    if(!mat){
        printf("Material is null\n");
        exit(0);
    }

    ss << mat->mat << " [ ";
    ss << v.x << " " << v.y << " " << v.z << " ]";
    return ss.str();
}

void pbrt_insert_dual_layer_box(SerializedShape *shape, pbrt_mat *mat, pbrt_opts *opts,
                                std::string strmat, std::string &data, int depth)
{
    data += "AttributeBegin\n\t";
    if(mat == nullptr){
        data += "Material " + strmat + "\n";
    }else{
        if(mat->built){
            data += "Material " + mat->mat + "\n";
        }else{
            data += "Material " + pbrt_build_material(mat, opts) + " \n";
        }
    }

    // insert the box with a small offset
    pbrt_insert_box(shape, data, depth, 0.01);
    data += "AttributeEnd\n";

    // insert a second box inside, if the material contains dielectric
    // properties it is nice for us to restore it so that thin properties
    // are captured
    data += "AttributeBegin\n\t";
    data += "Material \"dielectric\" \"float eta\" [ 1.0 ] \n";
    pbrt_insert_box(shape, data, depth);
    data += "AttributeEnd\n";

}

void pbrt_insert_mesh(SerializedShape *shape, pbrt_mat *mat, pbrt_opts *opts,
                      std::string strmat, int depth, std::string refPath)
{
    std::string meshName = shape->strParameters["Name"];
    std::string path = refPath + "/" + mesh_name_to_pbrt(meshName, opts);
    std::string bname = mesh_base_name(meshName);
    std::string data;

    if(opts->meshMap.find(meshName) != opts->meshMap.end()){
        data = opts->meshMap[meshName];
    }else{
        if(mat == nullptr){
            data += "Material " + strmat + "\n";
        }else{
            if(mat->built){
                data += "Material " + mat->mat + "\n";
            }else{
                data += "Material " + pbrt_build_material(mat, opts) + "\n";
            }
        }

        data += "AttributeBegin\n";
        data += "\tObjectBegin \"" + bname + "\"\n";
        data += "\t\tShape \"plymesh\" \"string filename\" [ \"";
        data += path + "\" ]\n";
        data += "\tObjectEnd\n";
        data += "AttributeEnd\n";
    }

    data += "AttributeBegin\n";

    // In order to matrch PBRT rendering coordinates we need to transpose
    // our transformations
    if(shape->transfParameters.find("Transform") != shape->transfParameters.end()){
        Transform transform = shape->transfParameters["Transform"];
        Matrix4x4 m = Transpose(transform.m);
        data += "\tTransform [ ";
        for(int i = 0; i < 4; i++){
            for(int j = 0; j < 4; j++){
                data += std::to_string(m.m[i][j]);
                data += " ";
            }
        }

        data += "]\n";
    }

    data += "\tObjectInstance \"" + bname + "\"\n";

    data += "AttributeEnd\n";
    opts->meshMap[meshName] = data;
}

void pbrt_mesh_gather(pbrt_opts *opts, std::string &data){
    for(auto it = opts->meshMap.begin(); it != opts->meshMap.end(); it++){
        data += it->second;
    }
}

void pbrt_command(int argc, char **argv){
    pbrt_opts opts;
    std::vector<SerializedParticle> particles;
    std::vector<SerializedShape> shapes;
    std::vector<int> boundaries;
    ParticleSetBuilder3 builder;
    std::vector<SerializedParticle> *rendergroups = nullptr;
    int count = 0;
    int pAdded = 0;
    std::string radiusString;
    
    default_pbrt_opts(&opts);
    argument_process(pbrt_argument_map, argc, argv, "pbrt", &opts);
    print_configs(&opts);
    
    if(opts.input.size() == 0){
        printf("No input given\n");
        return;
    }

    SerializerLoadSystem3(&builder, &shapes, opts.input.c_str(),
                          opts.flags, &boundaries);

    //TODO: Hacky implementation
    for(int i = 0; i < boundaries.size(); i++){
        SerializedParticle sp;
        sp.position = builder.positions[i];
        sp.boundary = boundaries[i];

        particles.push_back(sp);
    }

    count = particles.size();

    if(opts.mode != RenderMode::ALL && !(opts.flags & SERIALIZER_BOUNDARY)){
        printf("This mode requires LNM-based boundary output from Bubbles\n");
        return;
    }
    
    radiusString = __to_stringf(opts.radius);
    
    if(count > 0){
        int total = count;
        std::string data;
        std::ofstream ofs(opts.output, std::ofstream::out);
        
        if(!ofs.is_open()){
            std::cout << "Failed to open output file: " << opts.output << std::endl;
            return;
        }
        
        count = SplitByLayer(&rendergroups, &particles, total, &opts);

        std::string refPath(MESH_FOLDER);
        int has_mesh = 0;
        // add objects first for easier manual configuration
        for(SerializedShape &sh : shapes){
            std::string mat;
            const char *ptr = SerializerGetShapeName(sh.type);
            pbrt_mat *pmat = nullptr;
            if(opts.pickedMat.size() == 0){
                printf("[Object] Please enter material for type \'%s\': ", ptr);
                std::getline(std::cin, mat);
            }else{
                int ok = -1;
                mat = find_material(opts.pickedMat, &ok);
                if(ok < 0){
                    printf("Failed to find material\n");
                    exit(0);
                }

                pmat = &Materials[ok];
            }

            if(sh.type == ShapeBox){
                pbrt_insert_dual_layer_box(&sh, pmat, &opts, mat, data, 1);
                pAdded += 2;
            }else if(sh.type == ShapeMesh){
                has_mesh = 1;
                if(refPath.size() == 0){
                    printf("[Mesh] Simulation contains mesh objects, please specify reference path: ");
                    std::getline(std::cin, refPath);
                }

                pbrt_insert_mesh(&sh, pmat, &opts, mat, 1, refPath);
                pAdded += 1;
            }
        }

        if(has_mesh){
            pbrt_mesh_gather(&opts, data);
        }

        for(int i = 0; i < count; i++){
            std::vector<SerializedParticle> *layer = &rendergroups[i];
            printf("\rProcessing layer %d / %d ... ", i+1, count);
            fflush(stdout);
            
            if(layer->size() > 0){
                if(AcceptLayer(i, &opts) == 0) continue;
                
                data += std::string("###### Layer ");
                data += __to_stringi(i);
                data += " ("; data += __to_stringi(layer->size());
                data += ")\n";
                data += GetMaterialString(i, &opts);
                
                for(SerializedParticle &p : *layer){
                    if(opts.hasClipArgs){
                        vec3f ref(0);
                        vec3f at(0);
                        ref[opts.cutAxis] = opts.cutSource;
                        at[opts.cutAxis] = p.position[opts.cutAxis];
                        Float dist = Distance(ref, at);
                        if(dist < opts.cutDistance) continue;
                    }
                    data += "AttributeBegin\n\tTranslate ";
                    data += __to_stringf(p.position.x); data += " ";
                    data += __to_stringf(p.position.y); data += " ";
                    data += __to_stringf(p.position.z); data += "\n";
                    data += "\tShape \"sphere\" \"float radius\" [";
                    data += radiusString;
                    data += "]\nAttributeEnd\n";
                    pAdded += 1;
                }
            }
        }

        std::cout << "Done" << std::endl;

        std::cout << "Finished, created " << pAdded << " objects." << std::endl;

        ofs << data;
        ofs.close();
    }else{
        std::cout << "Empty file" << std::endl;
    }
    
    if(rendergroups) delete[] rendergroups;

    if(opts.warn_ply_mesh){
        printf("* Warning: PBRT only supports ply objects but the given simulation contains\n");
        printf("           alternative formats. Make sure to convert to ply before rendering.\n");
    }
}
