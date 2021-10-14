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

typedef enum{
    RENDERER_PBRT, RENDERER_LIT,
}Renderer;

typedef struct{
    Float cutSource;
    Float cutDistance;
    vec3f cutPointSource;
    vec3f cutPointAt;
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
    Renderer renderer;
    int is_legacy;
    std::string domain;
}pbrt_opts;

typedef struct{
    std::string name;
    std::string mat;
    int built;
    std::string warn_msg;
}pbrt_mat;

typedef pbrt_mat lit_mat;

const unsigned int Colors[] = {
    //0xffcccccc,
    //0xff00ff00,
    0xff00b218,
    0xffb20018,
    //0xffc60043,
    0xffe3b256,
    0xffe7dd96,
    0xff81c1f4,
    0xff4a94f4,
};

pbrt_mat Materials[] = {
    {.name = "glass-BK7", .mat = "\"dielectric\" \"spectrum eta\" \"glass-BK7\"", .built=1},
    {.name = "glass-thin", .mat = "\"dielectric\" \"float eta\" [ 1.1 ]", .built=1},
    {.name = "diffuse", .mat = "\"diffuse\" \"rgb reflectance\"", .built=0},
    {.name = "coated", .mat = "\"coateddiffuse\" \"float roughness\" [0] \"rgb reflectance\"", .built=0}
};

lit_mat MaterialsLit[] = {
    {.name = "glass", .mat = "reflectance [1.0] name[glass] eta[1.3] type [dielectric]", .built=1},
    {.name = "glass-thin", .mat = "reflectance [1.0] name[glass-thin] eta[1.05] type [dielectric]", .built=1},
    {.name = "diffuse", .mat = "type [diffuse] reflectance ", .built=0},
    {.name = "coated", .mat = "type [disney] rough[0.001 0.001] specular[0.5]"
          " clearcoat[1.0] clearcoatGloss [0.93] reflectance ", .built=0,
  .warn_msg = "Warning: Lit does not support PBRT coated material, using disney format instead"},
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
            if(Materials[i].warn_msg.size() > 0){
                std::cout << Materials[i].warn_msg << std::endl;
            }
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
    if(opts->renderer == RENDERER_PBRT){
        if(ext != ".ply"){
            opts->warn_ply_mesh = 1;
            pbrt_name = name.substr(0, at);
            pbrt_name += ".ply";
        }
    }else{
        if(ext != ".obj"){
            opts->warn_ply_mesh = 1;
            pbrt_name = name.substr(0, at);
            pbrt_name += ".obj";
        }
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

ARGUMENT_PROCESS(pbrt_renderer){
    pbrt_opts *opts = (pbrt_opts *)config;
    std::string value = ParseNext(argc, argv, i, "-renderer");
    if(value == "pbrt"){
        opts->renderer = RENDERER_PBRT;
    }else if(value == "lit"){
        opts->renderer = RENDERER_LIT;
    }else{
        return -1;
    }

    return 0;
}

ARGUMENT_PROCESS(pbrt_legacy){
    pbrt_opts *opts = (pbrt_opts *)config;
    opts->is_legacy = 1;
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

ARGUMENT_PROCESS(pbrt_clip_point_arg){
    pbrt_opts *opts = (pbrt_opts *)config;
    std::string value = ParseNext(argc, argv, i, "-clip-point", 7);
    const char *token = value.c_str();
    ParseV3(&opts->cutPointSource, &token);
    ParseV3(&opts->cutPointAt, &token);
    opts->cutDistance = ParseFloat(&token);
    opts->hasClipArgs = 2;
    return 0;
}

ARGUMENT_PROCESS(pbrt_with_domain_arg){
    pbrt_opts *opts = (pbrt_opts *)config;
    std::string value = ParseNext(argc, argv, i, "with-domain", 1);
    if(value.size() < 1) return -1;
    opts->domain = value;
    return 0;
}

std::map<const char *, arg_desc> pbrt_argument_map = {
    {"-in",
        { .processor = pbrt_input_arg, 
            .help = "Where to read input file." 
        }
    },
    {"-legacy",
        { .processor = pbrt_legacy,
            .help = "Inform the loader to use legacy format instead."
        }
    },
    {"-renderer",
        { .processor = pbrt_renderer,
            .help = "Inform target renderer for output generation (default: pbrt)"
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
            .help = "Translate input set."
        }
    },
    {"-clip-plane",
        { .processor = pbrt_clip_arg,
            .help = "Specify parameters to perform plane clip, <source> <distance> <axis>."
        }
    },
    {"-clip-point",
        { .processor = pbrt_clip_point_arg,
            .help = "Specify parameters to perform point clip, <source> <at> <distance>."
        }
    },
    {"-with-domain",
        { .processor = pbrt_with_domain_arg,
            .help = "Writes a grid in the renderer format."
        }
    }
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
    return GetStringColor(layer);
}

/*
* L1 = red; L2 = yellow;
* anything other should not get here but return gray
*/
static std::string GetMaterialStringLayered(int layer){
    int max_layer = sizeof(Colors) / sizeof(Colors[0]);
    if(layer > max_layer-1){
        layer = max_layer-1;
    }

    if(layer != 1 && layer != 2) return GetStringColor(0);
    return GetStringColor(layer);
}

std::string GetMaterialString(int layer, pbrt_opts *opts){
    std::string data("Material \"coateddiffuse\"\n\t\"rgb reflectance\" [ ");
    if(opts->renderer == RENDERER_LIT){
        data = "Material{ type [diffuse] name[layer_";
        data += __to_stringi(layer);
        data += "] reflectance[";
    }
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
    if(opts->renderer == RENDERER_PBRT){
        data += " ]\n\t\"float roughness\" [ 0 ]\n";
    }else{
        data += "] }\n";
    }
    return data;
}

void default_pbrt_opts(pbrt_opts *opts){
    opts->output = "";
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
    opts->renderer = RENDERER_PBRT;
    opts->is_legacy = 0;
}

void print_configs(pbrt_opts *opts){
    std::cout << "Configs: " << std::endl;
    std::cout << "    * Target file : " << opts->input << std::endl;
    std::cout << "    * Render radius : " << opts->radius << std::endl;
    std::cout << "    * Render mode : " << render_mode_string(opts->mode) << std::endl;
    if(opts->domain.size() > 0){
        std::cout << "    * Domain : " << opts->domain << std::endl;
    }

    if(opts->mode == RenderMode::LEVEL){
        std::cout << "    * Level : " << opts->level << std::endl;
    }
    
    if(opts->hasClipArgs == 1){
        std::cout << "    * Clip source : " << opts->cutSource << std::endl;
        std::cout << "    * Clip distance : " << opts->cutDistance << std::endl;
        std::cout << "    * Clip axis : " << opts->cutAxis << std::endl;
    }

    if(opts->hasClipArgs == 2){
        std::cout << "    * Clip source P : [" << opts->cutPointSource.x <<
        " " << opts->cutPointSource.y << " " << opts->cutPointSource.z << "]"
        << std::endl;
        std::cout << "    * Clip at P: [" << opts->cutPointAt.x <<
           " " << opts->cutPointAt.y << " " << opts->cutPointAt.z << "]" << std::endl;
        std::cout << "    * Clip distance : " << opts->cutDistance << std::endl;
    }

    if(opts->pickedMat.size() > 0){
        std::cout << "    * Material : " << opts->pickedMat << std::endl;
    }

    if(opts->renderer == RENDERER_PBRT){
        std::cout << "    * Renderer : PBRT" << std::endl;
    }else if(opts->renderer == RENDERER_LIT){
        std::cout << "    * Renderer : LIT" << std::endl;
    }
}

int SplitByLayer(std::vector<SerializedParticle> **renderflags, 
                 std::vector<SerializedParticle> *pSet, int pCount,
                 pbrt_opts *opts)
{
    int mCount = 0;
    int max_layer = sizeof(Colors) / sizeof(Colors[0]);
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

void pbrt_insert_box(SerializedShape *shape, std::string &data, int depth,
                     pbrt_opts *opts, Float of=0)
{
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

    if(opts->renderer == RENDERER_PBRT){
        data += tab + "AttributeBegin\n";
        data += tab + "\tShape \"trianglemesh\"\n";
        data += tab + "\t\t\"point3 P\" [\n";
    }else if(opts->renderer == RENDERER_LIT){
        data += tab + "\tShape{ type [trianglemesh]\n";
        data += tab + "\t\tvertex [\n";
    }

    // NOTE: I think PBRT transform computation are a bit different from ours,
    // lets transform the points here and give the geometry without transformations
    for(int i = 0; i < 8; i++){
        vec3f p = transform.Point(points[i]);
        ss << tab << "\t\t\t" << p.x << " " << p.y << " " << p.z << "\n";
    }

    ss << tab << "\t\t]\n";

    data += ss.str();
    ss.str(std::string());
    if(opts->renderer == RENDERER_PBRT){
        data += tab + "\t\t\"integer indices\" [\n";
    }else if(opts->renderer == RENDERER_LIT){
        data += tab + "\t\tindices[";
    }
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
    if(opts->renderer == RENDERER_PBRT){
        data += tab + "AttributeEnd\n";
    }else if(opts->renderer == RENDERER_LIT){
        data += tab + "}";
    }
}

std::string pbrt_build_material(pbrt_mat *mat, pbrt_opts *opts, int refname=-1){
    std::stringstream ss;
    vec3f v = opts->mat_value;
    if(!mat){
        printf("Material is null\n");
        exit(0);
    }
    if(opts->renderer == RENDERER_PBRT){
        ss << mat->mat << " [ ";
        ss << v.x << " " << v.y << " " << v.z << " ]";
    }else if(opts->renderer == RENDERER_LIT){
        ss << "{";
        if(refname >= 0){
            ss << "name[mat_ " << refname << "] ";
        }
        ss << mat->mat << " [ ";
        ss << v.x << " " << v.y << " " << v.z << " ] }";
    }
    return ss.str();
}

void pbrt_insert_single_box(SerializedShape *shape, pbrt_mat *mat, pbrt_opts *opts,
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

    pbrt_insert_box(shape, data, depth, opts);
    data += "AttributeEnd\n";
}

void pbrt_insert_mesh(SerializedShape *shape, pbrt_mat *mat, pbrt_opts *opts,
                      std::string strmat, int depth, std::string refPath)
{
    static int meshId = 0;
    std::string meshName = shape->strParameters["Name"];
    std::string path = refPath + "/" + mesh_name_to_pbrt(meshName, opts);
    std::string bname = mesh_base_name(meshName);
    std::string data;
    std::string mats("Material ");
    std::string mate("\n");
    if(opts->renderer == RENDERER_LIT){
        mats = "Material { ";
        mate = " }\n";
    }

    if(opts->meshMap.find(meshName) != opts->meshMap.end()){
        data = opts->meshMap[meshName];
    }else{
        if(mat == nullptr){
            data += mats + strmat + mate;
        }else{
            if(mat->built){
                if(opts->renderer == RENDERER_LIT){
                    mats += "name[mat_";
                    mats += std::to_string(meshId);
                    mats += "] ";
                }
                data += mats + mat->mat + mate;
            }else{
                data += mats + pbrt_build_material(mat, opts, meshId) + mate;
            }
        }

        if(opts->renderer == RENDERER_PBRT){
            data += "AttributeBegin\n";
            data += "\tObjectBegin \"" + bname + "\"\n";
            data += "\t\tShape \"plymesh\" \"string filename\" [ \"";
            data += path + "\" ]\n";
            data += "\tObjectEnd\n";
            data += "AttributeEnd\n";
        }else if(opts->renderer == RENDERER_LIT){
            data += "Shape{ type[mesh] geometry[";
            data += path; data += "] name[";
            data += bname; data += "]\n";
        }

        meshId++;
    }

    if(opts->renderer == RENDERER_PBRT){
        data += "AttributeBegin\n";
    }

    // In order to match PBRT rendering coordinates we need to transpose
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

    if(opts->renderer == RENDERER_PBRT){
        data += "\tObjectInstance \"" + bname + "\"\n";
        data += "AttributeEnd\n";
    }
    opts->meshMap[meshName] = data;
}

void pbrt_mesh_gather(pbrt_opts *opts, std::string &data){
    for(auto it = opts->meshMap.begin(); it != opts->meshMap.end(); it++){
        data += it->second;
    }
}

void pbrt_write_domain(pbrt_opts *opts){
    std::stringstream ss;
    std::vector<vec3f> points;
    if(opts->domain.size() < 1) return;
    SerializerLoadLegacySystem3(&points, opts->domain.c_str(),
                                SERIALIZER_POSITION, nullptr);
    Float h = 0.02 * 2.0;
    if(opts->renderer == RENDERER_LIT){
        std::string scale = "scale[";
        scale += __to_stringf(h); scale += " ";
        scale += __to_stringf(h); scale += " ";
        scale += __to_stringf(h);
        scale += "]";
        vec3f p0 = points[0];
        ss << "Shape{ type[mesh] geometry[models/wireframe_cube.obj] translate [";
        ss << p0.x << " " << p0.y << " " << p0.z << "] mat[grid_mat] name[grid_base]\n";
        ss << "       " << scale << " }\n";
        for(int i = 1; i < points.size(); i++){
            vec3f p = points[i];
            ss << "Shape{ type[instance] mat[grid_mat] base[grid_base] \n";
            ss << "       translate[" << p.x << " " << p.y << " " << p.z << "]\n";
            ss << "       " << scale << " }\n";
        }

        std::ofstream out("domain.lit");
        if(out.is_open()){
            out << ss.str();
            out.close();
        }
    }else{
        printf("[Warning] : PBRT grid not supported\n");
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

    if(opts.is_legacy){
        std::vector<vec3f> points;
        SerializerLoadLegacySystem3(&points, opts.input.c_str(),
                                    opts.flags, &boundaries);
        for(int i = 0; i < points.size(); i++){
            SerializedParticle sp;
            sp.position = points[i];
            sp.boundary = boundaries[i];
            particles.push_back(sp);
        }
    }else{
        SerializerLoadSystem3(&builder, &shapes, opts.input.c_str(),
                              opts.flags, &boundaries);
        //TODO: Hacky implementation
        for(int i = 0; i < boundaries.size(); i++){
            SerializedParticle sp;
            sp.position = builder.positions[i];
            sp.boundary = boundaries[i];

            particles.push_back(sp);
        }
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
        if(opts.output.size() == 0){
            opts.output = opts.renderer == RENDERER_PBRT ? "geometry.pbrt" : "geometry.lit";
        }

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
                //pbrt_insert_dual_layer_box(&sh, pmat, &opts, mat, data, 1);
                pbrt_insert_single_box(&sh, pmat, &opts, mat, data, 1);
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

        pbrt_write_domain(&opts);

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
                    if(opts.hasClipArgs == 1){
                        vec3f ref(0);
                        vec3f at(0);
                        ref[opts.cutAxis] = opts.cutSource;
                        at[opts.cutAxis] = p.position[opts.cutAxis];
                        Float dist = Distance(ref, at);
                        if(dist < opts.cutDistance) continue;
                    }else if(opts.hasClipArgs == 2){
                        vec3f s = opts.cutPointSource;
                        vec3f at = opts.cutPointAt;
                        vec3f pi = p.position;
                        vec3f ls = at - s;
                        vec3f ppi = s + Dot(pi - s, ls) / Dot(ls, ls) * ls;
                        Float dist = Distance(s, ppi);
                        if(dist < opts.cutDistance && pi.y < 0) continue;
                    }

                    if(opts.renderer == RENDERER_PBRT){
                        data += "AttributeBegin\n\tTranslate ";
                        data += __to_stringf(p.position.x); data += " ";
                        data += __to_stringf(p.position.y); data += " ";
                        data += __to_stringf(p.position.z); data += "\n";
                        data += "\tShape \"sphere\" \"float radius\" [";
                        data += radiusString;
                        data += "]\nAttributeEnd\n";
                    }else{
                        data += "Shape{ type[sphere] radius[";
                        data += radiusString;
                        data += "] mat[layer_";
                        data += __to_stringi(i);
                        data += "] center[";
                        data += __to_stringf(p.position.x); data += " ";
                        data += __to_stringf(p.position.y); data += " ";
                        data += __to_stringf(p.position.z); data += "] }\n";
                    }
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
        if(opts.renderer == RENDERER_PBRT){
            printf("* Warning: PBRT only supports ply objects but the given simulation contains\n");
            printf("           alternative formats. Make sure to convert to ply before rendering.\n");
        }else{
            printf("* Warning: LIT only supports obj objects but the given simulation contains\n");
            printf("           alternative formats. Make sure to convert to obj before rendering.\n");
        }
    }
}
