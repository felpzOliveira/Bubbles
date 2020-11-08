#include <args.h>
#include <serializer.h>
#include <sstream>
#include <fstream>
#include <obj_loader.h> // get parser utilities
#include <transform.h>

#define __to_stringf __to_string<Float>
#define __to_stringi __to_string<int>

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
    int level;
    int flags;
    int hasClipArgs;
}pbrt_opts;

const std::string ColorsString[] = {
    "0.78 0.78 0.78",
    "0.87 0.0 0.1",
    "0.8 0.34 0.1",
    "0.85 0.66 0.30",
    "0.35 0.60 0.84",
};

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

ARGUMENT_PROCESS(pbrt_serializer_pos_arg){
    pbrt_opts *opts = (pbrt_opts *)config;
    opts->flags |= SERIALIZER_POSITION;
    return 0;
}

ARGUMENT_PROCESS(pbrt_serializer_bod_arg){
    pbrt_opts *opts = (pbrt_opts *)config;
    opts->flags |= SERIALIZER_BOUNDARY;
    return 0;
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
    {"-ppos", 
        { .processor = pbrt_serializer_pos_arg, 
            .help = "Input format contains position. (default)" 
        }
    },
    {"-pbod", 
        { .processor = pbrt_serializer_bod_arg, 
            .help = "Input format contains boundary." 
        }
    },
    {"-layered", 
        { .processor = pbrt_layered_arg, 
            .help = "Generates geometry containing only CNM-based layered boundary particles." 
        }
    },
    {"-filtered", 
        { .processor = pbrt_filtered_arg, 
            .help = "Generates geometry containing CNM-based layer and interior particles as gray." 
        }
    },
    {"-level", 
        { .processor = pbrt_level_arg, 
            .help = "Generates geometry containing only a specific level of CNM classification." 
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
    if(layer != 1 && layer != 2) return ColorsString[0];
    return ColorsString[1];
}

/*
* Follow color map
*/
static std::string GetMaterialStringAll(int layer){
    if(layer > 4){
        printf("\nUnknown layer level (%d)\n", layer);
        exit(0);
    }
    return ColorsString[layer];
}

/*
* L1 = red; L2 = yellow;
* anything other should not get here but return gray
*/
static std::string GetMaterialStringLayered(int layer){
    if(layer != 1 && layer != 2) return ColorsString[0];
    return ColorsString[layer];
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
}

int SplitByLayer(std::vector<SerializedParticle> **renderflags, 
                 std::vector<SerializedParticle> *pSet, int pCount,
                 pbrt_opts *opts)
{
    int mCount = 0;
    std::vector<SerializedParticle> *groups;
    
    for(int i = 0; i < pCount; i++){
        SerializedParticle p = pSet->at(i);
        if((p.boundary + 1) > mCount) mCount = p.boundary + 1;
    }
    
    printf("Found %d layers\n", mCount);
    
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

void pbrt_command(int argc, char **argv){
    pbrt_opts opts;
    std::vector<SerializedParticle> particles;
    std::vector<SerializedParticle> *rendergroups = nullptr;
    int count = 0;
    int pAdded = 0;
    std::string radiusString;
    
    default_pbrt_opts(&opts);
    argument_process(pbrt_argument_map, argc, argv, &opts);
    print_configs(&opts);
    
    if(opts.input.size() == 0){
        printf("No input given\n");
        return;
    }
    
    if(opts.mode != RenderMode::ALL && !(opts.flags & SERIALIZER_BOUNDARY)){
        printf("This mode requires CNM-based boundary output from Bubbles\n");
        return;
    }
    
    radiusString = __to_stringf(opts.radius);
    
    count = SerializerLoadParticles3(&particles, opts.input.c_str(), opts.flags);
    
    if(count > 0){
        int total = count;
        std::string data;
        std::ofstream ofs(opts.output, std::ofstream::out);
        
        if(!ofs.is_open()){
            std::cout << "Failed to open output file: " << opts.output << std::endl;
            return;
        }
        
        count = SplitByLayer(&rendergroups, &particles, total, &opts);
        for(int i = 0; i < count; i++){
            std::vector<SerializedParticle> *layer = &rendergroups[i];
            printf("\rProcessing layer %d / %d ... ", i+1, count);
            fflush(stdout);
            
            if(layer->size() > 0){
                if(AcceptLayer(i, &opts) == 0) continue;
                
                data += std::string("###### Layer ");
                data += __to_stringi(i); data += "\n";
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
                    data += "TransformBegin\n\tTranslate ";
                    data += __to_stringf(p.position.x); data += " ";
                    data += __to_stringf(p.position.y); data += " ";
                    data += __to_stringf(p.position.z); data += "\n";
                    data += "\tShape \"sphere\" \"float radius\" [";
                    data += radiusString;
                    data += "]\nTransformEnd\n";
                    pAdded += 1;
                }
            }
        }
        
        ofs << data;
        ofs.close();
        std::cout << "Done\nCreated " << pAdded << " objects." << std::endl;
    }else{
        std::cout << "Empty file" << std::endl;
    }
    
    if(rendergroups) delete[] rendergroups;
}