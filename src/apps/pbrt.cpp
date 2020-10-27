#include <args.h>
#include <serializer.h>
#include <sstream>
#include <fstream>
#include <obj_loader.h> // get parser utilities
#include <transform.h>

#define __to_stringf __to_string<Float>
#define __to_stringi __to_string<int>

typedef enum{
    LAYERED=0, LEVEL, ALL
}RenderMode;

typedef struct{
    std::string input;
    std::string output;
    RenderMode mode;
    Float rotate;
    Float radius;
    int level;
    int flags;
}pbrt_opts;

ARGUMENT_PROCESS(pbrt_input_arg){
    pbrt_opts *opts = (pbrt_opts *)config;
    opts->input = ParseNext(argc, argv, i, "--in");
    return 0;
}

ARGUMENT_PROCESS(pbrt_output_arg){
    pbrt_opts *opts = (pbrt_opts *)config;
    opts->output = ParseNext(argc, argv, i, "--out");
    return 0;
}

ARGUMENT_PROCESS(pbrt_radius_arg){
    pbrt_opts *opts = (pbrt_opts *)config;
    Float radius = ParseNextFloat(argc, argv, i, "--radius");
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

ARGUMENT_PROCESS(pbrt_level_arg){
    pbrt_opts *opts = (pbrt_opts *)config;
    std::string value = ParseNext(argc, argv, i, "--level");
    const char *token = value.c_str();
    opts->level = (int)ParseFloat(&token);
    opts->mode = RenderMode::LEVEL;
    return 0;
}

ARGUMENT_PROCESS(pbrt_rotate_arg){
    pbrt_opts *opts = (pbrt_opts *)config;
    std::string value = ParseNext(argc, argv, i, "--rotate");
    const char *token = value.c_str();
    opts->rotate = ParseFloat(&token);
    return 0;
}

std::map<const char *, arg_desc> pbrt_argument_map = {
    {"--in", 
        { .processor = pbrt_input_arg, 
            .help = "Where to read input file." 
        }
    },
    {"--out", 
        { .processor = pbrt_output_arg, 
            .help = "Where to write output." 
        }
    },
    {"--radius", 
        { .processor = pbrt_radius_arg, 
            .help = "Radius to use for particle cloud. (default: 0.012)" 
        }
    },
    {"--ppos", 
        { .processor = pbrt_serializer_pos_arg, 
            .help = "Input format contains position. (default)" 
        }
    },
    {"--pbod", 
        { .processor = pbrt_serializer_bod_arg, 
            .help = "Input format contains boundary." 
        }
    },
    {"--layered", 
        { .processor = pbrt_layered_arg, 
            .help = "Generates geometry containing only CNM-based layered boundary particles." 
        }
    },
    {"--level", 
        { .processor = pbrt_level_arg, 
            .help = "Generates geometry containing only a specific level of CNM classification." 
        }
    },
    {"--rotate", 
        { .processor = pbrt_rotate_arg, 
            .help = "Rotate input in the Y direction. (degrees)" 
        }
    },
};

const char *render_mode_string(RenderMode mode){
    switch(mode){
        case RenderMode::LAYERED: return "Layered";
        case RenderMode::LEVEL: return "Level";
        case RenderMode::ALL: return "All";
        default: return "None";
    }
}

template<typename T> std::string __to_string(T value){
    std::stringstream ss;
    ss << value;
    return ss.str();
}

std::string GetMaterialString(int layer){
    std::string data("Material \"coateddiffuse\"\n\t\"rgb reflectance\" [ ");
    if(layer == 0){
        data += std::string("0.78 0.78 0.78");
    }else if(layer == 1){
        data += std::string("0.87 0.0 0.1");
    }else if(layer == 2){
        data += std::string("0.8 0.34 0.1");
    }else if(layer == 3){
        data += std::string("0.85 0.66 0.30");
    }else if(layer == 4){
        data += std::string("0.35 0.60 0.84");
    }else{
        printf("\nUnknown layer level (%d)\n", layer);
        exit(0);
    }
    
    data += " ]\n\t\"float roughness\" [ 0 ]\n";
    return data;
}

void default_pbrt_opts(pbrt_opts *opts){
    opts->output = "geometry.pbrt";
    opts->flags = SERIALIZER_POSITION;
    opts->radius = 0.012;
    opts->mode = RenderMode::ALL;
    opts->rotate = 0;
    opts->level = -1;
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
    
    if(!IsZero(opts->rotate)){
        std::cout  << "    * Rotate (Y) : " << opts->rotate << std::endl;
    }
}

int SplitByLayer(std::vector<SerializedParticle> **renderflags, 
                 std::vector<SerializedParticle> *pSet, int pCount,
                 pbrt_opts *opts)
{
    int mCount = 0;
    std::vector<SerializedParticle> *groups;
    Transform transform;
    if(!IsZero(opts->rotate)){
        transform = RotateY(opts->rotate);
    }
    
    for(int i = 0; i < pCount; i++){
        SerializedParticle p = pSet->at(i);
        if((p.boundary + 1) > mCount) mCount = p.boundary + 1;
    }
    
    groups = new std::vector<SerializedParticle>[mCount];
    for(int i = 0; i < pCount; i++){
        SerializedParticle p = pSet->at(i);
        p.position = transform.Point(p.position);
        groups[p.boundary].push_back(p);
    }
    
    *renderflags = groups;
    return mCount;
}

int AcceptLayer(int layer, pbrt_opts *opts){
    if(opts->mode == RenderMode::ALL) return 1;
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
                data += GetMaterialString(i);
                
                for(SerializedParticle &p : *layer){
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