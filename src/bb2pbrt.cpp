#include <iostream>
#include <string>
#include <serializer.h>
#include <sstream>
#include <fstream>
#include <obj_loader.h> // get parser utilities
#include <transform.h>

#define TARGET_FILE "/home/felipe/Documents/Bubbles/whale/output_300.txt"
#define TARGET_OUTFILE "geometry.pbrt"
#define DEFAULT_R "0.012"
#define __to_stringf __to_string<Float>
#define __to_stringi __to_string<int>

typedef enum{
    LAYERED=0, LEVEL, ALL
}RenderMode;

struct convert_opts{
    std::string input;
    std::string output;
    std::string radius;
    RenderMode mode;
    Float rotate;
    int level;
    int flags;
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

std::string ParseNext(int argc, char **argv, int &i, const char *arg, int count=1){
    int ok = (argc > i+count) ? 1 : 0;
    if(!ok){
        printf("Invalid argument for %s\n", arg);
        exit(0);
    }
    
    std::string res;
    for(int n = 1; n < count+1; n++){
        res += std::string(argv[n+i]);
        if(n < count) res += " ";
    }
    
    i += count;
    return res;
}

void InitializeArguments(convert_opts *opts, int argc, char **argv){
    opts->input = TARGET_FILE;
    opts->output = TARGET_OUTFILE;
    opts->flags = SERIALIZER_POSITION;
    opts->radius = DEFAULT_R;
    opts->mode = RenderMode::ALL;
    opts->rotate = 0;
    opts->level = -1;
    //TODO: flags
    for(int i = 1; i < argc; i++){
        std::string arg(argv[i]);
        if(arg == "--in"){
            opts->input = ParseNext(argc, argv, i, "--in");
        }else if(arg == "--out"){ // output file
            opts->output = ParseNext(argc, argv, i, "--out");
        }else if(arg == "--radius"){ // target radius
            opts->radius = ParseNext(argc, argv, i, "--radius");
        }else if(arg == "--ppos"){ // parse position
            opts->flags |= SERIALIZER_POSITION;
        }else if(arg == "--pbod"){ // parse boundary
            opts->flags |= SERIALIZER_BOUNDARY;
        }else if(arg == "--layered"){ // layered mode
            opts->mode = RenderMode::LAYERED;
        }else if(arg == "--level"){ // single level mode
            std::string value = ParseNext(argc, argv, i, "--level");
            const char *token = value.c_str();
            opts->level = (int)ParseFloat(&token);
            opts->mode = RenderMode::LEVEL;
        }else if(arg == "--rotate"){
            std::string value = ParseNext(argc, argv, i, "--rotate");
            const char *token = value.c_str();
            opts->rotate = ParseFloat(&token);
        }else{
            printf("Unknown argument %s\n", argv[i]);
            exit(0);
        }
    }
}

void PrintArgs(convert_opts *opts){
    printf("* Bubbles2Pbrt - Bubbles converter built %s at %s *\n", __DATE__, __TIME__);
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
                 convert_opts *opts)
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

int AcceptLayer(int layer, convert_opts *opts){
    if(opts->mode == RenderMode::ALL) return 1;
    if(opts->mode == RenderMode::LAYERED) return (layer > 0 && layer < 3 ? 1 : 0);
    if(opts->mode == RenderMode::LEVEL) return (layer == opts->level ? 1 : 0);
    printf("[Accept] Unsupported render mode\n");
    exit(0);
}

int main(int argc, char **argv){
    convert_opts opts;
    std::vector<SerializedParticle> particles;
    std::vector<SerializedParticle> *rendergroups = nullptr;
    int count = 0;
    int pAdded = 0;
    
    InitializeArguments(&opts, argc, argv);
    PrintArgs(&opts);
    
    count = SerializerLoadParticles3(&particles, opts.input.c_str(), opts.flags);
    
    if(count > 0){
        int total = count;
        std::string data;
        std::ofstream ofs(opts.output, std::ofstream::out);
        
        if(!ofs.is_open()){
            std::cout << "Failed to open output file: " << opts.output << std::endl;
            return 0;
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
                    data += opts.radius;
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
    
    return 0;
}