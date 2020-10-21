#include <cutil.h>
#include <transform.h>
#include <particle.h>
#include <emitter.h>
#include <obj_loader.h>
#include <serializer.h>
#include <iostream>

/*
* Happy Whale cmd: ./bbmesh2part --input HappyWhale.obj --scale 0.3 --spacing 0.02
*/

typedef struct{
    std::string input;
    std::string output;
    Float spacing;
    Float xrot, yrot, zrot;
    Float scale;
    Transform transform;
}emit_opts;

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


Float ParseNextFloat(int argc, char **argv, int &i, const char *arg){
    std::string value = ParseNext(argc, argv, i, arg);
    const char *token = value.c_str();
    return ParseFloat(&token);
}

void ParseArguments(int argc, char **argv, emit_opts *opts){
    opts->spacing = 0;
    opts->transform = Transform();
    opts->output = "output.txt";
    opts->spacing = 0.02;
    opts->scale = 1;
    opts->xrot = 0;
    opts->yrot = 0;
    opts->zrot = 0;
    for(int i = 1; i < argc; i++){
        std::string arg(argv[i]);
        if(arg == "--rotateX"){
            Float angle = ParseNextFloat(argc, argv, i, "--rotateX");
            opts->xrot = angle;
            opts->transform = RotateX(angle) * opts->transform;
        }else if(arg == "--rotateY"){
            Float angle = ParseNextFloat(argc, argv, i, "--rotateY");
            opts->yrot = angle;
            opts->transform = RotateY(angle) * opts->transform;
        }else if(arg == "--rotateZ"){
            Float angle = ParseNextFloat(argc, argv, i, "--rotateZ");
            opts->zrot = angle;
            opts->transform = RotateZ(angle) * opts->transform;
        }else if(arg == "--scale"){
            Float scale = ParseNextFloat(argc, argv, i, "--scale");
            opts->scale = scale;
            opts->transform = Scale(scale) * opts->transform;
        }else if(arg == "--input"){
            opts->input = ParseNext(argc, argv, i, "--input");
        }else if(arg == "--output"){
            opts->output = ParseNext(argc, argv, i, "--output");
        }else if(arg == "--spacing"){
            Float spacing = ParseNextFloat(argc, argv, i, "--spacing");
            opts->spacing = spacing;
        }else{
            std::cout << "Unknwon argument " << arg << std::endl;
            exit(0);
        }
    }
    
    if(opts->input.size() == 0){
        std::cout << "Missing input geometry" << std::endl;
        exit(0);
    }
}

void PrintArguments(emit_opts *opts){
    printf("* BubblesMesh2Part - Bubbles mesh2particle built %s at %s *\n", __DATE__, __TIME__);
    std::cout << "Configs: " << std::endl;
    std::cout << "    * Target file : " << opts->input << std::endl;
    std::cout << "    * Target output : " << opts->output << std::endl;
    std::cout << "    * Spacing : " << opts->spacing << std::endl;
    std::cout << "    * Transforms : " << std::endl;
    std::cout << "        - Rotate X : " << opts->xrot << std::endl;
    std::cout << "        - Rotate Y : " << opts->yrot << std::endl;
    std::cout << "        - Rotate Z : " << opts->zrot << std::endl;
    std::cout << "        - Scale : " << opts->scale << std::endl;
}

void MeshToParticles(const char *name, const Transform &transform,
                     Float spacing, const char *output)
{
    printf("===== Emitting particles from mesh\n");
    
    UseDefaultAllocatorFor(AllocatorType::CPU);
    ParsedMesh *mesh = LoadObj(name);
    
    Shape *shape = MakeMesh(mesh, transform);
    ParticleSetBuilder3 builder;
    VolumeParticleEmitter3 emitter(shape, shape->GetBounds(), spacing);
    
    emitter.Emit(&builder);
    
    SphParticleSet3 *sphSet = SphParticleSet3FromBuilder(&builder);
    SphSolverData3 *data = DefaultSphSolverData3();
    data->sphpSet = sphSet;
    data->domain = nullptr;
    
    SerializerSaveSphDataSet3(data, output, SERIALIZER_POSITION);
    
    printf("===== OK\n");
}


#if 0
MeshToParticles("/home/felpz/Documents/models/dragon_aligned.obj",
                Scale(0.7) * RotateY(-90), 0.02, "output.txt");
MeshToParticles("/home/felpz/Downloads/happy_whale.obj",
                Scale(0.3), 0.02, "output.txt");
#endif


int main(int argc, char **argv){
    emit_opts opts;
    ParseArguments(argc, argv, &opts);
    PrintArguments(&opts);
    MeshToParticles(opts.input.c_str(), opts.transform, 
                    opts.spacing, opts.output.c_str());
    return 0;
}