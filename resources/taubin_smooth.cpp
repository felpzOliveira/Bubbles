#include <vcg/complex/complex.h>

#include <vcg/complex/algorithms/clean.h>
#include <vcg/complex/algorithms/create/platonic.h>
#include <vcg/complex/algorithms/polygonal_algorithms.h>

#include <wrap/io_trimesh/import_ply.h>
#include <wrap/io_trimesh/export_ply.h>
#include <string>

using namespace vcg;
using namespace std;

/* Definition of a mesh of polygons that also supports half-edges
*/
class PFace;
class PVertex;

struct PUsedTypes: public vcg::UsedTypes<vcg::Use<PVertex>  ::AsVertexType,
                                          vcg::Use<PFace>	::AsFaceType>{};

class PVertex:public vcg::Vertex<	PUsedTypes,
    vcg::vertex::Coord3f,
    vcg::vertex::Normal3f,
    vcg::vertex::Mark,
    vcg::vertex::Qualityf,
    vcg::vertex::Mark,
    vcg::vertex::BitFlags>{} ;

class PFace:public vcg::Face<
     PUsedTypes
    ,vcg::face::PolyInfo // this is necessary  if you use component in vcg/simplex/face/component_polygon.h
                        // It says "this class is a polygon and the memory for its components (e.g. pointer to its vertices
                        // will be allocated dynamically")
    ,vcg::face::PFVAdj // Pointer to the vertices (just like FVAdj )
    ,vcg::face::PFVAdj
    ,vcg::face::PFFAdj  // Pointer to edge-adjacent face (just like FFAdj )
    ,vcg::face::BitFlags // bit flags
    ,vcg::face::Qualityf // quality
    ,vcg::face::Mark     // incremental mark
    ,vcg::face::Normal3f // normal
> {};

class PMesh: public
    vcg::tri::TriMesh<
    std::vector<PVertex>, // the vector of vertices
    std::vector<PFace > // the vector of faces
    >{};

using namespace vcg;
using namespace std;

bool arg_string_value(int argc, char **argv, int &iter, std::string &value, const char *arg){
    bool rv = false;
    if(argc > iter+1){
        value = argv[iter+1];
        iter += 1;
        rv = true;
    }else
        printf("Missing argument for %s\n", arg);
    return rv;
}

bool arg_float_value(int argc, char **argv, int &iter, float &value, const char *arg){
    bool rv = false;
    if(argc > iter+1){
        std::string str = argv[iter+1];
        value = std::stof(str);
        iter += 1;
        rv = true;
    }else
        printf("Missing argument for %s\n", arg);
    return rv;
}

int main(int argc, char **argv){
    PMesh mm;
    float lambda = 0.5;
    float mu = -0.53;
    int iterations = 10;
    std::string input;
    std::string output = "smoothed.ply";

    for(int i = 1; i < argc; ){
        std::string cmd(argv[i]);
        if(cmd == "-mu"){
            if(!arg_float_value(argc, argv, i, mu, "mu"))
                return 0;
        }else if(cmd == "-lambda"){
            if(!arg_float_value(argc, argv, i, lambda, "lambda"))
                return 0;
        }else if(cmd == "-in"){
            if(!arg_string_value(argc, argv, i, input, "in"))
                return 0;
        }else if(cmd == "-out"){
            if(!arg_string_value(argc, argv, i, output, "out"))
                return 0;
        }else{
            printf("Unknown flag '%s'\n", cmd.c_str());
            return 0;
        }
        i++;
    }

    if(input.size() == 0){
        printf("Missing input\n");
        return 0;
    }

    std::cout << " - Loading..." << std::flush;
    tri::io::ImporterPLY<PMesh>::Open(mm, input.c_str());
    std::cout << "done\n - Smoothing ( μ = " << mu << " λ = " << lambda << " ) ..." << std::flush;
    tri::Smooth<PMesh>::VertexCoordTaubin(mm, iterations, lambda, mu);
    std::cout << "done\n - Outputing..." << std::flush;
    tri::io::ExporterPLY<PMesh>::Save(mm, output.c_str());
    std::cout << "done" << std::endl;
    return 0;
}


