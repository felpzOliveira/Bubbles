#include <vcg/complex/complex.h>
#include <vcg/complex/algorithms/clean.h>
#include <vcg/complex/algorithms/hole.h>

#include <vcg/complex/algorithms/clean.h>
#include <vcg/complex/algorithms/create/platonic.h>
#include <vcg/complex/algorithms/refine_loop.h>
#include <vcg/complex/algorithms/polygonal_algorithms.h>

#include <wrap/io_trimesh/import_ply.h>
#include <wrap/io_trimesh/import_obj.h>
#include <wrap/io_trimesh/export_ply.h>
#include <wrap/io_trimesh/export_obj.h>
#include <string>
#include <fstream>

using namespace vcg;
using namespace std;

class TFace;
class TVertex;
class TEdge;

struct TUsedTypes: public vcg::UsedTypes< vcg::Use<TVertex>::AsVertexType,
                                          vcg::Use<TEdge>::AsEdgeType,
                                          vcg::Use<TFace>::AsFaceType >{};

class TEdge : public vcg::Edge<TUsedTypes>{};

class TVertex : public Vertex< TUsedTypes,
    vertex::BitFlags,
    vertex::VFAdj,
    vertex::Coord3f,
    vertex::Normal3f,
    vertex::Mark >{};

class TFace   : public Face<   TUsedTypes,
    face::VertexRef,    // three pointers to vertices
    face::Normal3f,     // normal
    face::BitFlags,     // flags
    face::VFAdj,
    face::FFAdj         // three pointers to adjacent faces
> {};

/* mesh container */
class TMesh   : public vcg::tri::TriMesh< vector<TVertex>, vector<TFace> > {
    public:
    bool HasVFAdjacency() const { return true; }
};
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

bool arg_int_value(int argc, char **argv, int &iter, int &value, const char *arg){
    bool rv = false;
    if(argc > iter+1){
        std::string str = argv[iter+1];
        value = std::stoi(str);
        iter += 1;
        rv = true;
    }else
        printf("Missing argument for %s\n", arg);
    return rv;
}

bool method_is_valid(std::string method){
    return method == "taubin" || method == "laplacian" || method == "loop" || method == "cl";
}

std::string methods_string(){
    return "taubin, laplacian, loop or cl";
}

bool _is_file_ply(const char *path){
    bool rv = false;
    std::ifstream ifs(path);
    std::string first_line;
    if(std::getline(ifs, first_line)){
        rv = first_line.substr(0, 3) == "ply";
    }
    ifs.close();
    return rv;
}

int main(int argc, char **argv){
    TMesh mm;
    float lambda = 0.5;
    float loopThreshold = 0.042739;
    float mu = -0.53;
    int iterations = 10;
    int mask = 0;
    bool cleanFaces = true;
    std::string input;
    std::string output = "smoothed.ply";
    std::string method;
    bool is_ply = false;

    for(int i = 1; i < argc; ){
        std::string cmd(argv[i]);
        if(cmd == "-mu"){
            if(!arg_float_value(argc, argv, i, mu, "mu"))
                return 0;
        }else if(cmd == "-iterations"){
            if(!arg_int_value(argc, argv, i, iterations, "iterations"))
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
        }else if(cmd == "-method"){
            if(!arg_string_value(argc, argv, i, method, "method"))
                return 0;
        }else if(cmd == "-no-rm"){
            cleanFaces = false;
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

    if(method.size() == 0){
        printf("Missing method, pick either: %s\n", methods_string().c_str());
        return 0;
    }

    if(!method_is_valid(method)){
        printf("Unsupported method\n");
        return 0;
    }

    if(method == "cl"){
        cleanFaces = true;
    }

    is_ply = _is_file_ply(input.c_str());

    std::cout << " - Loading..." << std::flush;
    if(is_ply)
        tri::io::ImporterPLY<TMesh>::Open(mm, input.c_str());
    else
        tri::io::ImporterOBJ<TMesh>::Open(mm, input.c_str(), mask);

    tri::UpdateTopology<TMesh>::FaceFace(mm);
    int count = tri::Clean<TMesh>::CountNonManifoldEdgeFF(mm);
    std::cout << "done\n   * Non Manifold edges: " <<
                    tri::Clean<TMesh>::CountNonManifoldEdgeFF(mm) << std::endl;
    std::cout << "   * Non Manifold vertices: " <<
                    tri::Clean<TMesh>::CountNonManifoldVertexFF(mm) << std::endl;

    if(cleanFaces){
        std::cout << " - Clearing..." << std::flush;
        vcg::tri::Clean<TMesh>::RemoveNonManifoldFace(mm);
        vcg::tri::Clean<TMesh>::RemoveUnreferencedVertex(mm);

        vcg::tri::UpdateTopology<TMesh>::VertexFace(mm);
        // Repair the mesh to fill in holes
        vcg::tri::UpdateTopology<TMesh>::FaceFace(mm);
        vcg::tri::UpdateTopology<TMesh>::VertexFace(mm);
        vcg::tri::UpdateFlags<TMesh>::FaceBorderFromFF(mm);
        vcg::tri::UpdateFlags<TMesh>::FaceBorderFromVF(mm);
        vcg::tri::Clean<TMesh>::RemoveDuplicateVertex(mm);
        vcg::tri::Clean<TMesh>::RemoveDuplicateFace(mm);
        vcg::tri::Allocator<TMesh>::CompactEveryVector(mm);
        std::cout << "done" << std::endl;
#if 0
        int numHoles = vcg::tri::Clean<TMesh>::CountHoles(mm);

        if(numHoles > 0){
            numHoles = vcg::tri::Hole<TMesh>::EarCuttingFill<tri::TrivialEar<TMesh>>(mm, 10, false);
            vcg::tri::UpdateFlags<TMesh>::FaceBorderFromFF(mm);
        }
#endif
        if(method == "cl"){
            goto __output;
        }
    }

    if(method == "taubin"){
        std::cout << " - Taubin Smoothing ( μ = " << mu << " λ = " << lambda << " )..." << std::flush;
        tri::Smooth<TMesh>::VertexCoordTaubin(mm, iterations, lambda, mu);
    }else if(method == "laplacian"){
        std::cout << " - Laplacian Smoothing..." << std::flush;
        tri::Smooth<TMesh>::VertexCoordLaplacian(mm, iterations);
    }else{
        std::cout << " - Loop Subdivision..." << std::flush;
        for(int i = 0; i < iterations; i++){
            tri::RefineOddEven<TMesh>(mm, tri::OddPointLoop<TMesh>(mm), tri::EvenPointLoop<TMesh>(),
                                      loopThreshold);
        }
    }
    std::cout << "done" << std::endl;
__output:
    std::cout << " - Outputing..." << std::flush;
    if(output.substr(output.size()-3) == "ply")
        tri::io::ExporterPLY<TMesh>::Save(mm, output.c_str());
    else
        tri::io::ExporterOBJ<TMesh>::Save(mm, output.c_str(),
                                          vcg::tri::io::Mask::IOM_BITPOLYGONAL);
    std::cout << "done" << std::endl;
    return 0;
}
