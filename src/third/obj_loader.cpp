#include <obj_loader.h>
#include <fstream>
#include <cutil.h>
#include <sstream>

#define AllocType(type, n) (type *)memoryAlloc(sizeof(type) * n)

struct vertex_index_t {
    int v_idx, vt_idx, vn_idx;
    vertex_index_t() : v_idx(-1), vt_idx(-1), vn_idx(-1) {}
    explicit vertex_index_t(int idx) : v_idx(idx), vt_idx(idx), vn_idx(idx) {}
    vertex_index_t(int vidx, int vtidx, int vnidx)
        : v_idx(vidx), vt_idx(vtidx), vn_idx(vnidx) {}
};

struct pack_data_t{
    int *pickedV, pickedVSize;
    int *pickedN, pickedNSize;
    int *pickedU, pickedUSize;
    pack_data_t() : pickedV(nullptr), pickedVSize(0), pickedN(nullptr),
    pickedNSize(0), pickedU(nullptr), pickedUSize(0) {}
};


static int memoryInitialized = 0;
AllocatorType memoryType = AllocatorType::GPU;
MemoryAllocator memoryAlloc;
MemoryFree memoryFree;

void * GPUMemAlloc(long size){
    void *ptr = cudaAllocate(size);
    return ptr;
}

void GPUMemFree(void *ptr){
    //if(ptr) cudaFree(ptr);
    (void)ptr;
}

void *CPUMemAlloc(long size){
    return malloc(size);
}

void CPUMemFree(void *ptr){
    if(ptr) free(ptr);
}

static void MemoryCheck(){
    if(memoryInitialized == 0){
        memoryAlloc = GPUMemAlloc;
        memoryFree  = GPUMemFree;
        memoryInitialized = 1;
    }
}

static bool _ParseDouble(const char *s, const char *s_end, Float *result){
    if (s >= s_end) {
        return false;
    }
    
    double mantissa = 0.0;
    // This exponent is base 2 rather than 10.
    // However the exponent we parse is supposed to be one of ten,
    // thus we must take care to convert the exponent/and or the
    // mantissa to a * 2^E, where a is the mantissa and E is the
    // exponent.
    // To get the final double we will use ldexp, it requires the
    // exponent to be in base 2.
    int exponent = 0;
    
    // NOTE: THESE MUST BE DECLARED HERE SINCE WE ARE NOT ALLOWED
    // TO JUMP OVER DEFINITIONS.
    char sign = '+';
    char exp_sign = '+';
    char const *curr = s;
    
    // How many characters were read in a loop.
    int read = 0;
    // Tells whether a loop terminated due to reaching s_end.
    bool end_not_reached = false;
    
    /*
            BEGIN PARSING.
    */
    
    // Find out what sign we've got.
    if (*curr == '+' || *curr == '-') {
        sign = *curr;
        curr++;
    } else if (IS_DIGIT(*curr)) { /* Pass through. */
    } else {
        goto fail;
    }
    
    // Read the integer part.
    end_not_reached = (curr != s_end);
    while (end_not_reached && IS_DIGIT(*curr)) {
        mantissa *= 10;
        mantissa += static_cast<int>(*curr - 0x30);
        curr++;
        read++;
        end_not_reached = (curr != s_end);
    }
    
    // We must make sure we actually got something.
    if (read == 0) goto fail;
    // We allow numbers of form "#", "###" etc.
    if (!end_not_reached) goto assemble;
    
    // Read the decimal part.
    if (*curr == '.') {
        curr++;
        read = 1;
        end_not_reached = (curr != s_end);
        while (end_not_reached && IS_DIGIT(*curr)) {
            static const double pow_lut[] = {
                1.0, 0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001, 0.0000001,
            };
            const int lut_entries = sizeof pow_lut / sizeof pow_lut[0];
            
            // NOTE: Don't use powf here, it will absolutely murder precision.
            mantissa += static_cast<int>(*curr - 0x30) *
                (read < lut_entries ? pow_lut[read] : std::pow(10.0, -read));
            read++;
            curr++;
            end_not_reached = (curr != s_end);
        }
    } else if (*curr == 'e' || *curr == 'E') {
    } else {
        goto assemble;
    }
    
    if (!end_not_reached) goto assemble;
    
    // Read the exponent part.
    if (*curr == 'e' || *curr == 'E') {
        curr++;
        // Figure out if a sign is present and if it is.
        end_not_reached = (curr != s_end);
        if (end_not_reached && (*curr == '+' || *curr == '-')) {
            exp_sign = *curr;
            curr++;
        } else if (IS_DIGIT(*curr)) { /* Pass through. */
        } else {
            // Empty E is not allowed.
            goto fail;
        }
        
        read = 0;
        end_not_reached = (curr != s_end);
        while (end_not_reached && IS_DIGIT(*curr)) {
            exponent *= 10;
            exponent += static_cast<int>(*curr - 0x30);
            curr++;
            read++;
            end_not_reached = (curr != s_end);
        }
        exponent *= (exp_sign == '+' ? 1 : -1);
        if (read == 0) goto fail;
    }
    
    assemble:
    *result = (sign == '+' ? 1 : -1) *
        (exponent ? std::ldexp(mantissa * std::pow(5.0, exponent), exponent)
         : mantissa);
    return true;
    fail:
    return false;
}


static inline bool fixIndex(int idx, int n, int *ret){
    if(!ret){
        return false;
    }
    
    if(idx > 0){
        (*ret) = idx - 1;
        return true;
    }
    
    if(idx == 0){
        // zero is not allowed according to the spec.
        return false;
    }
    
    if(idx < 0){
        (*ret) = n + idx;  // negative value = relative
        return true;
    }
    
    return false;  // never reach here.
}

static bool parseTriple(const char **token, int vsize, int vnsize, int vtsize,
                        vertex_index_t *ret)
{
    if(!ret){
        return false;
    }
    
    vertex_index_t vi(-1);
    
    if(!fixIndex(atoi((*token)), vsize, &(vi.v_idx))){
        return false;
    }
    
    (*token) += strcspn((*token), "/ \t\r");
    if((*token)[0] != '/'){
        (*ret) = vi;
        return true;
    }
    (*token)++;
    
    // i//k
    if((*token)[0] == '/'){
        (*token)++;
        if(!fixIndex(atoi((*token)), vnsize, &(vi.vn_idx))){
            return false;
        }
        (*token) += strcspn((*token), "/ \t\r");
        (*ret) = vi;
        return true;
    }
    
    // i/j/k or i/j
    if(!fixIndex(atoi((*token)), vtsize, &(vi.vt_idx))){
        return false;
    }
    
    (*token) += strcspn((*token), "/ \t\r");
    if((*token)[0] != '/'){
        (*ret) = vi;
        return true;
    }
    
    // i/j/k
    (*token)++;  // skip '/'
    if(!fixIndex(atoi((*token)), vnsize, &(vi.vn_idx))){
        return false;
    }
    (*token) += strcspn((*token), "/ \t\r");
    
    (*ret) = vi;
    
    return true;
}



std::istream &GetLine(std::istream &is, std::string &t){
    t.clear();
    std::istream::sentry se(is, true);
    std::streambuf *sb = is.rdbuf();
    if(se){
        for(;;){
            int c = sb->sbumpc();
            switch(c){
                case '\n': return is;
                case '\r': if(sb->sgetc() == '\n') sb->sbumpc(); return is;
                case EOF: if(t.empty()) is.setstate(std::ios::eofbit); return is;
                default: t += static_cast<char>(c);
            }
        }
    }
    
    return is;
}

Float ParseFloat(const char **token){
    (*token) += strspn((*token), " \t");
    const char *end = (*token) + strcspn((*token), " \t\r");
    Float val = 0;
    _ParseDouble((*token), end, &val);
    Float f = static_cast<Float>(val);
    (*token) = end;
    return f;
}

void ParseV3(vec3f *v, const char **token){
    Float f0 = ParseFloat(token);
    Float f1 = ParseFloat(token);
    Float f2 = ParseFloat(token);
    *v = vec3f(f0, f1, f2);
}

void ParseV2(vec2f *v, const char **token){
    Float f0 = ParseFloat(token);
    Float f1 = ParseFloat(token);
    *v = vec2f(f0, f1);
}

void ParseTransform(Transform *t, const char **token){
    Float mat[4][4];
    for(int i = 0; i < 4; i++){
        for(int j = 0; j < 4; j++){
            Float f = ParseFloat(token);
            mat[i][j] = f;
        }
    }

    *t = Transform(mat);
}

static inline void AssureBufferInit(int **buffer, int &curr, int max_size){
    if(!(*buffer)){
        *buffer = new int[max_size];
        curr = max_size;
    }else if(curr < max_size && max_size > 0){
        delete[] (*buffer);
        *buffer = new int[max_size];
        curr = max_size;
    }
}

static inline void AssurePackInit(pack_data_t *pack, int maxv, int maxvn, int maxvt){
    (void)maxvn; (void)maxvt;
    AssureBufferInit(&pack->pickedV, pack->pickedVSize, maxv);
    AssureBufferInit(&pack->pickedN, pack->pickedNSize, maxvn);
    AssureBufferInit(&pack->pickedU, pack->pickedUSize, maxvt);
    int max = maxvt > maxvn ?((maxvt > maxv) ? maxvt : maxv):((maxvn > maxv) ? maxvn : maxv);
    for(int i = 0; i < max; i++){
        if(i < maxv) pack->pickedV[i] = -1;
        if(i < maxvn) pack->pickedN[i] = -1;
        if(i < maxvt) pack->pickedU[i] = -1;
    }
}

static inline void FreePack(pack_data_t *pack){
    if(pack->pickedV) delete[] pack->pickedV;
    if(pack->pickedN) delete[] pack->pickedN;
    if(pack->pickedU) delete[] pack->pickedU;
}

static inline void FillMesh(ParsedMesh *mesh, std::vector<vec3f> *v, 
                            std::vector<vec3f> *vn, std::vector<vec2f> *vt,
                            pack_data_t *pack, std::vector<vertex_index_t> *indexes)
{
    int cV = 0, cN = 0, cU = 0;
    std::vector<Point3f> p2;
    std::vector<Point2f> uv;
    std::vector<Normal3f> nor;
    
    AssurePackInit(pack, v->size(), vn->size(), vt->size());
    int *picked  = pack->pickedV;
    int *pickedN = pack->pickedN;
    int *pickedU = pack->pickedU;
    
    for(int i = 0; i < indexes->size(); i++){
        int whichV = indexes->operator[](i).v_idx;
        int whichN = indexes->operator[](i).vn_idx;
        int whichU = indexes->operator[](i).vt_idx;
        if(picked[whichV] == -1){
            picked[whichV] = cV;
            Point3f p(v->at(whichV));
            p2.push_back(p);
            cV++;
        }
        
        if(whichN > -1){
            if(pickedN[whichN] == -1){
                pickedN[whichN] = cN;
                Normal3f n(vn->at(whichN));
                nor.push_back(n);
                cN++;
            }
        }
        
        if(whichU > -1){
            if(pickedU[whichU] == -1){
                pickedU[whichU] = cU;
                Point2f u(vt->at(whichU));
                uv.push_back(u);
                cU++;
            }
        }
    }
    
    mesh->p = AllocType(Point3f, p2.size());
    mesh->indices = AllocType(Point3i, indexes->size());
    mesh->nTriangles = indexes->size() / 3;
    mesh->nVertices = p2.size();
    mesh->nUvs = 0;
    mesh->nNormals = 0;
    memcpy(mesh->p, p2.data(), p2.size() * sizeof(Point3f));
    
    if(cN > 0){
        mesh->n = AllocType(Normal3f, cN);
        memcpy(mesh->n, nor.data(), cN * sizeof(Normal3f));
        mesh->nNormals = cN;
    }
    
    if(cU > 0){
        mesh->uv = AllocType(Point2f, cU);
        memcpy(mesh->uv, uv.data(), cU * sizeof(Point2f));
        mesh->nUvs = cU;
    }
    
    for(int i = 0; i < indexes->size(); i++){
        int ip = indexes->operator[](i).v_idx;
        int in = indexes->operator[](i).vn_idx;
        int it = indexes->operator[](i).vt_idx;
        
        int iip = picked[ip];
        int iin = (in > -1) ? pickedN[in] : -1;
        int iit = (it > -1) ? pickedU[it] : -1;
        if(mesh->uv){
            if(iin > cN) {
                printf("[OBJ LOADER]Invalid index for normal [%d > %d]\n", iin, cN);
                iin = -1;
            }
            
            if(iit > cU){
                printf("[OBJ LOADER]Invalid index for uv [%d > %d]\n", iit, cU);
                iit = -1;
            }
        }
        
        mesh->indices[i] = Point3i(iip, iin, iit);
    }
    
}

__host__ int FindName(const char *path){
    char sep = '/';
    int size = strlen(path);
    for(int i = size-1; i >= 0; i--){
        if(path[i] == sep) return i+1;
    }

    return 0;
}

__host__ ParsedMesh *LoadObj(const char *path){
    int p = FindName(path);
    std::vector<ParsedMesh *> *meshes = LoadObj(path, nullptr, false);
    ParsedMesh *mesh = meshes->at(0);
    snprintf(mesh->name, sizeof(mesh->name), "%s", &path[p]);
    delete meshes;
    return mesh;
}

__host__ std::vector<ParsedMesh*> *LoadObj(const char *path, std::vector<MeshMtl> *mtls,
                                           bool split_mesh)
{
    ParsedMesh *currentMesh = nullptr;
    std::vector<vec3f> v, vn;
    std::vector<vec2f> vt;
    std::vector<vertex_index_t> indexes;
    std::vector<ParsedMesh*> *meshes = new std::vector<ParsedMesh *>();
    int p = FindName(path);

    MemoryCheck();
    
    vertex_index_t face[4];
    int facen = 0;
    
    pack_data_t pack;
    printf("[OBJ LOADER] Attempting to parse %s\n", &path[p]);
    
    std::ifstream ifs(path);
    if(!ifs){
        printf("[OBJ LOADER] Could not open file %s\n", path);
        return meshes;
    }
    
    std::string linebuf;
    
    int making_mesh = 0;
    std::string currentMtlFile;
    std::string currentMaterialName;
    int matNameCounter = 0;
    
    clock_t start = clock();
    
    while(ifs.peek() != -1){
        GetLine(ifs, linebuf);
        
        if(linebuf.size() > 0){ //remove '\n'
            if(linebuf[linebuf.size()-1] == '\n') linebuf.erase(linebuf.size() - 1);
        }
        
        if(linebuf.size() > 0){ //remove '\r'
            if(linebuf[linebuf.size()-1] == '\r') linebuf.erase(linebuf.size() - 1);
        }
        
        // skip empty
        if(linebuf.empty()) continue;
        const char *token = linebuf.c_str();
        token += strspn(token, " \t");
        
        Assert(token);
        if(token[0] == '\0') continue; //empty line
        if(token[0] == '#') continue; //comment line
        
        //if we just ended a mesh
        if(token[0] != 'f' && making_mesh && split_mesh){
            making_mesh = 0;
            FillMesh(currentMesh, &v, &vn, &vt, &pack, &indexes);
        }
        
        if(token[0] == 'o' && IS_SPACE((token[1]))){
            //TODO: grab name
            token += 2;
            std::stringstream ss;
            ss << token;
            //printf("Found %s\n", token);
        }
        
        if(token[0] == 'v' && IS_SPACE((token[1]))){
            token += 2;
            vec3f vertex;
            ParseV3(&vertex, &token);
            v.push_back(vertex);
            continue;
        }
        
        if(token[0] == 'v' && token[1] == 'n' && IS_SPACE((token[2]))){
            token += 3;
            vec3f normal;
            ParseV3(&normal, &token);
            vn.push_back(normal);
            continue;
        }
        
        if(token[0] == 'v' && token[1] == 't' && IS_SPACE((token[2]))){
            token += 3;
            vec2f uv;
            ParseV2(&uv, &token);
            vt.push_back(uv);
            continue;
        }
        
        if((0 == strncmp(token, "mtllib", 6)) && IS_SPACE((token[6]))){
            token += 7;
            currentMtlFile = std::string(token);
            //printf("Materials to lookup here: %s\n", currentMtlFile.c_str());
            continue;
        }
        
        if((0 == strncmp(token, "usemtl", 6)) && IS_SPACE((token[6]))){
            token += 7;
            std::stringstream ss;
            ss << token;
            currentMaterialName = ss.str();
            matNameCounter ++;
            //printf("Found material %s\n", currentMaterialName.c_str());
            continue;
        }
        
        if(token[0] == 'f' && IS_SPACE((token[1]))){
            //NOTE: entering here means we discovered a new mesh
            if(!making_mesh){
                MeshMtl mtl;
                currentMesh = AllocType(ParsedMesh, 1);
                currentMesh->allocator = memoryType;
                currentMesh->nVertices = 0;
                currentMesh->nTriangles = 0;
                snprintf(currentMesh->name, sizeof(currentMesh->name), "%s", &path[p]);
#if defined(WITH_TRANSFORM)
                currentMesh->toWorld = Translate(0,0,0);
#endif
                meshes->push_back(currentMesh);
                mtl.file = currentMtlFile;
                mtl.name = currentMaterialName;
                if(mtls){
                    mtls->push_back(mtl);
                }
                indexes.clear();
                making_mesh = 1;
            }
            
            token += 2;
            token += strspn(token, " \t");
            facen = 0;
            while(!IS_NEW_LINE(token[0])){
                vertex_index_t vi;
                if (!parseTriple(&token, static_cast<int>(v.size()),
                                 static_cast<int>(vn.size()),
                                 static_cast<int>(vt.size()), &vi)) 
                {
                    printf("[OBJ LOADER] Failed parsing face\n");
                    break;
                }
                
                size_t n = strspn(token, " \t\r");
                token += n;
                if(facen >= 4){
                    printf("[OBJ LOADER] Error: Not a supported face description\n");
                    exit(0);
                }
                
                face[facen++] = vi;
            }
            
            if(facen == 3){
                indexes.push_back(face[0]); indexes.push_back(face[1]);
                indexes.push_back(face[2]);
            }else if(facen == 4){
                indexes.push_back(face[0]); indexes.push_back(face[1]);
                indexes.push_back(face[2]); indexes.push_back(face[0]); 
                indexes.push_back(face[2]); indexes.push_back(face[3]);
            }else{
                printf("[OBJ LOADER] Warning unsupported face with %d vertices\n", facen);
            }
            
            continue;
        }
    }
    
    if(making_mesh){
        making_mesh = 0;
        FillMesh(currentMesh, &v, &vn, &vt, &pack, &indexes);
    }
    
    FreePack(&pack);
    
    clock_t end = clock();
    
    double time_taken = to_cpu_time(start, end);
    printf("[OBJ LOADER] Took %g seconds, #v [%d] #vn [%d] #vt [%d]. Decomposed in %d meshe(s)\n",
           time_taken, (int)v.size(), (int)vn.size(), (int)vt.size(),
           (int)meshes->size());
    
    return meshes;
}

__host__ ParsedMesh *DuplicateMesh(ParsedMesh *mesh, MeshProperties *props){
    ParsedMesh *duplicated = nullptr;
    MemoryCheck();
    if(mesh){
        duplicated = AllocType(ParsedMesh, 1);
#if defined(WITH_TRANSFORM)
        duplicated->toWorld = mesh->toWorld;
#endif
        duplicated->nVertices = mesh->nVertices;
        duplicated->nTriangles = mesh->nTriangles;
        duplicated->nUvs = mesh->nUvs;
        duplicated->nNormals = mesh->nNormals;
        duplicated->allocator = memoryType;
        duplicated->transform = mesh->transform;
        strcpy(duplicated->name, mesh->name);
        vec3f center;
        Float w = 0.f;
        if(mesh->p){
            duplicated->p = AllocType(Point3f, mesh->nVertices);
            for(int i = 0; i < mesh->nVertices; i++){
                center += vec3f(mesh->p[i]);
                w += 1.f;
            }
            
            Float invW = 1.f / w;
            center = center * invW;
        }
        
        if(mesh->uv){
            duplicated->uv = AllocType(Point2f, mesh->nUvs);
        }
        
        if(mesh->indices){
            duplicated->indices = AllocType(Point3i, mesh->nTriangles * 3);
        }
        
        //TODO:Tangents
        duplicated->s = nullptr;
        
        if(mesh->n){
            duplicated->n = AllocType(Normal3f, mesh->nNormals);
        }
        
        int ia = mesh->nTriangles * 3;
        int ib = mesh->nVertices;
        int ic = mesh->nNormals;
        int id = mesh->nUvs;
        
        int maxl = Max(ia, Max(ib, Max(ic, id)));
        
        for(int i = 0; i < maxl; i++){
            if(i < mesh->nVertices){
                Point3f p = mesh->p[i];
                if(props){
                    if(props->flip_x){
                        vec3f vp(p);
                        vec3f op = vp - center;
                        vec3f wp(-op.x, op.y, op.z);
                        p = Point3f(wp.x + center.x,
                                    wp.y + center.y,
                                    wp.z + center.z);
                    }
                }
                
                duplicated->p[i] = p;
            }
            
            if(i < mesh->nUvs){
                duplicated->uv[i] = mesh->uv[i];
            }
            
            if(i < mesh->nNormals){
                duplicated->n[i] = mesh->n[i];
            }
            
            if(i < ia){
                duplicated->indices[i] = mesh->indices[i];
            }
        }
    }
    
    return duplicated;
}

__host__ void UseDefaultAllocatorFor(AllocatorType type){
    if(type == AllocatorType::CPU){
        memoryAlloc = CPUMemAlloc;
        memoryFree  = CPUMemFree;
    }else{
        memoryAlloc = GPUMemAlloc;
        memoryFree  = GPUMemFree;
    }
    
    memoryType = type;
    memoryInitialized = 1;
}

void writeObj(HostTriangleMesh3 *mesh, std::ostream *strm){
    // vertex
    for(const auto& pt : mesh->points){
        (*strm) << "v " << pt.x << " " << pt.y << " " << pt.z << std::endl;
    }

    // uv coords
    for(const auto& uv : mesh->uvs){
        (*strm) << "vt " << uv.x << " " << uv.y << std::endl;
    }

    // normals
    for(const auto& n : mesh->normals){
        (*strm) << "vn " << n.x << " " << n.y << " " << n.z << std::endl;
    }

    // faces
    bool hasUvs_ = mesh->hasUvs();
    bool hasNormals_ = mesh->hasNormals();
    for(size_t i = 0; i < mesh->numberOfTriangles(); ++i){
        (*strm) << "f ";
        for(int j = 0; j < 3; ++j){
            (*strm) << mesh->pointIndices[i][j] + 1;
            if(hasNormals_ || hasUvs_){
                (*strm) << '/';
            }
            if(hasUvs_){
                (*strm) << mesh->uvIndices[i][j] + 1;
            }
            if(hasNormals_){
                (*strm) << '/' << mesh->normalIndices[i][j] + 1;
            }
            (*strm) << ' ';
        }
        (*strm) << std::endl;
    }
}

void HostTriangleMesh3::writeToDisk(const char *filename){
    std::ofstream file(filename);
    if(file){
        writeObj(this, &file);
        file.close();
    }else{
        printf("[OBJ_LOADER] Cannot open file %s\n", filename);
    }
}

void *DefaultAllocatorMemory(long size){
    MemoryCheck();
    return memoryAlloc(size);
}

void DefaultAllocatorFree(void *ptr){
    MemoryCheck();
    memoryFree(ptr);
}
