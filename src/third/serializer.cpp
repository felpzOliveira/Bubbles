#include <serializer.h>
#include <obj_loader.h>
#include <string.h>
#include <map>

#define CAN_SKIP(x) (IS_SPACE(x) && !IS_NEW_LINE(x))

static void PrintToFile(FILE *fp, const Float &value, int spacing=0){
    if(spacing)
        fprintf(fp, " %g", value);
    else
        fprintf(fp, "%g", value);
}

static void PrintToFile(FILE *fp, const int &value, int spacing=0){
    if(spacing)
        fprintf(fp, " %d", value);
    else
        fprintf(fp, "%d", value);
}

static void PrintToFile(FILE *fp, const vec3f &value, int spacing=0){
    if(spacing)
        fprintf(fp, " %g %g %g", value.x, value.y, value.z);
    else
        fprintf(fp, "%g %g %g", value.x, value.y, value.z);
}

static void PrintToFile(FILE *fp, const vec2f &value, int spacing=0){
    if(spacing)
        fprintf(fp, " %g %g", value.x, value.y);
    else
        fprintf(fp, "%g %g", value.x, value.y);
}

std::string SerializerStringFromFlags(int flags){
    std::string str;
    if(flags & SERIALIZER_POSITION) str += "p";
    if(flags & SERIALIZER_VELOCITY) str += "v";
    if(flags & SERIALIZER_DENSITY) str += "d";
    if(flags & SERIALIZER_MASS) str += "m";
    if(flags & SERIALIZER_BOUNDARY) str += "b";
    if(flags & SERIALIZER_NORMAL) str += "n";
    if(flags & SERIALIZER_LAYERS) str += "l";
    if(flags & SERIALIZER_RULE_BOUNDARY_EXCLUSIVE) str += "o";
    return str;
}

int SerializerFlagsFromString(const char *spec){
    int flags = 0;
    std::string str(spec);
    for(int i = 0; i < str.size(); i++){
        char c = spec[i];
        if(c == 'p' || c == 'P') flags |= SERIALIZER_POSITION;
        else if(c == 'v' || c == 'V') flags |= SERIALIZER_VELOCITY;
        else if(c == 'd' || c == 'D') flags |= SERIALIZER_DENSITY;
        else if(c == 'm' || c == 'M') flags |= SERIALIZER_MASS;
        else if(c == 'b' || c == 'B') flags |= SERIALIZER_BOUNDARY;
        else if(c == 'n' || c == 'N') flags |= SERIALIZER_NORMAL;
        else if(c == 'l' || c == 'L') flags |= SERIALIZER_LAYERS;
        else if(c == 'o' || c == 'O') flags |= SERIALIZER_RULE_BOUNDARY_EXCLUSIVE;
        else if(c == 'z' || c == 'Z') flags |= SERIALIZER_XYZ;
        else{
            printf("Unknown flag argument %c\n", c);
            return -1;
        }
    }
    
    return flags;
}

int SerializerFluidProcessToken(std::string &token, int &in_region,
                                std::string &format, int &pCount,
                                std::string linebuf, int i)
{
    int r = 0;
    if(token.size() > 0){
        if(in_region == 0){
            if(token == "FluidBegin"){
                in_region = 1;
            }
        }else{
            if(token == "\"Count\""){
                int j = 0;
                std::string sub = linebuf.substr(i+1);
                while(CAN_SKIP(sub[j])) j++;
                sub = sub.substr(j);
                const char *t = sub.c_str();
                pCount = (int)ParseFloat(&t);
            }else if(token == "\"Format\""){
                int j = 0;
                std::string sub = linebuf.substr(i+1);
                while(CAN_SKIP(sub[j])) j++;
                format = std::string();

                while(!CAN_SKIP(sub[j])){
                    format += sub[j];
                    j++;
                }
            }else if(token == "DataBegin"){
                // ???
                r = 1;
            }else if(token == "DataEnd"){
                // ???
            }else if(token == "FluidEnd"){
                in_region = 0;
            }
        }

        token.clear();
    }

    return r;
}

int SerializerNextToken(std::istream &ifs, std::string &value){
    std::string linebuf;
    while(ifs.peek() != -1){
        GetLine(ifs, linebuf);
        std::string token;
        for(int i = 0; i < linebuf.size(); i++){
            if(IS_SPACE(linebuf[i]) || IS_NEW_LINE(linebuf[i])){
                if(token.size() > 0){
                    value = token;
                    return 1;
                }
            }

            token += linebuf[i];
        }

        if(token.size() > 0){
            value = token;
            return 1;
        }
    }

    return 0;
}

int SerializerFindFluidSection(std::istream &is, std::string &format, int region=0){
    std::string linebuf;
    int pCount = 0;
    int in_region = region;
    int r = 0;
    while(is.peek() != -1){
        GetLine(is, linebuf);
        std::string token;
        for(int i = 0; i < linebuf.size(); i++){
            if(IS_SPACE(linebuf[i]) || IS_NEW_LINE(linebuf[i])){
                r = SerializerFluidProcessToken(token, in_region, format,
                                                pCount, linebuf, i);
                if(r){
                    return pCount;
                }

                continue;
            }

            token += linebuf[i];
        }

        r = SerializerFluidProcessToken(token, in_region, format,
                                        pCount, linebuf, 0);
        if(r) return pCount;
    }

    return pCount;
}

void SerializerLoadPoints3(std::vector<vec3f> *points,
                           const char *filename, int &flags)
{
    std::ifstream ifs(filename);
    int found_end = 0;
    if(ifs){
        std::string format;
        int pCount = 0;
        std::string linebuf;
        vec3f vel(0), nor(0);
        int boundary = 0;

        pCount = SerializerFindFluidSection(ifs, format);
        if(format.size() > 0){
            flags = SerializerFlagsFromString(format.c_str());
        }

        points->clear();
        points->reserve(pCount);

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
            vec3f pos(0);
            const char *token = linebuf.c_str();
            token += strspn(token, " \t");

            if(std::string(token).find("DataEnd") != std::string::npos){
                found_end = 1;
                break;
            }

            if(flags & SERIALIZER_POSITION){
                ParseV3(&pos, &token);
                while(CAN_SKIP(token[0])) token++;
            }
            
            if(flags & SERIALIZER_VELOCITY){
                ParseV3(&vel, &token);
                while(CAN_SKIP(token[0])) token++;
            }
            
            if(flags & SERIALIZER_DENSITY){
                (void)ParseFloat(&token);
                while(CAN_SKIP(token[0])) token++;
            }
            
            if(flags & SERIALIZER_MASS){
                (void)ParseFloat(&token);
                while(CAN_SKIP(token[0])) token++;
            }
            
            if(flags & SERIALIZER_BOUNDARY){
                boundary = (int)ParseFloat(&token);
                while(CAN_SKIP(token[0])) token++;
                if((flags & SERIALIZER_RULE_BOUNDARY_EXCLUSIVE) && !boundary){
                    continue;
                }
            }
            
            points->push_back(pos);
        }
    }

    if(!found_end){
        printf("Error: Unterminated fluid description, missing \'DataEnd\'\n");
        exit(0);
    }
}

void SerializerLoadLegacySystem3(std::vector<vec3f> *points, const char *filename,
                                 int flags, std::vector<int> *boundaries)
{
    std::ifstream ifs(filename);
    if(ifs){
        std::string format;
        int pCount = 0;
        std::string linebuf;
        vec3f vel(0), nor(0);
        int boundary = 0;
        int readCount = 0;

        GetLine(ifs, linebuf);
        if(linebuf.size() > 0){ //remove '\n'
            if(linebuf[linebuf.size()-1] == '\n') linebuf.erase(linebuf.size() - 1);
        }

        if(linebuf.size() > 0){ //remove '\r'
            if(linebuf[linebuf.size()-1] == '\r') linebuf.erase(linebuf.size() - 1);
        }

        points->clear();
        if(flags & SERIALIZER_XYZ){
            flags = SERIALIZER_POSITION | SERIALIZER_XYZ;
            vec3f pos(0);
            const char *token = linebuf.c_str();
            token += strspn(token, " \t");

            ParseV3(&pos, &token);
            points->push_back(pos);
            readCount = 1;
        }else{
            pCount = std::stoi(linebuf);
            if(pCount < 1){
                printf("Failed to read particle count\n");
                return;
            }

            points->reserve(pCount);
        }

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
            vec3f pos(0);
            const char *token = linebuf.c_str();
            token += strspn(token, " \t");

            if(readCount == pCount && !(flags & SERIALIZER_XYZ)){
                break;
            }

            if(flags & SERIALIZER_POSITION){
                ParseV3(&pos, &token);
                while(CAN_SKIP(token[0])) token++;
            }

            if(flags & SERIALIZER_VELOCITY){
                ParseV3(&vel, &token);
                while(CAN_SKIP(token[0])) token++;
            }

            if(flags & SERIALIZER_DENSITY){
                (void)ParseFloat(&token);
                while(CAN_SKIP(token[0])) token++;
            }

            if(flags & SERIALIZER_MASS){
                (void)ParseFloat(&token);
                while(CAN_SKIP(token[0])) token++;
            }

            if(flags & SERIALIZER_BOUNDARY){
                boundary = (int)ParseFloat(&token);
                while(CAN_SKIP(token[0])) token++;
                if((flags & SERIALIZER_RULE_BOUNDARY_EXCLUSIVE) && !boundary){
                    continue;
                }
            }

            points->push_back(pos);
            if(boundaries){
                boundaries->push_back(boundary);
            }

            readCount += 1;
        }
    }
}

ShapeType _SerializerGetShapeType(std::string name){
    if(name == "box") return ShapeBox;
    if(name == "sphere") return ShapeSphere;
    if(name == "mesh") return ShapeMesh;
    //TODO
    printf("Unknown shape type \'%s\'\n", name.c_str());
    exit(0);
}

const char *SerializerGetShapeName(ShapeType type){
    if(type == ShapeBox) return "box";
    if(type == ShapeSphere) return "sphere";
    if(type == ShapeMesh) return "mesh";
    printf("Unknown shape name for \'%d\'\n", (int)type);
    exit(0);
}

void _SerializerLoadShape(SerializedShape *shape, std::ifstream &ifs){
    std::string linebuf;
    int found_end = 0;
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

        if(std::string(token).find("ShapeEnd") != std::string::npos){
            found_end = 1;
            break;
        }

        size_t type = std::string(token).find("\"Type\"");
        size_t len  = std::string(token).find("\"Length\"");
        size_t tran = std::string(token).find("\"Transform\"");
        size_t name = std::string(token).find("\"Name\"");
        if(type != std::string::npos){
            const char *val = token + type + std::string("\"Type\"").size();
            while(CAN_SKIP(val[0])) val++;
            shape->type = _SerializerGetShapeType(std::string(val));
        }else if(len != std::string::npos){
            vec3f v;
            size_t ss = std::string("\"Length\"").size();
            const char *val = token + len + ss;
            while(CAN_SKIP(val[0])) val++;
            ParseV3(&v, &val);
            shape->numParameters["Length"] = vec4f(v.x, v.y, v.z, 0.0);
        }else if(tran != std::string::npos){
            Transform t;
            size_t ss = std::string("\"Transform\"").size();
            const char *val = token + tran + ss;
            while(CAN_SKIP(val[0])) val++;
            ParseTransform(&t, &val);
            shape->transfParameters["Transform"] = t;
        }else if(name != std::string::npos){
            const char *val = token + name + std::string("\"Name\"").size();
            while(CAN_SKIP(val[0])) val++;
            shape->strParameters["Name"] = std::string(val);
        }else{
            printf("Unknown %s\n", token);
        }
    }

    if(!found_end){
        printf("Error: Unterminated shape description, missing \'ShapeEnd\'\n");
        exit(0);
    }
}

int _SerializerLoadParticles3(std::vector<SerializedParticle> *pSet,
                              std::ifstream &ifs, int &flags, int pCount)
{
    std::string linebuf;
    int found_end = 0;
    int size = 0;

    pSet->clear();
    pSet->reserve(pCount);

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

        if(std::string(token).find("DataEnd") != std::string::npos){
            found_end = 1;
            break;
        }

        // Parse a particle data
        /*
        * Order must be:
        *    - Position;
        *    - Velocity;
        *    - Density;
        *    - Boundary;
        */
        SerializedParticle particle;
        particle.position = vec3f(0);
        particle.velocity = vec3f(0);
        particle.normal = vec3f(0);
        particle.density = 0;
        particle.boundary = 0;
        if(flags & SERIALIZER_POSITION){ // get position
            ParseV3(&particle.position, &token);
            while(CAN_SKIP(token[0])) token++;
        }
        
        if(flags & SERIALIZER_VELOCITY){ // get velocity
            ParseV3(&particle.velocity, &token);
            while(CAN_SKIP(token[0])) token++;
        }
        
        if(flags & SERIALIZER_DENSITY){ // get density
            particle.density = ParseFloat(&token);
            while(CAN_SKIP(token[0])) token++;
        }
        
        if(flags & SERIALIZER_MASS){ // get mass
            particle.mass = ParseFloat(&token);
            while(CAN_SKIP(token[0])) token++;
        }
        
        if(flags & SERIALIZER_BOUNDARY){ // get boundary
            particle.boundary = (int)ParseFloat(&token);
            while(CAN_SKIP(token[0])) token++;
            if((flags & SERIALIZER_RULE_BOUNDARY_EXCLUSIVE)
               && !particle.boundary)
            {
                continue;
            }
        }
        
        if(flags & SERIALIZER_NORMAL){
            ParseV3(&particle.normal, &token); // get normal
            while(CAN_SKIP(token[0])) token++;
        }
        
        pSet->push_back(particle);
        size += 1;
    }

    if(!found_end){
        printf("Error: Unterminated fluid description, missing \'DataEnd\'\n");
        exit(0);
    }
    
    if(size != pCount) printf(" found %d\n", size);
    
    return size;
}

int SerializerLoadParticles3(std::vector<SerializedParticle> *pSet,
                             const char *filename, int &flags)
{
    int pCount = 0;
    std::string format;
    std::ifstream ifs(filename);
    if(!ifs){
        printf("Could not open file %s\n", filename);
        return -1;
    }

    pCount = SerializerFindFluidSection(ifs, format);
    if(format.size() > 0){
        flags = SerializerFlagsFromString(format.c_str());
    }

    return _SerializerLoadParticles3(pSet, ifs, flags, pCount);
}

int SerializerLoadSphDataSet3(ParticleSetBuilder3 *builder,
                              const char *filename, int &flags,
                              std::vector<int> *boundary)
{
    std::vector<SerializedParticle> particles;
    int pCount = SerializerLoadParticles3(&particles, filename, flags);
    
    for(int i = 0; i < pCount; i++){
        builder->AddParticle(particles[i].position, particles[i].velocity);
        if(boundary){
            boundary->push_back(particles[i].boundary);
        }
    }
    
    return pCount;
}

void SerializerLoadSystem3(ParticleSetBuilder3 *builder,
                           std::vector<SerializedShape> *shapes,
                           const char *filename, int &flags,
                           std::vector<int> *boundary)
{
    std::vector<SerializedParticle> particles;
    std::ifstream ifs(filename);
    if(!ifs){
        printf("Could not open file %s\n", filename);
    }

    while(ifs.peek() != -1){
        std::string token;
        int found = SerializerNextToken(ifs, token);
        if(!found){
            break;
        }

        if(token == "FluidBegin"){
            std::string format;
            int pCount = 0;
            pCount = SerializerFindFluidSection(ifs, format, 1);
            if(format.size() > 0){
                flags = SerializerFlagsFromString(format.c_str());
            }

            pCount = _SerializerLoadParticles3(&particles, ifs, flags, pCount);
            for(int i = 0; i < pCount; i++){
                builder->AddParticle(particles[i].position, particles[i].velocity);
                if(boundary){
                    boundary->push_back(particles[i].boundary);
                }
            }

            // go to FluidEnd
            int end = 0;
            while(ifs.peek() != -1){
                found = SerializerNextToken(ifs, token);
                if(!found) break;
                if(token == "FluidEnd"){
                    end = 1;
                    break;
                }
            }

            if(!end){
                printf("Error: Unterminated fluid description, missing \'FluidEnd\'\n");
                exit(0);
            }
        }else if(token == "ShapeBegin"){
            SerializedShape shp;
            _SerializerLoadShape(&shp, ifs);
            shapes->push_back(shp);
        }else{
            printf("Unknown token %s\n", token.c_str());
        }
    }
}

int SerializerLoadMany3(std::vector<vec3f> ***data, const char *basename, int &flags,
                        int start, int end, int legacy)
{
    // Get amount of particles in first
    std::string filename(basename);
    filename += std::to_string(start);
    filename += ".txt";
    std::ifstream ifs(filename);

    int pCount = 0;
    std::string format;
    if(ifs){
        *data = new std::vector<vec3f>*[end-start];
        if(!legacy){
            pCount = SerializerFindFluidSection(ifs, format);
        }
        ifs.close();

        for(int i = 0; i < end-start; i++){
            std::string file(basename);
            file += std::to_string(i + start);
            file += ".txt";
            (*data)[i] = new std::vector<vec3f>();
            if(!legacy){
                SerializerLoadPoints3((*data)[i], file.c_str(), flags);
            }else{
                SerializerLoadLegacySystem3((*data)[i], file.c_str(), flags);
            }
            printf("\rLoaded %d / %d", i, end-start);
            fflush(stdout);
            if(pCount < (*data)[i]->size()){
                pCount = (*data)[i]->size();
            }
        }
        printf("\rLoaded %d\n", end-start);
    }else{
        printf("Failed to open file %s\n", filename.c_str());
        exit(0);
    }
    
    return pCount;
}

template<typename SolverData = SphSolverData3, typename T>
void SaveSimulationDomain(SolverData *data, const char *filename){
    FILE *fp = fopen(filename, "a+");
    int dims = T(0).Dimensions();

    if(fp){
        auto grid = data->domain;
        unsigned int count = grid->GetCellCount();
        fprintf(fp, "%u\n", count);
        for(unsigned int i = 0; i < count; i++){
            auto cell = grid->GetCell(i);
            T center = cell->bounds.Center();
            vec3f pi;
            if(dims == 2){
                pi = vec3f(center[0], center[1], 0);
            }else{
                pi = vec3f(center[0], center[1], center[2]);
            }

            PrintToFile(fp, pi);
            int n = cell->GetChainLength();

            PrintToFile(fp, n, 1);
            fprintf(fp, "\n");
        }
        fclose(fp);
    }
}

template<typename SolverData = SphSolverData3, typename ParticleSet = ParticleSet3,
typename Domain = Grid3, typename T>
void SaveSphParticleSetLegacy(SolverData *data, const char *filename, int flags,
                              std::vector<int> *boundary = nullptr)
{
    ParticleSet *pSet = data->sphpSet->GetParticleSet();
    FILE *fp = fopen(filename, "a+");
    std::string format = SerializerStringFromFlags(flags);
    int logged = 0;
    int dims = T(0).Dimensions();
    if(flags & SERIALIZER_XYZ){ // cleanup
        if(flags & SERIALIZER_RULE_BOUNDARY_EXCLUSIVE){
            flags = SERIALIZER_POSITION | SERIALIZER_XYZ |
                    SERIALIZER_RULE_BOUNDARY_EXCLUSIVE;
        }else{
            flags = SERIALIZER_POSITION | SERIALIZER_XYZ;
        }
    }

    if(fp){
        int pCount = pSet->GetParticleCount();
        if((flags & SERIALIZER_RULE_BOUNDARY_EXCLUSIVE) && boundary){
            pCount = 0;
            for(int i = 0; i < boundary->size(); i++){
                pCount += boundary->at(i) > 0 ? 1 : 0;
            }
        }else if((flags & SERIALIZER_RULE_BOUNDARY_EXCLUSIVE) && !boundary){
            printf("Invalid configuration for Serialized particles\n");
            return;
        }

        if(!(flags & SERIALIZER_XYZ)){
            fprintf(fp, "%d\n", pCount);
        }

        for(int i = 0; i < pSet->GetParticleCount(); i++){
            int needs_space = 0;
            int boundary_value = 0;
            T pi = pSet->GetParticlePosition(i);

            if(boundary) boundary_value = boundary->at(i);

            if((flags & SERIALIZER_RULE_BOUNDARY_EXCLUSIVE) && !boundary_value){
                continue;
            }

            if(flags & SERIALIZER_POSITION){
                if(dims == 2){
                    vec3f ps(pi.x, pi.y, 0);
                    PrintToFile(fp, ps);
                }else{
                    PrintToFile(fp, pi);
                }
                needs_space = 1;
            }

            if(flags & SERIALIZER_VELOCITY){
                T vi = pSet->GetParticleVelocity(i);
                PrintToFile(fp, vi, needs_space);
                needs_space = 1;
            }

            if(flags & SERIALIZER_DENSITY){
                Float di = pSet->GetParticleDensity(i);
                PrintToFile(fp, di, needs_space);
                needs_space = 1;
            }

            if(flags & SERIALIZER_MASS){
                Float mi = pSet->GetMass();
                PrintToFile(fp, mi, needs_space);
                needs_space = 1;
            }

            if(flags & SERIALIZER_BOUNDARY){
                if(!boundary && !logged){
                    printf("Warning: Not a valid boundary given\n");
                    logged = 1;
                }else if(boundary){
                    PrintToFile(fp, boundary->at(i), needs_space);
                    needs_space = 1;
                }
            }

            if(flags & SERIALIZER_NORMAL){
                T ni = pSet->GetParticleNormal(i);
                PrintToFile(fp, ni, needs_space);
                needs_space = 1;
            }

            fprintf(fp, "\n");
        }

        fclose(fp);
    }else{
        printf("Error: Failed to open %s\n", filename);
    }
}

template<typename ParticleSet = ParticleSet3, typename T>
void PushParticleSetToFile(ParticleSet *pSet, const char *filename, int flags,
                           FILE *fp, std::vector<int> *boundary = nullptr)
{
    if(fp){
        int logged = 0;
        int pCount = pSet->GetParticleCount();
        if((flags & SERIALIZER_RULE_BOUNDARY_EXCLUSIVE) && boundary){
            pCount = 0;
            for(int i = 0; i < boundary->size(); i++){
                pCount += boundary->at(i) > 0 ? 1 : 0;
            }
        }else if((flags & SERIALIZER_RULE_BOUNDARY_EXCLUSIVE) && !boundary){
            printf("Invalid configuration for Serialized particles\n");
            return;
        }
        for(int i = 0; i < pSet->GetParticleCount(); i++){
            int needs_space = 0;
            int boundary_value = 0;
            T pi = pSet->GetParticlePosition(i);

            if(boundary) boundary_value = boundary->at(i);

            if((flags & SERIALIZER_RULE_BOUNDARY_EXCLUSIVE) && !boundary_value){
                continue;
            }

            fprintf(fp, "\t\t");
            if(flags & SERIALIZER_POSITION){
                PrintToFile(fp, pi);
                needs_space = 1;
            }

            if(flags & SERIALIZER_VELOCITY){
                T vi = pSet->GetParticleVelocity(i);
                PrintToFile(fp, vi, needs_space);
                needs_space = 1;
            }

            if(flags & SERIALIZER_DENSITY){
                Float di = pSet->GetParticleDensity(i);
                PrintToFile(fp, di, needs_space);
                needs_space = 1;
            }

            if(flags & SERIALIZER_MASS){
                Float mi = pSet->GetMass();
                PrintToFile(fp, mi, needs_space);
                needs_space = 1;
            }

            if(flags & SERIALIZER_BOUNDARY){
                if(!boundary && !logged){
                    printf("Warning: Not a valid boundary given\n");
                    logged = 1;
                }else if(boundary){
                    PrintToFile(fp, boundary->at(i), needs_space);
                    needs_space = 1;
                }
            }

            if(flags & SERIALIZER_NORMAL){
                T ni = pSet->GetParticleNormal(i);
                PrintToFile(fp, ni, needs_space);
                needs_space = 1;
            }

            fprintf(fp, "\n");
        }
    }
}

template<typename SolverData = SphSolverData3, typename ParticleSet = ParticleSet3, 
typename Domain = Grid3, typename T>
void SaveSphParticleSet(SolverData *data, const char *filename, int flags,
                        std::vector<int> *boundary = nullptr)
{
    ParticleSet *pSet = data->sphpSet->GetParticleSet();
    Float spacing = data->sphpSet->GetTargetSpacing();
    FILE *fp = fopen(filename, "a+");
    std::string format = SerializerStringFromFlags(flags);
    if(fp){
        int pCount = pSet->GetParticleCount();
        if((flags & SERIALIZER_RULE_BOUNDARY_EXCLUSIVE) && boundary){
            pCount = 0;
            for(int i = 0; i < boundary->size(); i++){
                pCount += boundary->at(i) > 0 ? 1 : 0;
            }
        }else if((flags & SERIALIZER_RULE_BOUNDARY_EXCLUSIVE) && !boundary){
            printf("Invalid configuration for Serialized particles\n");
            return;
        }

        fprintf(fp, "FluidBegin\n");
        fprintf(fp, "\t\"Type\" particles\n");
        fprintf(fp, "\t\"Count\" %d\n", pCount);
        fprintf(fp, "\t\"Format\" %s\n", format.c_str());
        fprintf(fp, "\t\"Spacing\" %g\n", spacing);
        fprintf(fp, "\tDataBegin\n");

        PushParticleSetToFile<ParticleSet, T>(pSet, filename, flags, fp, boundary);

        fprintf(fp, "\tDataEnd\n");
        fprintf(fp, "FluidEnd\n");
        fclose(fp);
    }else{
        printf("Error: Failed to open %s\n", filename);
    }
}

template<typename SolverData = SphSolverData3, typename ParticleSet = ParticleSet3,
typename Domain = Grid3, typename T>
void SaveSphParticleSetMany(SolverData *data, const char *filename, int flags,
                            std::vector<ParticleSet *> pSets)
{
    ParticleSet *pSet = data->sphpSet->GetParticleSet();
    Float spacing = data->sphpSet->GetTargetSpacing();
    std::string format = SerializerStringFromFlags(flags);
    if(flags & SERIALIZER_RULE_BOUNDARY_EXCLUSIVE){
        printf("Data set concatenation does not support boundary information\n");
        return;
    }

    FILE *fp = fopen(filename, "a+");

    if(fp){
        int pCount = 0;
        for(ParticleSet *set : pSets){
            pCount += set->GetParticleCount();
        }

        fprintf(fp, "FluidBegin\n");
        fprintf(fp, "\t\"Type\" particles\n");
        fprintf(fp, "\t\"Count\" %d\n", pCount);
        fprintf(fp, "\t\"Format\" %s\n", format.c_str());
        fprintf(fp, "\t\"Spacing\" %g\n", spacing);
        fprintf(fp, "\tDataBegin\n");

        for(ParticleSet *set : pSets){
            PushParticleSetToFile<ParticleSet, T>(set, filename, flags, fp, nullptr);
        }

        fprintf(fp, "\tDataEnd\n");
        fprintf(fp, "FluidEnd\n");
        fclose(fp);
    }else{
        printf("Error: Failed to open %s\n", filename);
    }
}

void SerializerWriteShapes(std::vector<SerializedShape> *shapes, const char *filename){
    FILE *fp = fopen(filename, "a+");
    if(!fp){
        printf("Failed to open file %s\n", filename);
        exit(0);
    }

    for(SerializedShape sh : *shapes){
        fprintf(fp, "ShapeBegin\n");
        fprintf(fp, "\t\"Type\" %s\n", SerializerGetShapeName(sh.type));
        for(auto it = sh.numParameters.begin(); it != sh.numParameters.end(); it++){
            vec4f v = it->second;
            fprintf(fp, "\t\"%s\" %g %g %g\n", it->first.c_str(), v.x, v.y, v.z);
        }

        for(auto it = sh.strParameters.begin();
            it != sh.strParameters.end(); it++)
        {
            std::string name  = it->first;
            std::string value = it->second;
            fprintf(fp, "\t\"%s\" %s\n", name.c_str(), value.c_str());
        }

        for(auto it = sh.transfParameters.begin();
            it != sh.transfParameters.end(); it++)
        {
            Transform t = it->second;
            fprintf(fp, "\t\"%s\" ", it->first.c_str());
            for(int i = 0; i < 4; i++){
                for(int j = 0; j < 4; j++){
                    if(i == 3 && j == 3){
                        fprintf(fp, "%g", t.m.m[i][j]);
                    }else{
                        fprintf(fp, "%g ", t.m.m[i][j]);
                    }
                }
            }

            fprintf(fp, "\n");
        }

        fprintf(fp, "ShapeEnd\n");
    }

    fclose(fp);
}

void SerializerSaveDomain(SphSolverData3 *pSet, const char *filename){
    SaveSimulationDomain<SphSolverData3, vec3f>(pSet, filename);
}

void SerializerSaveDomain(SphSolverData2 *pSet, const char *filename){
    SaveSimulationDomain<SphSolverData2, vec2f>(pSet, filename);
}

void SerializerSaveSphDataSet3Legacy(SphSolverData3 *pSet, const char *filename,
                                     int flags, std::vector<int> *boundary)
{
    SaveSphParticleSetLegacy<SphSolverData3, ParticleSet3, Grid3, vec3f>(pSet, filename,
                                                                         flags, boundary);
}

void SerializerSaveSphDataSet3(SphSolverData3 *pSet, const char *filename,
                               int flags, std::vector<int> *boundary)
{
    SaveSphParticleSet<SphSolverData3, ParticleSet3, Grid3, vec3f>(pSet, filename, 
                                                                   flags, boundary);
}

void SerializerSaveSphDataSet3Many(SphSolverData3 *data,
                                   std::vector<ParticleSet3 *> pSets,
                                   const char *filename, int flags)
{
    SaveSphParticleSetMany<SphSolverData3, ParticleSet3, Grid3, vec3f>(data, filename,
                                                                       flags, pSets);
}

void SerializerSaveSphDataSet2Legacy(SphSolverData2 *pSet, const char *filename,
                                     int flags, std::vector<int> *boundary)
{
    SaveSphParticleSetLegacy<SphSolverData2, ParticleSet2, Grid2, vec2f>(pSet, filename,
                                                                         flags, boundary);
}

void SerializerSaveSphDataSet2(SphSolverData2 *pSet, const char *filename, 
                               int flags, std::vector<int> *boundary)
{
    SaveSphParticleSet<SphSolverData2, ParticleSet2, Grid2, vec2f>(pSet, filename, 
                                                                   flags, boundary);
}
