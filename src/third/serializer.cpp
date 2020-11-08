#include <serializer.h>
#include <obj_loader.h>
#include <string.h>

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

int SerializerFlagsFromString(const char *spec){
    int flags = 0;
    std::string str(spec);
    for(int i = 0; i < str.size(); i++){
        char c = spec[i];
        if(c == 'p' || c == 'P') flags |= SERIALIZER_POSITION;
        else if(c == 'd' || c == 'D') flags |= SERIALIZER_DENSITY;
        else if(c == 'n' || c == 'N') flags |= SERIALIZER_NORMAL;
        else if(c == 'b' || c == 'B') flags |= SERIALIZER_BOUNDARY;
        else if(c == 'v' || c == 'V') flags |= SERIALIZER_VELOCITY;
        else if(c == 'm' || c == 'M') flags |= SERIALIZER_MASS;
        else if(c == 'o' || c == 'O') flags |= SERIALIZER_RULE_BOUNDARY_EXCLUSIVE;
        else{
            printf("Unknown flag argument %c\n", c);
            return -1;
        }
    }
    
    return flags;
}

void SerializerLoadPoints3(std::vector<vec3f> *points,
                           const char *filename, int flags)
{
    std::ifstream ifs(filename);
    if(ifs){
        int start = 1;
        int pCount = 0;
        std::string linebuf;
        vec3f vel(0), nor(0);
        int boundary = 0;
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
            
            if(start){ // particle count
                pCount = (int)ParseFloat(&token);
                start = 0;
                if(pCount <= 0){
                    printf("Invalid particle count %d\n", pCount);
                    return;
                }
                
                continue;
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
}

int SerializerLoadParticles3(std::vector<SerializedParticle> *pSet, 
                             const char *filename, int flags)
{
    std::string linebuf;
    int start = 1;
    int pCount = 0;
    std::ifstream ifs(filename);
    if(!ifs){
        printf("Could not open file %s\n", filename);
        return -1;
    }
    
    int size = 0;
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
        
        if(start){ // particle count
            pCount = (int)ParseFloat(&token);
            start = 0;
            if(pCount > 0){
                printf("Attempting to parse %d particles... ", pCount);
                pSet->clear();
                pSet->reserve(pCount);
            }else{
                printf("Invalid particle count %d\n", pCount);
                return -1;
            }
            
            continue;
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
    
    if(size == pCount) printf("OK\n");
    else printf(" found %d\n", size);
    
    return size;
}

int SerializerLoadSphDataSet3(ParticleSetBuilder3 *builder,
                              const char *filename, int flags,
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

int SerializerLoadMany3(std::vector<vec3f> ***data, const char *basename, int flags,
                        int start, int end)
{
    // Get amount of particles in first
    std::string filename(basename);
    filename += std::to_string(start);
    filename += ".txt";
    FILE *fp = fopen(filename.c_str(), "r");
    int pCount = 0;
    if(fp){
        *data = new std::vector<vec3f>*[end-start];
        fscanf(fp, "%d\n", &pCount);
        fclose(fp);
        
        for(int i = 0; i < end-start; i++){
            std::string file(basename);
            file += std::to_string(i + start);
            file += ".txt";
            (*data)[i] = new std::vector<vec3f>();
            SerializerLoadPoints3((*data)[i], file.c_str(), flags);
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

template<typename SolverData = SphSolverData3, typename ParticleSet = ParticleSet3, 
typename Domain = Grid3, typename T>
void SaveSphParticleSet(SolverData *data, const char *filename, int flags,
                        std::vector<int> *boundary = nullptr)
{
    ParticleSet *pSet = data->sphpSet->GetParticleSet();
    FILE *fp = fopen(filename, "w+");
    int logged = 0;
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
        
        fprintf(fp, "%d\n", pCount);
        for(int i = 0; i < pSet->GetParticleCount(); i++){
            int needs_space = 0;
            int boundary_value = 0;
            T pi = pSet->GetParticlePosition(i);
            
            if(boundary) boundary_value = boundary->at(i);
            
            if((flags & SERIALIZER_RULE_BOUNDARY_EXCLUSIVE) && !boundary_value){
                continue;
            }
            
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
        
        fclose(fp);
    }else{
        printf("Error: Failed to open %s\n", filename);
    }
}

void SerializerSaveSphDataSet3(SphSolverData3 *pSet, const char *filename, 
                               int flags, std::vector<int> *boundary)
{
    SaveSphParticleSet<SphSolverData3, ParticleSet3, Grid3, vec3f>(pSet, filename, 
                                                                   flags, boundary);
}

void SerializerSaveSphDataSet2(SphSolverData2 *pSet, const char *filename, 
                               int flags, std::vector<int> *boundary)
{
    SaveSphParticleSet<SphSolverData2, ParticleSet2, Grid2, vec2f>(pSet, filename, 
                                                                   flags, boundary);
}