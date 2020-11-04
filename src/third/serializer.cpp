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

void SerializerLoadPoints3(std::vector<vec3f> *points,
                           const char *filename, int flags)
{
    std::ifstream ifs(filename);
    if(ifs){
        int start = 1;
        int pCount = 0;
        std::string linebuf;
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
                vec3f p;
                ParseV3(&p, &token);
                while(CAN_SKIP(token[0])) token++;
                points->push_back(p);
            }
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
        particle.density = 0;
        particle.boundary = 1;
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
        
        if(flags & SERIALIZER_BOUNDARY){ // get boundary
            particle.boundary = (int)ParseFloat(&token);
            while(CAN_SKIP(token[0])) token++;
        }
        
        pSet->push_back(particle);
        size += 1;
    }
    
    if(size == pCount) printf("OK\n");
    else printf(" found %d\n", size);
    
    return size;
}

//TODO: I'm feeling lazy so let fscanf rule this, but change sometime
int SerializerLoadSphDataSet3(ParticleSetBuilder3 *builder,
                              const char *filename, int flags)
{
    int pCount = 0;
    FILE *fp = fopen(filename, "r");
    if(fp){
        Float x, y, z;
        fscanf(fp, "%d\n", &pCount);
        for(int i = 0; i < pCount; i++){
            if(flags & SERIALIZER_POSITION){
                fscanf(fp, "%g %g %g", &x, &y, &z);
                builder->AddParticle(vec3f(x,y,z));
            }
            
            fscanf(fp, "\n");
        }
        
        fclose(fp);
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
void SaveSphParticleSet(SolverData *data, const char *filename, int flags){
    ParticleSet *pSet = data->sphpSet->GetParticleSet();
    Domain *grid = data->domain;
    FILE *fp = fopen(filename, "w+");
    int logged = 0;
    if(fp){
        int pCount = pSet->GetParticleCount();
        fprintf(fp, "%d\n", pCount);
        for(int i = 0; i < pCount; i++){
            int needs_space = 0;
            T pi = pSet->GetParticlePosition(i);
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
            
            if(flags & SERIALIZER_BOUNDARY){
                if(!grid){
                    if(!logged){
                        printf("Warning: Not a valid grid given\n");
                        logged = 1;
                    }
                }else{
                    unsigned int cellId = grid->GetLinearHashedPosition(pi);
                    int level = grid->GetCellLevel(cellId);
                    PrintToFile(fp, level, needs_space);
                    needs_space = 1;
                }
            }
            
            fprintf(fp, "\n");
        }
        
        fclose(fp);
    }else{
        printf("Error: Failed to open %s\n", filename);
    }
}

void SerializerSaveSphDataSet3(SphSolverData3 *pSet, const char *filename, int flags){
    SaveSphParticleSet<SphSolverData3, ParticleSet3, Grid3, vec3f>(pSet, filename, flags);
}

void SerializerSaveSphDataSet2(SphSolverData2 *pSet, const char *filename, int flags){
    SaveSphParticleSet<SphSolverData2, ParticleSet2, Grid2, vec2f>(pSet, filename, flags);
}