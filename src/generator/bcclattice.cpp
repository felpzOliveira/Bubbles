#include <point_generator.h>

__host__ BccLatticePointGenerator::BccLatticePointGenerator() : PointGenerator3(){}

__host__ void
BccLatticePointGenerator::ForEach(const Bounds3f &domain, Float spacing,
                                  const std::function<bool(const vec3f &)> &callback) const
{
    Float halfSpacing = spacing / 2;
    Float width  = domain.ExtentOn(0);
    Float height = domain.ExtentOn(1);
    Float depth  = domain.ExtentOn(2);
    
    bool hasOff = false;
    bool shouldStop = false;
    
    vec3f pos;
    for(int k = 0; k * halfSpacing <= depth && !shouldStop; k++){
        pos.z = k * halfSpacing + domain.pMin.z;
        Float offset = hasOff ? halfSpacing : 0;
        
        for(int j = 0; j * spacing + offset <= height && !shouldStop; j++){
            pos.y = j * spacing + offset + domain.pMin.y;
            
            for(int i = 0; i * spacing + offset <= width && !shouldStop; i++){
                pos.x = i * spacing + offset + domain.pMin.x;
                if(!callback(pos)){
                    shouldStop = true;
                    break;
                }
            }
        }
        
        hasOff = !hasOff;
    }
}

__bidevice__ BccLatticePointGeneratorDevice::BccLatticePointGeneratorDevice(){}

__bidevice__ int BccLatticePointGeneratorDevice::Generate(const Bounds3f &domain, 
                                                          Float spacing, vec3f *points, 
                                                          int maxn)
{
    Float halfSpacing = spacing / 2;
    Float width  = domain.ExtentOn(0);
    Float height = domain.ExtentOn(1);
    Float depth  = domain.ExtentOn(2);
    
    bool hasOff = false;
    bool shouldStop = false;
    
    vec3f pos;
    int pi = 0;
    for(int k = 0; k * halfSpacing <= depth && !shouldStop; k++){
        pos.z = k * halfSpacing + domain.pMin.z;
        Float offset = hasOff ? halfSpacing : 0;
        
        for(int j = 0; j * spacing + offset <= height && !shouldStop; j++){
            pos.y = j * spacing + offset + domain.pMin.y;
            
            for(int i = 0; i * spacing + offset <= width && !shouldStop; i++){
                pos.x = i * spacing + offset + domain.pMin.x;
                AssertA(pi < maxn, "Not enough memory for BCCLattice point generation");
                points[pi++] = pos;
            }
        }
        
        hasOff = !hasOff;
    }
    
    return pi;
}