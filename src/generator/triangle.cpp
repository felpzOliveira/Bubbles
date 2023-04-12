#include <point_generator.h>

__host__ TrianglePointGenerator::TrianglePointGenerator() : PointGenerator2(){}

__host__ void
TrianglePointGenerator::ForEach(const Bounds2f &domain, Float spacing,
                                const std::function<bool(const vec2f &)> &callback) const
{
    Float halfSpacing = spacing / 2;
    Float ySpacing = spacing * std::sqrt(3.0) / 2.0;
    Float bWidth = domain.ExtentOn(0);
    Float bHeight = domain.ExtentOn(1);
    vec2f position;

    bool hasOff = false;
    bool shouldStop = false;

    for(int j = 0; j * ySpacing <= bHeight && !shouldStop; j++){
        position.y = j * ySpacing + domain.pMin.y;
        Float offset = hasOff ? halfSpacing : 0;

        for(int i = 0; i * spacing + offset <= bWidth && !shouldStop; i++){
            position.x = i * spacing + offset + domain.pMin.x;
            if(!callback(position)){
                shouldStop = true;
                break;
            }
        }

        hasOff = !hasOff;
    }
}


__bidevice__ TrianglePointGeneratorDevice::TrianglePointGeneratorDevice(){}

__bidevice__ int TrianglePointGeneratorDevice::Generate(const Bounds2f &domain, Float spacing,
                                                        vec2f *points, int maxn)
{
    Float halfSpacing = spacing / 2;
    Float ySpacing = spacing * std::sqrt(3.0) / 2.0;
    Float bWidth = domain.ExtentOn(0);
    Float bHeight = domain.ExtentOn(1);
    vec2f position;

    bool hasOff = false;
    bool shouldStop = false;

    int pi = 0;
    for(int j = 0; j * ySpacing <= bHeight && !shouldStop; j++){
        position.y = j * ySpacing + domain.pMin.y;
        Float offset = hasOff ? halfSpacing : 0;

        for(int i = 0; i * spacing + offset <= bWidth && !shouldStop; i++){
            position.x = i * spacing + offset + domain.pMin.x;
            AssertA(pi < maxn, "Not enough memory for triangular point generation");
            points[pi++] = position;
        }

        hasOff = !hasOff;
    }

    return pi;
}
