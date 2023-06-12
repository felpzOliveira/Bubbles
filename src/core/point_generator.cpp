#include <point_generator.h>

PointGenerator2::PointGenerator2(){}

void PointGenerator2::Generate(const Bounds2f &domain, Float spacing,
                               std::vector<vec2f> *points) const
{
    auto insert = [&points](const vec2f &point) -> bool{
        points->push_back(point);
        return true;
    };

    ForEach(domain, spacing, insert);
}

PointGenerator3::PointGenerator3(){}

void PointGenerator3::Generate(const Bounds3f &domain, Float spacing,
                               std::vector<vec3f> *points) const
{
    auto insert = [&points](const vec3f &point) -> bool{
        points->push_back(point);
        return true;
    };

    ForEach(domain, spacing, insert);
}
