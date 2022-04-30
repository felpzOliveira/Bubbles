#include <explicit_grid.h>

/*
* TODO: Implement methods here
*/

__bidevice__ ExplicitGrid2::ExplicitGrid2(){}
__bidevice__ vec2ui ExplicitGrid2::GetResolution(){ return resolution; }
__bidevice__ vec2f ExplicitGrid2::GetSpacing(){ return spacing; }
__bidevice__ Bounds2f ExplicitGrid2::GetBounds(){ return bounds; }

__bidevice__ vec2f ExplicitGrid2::GetCellCenteredPosition(const size_t &i, const size_t &j){
    return origin + spacing * vec2f((Float)i + 0.5, (Float)j + 0.5);
}

__bidevice__ vec2f ExplicitGrid2::GetVertexCenteredPosition(const size_t &i,
                                                            const size_t &j)
{
    return origin + spacing * vec2f((Float)i, (Float)j);
}

__bidevice__ void ExplicitGrid2::Set(const vec2ui &res, const vec2f &h, const vec2f &o){
    resolution = res;
    origin = o;
    spacing = h;
    invSpacing = vec2f(1.0 / h.x, 1.0 / h.y);
    invSpacing2 = vec2f(invSpacing.x * invSpacing.x, invSpacing.y * invSpacing.y);
    bounds = Bounds2f(origin, origin + spacing * vec2f((Float)res.x, (Float)res.y));
}

