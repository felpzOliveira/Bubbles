/* date = May 19th 2021 15:25 */
#pragma once
#include <grid.h>
#include <cutil.h>
#include <bound_util.h>

/**************************************************************/
//               I N T E R V A L   M E T H O D                //
//                  Geometrical intersection                  //
/**************************************************************/

/*
* Implementation of Sandim's new interval method for boundary detection in:
* Simple and reliable boundary detection for meshfree particle methods
*                       using interval analysis
*
* Sandim uses recursive methods for classification, since we are running
* on GPU I'll implement a simple GPU stack (not very efficient tho)
* and unroll the recursion to be stack-based.
*/

#define INTERVAL_LABEL_COVERED 0
#define INTERVAL_LABEL_UNCOVERED 1
#define INTERVAL_LABEL_PARTIALLY_COVERED 2

#define INTERVAL_DEBUG

typedef enum{
    PolygonSubdivision,
    BoundingBoxSubdivision,
}SubdivisionMethod;

template<typename T>
class Stack{
    public:
    T data[256];
    unsigned int _top;
    unsigned int capacity;
    unsigned int size;
    __bidevice__ Stack(){
        _top = 0;
        capacity = 256;
        size = 0;
    }

    __bidevice__ void push(T item){
        if(!(_top < capacity)){
            printf("Not enough space\n");
            return;
        }

        memcpy(&data[_top++], &item, sizeof(T));
        size += 1;
    }

    __bidevice__ bool empty(){ return size < 1; }

    __bidevice__ void pop(){
        if(size > 0 && _top > 0){
            _top--;
            size--;
        }
    }

    __bidevice__ void top(T *item){
        if(size > 0 && _top > 0){
            memcpy(item, &data[_top-1], sizeof(T));
        }
    }

    __bidevice__ ~Stack(){}
};

/*
* Geometry representation for each method described in the paper.
*/

// The bouding box one using interval arithmetic:
template<typename T, typename Q, unsigned int dims>
class BBSubdivisionGeometry{
    public:
    BBSubdivisionGeometry() = default;
    ~BBSubdivisionGeometry() = default;

    __bidevice__ vec2f ComputeInterval(T pj, Float h){
        auto square = [](Float tmin, Float tmax, Float t) -> vec2f{
            vec2f f;
            tmin = tmin - t;
            tmax = tmax - t;
            Float tmax2 = tmax * tmax;
            Float tmin2 = tmin * tmin;
            if(tmin >= 0){
                f.x = tmin2;
                f.y = tmax2;
            }else if(tmax <= 0){
                f.x = tmax2;
                f.y = tmin2;
            }else{
                f.x = 0;
                f.y = Max(tmin2, tmax2);
            }

            return f;
        };

        vec2f vals[3];
        T pmin = bounds.pMin;
        T pmax = bounds.pMax;
        for(int i = 0; i < dims; i++){
            vals[i] = square(pmin[i], pmax[i], pj[i]);
        }

        Float rmin = 0, rmax = 0;
        for(int i = 0; i < dims; i++){
            rmin += vals[i].x;
            rmax += vals[i].y;
        }

        return vec2f(rmin - h * h, rmax - h * h);
    }

    __bidevice__ int CountVerticesInside(T p, Float h, int *max_query){
        int inside = 1;
        vec2f ival = ComputeInterval(p, h);

        if(ival.y <= 0){
            inside = 2;
        }else if(ival.x > 0){
            inside = 0;
        }

        *max_query = 2;
        return inside;
    }

    /* External routines for the solver */
    __bidevice__ void Subdivide(BBSubdivisionGeometry *slabs,
                                int *nSlabs, T pi, Float h)
    {
        (void)pi;
        (void)h;
        Q inner[8];
        int s = SplitBounds(bounds, &inner[0]);
        for(int i = 0; i < s; i++){
            slabs[i].bounds = inner[i];
        }

        *nSlabs = s;
    }

    __bidevice__ int BallIntersectionCount(T pi, T pj, Float h, int *max_query){
        return CountVerticesInside(pj, h, max_query);
    }

    __bidevice__ bool IsGeometryClipped(T pi, Float h){
        int m_query = 0;
        int inside = CountVerticesInside(pi, h, &m_query);
        return inside == m_query;
    }

    static __bidevice__
    void SlabsForParticle(T p, Float h, BBSubdivisionGeometry *slabs, int *nSlabs){
        slabs[0].bounds = Q(p - T(h), p + T(h));
        *nSlabs = 1;
    }

    Q bounds;
};

// The method using 2D n-gon division
class PolygonSubdivisionGeometry{
    public:
    PolygonSubdivisionGeometry() = default;
    ~PolygonSubdivisionGeometry() = default;

    __bidevice__ void Subdivide(PolygonSubdivisionGeometry *slabs,
                                int *nSlabs, vec2f pi, Float h)
    {
        // We need to subdivide a segment that is a quad. The main idea is to generate
        // the (2n)-gon, luckly we can avoid doing that because generating the (2n)
        // vertices will be the same as splitting the segment in the middle and generating
        // 2 new slabs. This also avoids the need to subdivide all segments, we only
        // subdivide the relevant segments, allowing for faster (?) tests.

        // Compute the midpoint and relative angles
        Float tn = (t0 + t1) * 0.5; // half angle between p0 and p1
        Float tm = (t1 + tn) * 0.5; // 1/4 angle between p0 and p1
        Float tk = (t0 + tn) * 0.5; // 3/4 angle between p0 and p1

        // offset from p0 and p1 the rect must be constructed
        Float theta = Absf(t1-t0) * 0.25;
        Float dh = h * (1.f - cos(Radians(theta)));

        // actual relevant points
        Float an = Radians(tn);
        Float am = Radians(tm);
        Float ak = Radians(tk);
        vec2f pn = vec2f(cos(an), sin(an)) * h + pi; // midpoint for p0 and p1
        vec2f pk = vec2f(cos(ak), sin(ak)) * h + pi; // midpoint for pn and p0
        vec2f pm = vec2f(cos(am), sin(am)) * h + pi; // midpoint for pn and p1

        // vectors for offset
        vec2f dv = Normalize(pk - pi);
        vec2f ds = Normalize(pm - pi);

        // other points from the rect that are outside the circle
        vec2f pfu = p0 + dv * dh;
        vec2f pfd = pn + dv * dh;

        vec2f puu = pn + ds * dh;
        vec2f pud = p1 + ds * dh;

        // finally, the two new slabs
        slabs[0].p0 = p0;
        slabs[0].p1 = pn;
        slabs[0].p2 = pfd;
        slabs[0].p3 = pfu;
        slabs[0].t0 = t0;
        slabs[0].t1 = tn;

        slabs[1].p0 = pn;
        slabs[1].p1 = p1;
        slabs[1].p2 = pud;
        slabs[1].p3 = puu;
        slabs[1].t0 = tn;
        slabs[1].t1 = t1;
        *nSlabs = 2;
    }

    __bidevice__ bool IsGeometryClipped(vec2f pi, Float h){
        return false;
    }

    __bidevice__ int BallIntersectionCount(vec2f pi, vec2f pj, Float h, int *max_query){
        int inside = 0;
        inside += (Distance(p0, pj) < h ? 1 : 0);
        inside += (Distance(p1, pj) < h ? 1 : 0);
        inside += (Distance(p2, pj) < h ? 1 : 0);
        inside += (Distance(p3, pj) < h ? 1 : 0);
        *max_query = 4;
        return inside;
    }

    static __bidevice__
    void SlabsForParticle(vec2f p, Float h, PolygonSubdivisionGeometry *slabs, int *nSlabs){
        // In the 2D circle we have P(t) = r(cos(t), sin(t)) + P
        // we take the first point at 90 so we get P0 = (0, r) + P
        // rotating this point 60 degrees we find that:
        // P1 = r(cos(210), sin(210)) + P = (-sqrt(3)/2 r, -r/2) + P
        // P2 = r(cos(-30), sin(-30)) + P = (sqrt(3)/2 r, -r/2) + P
        // For each slab we compute the midpoint between each pair of
        // vertices and offset by the vector d between P and midpoint:
        // M0 = r(cos(150), sin(150)) + P = (-sqrt(3)/2 r, r/2) + P
        // M1 = r(cos(270), sin(270)) + P = (0, -r) + P
        // M2 = r(cos(30), sin(30)) + P = (sqrt(3)/2 r, r/2) + P
        const Float h_sqrt3 = 0.866025404; // sqrt(3) / 2
        const Float half = 0.5f;
        const Float half_h = 0.5 * h;

        //        p0
        //       /  \
        //    M0/    \M2
        //     /      \
        //    /________\
        //    p1  M1   p2

        const vec2f vertices[] = {
            p + vec2f(0, 1) * h,
            p + vec2f(-h_sqrt3, -half) * h,
            p + vec2f(h_sqrt3, -half) * h
        };

        // the result of Normalize(Mn - Pn)
        const vec2f directions[] = {
            vec2f(-h_sqrt3, half),
            vec2f(0, -1),
            vec2f(h_sqrt3, half),
        };

        // we are also going to store the base angle so that we can
        // easily subdivide the polygon
        slabs[0].p0 = vertices[0];
        slabs[0].p1 = vertices[1];
        slabs[0].p2 = vertices[1] + directions[0] * half_h;
        slabs[0].p3 = vertices[0] + directions[0] * half_h;
        slabs[0].t0 = 90;
        slabs[0].t1 = 210;

        slabs[1].p0 = vertices[1];
        slabs[1].p1 = vertices[2];
        slabs[1].p2 = vertices[2] + directions[1] * half_h;
        slabs[1].p3 = vertices[1] + directions[1] * half_h;
        slabs[1].t0 = 210;
        slabs[1].t1 = 330;

        slabs[2].p0 = vertices[0];
        slabs[2].p1 = vertices[2];
        slabs[2].p2 = vertices[2] + directions[2] * half_h;
        slabs[2].p3 = vertices[0] + directions[2] * half_h;
        slabs[2].t0 = 90;
        slabs[2].t1 = -30;

        *nSlabs = 3;
    }

    vec2f p0, p1, p2, p3;
    Float t0, t1;
};

// The method using 3D tetrahedron
class VolumetricSubdivisionGeometry{
    public:
    VolumetricSubdivisionGeometry() = default;
    ~VolumetricSubdivisionGeometry() = default;

    __bidevice__ int BallIntersectionCount(vec3f pi, vec3f pj, Float h, int *max_query){
        // The only difference from previous implementations is that since the
        // testing particle is located at the origin we need to shift the neighboor
        // particle to the testing particle's coordinates, i.e.: (pj - pi) / h.
        int inside = 0;
        pj = (pj - pi) / h;
        inside += (Distance(p0, pj) < 1 ? 1 : 0);
        inside += (Distance(p1, pj) < 1 ? 1 : 0);
        inside += (Distance(p2, pj) < 1 ? 1 : 0);
        inside += (Distance(a, pj) < 1 ? 1 : 0);
        inside += (Distance(b, pj) < 1 ? 1 : 0);
        inside += (Distance(c, pj) < 1 ? 1 : 0);
        *max_query = 6;
        return inside;
    }

    __bidevice__ bool IsGeometryClipped(vec3f pi, Float h){
        return false;
    }

    __bidevice__ void Subdivide(VolumetricSubdivisionGeometry *slabs,
                                int *nSlabs, vec3f pi, Float h)
    {
        // Subdividing a slab is simply taking the half point over all edges
        // and creating three new triangles-slabs
        vec3f M0 = (p0 + p1) * .5;
        vec3f M1 = (p0 + p2) * .5;
        vec3f M2 = (p1 + p2) * .5;

        M0 = Normalize(M0);
        M1 = Normalize(M1);
        M2 = Normalize(M2);

        ComputeSlab(p0, M0, M1, &slabs[0]);
        ComputeSlab(M0, p1, M2, &slabs[1]);
        ComputeSlab(M2, p2, M1, &slabs[2]);
        ComputeSlab(M0, M1, M2, &slabs[3]);
        *nSlabs = 4;
    }

    static __bidevice__ void ComputeSlab(vec3f A, vec3f B, vec3f C,
                                         VolumetricSubdivisionGeometry *slab)
    {
        const Float one_over_three = 0.333333333;
        vec3f a = A;
        vec3f b = B;
        vec3f c = C;

        vec3f M = (a + b + c) * one_over_three;
        vec3f dir = Normalize(M);

        vec3f e0 = a / Dot(a, dir);
        vec3f e1 = b / Dot(b, dir);
        vec3f e2 = c / Dot(c, dir);

        slab->p0 = A;
        slab->p1 = B;
        slab->p2 = C;
        slab->a = e0;
        slab->b = e1;
        slab->c = e2;
    }

    static __bidevice__ void
    SlabsForParticle(vec3f p, Float R, VolumetricSubdivisionGeometry *slabs, int *nSlabs){
        // Building the tetrahedron over the sphere of center 'p' and radius R
        // is kinda of annoying because there are infinite solutions so we would
        // need to pick special vectors for moving around, which can be troublesome.

        // Lets use the demicube with edge 2, this generates the tetrahedron
        // given by (1,1,1), (1,−1,−1), (−1,1,−1), (−1,−1,1)
        // we can take these points as directions from the origin to walk
        // a standard distance 'd' and still arrive on a valid tetrahedron
        // over the sphere centered at the origin and radius 'd'. Transforming
        // these points into directions we get the following set:
        const vec3f u0 = vec3f(0.57735, 0.57735, 0.57735);
        const vec3f u1 = vec3f(0.57735, -0.57735, -0.57735);
        const vec3f u2 = vec3f(-0.57735, 0.57735, -0.57735);
        const vec3f u3 = vec3f(-0.57735, -0.57735, 0.57735);

        // To build the tetrahedron we need to walk 'R' over the set of directions
        // previously obtained. I'm however going to follow Sandim's suggestion
        // to solve entirely on local coordinates, i.e.: each particle is centered
        // at the origin and has radius 1 and therefore the directions match exactly
        // with the vertex of the regular tetrahedron
        vec3f A = u0;
        vec3f B = u1;
        vec3f C = u2;
        vec3f D = u3;

        // The faces of the tetrahedron are given by: ABD, ABC, BCD and ACD.
        // for each of these faces we need to construct the frustum based
        // on the vector that goes through the center of the particle 'p' and the
        // barycenter of each of these triangles, call it O.
        // The frustum is built by computing the plane where the vector O intersects
        // the sphere centered at 'p' and radius 'R'.
        ComputeSlab(A, B, D, &slabs[0]);
        ComputeSlab(A, B, C, &slabs[1]);
        ComputeSlab(B, C, D, &slabs[2]);
        ComputeSlab(A, C, D, &slabs[3]);
        *nSlabs = 4;
    }

    vec3f p0, p1, p2;
    vec3f a, b, c;
};


// The method
template<typename T, typename U, typename Q, typename Geometry> __bidevice__
bool IntervalParticleIsInterior(Grid<T, U, Q> *domain, ParticleSet<T> *pSet,
                                int max_depth, Float h, int pId)
{
    struct IntervalData{
        Geometry bq;
        int depth;
    };

    Geometry Bq;
    Geometry initial[8];
    int initialSlabs = 8;
    Stack<IntervalData> stack;
    int q = INTERVAL_LABEL_UNCOVERED;

    T pi = pSet->GetParticlePosition(pId);
    Geometry::SlabsForParticle(pi, h, &initial[0], &initialSlabs);

    for(int i = 0; i < initialSlabs; i++){
        stack.push({initial[i], 0});
    }

    bool reached_max_depth = false;
    while(!stack.empty()){
        IntervalData iBq;
        stack.top(&iBq);
        stack.pop();

        // start query on the given depth, assume it is uncovered, i.e.: boundary
        q = INTERVAL_LABEL_UNCOVERED;
        Bq = iBq.bq;

        // check if the query is being made from a 'useless' geometry, i.e.:
        // completely inside the original particle
        if(Bq.IsGeometryClipped(pi, h)) continue;

        // check neighboors to see if this assumption is correct
        ForAllNeighbors(domain, pSet, pId, [&](int j) -> int{
            int max_query = 0;
            T pj = pSet->GetParticlePosition(j);
            int inside = Bq.BallIntersectionCount(pi, pj, h, &max_query);
            // Possible cases:
            // 1- inside = max_query => the Geometry is entirely inside the ball
            //                          this means we don't need to continue
            if(inside == max_query){
                q = INTERVAL_LABEL_COVERED;
                return 1;
            }

            // 2- inside > 0 => the Geometry is partially covered we need to continue
            if(inside > 0){
                q = INTERVAL_LABEL_PARTIALLY_COVERED;
            }

            return 0;
        });

        // if the label changed, adjust accordingly
        if(q == INTERVAL_LABEL_COVERED){
            // if the current geometry is covered we need to inspect
            // others but not subdivide this one
        }else if(q == INTERVAL_LABEL_UNCOVERED){
            // if we found a geometry that is completely uncovered
            // than it means we are done, because there is a portion of the
            // particle that is not being touched by its neighboors
            return false;
        }else{
            // we need to subdivide and further query the subdivisons, if depth allows
            if(iBq.depth < max_depth){
                Geometry inner[8];
                int n = 0;
                Bq.Subdivide(&inner[0], &n, pi, h);
                // push the new slabs into the stack
                for(int k = 0; k < n; k++){
                    stack.push({inner[k], iBq.depth+1});
                }
            }else{
                // if we are unsure about the result and depth is already too large
                // assume it is a boundary particle. Since the query is always
                // exclusive I think this means we can terminate the query
                // as there will always be one portion of the original geometry
                // that will be unresolved
                reached_max_depth = true;
                break;
            }
        }
    }

    // i'm not really sure what reaching here means. One idea might be
    // that if too many slabs are completely covered it loops
    // with label 'INTERVAL_LABEL_COVERED' and the exit is the exhaust
    // of all subdivisions. In this situation I guess this particle is interior.
    // However we need to account for the order in which the stack is
    // processed. If we reach the max depth it means we were subdividing
    // and we are unsure about a segment of the geometry, for this case it is
    // best to mark the particle as boundary.
    return !reached_max_depth;
}

inline __host__
void IntervalBoundary(ParticleSet2 *pSet, Grid2 *domain,
                      Float h, SubdivisionMethod method = PolygonSubdivision,
                      int max_depth = 4)
{
    using T = vec2f;
    using U = vec2ui;
    using Q = Bounds2f;
    int N = pSet->GetParticleCount();
    AutoParallelFor("IntervalBoundary2D", N, AutoLambda(int i){
        int is_interior = 0;
        if(method == PolygonSubdivision){
            is_interior =
                IntervalParticleIsInterior<T, U, Q, PolygonSubdivisionGeometry>
                    (domain, pSet, max_depth, h, i);
        }else{
            is_interior =
                IntervalParticleIsInterior<T, U, Q, BBSubdivisionGeometry<T, Q, 2>>
                    (domain, pSet, max_depth, h, i);
        }

        pSet->SetParticleV0(i, 1-is_interior);
    });
}

inline __host__
void IntervalBoundary(ParticleSet3 *pSet, Grid3 *domain,
                      Float h, SubdivisionMethod method = PolygonSubdivision,
                      int max_depth = 4)
{
    using T = vec3f;
    using U = vec3ui;
    using Q = Bounds3f;
    int N = pSet->GetParticleCount();
    AutoParallelFor("IntervalBoundary3D", N, AutoLambda(int i){
        int is_interior = 0;
        if(method == PolygonSubdivision){
            is_interior =
                IntervalParticleIsInterior<T, U, Q, VolumetricSubdivisionGeometry>
                    (domain, pSet, max_depth, h, i);
        }else{
            is_interior =
                IntervalParticleIsInterior<T, U, Q, BBSubdivisionGeometry<T, Q, 3>>
                    (domain, pSet, max_depth, h, i);
        }

        pSet->SetParticleV0(i, 1-is_interior);
    });
}

