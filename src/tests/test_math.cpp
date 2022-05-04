#include <graphy-inl.h>
#include <point_generator.h>
#include <kernel.h>
#include <graphy.h>
#include <unistd.h>
#include <tests.h>
#include <transform.h>
#include <shape.h>
#include <collider.h>
#include <emitter.h>
#include <sph_solver.h>
#include <sampling.h>
#include <doring.h>
#include <algorithm>
#include <explicit_grid.h>

#define SolverGrid2 CatmullMarkerAndCellGrid2
//#define SolverGrid2 LinearMarkerAndCellGrid2

class FluidSolver {
    public:
    /* Fluid quantities */
    SolverGrid2 *_d;
    SolverGrid2 *_u;
    SolverGrid2 *_v;
    PressureSolver *pSolver;

    /* Width and height */
    int _w;
    int _h;

    /* Grid cell size and fluid density */
    Float _hx;
    Float _density;

    /* Arrays for: */
    Float *_r; /* Right hand side of pressure solve */
    Float *_p; /* Pressure solution */

    /* Conjugate gradients solver */
    void project(int limit, Float timestep){
        pSolver->SolvePressure(_u, _v, limit, _p, _density, timestep);
        //pSolver->Update(limit, _p, _density, timestep);
    }

    /* Applies the computed pressure to the velocity field */
    void applyPressure(Float timestep) {
        Float scale = timestep/(_density*_hx);
        int w = _w;
        int h = _h;
        size_t items = w * h;

        Float *p = _p;
        SolverGrid2 *u = _u;
        SolverGrid2 *v = _v;

        ParallelFor((size_t)0, items, [&](int i) -> void
        //AutoParallelFor("Apply_Pressure", items, AutoLambda(size_t i)
        {
            int x = i % _w;
            int y = i / _w;
            Float pi = p[i];
            Float pxpy = 0;
            Float pxyp = 0;
            if(x >= 1){
                pxpy = Accessor2D(p, x-1, y, w);
            }

            if(y >= 1){
                pxyp = Accessor2D(p, x, y-1, w);
            }

            u->at(x, y) -= scale*(pi - pxpy);
            v->at(x, y) -= scale*(pi - pxyp);
        });

        for(int i = 0; i < Max(_h, _w); i++){
            if(i < _h){
                _u->at(0, i) = _u->at(_w, i) = 0.0;
            }
            if(i < _w){
                _v->at(i, 0) = _v->at(i, _h) = 0.0;
            }
        }
    }

    FluidSolver(int w, int h, Float density, Float targetTimestep) :
      _w(w), _h(h), _density(density)
    {
        size_t n = _w * _h;
        _hx = 1.0/min(w, h);

        _d = SolverGrid2::Create(_w,     _h,     0.5, 0.5, _hx);
        _u = SolverGrid2::Create(_w + 1, _h,     0.0, 0.5, _hx);
        _v = SolverGrid2::Create(_w,     _h + 1, 0.5, 0.0, _hx);

        _p = cudaAllocateVx(Float, n);

        pSolver = new PressureSolverJacobi;
        pSolver->BuildSolver(n, _hx, vec2ui(_w, _h));

        memset(_p, 0, _w*_h*sizeof(Float));
        _r = pSolver->RHS();
    }

    ~FluidSolver(){
        delete pSolver;
    }

    void update(Float timestep) {
        //PressureSolverBuildRHS(_u, _v, _hx, _r, vec2ui(_w, _h));
        project(600, timestep);
        applyPressure(timestep);

        Advect(timestep, _d, _u, _v);
        Advect(timestep, _u, _u, _v);
        Advect(timestep, _v, _u, _v);

        /* Make effect of advection visible, since it's not an in-place operation */
        _d->flip();
        _u->flip();
        _v->flip();
    }

    /* Set density and x/y velocity in given rectangle to d/u/v, respectively */
    void addInflow(Float x, Float y, Float w, Float h, Float d, Float u, Float v){
        _d->addInflow(x, y, x + w, y + h, d);
        _u->addInflow(x, y, x + w, y + h, u);
        _v->addInflow(x, y, x + w, y + h, v);
    }

    /* Convert fluid density to RGBA image */
    void toImage(float *rgb) {
        for (int i = 0; i < _w*_h; i++) {
            float shade = ((1.0 - _d->src()[i]));
            shade = max(min(shade, 1.f), 0.f);

            rgb[i*3 + 0] = shade;
            rgb[i*3 + 1] = shade;
            rgb[i*3 + 2] = shade;
        }
    }

    GVec4f ValueAt(int i, int j){
        int f = i + j * _w;
        float shade = ((1.0 - _d->src()[f]));
        return GVec4f(shade, shade, shade, 1.f);
    }
};

#include <interval.h>
#define TypedPolygon PolygonSubdivisionGeometry
void gui_tri(Graphy2DCanvas *canvas, TypedPolygon *sub,
            GVec4f circ_color = GVec4f(0,0,1,1), GVec4f line_color = GVec4f(0,1,0,1))
{
    canvas->circle(sub->p0.x, sub->p0.y).color(circ_color);
    canvas->circle(sub->p1.x, sub->p1.y).color(circ_color);
    canvas->circle(sub->p2.x, sub->p2.y).color(circ_color);
    canvas->circle(sub->p3.x, sub->p3.y).color(circ_color);
    canvas->path(sub->p0.x, sub->p0.y, sub->p1.x, sub->p1.y).color(line_color).width(2.f);
    canvas->path(sub->p1.x, sub->p1.y, sub->p2.x, sub->p2.y).color(line_color).width(2.f);
    canvas->path(sub->p2.x, sub->p2.y, sub->p3.x, sub->p3.y).color(line_color).width(2.f);
    canvas->path(sub->p3.x, sub->p3.y, sub->p0.x, sub->p0.y).color(line_color).width(2.f);
}

void test_mac(int argc, char **argv){
    GWindow gui("MAC Test");
    auto canvas = gui.get_canvas();
#if 0
    TypedPolygon sub[8], div[6], subdivs[12];
    int nSlabs = 8, nDivs = 2;
    vec2f p = vec2f(0.5, 0.5);
    Float h = 0.25;

    TypedPolygon::SlabsForParticle(p, h, &sub[0], &nSlabs);
    int divs = 0;
    sub[0].Subdivide(&div[0], &nDivs, p, h); divs += nDivs;
    sub[1].Subdivide(&div[2], &nDivs, p, h); divs += nDivs;
    sub[2].Subdivide(&div[4], &nDivs, p, h); divs += nDivs;

    int at = 0;
    for(int i = 0; i < 6; i++){
        int ns = 0;
        div[i].Subdivide(&subdivs[at], &ns, p, h);
        at += ns;
    }
#endif

    VolumetricSubdivisionGeometry slabs[4];
    int nSlabs = 4;
    VolumetricSubdivisionGeometry::SlabsForParticle(vec3f(0), 0.05, slabs, &nSlabs);

    return;
    while(1){
        canvas.Color(0x112F41);
#if 0
        canvas.Radius(4.f);
        for(int i = 0; i < nSlabs; i++){
            gui_tri(&canvas, &sub[i]);
        }

        for(int i = 0; i < divs; i++){
            gui_tri(&canvas, &div[i], GVec4f(1,0,0,1), GVec4f(1,1,0,1));
        }

        for(int i = 0; i < at; i++){
            gui_tri(&canvas, &subdivs[i], GVec4f(1,0,1,1), GVec4f(0,1,1,1));
        }
#endif
        gui.update();
    }

    return;

    int res = 128;
    if(argc > 1){
        res = atoi(argv[1]);
    }

    printf("Resolution %d x %d\n", res, res);
    int _w = res, _h = res;
    Float density = 0.1;
    Float timestep = 0.005;
    FluidSolver *solver = new FluidSolver(_w, _h, density, timestep);

    int it = 0;
    canvas.Color(0x112F41);

    auto fetcher = [&](int x, int y) -> GVec4f{ return solver->ValueAt(x, y); };
    while(1){
        solver->addInflow(0.45, 0.2, 0.15, 0.03, 1.6, 2.0, 4.0);
        solver->update(timestep);

        canvas.for_each_pixel([&](int x, int y) -> GVec4f{
            return canvas.upsample_from(x, y, _w, _h, fetcher, GUpsampleMode::Bilinear);
        });

        gui.update();

        //solver->toImage(colors);
        //graphy_display_pixels();
        if(it++ == 181){
            //graphy_write_image(nullptr, 0, 0, 0, "image180.png");
            //printf("*Saved*\n");
        }
    }

    delete solver;
}

// God, float point precision is hard. Don't want to use double but accumulating
// float multiplications in matrix breaks precision on zero computations
inline bool IsLowZero(Float v){
    if(sizeof(Float) == sizeof(float)){
        return Absf(v) < 1e-2;
    }else{
        return Absf(v) < 1e-5;
    }
}

inline vec2f PointOnBox(Float length){
    return vec2f(rand_float() * length * (rand_float() > 0.5 ? 1 : -1),
                 rand_float() * length * (rand_float() > 0.5 ? 1 : -1));
}

inline vec2f PointOnSphere2(Float radius){
    vec2f p;
    do{
        p = PointOnBox(radius);
    }while(p.Length() > radius);
    return p;
}

void test_sampling_barycentric(){
    printf("===== Test Sampling Barycentric\n");
    vec3f p(5.3, 5.2, 5.4);
    vec3f o(0);
    Float spacing = 1;
    Float fx, fy, fz;
    int iSize = 11;
    int i, j, k;

    vec3f N = (p - o) / spacing;

    GetBarycentric(N.x, 0, iSize-1, &i, &fx);
    GetBarycentric(N.y, 0, iSize-1, &j, &fy);
    GetBarycentric(N.z, 0, iSize-1, &k, &fz);

    printf("(i,j,k) = (%d,%d,%d)\n", i, j, k);
    printf("(fx,fy,fz) = (%g,%g,%g)\n", fx, fy, fz);

    printf("===== OK\n");
}

void test_simple_triangle_distance(){
    printf("===== Test Triangle Distance 3D\n");

    // 2 triangles
    Float pos[] = {
        0.4, -0.5, 0.0,
        0.8, -0.5, 0.0,
        0.6, 1.0, 0.0,

        -0.4, -0.5, 0.0,
        -0.8, -0.5, 0.0,
        -0.6, 1.0, 0.0
    };

    unsigned int indices[] = {
        0, 1, 2,
        3, 4, 5
    };

    int pointCount = (sizeof(pos) / sizeof(Float)) / 3;
    int iCount = (sizeof(indices) / sizeof(unsigned int));
    int it = Max(pointCount, iCount);

    /* manually construct ParsedMesh */
    ParsedMesh mesh;
    memset(&mesh, 0x00, sizeof(ParsedMesh));
    mesh.p = new Point3f[pointCount];
    mesh.indices = new Point3i[iCount];

    for(int i = 0; i < it; i++){
        if(i < pointCount){
            int st = 3 * i;
            mesh.p[i] = Point3f(pos[st+0], pos[st+1], pos[st+2]);
        }

        if(i < iCount){
            mesh.indices[i] = Point3i(indices[i], 0, 0);
        }
    }

    mesh.nTriangles = iCount / 3;
    mesh.nVertices = pointCount;

    Node *bvh = MakeBVH(&mesh, 8);
    int tri = -1;
    Float dist = BVHMeshClosestDistance(vec3f(-0.1, 0.0, 0.0), &tri, &mesh, bvh);
    TEST_CHECK(tri == 1, "Failed to find closest triangle");
    dist = BVHMeshClosestDistance(vec3f(0.1, 0.0, 0.0), &tri, &mesh, bvh);
    TEST_CHECK(tri == 0, "Failed to find closest triangle");
    printf("===== OK\n");
}

// Theres not really a way to validate particles are indeed from boundary
// without performing a comparation with Dilts method, don't want to implement
// that right now, so look at graphy output and see if it looks like boundary.
// TODO: Implement Dilts.
void test_color_field_2D(){
    printf("===== Test Color Field 2D\n");
    Float spacing = 0.05;
    Float targetDensity = WaterDensity;
    vec2f center(0,0);
    Float r1 = 0.5;
    Float r2 = 1.0;
    vec2f pMin, pMax;
    Bounds2f containerBounds;
    ParticleSetBuilder2 builder;
    Grid2 grid;
    SphSolver2 solver;
    solver.Initialize(DefaultSphSolverData2());
    vec2ui res(10, 10);

    Shape2 sphere; sphere.InitSphere2(Translate2(center.x, center.y), r1);
    Shape2 container; container.InitSphere2(Translate2(center.x, center.y), r2, true);

    containerBounds = container.GetBounds();

    // NOTE: Need to make sure bounds are expanded to fit all particles
    pMin = containerBounds.pMin - vec2f(spacing);
    pMax = containerBounds.pMax + vec2f(spacing);

    grid.Build(res, pMin, pMax);

    VolumeParticleEmitter2 emitter(&sphere, sphere.GetBounds(), spacing);

    emitter.Emit(&builder);
    SphParticleSet2 *sphSet = SphParticleSet2FromBuilder(&builder);

    solver.Setup(targetDensity, spacing, 1.8, &grid, sphSet);

    solver.UpdateDensity();

    // Do boundary computation by hand
    ParticleSet2 *pSet = solver.solverData->sphpSet->GetParticleSet();
    Grid2 *domain = solver.solverData->domain;
    SphStdKernel2 kernel(solver.GetKernelRadius());
    int *neighbors = nullptr;
    Float mass = pSet->GetMass();

    float *bpos = new float[pSet->GetParticleCount() * 3];
    float *cols = new float[pSet->GetParticleCount() * 3];
    int currIt = 0;

    Float *lens = new Float[pSet->GetParticleCount()];
    Float maxS = 0;
    for(int i = 0; i < pSet->GetParticleCount(); i++){
        vec2f pi = pSet->GetParticlePosition(i);
        unsigned int cellId = domain->GetLinearHashedPosition(pi);

        int count = domain->GetNeighborsOf(cellId, &neighbors);
        vec2f s(0, 0);
        for(int id = 0; id < count; id++){
            Cell<Bounds2f> *cell = domain->GetCell(neighbors[id]);
            ParticleChain *pChain = cell->GetChain();
            int size = cell->GetChainLength();

            for(int j = 0; j < size; j++){
                vec2f pj = pSet->GetParticlePosition(pChain->pId);
                Float distance = Distance(pi, pj);
                Float density = pSet->GetParticleDensity(pChain->pId);
                Float volume = mass / density;
                vec2f dir(0,0);

                if(distance > 0){
                    dir = (pj - pi) / distance;
                }

                s += volume * kernel.gradW(distance, dir);
                pChain = pChain->next; // NOTE: Don't forget to move the chain node
            }
        }

        Float sLen = s.Length();
        lens[i] = sLen;
        maxS = sLen > maxS ? sLen : maxS;
    }

    for(int i = 0; i < pSet->GetParticleCount(); i++){
        vec2f pi = pSet->GetParticlePosition(i);
        Float ls = lens[i];
        bpos[currIt * 3 + 0] = pi[0];
        bpos[currIt * 3 + 1] = pi[1];
        bpos[currIt * 3 + 2] = 0;

        if(ls > 3){
            cols[currIt * 3 + 0] = 0;
            cols[currIt * 3 + 1] = 1;
            cols[currIt * 3 + 2] = 0;
        }else{
            cols[currIt * 3 + 0] = 1;
            cols[currIt * 3 + 1] = 0;
            cols[currIt * 3 + 2] = 0;
        }
        currIt ++;
    }

    graphy_render_pointsEx(bpos, cols, currIt, -2, 2, 2, -2);
    printf(" * Graphy integration\n");
    sleep(4);
    delete[] bpos;
    delete[] cols;
    delete[] lens;
    printf("===== OK\n");
}

void test_closest_point_sphere2D(){
    printf("===== Test Closest Point Sphere 2D\n");
    Float spacing = 0.09;
    vec2f center(0,0);
    Float radius = 1.0;
    Shape2 sphere; sphere.InitSphere2(Translate2(center.x, center.y), radius);
    Shape2 container; container.InitSphere2(Translate2(center.x, center.y), radius, true);

    ParticleSetBuilder2 builder;
    VolumeParticleEmitter2 emitter(&sphere, sphere.GetBounds(), spacing);
    emitter.Emit(&builder);
    SphParticleSet2 *sphSet = SphParticleSet2FromBuilder(&builder);
    ParticleSet2 *pSet = sphSet->GetParticleSet();

    Collider2 collider(&container);

    float *pos = new float[pSet->GetParticleCount() * 3 * 2];
    float *colors = new float[pSet->GetParticleCount() * 3 * 2];

    for(int i = 0; i < pSet->GetParticleCount(); i++){
        ClosestPointQuery2 query;
        vec2f p = pSet->GetParticlePosition(i);
        vec2f p1 = p;
        sphere.ClosestPoint(p, &query);
        p = query.point;
        Float distance = Distance(query.point, center);
        Float dr = Absf(distance - radius);
        bool is_zero = IsLowZero(dr);
        TEST_CHECK(is_zero, "Particle closest not on sphere");

        p = PointOnBox(radius);
        sphere.ClosestPoint(p, &query);
        p = query.point;
        distance = Distance(query.point, center);
        dr = Absf(distance - radius);
        is_zero = IsLowZero(dr);
        TEST_CHECK(is_zero, "Random point not on sphere");

        if(!is_zero){
            printf("Distance is %g {dr: %g} [%g %g]\n", distance,
                   dr, query.point.x, query.point.y);
        }

        TEST_CHECK(is_zero, "Query point not on sphere");

        pos[3 * i + 0] = p1[0]; colors[3 * i + 0] = 1;
        pos[3 * i + 1] = p1[1]; colors[3 * i + 1] = 0;
        pos[3 * i + 2] = 0; colors[3 * i + 2] = 0;
    }

    Float h = 1.5f;
    int c = pSet->GetParticleCount();

    for(int i = 0; i < 1024; i++){
        vec2f p = PointOnSphere2(radius);
        vec2f p1 = p;
        vec2f v = PointOnBox(radius);
        Float d0 = Distance(p, center);
        if(collider.ResolveCollision(spacing, 0, &p, &v)){
            Float d1 = Distance(p, center);
            if(!(d1 < d0)){
                printf("Distance before: %g, Distance after: %g\n", d0, d1);
            }

            TEST_CHECK(d1 < d0, "Collision startup is further away from collider");
            pos[3 * c + 0] = p1[0]; colors[3 * c + 0] = 0;
            pos[3 * c + 1] = p1[1]; colors[3 * c + 1] = 0;
            pos[3 * c + 2] = 0; colors[3 * c + 2] = 1;
            c ++;

            pos[3 * c + 0] = p[0]; colors[3 * c + 0] = 0;
            pos[3 * c + 1] = p[1]; colors[3 * c + 1] = 1;
            pos[3 * c + 2] = 0; colors[3 * c + 2] = 0;
            c ++;
        }
    }

    graphy_render_pointsEx(pos, colors, c, -h, h, h, -h);
    printf(" * Graphy integration");
    sleep(4);
    //getchar();

    delete[] pos;
    delete[] colors;

    printf("===== OK\n");
}

void test_ray2_intersect(){
    printf("===== Test Ray/Sphere Intersect 2D\n");
    // sample points in center axis and just shoot through there 
    for(int i = 0; i < 2048; i++){
        SurfaceInteraction2 isect;
        Float u1 = rand_float() * 38 * (rand_float() > 0.5 ? 1 : -1);
        Float u2 = rand_float() * 38 * (rand_float() > 0.5 ? 1 : -1);
        vec2f center(u1, u2);
        Float u = 2.0 * rand_float() - 1.0;
        vec2f sample = center + vec2f(0, u);
        Float e = 1.5 + rand_float() * 10.0;
        vec2f source = center - vec2f(e, 0);
        vec2f dir = Normalize(sample - source);
        vec2f bi(u1 - 1.0, u2 - 1.0);
        vec2f bo(u1 + 1.0, u2 + 1.0);

        Ray2 ray(source, dir);
        Float tHit = 0;

        Shape2 sphere; sphere.InitSphere2(Translate2(u1, u2), 1.0);

        bool hit = sphere.Intersect(ray, &isect, &tHit);
        Float d = sphere.ClosestDistance(source);
        Bounds2f b = sphere.GetBounds();
        vec2f pMin = b.pMin;
        vec2f pMax = b.pMax;
        vec2f zi = pMin - bi;
        vec2f zo = pMax - bo;
        if(!hit){
            printf("Failed at source: {%g %g} sample: {%g %g} center: {%g %g} dir: {%g %g} dist: %g [%d]\n",
                   source.x, source.y, sample.x, sample.y, center.x, center.y,
                   dir.x, dir.y, d, i);
        }

        for(int i = 0; i < 2; i++){
            Float ai = zi[i];
            Float ao = zo[i];
            TEST_CHECK(IsZero(ai) && IsZero(ao), "Invalid axis for Sphere2::GetBounds");
        }

        TEST_CHECK(IsLowZero(d - (e - 1.0)), "Failed closest distance computation");
        TEST_CHECK(hit, "Did not hit sphere");
    }

    printf("===== OK\n");
}

vec2f sample_outside(Bounds2f bound){
    vec2f p(0,0);
    Float lenx = bound.ExtentOn(0) * 1.5;
    Float leny = bound.ExtentOn(1) * 1.5;
    do{
        p.x = (2 * rand_float() - 1) * lenx;
        p.y = (2 * rand_float() - 1) * leny;
    }while(Inside(p, bound));
    return p;
}

vec2f sample_inside(Bounds2f bound){
    vec2f p(0,0);
    Float lenx = bound.ExtentOn(0) * 0.5;
    Float leny = bound.ExtentOn(1) * 0.5;
    p.x = (2 * rand_float() - 1) * lenx;
    p.y = (2 * rand_float() - 1) * leny;
    return p;
}

void test_rectangle_distance_inside(){
    printf("===== Test Inside Rectangle Distance\n");
    Shape2 rectangle;
    ClosestPointQuery2 query;
    rectangle.InitRectangle2(Translate2(0, 0), vec2f(2, 2));
    int points = 2048;

    float *pos = new float[points * 3 * 2];
    float *col = new float[points * 3 * 2];
    int it = 0;

    for(int i = 0; i < points; i++){
        vec2f p = sample_inside(rectangle.GetBounds());
        rectangle.ClosestPoint(p, &query);

        Float dmin = Min(Min(Absf(1 - p.x), Absf(1 + p.x)),
                         Min(Absf(1 - p.y), Absf(1 + p.y)));

        if(!IsZero(dmin - query.signedDistance)){
            printf("Distance: %g, Queried: %g, Point {%g %g}\n",
                   dmin, query.signedDistance, p.x, p.y);
        }

        TEST_CHECK(IsZero(dmin - query.signedDistance), "Distance is not minimal");

        vec2f n = query.normal;
        pos[3 * it + 0] = p[0]; pos[3 * it + 1] = p[1]; pos[3 * it + 2] = 0;
        col[3 * it + 0] = Absf(n.x);
        col[3 * it + 1] = 0;
        col[3 * it + 2] = Absf(n.y);
        it++;

        p = query.point;
        pos[3 * it + 0] = p[0]; pos[3 * it + 1] = p[1]; pos[3 * it + 2] = 0;
        col[3 * it + 0] = 0; col[3 * it + 1] = 1; col[3 * it + 2] = 0;
        it++;
    }

    Float h = 2;
    graphy_render_pointsEx(pos, col, 2 * points, -h, h, h, -h);
    printf(" * Graphy integration\n");
    sleep(4);
    delete[] pos;
    delete[] col;

    printf("===== OK\n");
}

void test_rectangle_distance_outside(){
    printf("===== Test Outside Rectangle Distance\n");
    Shape2 rectangle;
    ClosestPointQuery2 query;
    rectangle.InitRectangle2(Translate2(0, 0), vec2f(1,1));
    int points = 1024;

    float *pos = new float[points * 3 * 2];
    float *col = new float[points * 3 * 2];
    int it = 0;

    for(int i = 0; i < points; i++){
        vec2f p = sample_outside(rectangle.GetBounds());
        rectangle.ClosestPoint(p, &query);
        vec2f n = query.normal;
        pos[3 * it + 0] = p[0]; pos[3 * it + 1] = p[1]; pos[3 * it + 2] = 0;
        col[3 * it + 0] = Absf(n.x);
        col[3 * it + 1] = 0;
        col[3 * it + 2] = Absf(n.y);
        it++;

        p = query.point;
        pos[3 * it + 0] = p[0]; pos[3 * it + 1] = p[1]; pos[3 * it + 2] = 0;
        col[3 * it + 0] = 0; col[3 * it + 1] = 1; col[3 * it + 2] = 0;
        it++;
    }

    Float h = 2;
    graphy_render_pointsEx(pos, col, 2 * points, -h, h, h, -h);
    printf(" * Graphy integration\n");
    sleep(4);
    delete[] pos;
    delete[] col;

    printf("===== OK\n");
}

void test_rectangle_emit(){
    printf("===== Test Rectangle Emittion\n");
    Shape2 rectangle;
    Float spacing = 0.09;
    rectangle.InitRectangle2(Translate2(0,0), vec2f(1,1));

    ParticleSetBuilder2 builder;
    VolumeParticleEmitter2 emitter(&rectangle, rectangle.GetBounds(), spacing);
    emitter.Emit(&builder);

    SphParticleSet2 *sphSet = SphParticleSet2FromBuilder(&builder);
    ParticleSet2 *pSet = sphSet->GetParticleSet();

    float *pos = new float[pSet->GetParticleCount() * 3];
    for(int i = 0; i < pSet->GetParticleCount(); i++){
        vec2f p = pSet->GetParticlePosition(i);
        pos[3 * i + 0] = p[0];
        pos[3 * i + 1] = p[1];
        pos[3 * i + 2] = 0;
    }

    float col[3] = {1,0,0};

    Float h = 1;
    graphy_render_points(pos, col, pSet->GetParticleCount(), -h, h, h, -h);
    sleep(4);

    delete[] pos;
    printf("===== OK\n");
}

void test_box_distance(){
    printf("===== Test Box Distance\n");
    Shape shape;
    shape.InitBox(Transform(), 1, 1, 1);

    Bounds3f testBound(vec3f(-0.48), vec3f(0.48));

    int points = 500;

    std::vector<vec3f> targets;

    for(int i = 0; i < points; i++){
        Float x = 2 * rand_float() - 1;
        Float y = 2 * rand_float() - 1;
        Float z = 2 * rand_float() - 1;

        ClosestPointQuery query;
        shape.ClosestPoint(vec3f(x,y,z), &query);
        targets.push_back(query.point);

        TEST_CHECK(!Inside(query.point, testBound), "Point is not on box face");
    }

    float *pos = new float[3 * targets.size()];

    int it = 0;
    for(vec3f &p : targets){
        pos[3 * it + 0] = p.x;
        pos[3 * it + 1] = p.y;
        pos[3 * it + 2] = p.z;
        it ++;
    }

    vec3f origin(0,0,3);
    vec3f target(0,0,0);
    float rgb[3] = {1, 0, 0};

    graphy_vector_set(origin, target);
    graphy_render_points3(pos, rgb, targets.size(), 0.01);

    sleep(4);
    delete[] pos;
    printf("===== OK\n");
}

void test_bounds_split3(){
    printf("===== Test Bounds3 Split\n");
    vec3f A(-1), B(1);
    Bounds3f bounds(vec3f(-1), vec3f(1));
    Bounds3f splits[8];

    Float lx = (B.x - A.x) * 0.5;
    Float ly = (B.y - A.y) * 0.5;
    Float lz = (B.z - A.z) * 0.5;
    vec3f res[] = {
        vec3f(A.x, A.y, A.z),
        vec3f(A.x + lx, A.y + ly, A.z + lz),
        vec3f(A.x + lx, A.y, A.z),
        vec3f(B.x, A.y + ly, A.z + lz),
        vec3f(A.x, A.y + ly, A.z),
        vec3f(A.x + lx, B.y, A.z + lz),
        vec3f(A.x + lx, A.y + ly, A.z),
        vec3f(B.x, B.y, A.z + lz),

        vec3f(A.x, A.y, A.z + lz),
        vec3f(A.x + lx, A.y + ly, A.z + lz + lz),
        vec3f(A.x + lx, A.y, A.z + lz),
        vec3f(B.x, A.y + ly, A.z + lz + lz),
        vec3f(A.x, A.y + ly, A.z + lz),
        vec3f(A.x + lx, B.y, A.z + lz + lz),
        vec3f(A.x + lx, A.y + ly, A.z + lz),
        vec3f(B.x, B.y, A.z + lz + lz),
    };

    printf("Splitting : ");
    bounds.PrintSelf();
    printf("\n");

    int k = SplitBounds(bounds, &splits[0]);
    TEST_CHECK(k == 8, "Invalid returned bound split count");

    int it = 0;
    for(int i = 0; i < 8; i++){
        Bounds3f b = splits[i];
        vec3f pMin = b.pMin;
        vec3f pMax = b.pMax;
        TEST_CHECK((IsZero(pMin.x - res[it].x) &&
                   IsZero(pMin.y - res[it].y) &&
                   IsZero(pMax.x - res[it+1].x) &&
                   IsZero(pMax.y - res[it+1].y) &&
                   IsZero(pMax.z - res[it+1].z) &&
                   IsZero(pMax.z - res[it+1].z)),
                   "Unexpected position");
        it += 2;

        // print for easy inspection
        b.PrintSelf();
        printf("\n");

        TEST_CHECK(Inside(b, bounds), "Not inside");
    }

    Bounds3f bout(vec3f(-2), vec3f(1));
    TEST_CHECK(!Inside(bout, bounds), "Invalid computation of bounds inside");

    printf("===== OK\n");
}

void test_bounds_split2(){
    printf("===== Test Bounds2 Split\n");
    Bounds2f bounds(vec2f(-1), vec2f(1));
    Bounds2f splits[4];

    vec2f res[] = {
        vec2f(-1, -1), vec2f(0, 0),
        vec2f(-1, 0), vec2f(0, 1),
        vec2f(0, 0), vec2f(1, 1),
        vec2f(0, -1), vec2f(1, 0),
    };

    printf("Splitting : ");
    bounds.PrintSelf();
    printf("\n");

    int k = SplitBounds(bounds, &splits[0]);
    TEST_CHECK(k == 4, "Invalid returned bound split count");

    int it = 0;
    for(int i = 0; i < 4; i++){
        Bounds2f b = splits[i];
        vec2f pMin = b.pMin;
        vec2f pMax = b.pMax;
        TEST_CHECK((IsZero(pMin.x - res[it].x) &&
                   IsZero(pMin.y - res[it].y) &&
                   IsZero(pMax.x - res[it+1].x) &&
                   IsZero(pMax.y - res[it+1].y)),
                   "Unexpected position");
        it += 2;

        // print for easy inspection
        b.PrintSelf();
        printf("\n");

        TEST_CHECK(Inside(b, bounds), "Not inside");
    }

    Bounds2f bout(vec2f(-2), vec2f(1));
    TEST_CHECK(!Inside(bout, bounds), "Invalid computation of bounds inside");

    printf("===== OK\n");
}

void test_matrix_operations2(){
    printf("===== Test MatrixNxN operations\n");
    Matrix2x2 m(5, 4, 3, 2);
    m.PrintSelf();

    Matrix2x2 s = Inverse(m);
    s.PrintSelf();

    Float L = ComputeMinEigenvalue(m);
    printf("Min : %g\n", L);

    Matrix3x3 t(-2, -4, 2, -2, 1, 2, 4, 2, 5);
    L = ComputeMinEigenvalue(t);
    printf("Min : %g\n", L);

    printf("===== OK\n");
}

void test_matrix_operations(){
    printf("===== Test Matrix3x3\n");

    for(int i = 0; i < 2048; i++){
        Float u1 = rand_float() * 1.0 + 1.0;
        Float u2 = rand_float() * 1.0 + 1.0;
        Float s1 = rand_float() * 6.0 + 1.0;
        Float s2 = rand_float() * 5.0 + 1.0;

        vec2f p(u1,  u2);
        Transform2 scale = Scale2(s1, s2);
        Transform2 translate = Translate2(s1, s2);
        vec2f t = scale.Point(p);
        vec2f s = translate.Point(p);

        Matrix3x3 identity = Matrix3x3::Mul(scale.m, scale.mInv);

        TEST_CHECK(IsZero(t.x - u1 * s1), "Invalid Scale in X axis");
        TEST_CHECK(IsZero(t.y - u2 * s2), "Invalid Scale in Y axis");
        TEST_CHECK(IsZero(s.x - (u1 + s1)), "Invalid Translate in X axis");
        TEST_CHECK(IsZero(s.y - (u2 + s2)), "Invalid Translate in Y axis");

        for(int i = 0; i < 3; i++){
            for(int j = 0; j < 3; j++){
                Float a = identity.m[i][j];
                if(i == j){
                    Float da = a - 1.0;
                    if(!IsLowZero(da)){
                        printf("d: %g\n", da);
                        identity.PrintSelf();
                    }
                    TEST_CHECK(IsLowZero(da), "Not 1 at i == j on identity matrix");
                }
                else{
                    if(!IsLowZero(a)){
                        printf("a: %g\n", a);
                        identity.PrintSelf();
                    }
                    TEST_CHECK(IsLowZero(a), "Not 0 at i != j on identity matrix");
                }
            }
        }

        Float mat[3][3];
        for(int i = 0; i < 3; i++){
            for(int j = 0; j < 3; j++){
                mat[i][j] = (rand_float() * 15.0 + 0.5) * (rand_float() > 0.5 ? 1 : -1);
            }
        }

        Matrix3x3 m1(mat);
        Matrix3x3 m1Inv = Inverse(m1);
        identity = Matrix3x3::Mul(m1, m1Inv);
        for(int i = 0; i < 3; i++){
            for(int j = 0; j < 3; j++){
                Float a = identity.m[i][j];
                if(i == j){
                    Float da = a - 1.0;
                    if(!IsLowZero(da)){
                        identity.PrintSelf();
                        printf("------------\n");
                        m1.PrintSelf();
                        printf("------------\n");
                        m1Inv.PrintSelf();
                        printf("d: %g\n", da);
                    }
                    TEST_CHECK(IsLowZero(da), "Not 1 at i == j on identity matrix");
                }
                else{
                    if(!IsLowZero(a)){
                        identity.PrintSelf();
                        printf("------------\n");
                        m1.PrintSelf();
                        printf("------------\n");
                        m1Inv.PrintSelf();
                        printf("a: %g\n", a);
                    }
                    TEST_CHECK(IsLowZero(a), "Not 0 at i != j on identity matrix");
                }
            }
        }
    }

    printf("===== OK\n");
}
