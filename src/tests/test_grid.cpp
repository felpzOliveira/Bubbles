#include <tests.h>
#include <particle.h>
#include <grid.h>
#include <graphy.h>
#include <unistd.h>
#include <espic_solver.h>
#include <obj_loader.h>

static int with_graphy = 1;

void test_particle_to_node_2D(){
    printf("===== Test 2D Node Grid Particle To Node - Simple\n");
    Grid2 grid;
    NodeEdgeGrid2f nodeGrid;
    
    // use prefix values from book
    vec2f p0(-0.1,-0.1), p1(0.1,0.1);
    vec2ui res(1, 1);
    
    grid.Build(res, p0, p1);
    nodeGrid.Build(&grid);
    
    TEST_CHECK(nodeGrid.GetNodeCount() == 4, "Failed to create 2D node grid");
    TEST_CHECK(nodeGrid.GetNodesPerCell() == 4, "Invalid nodes per cell");
    
    vec2f h = nodeGrid.GetSpacing();
    vec2f center = grid.GetBounds().Center();
    Float val = 5.0f;
    for(int i = 0; i < 4; i++){
        nodeGrid.ParticleToNodes(center, val, i);
    }
    
    for(int i = 0; i < 4; i++){
        Float f = nodeGrid.GetValue(i);
        printf("[%d] = %g\n", i, f);
    }
    
    Float g = nodeGrid.NodesToParticle(center);
    printf("To Part: %g\n", g);
    
    printf("===== OK\n");
}


void test_node_grid_2D(){
    printf("===== Test 2D Node Grid Access/Hash\n");
    Grid2 grid;
    NodeEdgeGrid2f nodeGrid;
    
    // use prefix values from book
    vec2f p0(-0.1,-0.1), p1(0.1,0.1);
    vec2ui res(21, 25);
    
    grid.Build(res, p0, p1);
    nodeGrid.Build(&grid);
    
    vec2f cellLen = grid.GetCellSize();
    
    int expected = (res.x + 1) * (res.y + 1);
    
    TEST_CHECK(nodeGrid.GetNodeCount() == expected, "Invalid 2D setup ");
    TEST_CHECK(nodeGrid.GetNodesPerCell() == 4, "Invalid nodes per cell");
    printf(" * Cell Len: {%g %g}\n", cellLen.x, cellLen.y);
    
    int count = nodeGrid.GetNodeCount();
    for(int x = 0; x <= res.x; x++){
        for(int y = 0; y <= res.y; y++){
            int idx = y * (res.x+1) + x;
            Float f = (Float)idx;
            vec2ui u(x, y);
            Float g = 0;
            
            nodeGrid.SetValue(u, f);
            g = nodeGrid.GetValue(idx);
            if(!IsZero(f - g)){
                int h = LinearIndex(u, vec2ui(res.x+1, res.y+1), 2);
                printf("Id: %d  {%d %d} H = %d F = %g  G = %g\n", idx, u.x, u.y, h, f, g);
            }
            TEST_CHECK(IsZero(f - g), "Invalid node hash");
            nodeGrid.SetValue(u, 0);
        }
    }
    
    NodeEdgeGrid2f rho;
    
    rho.Build(&grid);
    vec2ui count2 = rho.GetNodeIndexCount();
    
    for(int i = 0; i < count2.x; i++){
        for(int j = 0; j < count2.y; j++){
            vec2ui p(i, j);
            int hash = LinearIndex(p, vec2ui(22, 22), 2);
            Float value = i + 1000 * j;
            rho.SetValue(p, value);
        }
    }
    
    int cellCount = grid.GetCellCount();
    int nodes[4];
    
    for(int i = 0; i < cellCount; i++){
        int n = rho.GetNodesFrom(i, &nodes[0]);
        Float r0 = rho.GetValue(nodes[0]);
        Float r1 = rho.GetValue(nodes[1]);
        Float r2 = rho.GetValue(nodes[2]);
        Float r3 = rho.GetValue(nodes[3]);
        
        Float dif0 = r1 - r0;
        Float dif1 = r2 - r0;
        Float dif2 = r3 - r1;
        Float dif3 = r3 - r2;
        
        TEST_CHECK(IsZero(dif0 - 1), "Failed linear x diff[1]");
        TEST_CHECK(IsZero(dif1 - 1000), "Failed linear y diff[1]");
        TEST_CHECK(IsZero(dif2 - 1000), "Failed linear y diff[2]");
        TEST_CHECK(IsZero(dif3 - 1), "Failed linear x diff[2]");
        
        vec2ui pij   = DimensionalIndex(nodes[0], vec2ui(22,22), 2);
        vec2ui pi_j  = DimensionalIndex(nodes[1], vec2ui(22,22), 2);
        vec2ui pij_  = DimensionalIndex(nodes[2], vec2ui(22,22), 2);
        vec2ui pi_j_ = DimensionalIndex(nodes[3], vec2ui(22,22), 2);
        
        int d0x = pi_j[0] - pij[0];
        int d0y = pi_j[1] - pij[1];
        int d1x = pij_[0] - pij[0];
        int d1y = pij_[1] - pij[1];
        int d2x = pi_j_[0] - pij[0];
        int d2y = pi_j_[1] - pij[1];
        int d3x = pi_j_[0] - pi_j[0];
        int d3y = pi_j_[1] - pi_j[1];
        int d4x = pi_j_[0] - pij_[0];
        int d4y = pi_j_[1] - pij_[1];
        
        TEST_CHECK(d0x == 1 && d0y == 0, "R0 and R1 mismatch");
        TEST_CHECK(d1x == 0 && d1y == 1, "R0 and R2 mismatch");
        TEST_CHECK(d2x == 1 && d2y == 1, "R0 and R3 mismatch");
        TEST_CHECK(d3x == 0 && d3y == 1, "R1 and R3 mismatch");
        TEST_CHECK(d4x == 1 && d4y == 0, "R2 and R3 mismatch");
        
        TEST_CHECK(LinearIndex(pij  , vec2ui(22,22), 2) == nodes[0], "Inverse hash failed");
        TEST_CHECK(LinearIndex(pi_j , vec2ui(22,22), 2) == nodes[1], "Inverse hash failed");
        TEST_CHECK(LinearIndex(pij_ , vec2ui(22,22), 2) == nodes[2], "Inverse hash failed");
        TEST_CHECK(LinearIndex(pi_j_, vec2ui(22,22), 2) == nodes[3], "Inverse hash failed");
    }
    
    printf("===== OK\n");
}

void test_neighbor_query_grid2D(){
    printf("===== Test neighbor querying 2D grid\n");
    Grid2 grid;
    ParticleSetBuilder2 builder;
    
    vec2f p0(-1,-1), p1(1,1);
    vec2ui res(10, 10);
    grid.Build(res, p0, p1);
    int *neighbors = nullptr;
    int count = 0;
    Bounds2f bound = grid.GetBounds();
    
    vec2f halfExt(bound.ExtentOn(0), bound.ExtentOn(1));
    halfExt *= 0.5;
    
    for(int i = 0; i < grid.GetCellCount(); i++){
        Cell<Bounds2f> *cell = grid.GetCell(i);
        vec2f center = cell->bounds.Center();
        builder.AddParticle(center);
        for(int k = 0; k < 5; k++){
            Float r1 = 2.f * rand_float() - 1.f;
            Float r2 = 2.f * rand_float() - 1.f;
            builder.AddParticle(vec2f(r1 * halfExt.x, r2 * halfExt.y));
        }
    }
    
    for(int rows = 0; rows < 10; rows++){
        for(int cols = 0; cols < 10; cols++){
            int id = cols + rows * 10;
            count = grid.GetNeighborsOf(id, &neighbors);
            if(rows > 0 && rows < 9 && cols > 0 && cols < 9){
                // middle of the grid, should have all neighbors 8 + self
                TEST_CHECK(count == 9, "Failed to get all neighbors");
            }else if(rows > 0 && rows < 9){
                // bottom or top without left or right, 9 - 3
                TEST_CHECK(count == 6, "Failed to get all neighbors for bottom/top cell");
            }else if(cols > 0 && cols < 9){
                // left or right without bottom or top, 9 - 3
                TEST_CHECK(count == 6, "Failed to get all neighbors for left/right cell");
            }else{
                // corner cell, 9 - 3 - 2
                TEST_CHECK(count == 4, "Failed to get all neighbors for corner cell");
            }
            
            for(int i = 0; i < count; i++){
                vec2ui u = grid.GetCellIndex(neighbors[i]);
                int dx = Absf((int)u.y - rows);
                int dy = Absf((int)u.x - cols);
                TEST_CHECK(dx < 2 && dy < 2, "Neighbor is too far");
            }
        }
    }
    
    if(with_graphy){
        ParticleSet2 *pSet = builder.MakeParticleSet();
        
        for(int i = 0; i < grid.GetCellCount(); i++){
            grid.DistributeToCell(pSet, i);
        }
        
        float *pos = new float[pSet->GetParticleCount() * 3 * 2];
        float *colors = new float[pSet->GetParticleCount() * 3 * 2];
        
        for(int i = 0; i < pSet->GetParticleCount(); i++){
            vec2f p = pSet->GetParticlePosition(i);
            pos[3 * i + 0] = p[0]; colors[3 * i + 0] = 1;
            pos[3 * i + 1] = p[1]; colors[3 * i + 1] = 0;
            pos[3 * i + 2] = 0; colors[3 * i + 2] = 0;
        }
        
        int nShows = 32;
        for(int k = 0; k < nShows; k++){
            int cellU = (int)(rand_float() * grid.GetCellCount());
            count = grid.GetNeighborsOf(cellU, &neighbors);
            
            int c = pSet->GetParticleCount();
            for(int i = 0; i < count; i++){
                Cell<Bounds2f> *cell = grid.GetCell(neighbors[i]);
                ParticleChain *pChain = cell->GetChain();
                int size = cell->GetChainLength();
                Float rgb[3] = {0, 1, 0};
                if(cellU == neighbors[i]){
                    rgb[1] = 0; rgb[2] = 1;
                }
                
                for(int j = 0; j < cell->GetChainLength(); j++){
                    ParticleChain *pNode = pChain;
                    vec2f p = pSet->GetParticlePosition(pNode->pId);
                    pos[3 * c + 0] = p[0]; colors[3 * c + 0] = rgb[0];
                    pos[3 * c + 1] = p[1]; colors[3 * c + 1] = rgb[1];
                    pos[3 * c + 2] = 0; colors[3 * c + 2] = rgb[2];
                    c += 1;
                    pChain = pChain->next;
                }
            }
            
            Float h = 1.5f;
            graphy_render_pointsEx(pos, colors, c, -h, h, h, -h);
            usleep(100000);
        }
        
        delete[] pos;
        delete[] colors;
        printf(" * Graphy integration\n");
    }
    
    printf("===== OK\n");
}

void test_neighbor_query_grid3D(){
    printf("===== Test neighbor querying 3D grid\n");
    Grid3 grid;
    
    vec3f p0(-1,-1,-1), p1(1,1,1);
    vec3ui res(10, 10, 10);
    grid.Build(res, p0, p1);
    int *neighbors = nullptr;
    
    for(int rows = 0; rows < 10; rows++){
        for(int cols = 0; cols < 10; cols++){
            for(int depth = 0; depth < 10; depth++){
                int id = cols + rows * 10 + depth * 10 * 10;
                int count = grid.GetNeighborsOf(id, &neighbors);
                
                bool is_corner = 
                    (cols == 0 && rows == 0 && depth == 0) ||
                    (cols == 9 && rows == 0 && depth == 0) ||
                    (cols == 0 && rows == 9 && depth == 0) ||
                    (cols == 0 && rows == 0 && depth == 9) ||
                    (cols == 0 && rows == 9 && depth == 9) ||
                    (cols == 9 && rows == 9 && depth == 9) ||
                    (cols == 9 && rows == 0 && depth == 9) ||
                    (cols == 9 && rows == 9 && depth == 0);
                
                bool is_interior = 
                    (cols > 0 && rows > 0 && depth > 0) &&
                    (cols < 9 && rows < 9 && depth < 9);
                
                if(is_corner){
                    // corner cell, 2 * 2 * 2
                    TEST_CHECK(count == 8, "Failed to get all neighbors for corner cell");
                }else if(is_interior){
                    // interaior cell, 3 * 3 * 3
                    TEST_CHECK(count == 27, "Failed to get all neighbors for interior cell");
                }else{
                    // either 1 face contact or 2
                    // for 1 face, 3 * 3 * 2
                    // for 2 face, 3 * 2 * 2
                    TEST_CHECK(count == 12 || count == 18,
                               "Failed to get all neighbors for face conection cells");
                }
                
                for(int i = 0; i < count; i++){
                    vec3ui u = grid.GetCellIndex(neighbors[i]);
                    int dx = Absf((int)u.y - rows);
                    int dy = Absf((int)u.x - cols);
                    int dz = Absf((int)u.z - depth);
                    TEST_CHECK(dx < 2 && dy < 2 && dz < 2, "Neighbor is too far");
                }
            }
        }
    }
    
    printf("===== OK\n");
}

void test_distribute_random_grid2D(){
    printf("===== Test distribute random 2D grid\n");
    Grid2 grid;
    ParticleSetBuilder2 builder;
    
    vec2f p0(-1,-1), p1(1,1);
    vec2ui res(10, 10);
    grid.Build(res, p0, p1);
    
    vec2f len = grid.GetCellSize();
    Bounds2f bound = grid.GetBounds();
    
    vec2f halfExt(bound.ExtentOn(0), bound.ExtentOn(1));
    halfExt *= 0.5;
    Float hlenx = 0.5 * len.x;
    Float hleny = 0.5 * len.y;
    int count = 0;
    for(Float x = p0.x; x < p1.x; x += len.x){
        if(p1.x - x < hlenx) continue; // test does not fix out points
        for(Float y = p0.y; y < p1.y; y += len.y){
            if(p1.y - y < hleny) continue; // test does not fix out points
            Float r1 = 2.f * rand_float() - 1.f;
            Float r2 = 2.f * rand_float() - 1.f;
            builder.AddParticle(vec2f(x + hlenx + r1 * hlenx * 0.5, 
                                      y + hleny + r2 * hleny * 0.5));
            builder.AddParticle(vec2f(r1 * halfExt.x, r2 * halfExt.y)); // centered at origin
            count += 2;
        }
    }
    
    TEST_CHECK(builder.GetParticleCount() == count, "Failed to insert particles in builder");
    ParticleSet2 *pSet = builder.MakeParticleSet();
    TEST_CHECK(pSet, "Failed to create particle set");
    TEST_CHECK(pSet->GetParticleCount() == count, 
               "Failed to insert correct particle count");
    
    int rc = 0;
    for(int i = 0; i < grid.GetCellCount(); i++){
        grid.DistributeToCell(pSet, i);
        Cell<Bounds2f> *cell = grid.GetCell(i);
        TEST_CHECK(cell->GetChainLength() >= 1, "Failed to set chain particle");
        rc += cell->GetChainLength();
    }
    
    TEST_CHECK(rc == count, "Failed to distribute all paticles");
    
    printf("===== OK\n");
}

void test_distribute_random_grid3D(){
    printf("===== Test distribute random 3D grid\n");
    Grid3 grid;
    ParticleSetBuilder3 builder;
    
    vec3f p0(-1,-1,-1), p1(1,1,1);
    vec3ui res(10, 10, 10);
    grid.Build(res, p0, p1);
    
    vec3f len = grid.GetCellSize();
    Bounds3f bound = grid.GetBounds();
    
    vec3f halfExt(bound.ExtentOn(0), bound.ExtentOn(1), bound.ExtentOn(2));
    halfExt *= 0.5;
    Float hlenx = 0.5 * len.x;
    Float hleny = 0.5 * len.y;
    Float hlenz = 0.5 * len.z;
    int count = 0;
    for(Float x = p0.x; x < p1.x; x += len.x){
        if(p1.x - x < hlenx) continue; // test does not fix out points
        for(Float y = p0.y; y < p1.y; y += len.y){
            if(p1.y - y < hleny) continue; // test does not fix out points
            for(Float z = p0.z; z < p1.z; z += len.z){
                if(p1.z - z < hlenz) continue; // test does not fix out points
                Float r1 = 2.f * rand_float() - 1.f;
                Float r2 = 2.f * rand_float() - 1.f;
                Float r3 = 2.f * rand_float() - 1.f;
                builder.AddParticle(vec3f(x + hlenx + r1 * hlenx * 0.5, 
                                          y + hleny + r2 * hleny * 0.5,
                                          z + hlenz + r3 * hlenz * 0.5));
                
                // centered at origin
                builder.AddParticle(vec3f(r1 * halfExt.x, r2 * halfExt.y, r3 * halfExt.z));
                count += 2;
            }
        }
    }
    
    
    TEST_CHECK(builder.GetParticleCount() == count, "Failed to insert particles in builder");
    ParticleSet3 *pSet = builder.MakeParticleSet();
    TEST_CHECK(pSet, "Failed to create particle set");
    TEST_CHECK(pSet->GetParticleCount() == count, 
               "Failed to insert correct particle count");
    
    int rc = 0;
    for(int i = 0; i < grid.GetCellCount(); i++){
        grid.DistributeToCell(pSet, i);
        Cell<Bounds3f> *cell = grid.GetCell(i);
        TEST_CHECK(cell->GetChainLength() >= 1, "Failed to set chain particle");
        rc += cell->GetChainLength();
    }
    
    TEST_CHECK(rc == count, "Failed to distribute all paticles");
    printf("===== OK\n");
}

void test_distribute_uniform_grid2D(){
    printf("===== Test distribute 2D grid\n");
    Grid2 grid;
    ParticleSetBuilder2 builder;
    
    vec2f p0(-1,-1), p1(1,1);
    vec2ui res(10, 10);
    grid.Build(res, p0, p1);
    
    vec2f len = grid.GetCellSize();
    Bounds2f bound = grid.GetBounds();
    
    Float hlenx = 0.5 * len.x;
    Float hleny = 0.5 * len.y;
    int count = 0;
    for(Float x = p0.x; x < p1.x; x += len.x){
        if(p1.x - x < hlenx) continue; // test does not fix out points
        for(Float y = p0.y; y < p1.y; y += len.y){
            if(p1.y - y < hleny) continue; // test does not fix out points
            builder.AddParticle(vec2f(x + hlenx, y + hleny));
            count++;
        }
    }
    
    
    TEST_CHECK(builder.GetParticleCount() == count, "Failed to insert particles in builder");
    ParticleSet2 *pSet = builder.MakeParticleSet();
    TEST_CHECK(pSet, "Failed to create particle set");
    TEST_CHECK(pSet->GetParticleCount() == count, 
               "Failed to insert correct particle count");
    
    for(int i = 0; i < grid.GetCellCount(); i++){
        grid.DistributeToCell(pSet, i);
        Cell<Bounds2f> *cell = grid.GetCell(i);
        TEST_CHECK(cell->GetChainLength() == 1, "Failed to set chain particle");
        ParticleChain *chain = cell->head;
        TEST_CHECK(chain->cId == i, "Failed to set chain owner");
    }
    
    if(with_graphy && 0){
        float *pos = new float[pSet->GetParticleCount() * 3];
        float rgb[3] = {1, 0, 0};
        for(int i = 0; i < pSet->GetParticleCount(); i++){
            vec2f pi = pSet->GetParticlePosition(i);
            pos[3 * i + 0] = pi[0];
            pos[3 * i + 1] = pi[1];
            pos[3 * i + 2] = 0;
        }
        
        graphy_render_points(pos, rgb, pSet->GetParticleCount(), -3.f, 3.f, 3.f, -3.f);
        delete[] pos;
        printf(" * Graphy integration, press anything...");
        getchar();
    }
    
    printf("===== OK\n");
}

void test_distribute_uniform_grid3D(){
    printf("===== Test distribute 3D grid\n");
    Grid3 grid;
    ParticleSetBuilder3 builder;
    
    vec3f p0(-1,-1,-1), p1(1,1,1);
    vec3ui res(10, 10, 10);
    grid.Build(res, p0, p1);
    
    vec3f len = grid.GetCellSize();
    Bounds3f bound = grid.GetBounds();
    
    Float hlenx = 0.5 * len.x;
    Float hleny = 0.5 * len.y;
    Float hlenz = 0.5 * len.z;
    int count = 0;
    for(Float x = p0.x; x < p1.x; x += len.x){
        if(p1.x - x < hlenx) continue; // test does not fix out points
        for(Float y = p0.y; y < p1.y; y += len.y){
            if(p1.y - y < hleny) continue; // test does not fix out points
            for(Float z = p0.z; z < p1.z; z += len.z){
                if(p1.z - z < hlenz) continue; // test does not fix out points
                builder.AddParticle(vec3f(x + hlenx, y + hleny, z + hlenz));
                count++;
            }
        }
    }
    
    
    TEST_CHECK(builder.GetParticleCount() == count, "Failed to insert particles in builder");
    ParticleSet3 *pSet = builder.MakeParticleSet();
    TEST_CHECK(pSet, "Failed to create particle set");
    TEST_CHECK(pSet->GetParticleCount() == count, 
               "Failed to insert correct particle count");
    
    for(int i = 0; i < grid.GetCellCount(); i++){
        grid.DistributeToCell(pSet, i);
        Cell<Bounds3f> *cell = grid.GetCell(i);
        TEST_CHECK(cell->GetChainLength() == 1, "Failed to set chain particle");
        ParticleChain *chain = cell->head;
        TEST_CHECK(chain->cId == i, "Failed to set chain owner");
    }
    
    
    printf("===== OK\n");
}

void test_uniform_grid2D(){
    printf("===== Test minimal 2D grid\n");
    Grid2 grid;
    vec2f p0(-1,-1), p1(1,1);
    vec2ui res(10, 10);
    Float expLen = 0.2;
    
    grid.Build(res, p0, p1);
    TEST_CHECK(IsZero(grid.cellsLen[0] - expLen), "Failed grid cell length");
    TEST_CHECK(grid.total == 100, "Failed to build correct cell count");
    TEST_CHECK(grid.usizes[0] == 10 && grid.usizes[1] == 10, 
               "Invalid cell displacement");
    vec2f dif = grid.minPoint - p0;
    TEST_CHECK(dif.IsZeroVector(), "Invalid minimal point");
    
    /* Grid boundary not necessary is pMin = p0 and pMax = p1, so a inclusion test is better */
    Bounds2f bound = grid.GetBounds();
    TEST_CHECK(Inside(p0, bound), "Invalid grid boundary for minimal point");
    TEST_CHECK(Inside(p1, bound), "Invalid grid boundary for maximal point");
    dif = bound.pMin - p0;
    TEST_CHECK(dif.Length() < expLen, "Offset from minimal point to big");
    dif = bound.pMax - p1;
    TEST_CHECK(dif.Length() < expLen, "Offset from maximal point to big");
    
    Float hLen = expLen * 0.5;
    Float hhLen = hLen * 0.5;
    int ci = 0;
    int cj = 0;
    for(Float x = p0.x; x < p1.x; x += expLen){
        if(p1.x - x < hLen) continue; // test does not fix out points
        
        cj = 0;
        for(Float y = p0.y; y < p1.y; y += expLen){
            if(p1.y - y < hLen) continue; // test does not fix out points
            
            vec2f c(x + hLen, y + hLen);
            vec2f v[] = {
                /* half points */
                vec2f(c.x - hhLen, c.y - hhLen),
                vec2f(c.x - hhLen, c.y + hhLen),
                vec2f(c.x + hhLen, c.y - hhLen),
                vec2f(c.x + hhLen, c.y + hhLen),
                /* center point */
                vec2f(c.x, c.y),
            };
            
            /* corners can be in other cell as long as it is a valid cell 
            * and neighbor to this cell.
            */
            vec2f cor[] = {
                vec2f(c.x - hLen, c.y - hLen),
                vec2f(c.x + hLen, c.y + hLen),
                vec2f(c.x - hLen, c.y + hLen),
                vec2f(c.x + hLen, c.y - hLen)
            };
            
            for(int i = 0; i < 5; i++){
                vec2f p = v[i];
                vec2ui hashu = grid.GetHashedPosition(p);
                unsigned int hashid = grid.GetLinearHashedPosition(p);
                TEST_CHECK(hashu[0] == ci && hashu[1] == cj, "Failed hash position");
                TEST_CHECK(hashid == (ci + 10 * cj), "Failed linear hash position");
            }
            
            for(int i = 0; i < 4; i++){
                vec2f p = cor[i];
                vec2ui hashu = grid.GetHashedPosition(p);
                unsigned int hashid = grid.GetLinearHashedPosition(p);
                TEST_CHECK(hashid < 100, "Corner not in a valid cell");
                if(hashu[0] != ci || hashu[1] != cj){
                    int dx = (int)hashu[0] - ci;
                    int dy = (int)hashu[1] - cj;
                    TEST_CHECK(Absf(dx) < 2 && Absf(dy) < 2, "Corner not in neighor cell");
                }
            }
            
            cj++;
        }
        
        ci++;
    }
    
    printf("===== OK\n");
}

void test_field_grid(){
    printf("===== Test Field Grid\n");
    FieldGrid3f *grid = cudaAllocateVx(FieldGrid3f, 1);
    int resolution = 64;
    Float margin = 0.01;
    
    const char *whaleObj = "/home/felipe/Documents/CGStuff/models/HappyWhale.obj";
    //Transform transform = Scale(0.02); //dragon
    Transform transform = Scale(0.3); // happy whale
    
    UseDefaultAllocatorFor(AllocatorType::GPU);
    
    ParsedMesh *mesh = LoadObj(whaleObj);
    Shape *shape = MakeMesh(mesh, transform);
    
    Bounds3f bounds = shape->GetBounds();
    vec3f scale(bounds.ExtentOn(0), bounds.ExtentOn(1), bounds.ExtentOn(2));
    bounds.pMin -= margin * scale;
    bounds.pMax += margin * scale;
    
    Float dx = 0.024;
    Float width = bounds.ExtentOn(0);
    Float height = bounds.ExtentOn(1);
    Float depth = bounds.ExtentOn(2);
    
    resolution = (int)std::ceil(width / dx);
    int resolutionY = (int)std::ceil(resolution * height / width);
    int resolutionZ = (int)std::ceil(resolution * depth / width);
    
    unsigned int expec = (resolution+1) * (resolutionY+1) * (resolutionZ+1);
    
    printf("Using resolution %d x %d x %d\n", resolution, resolutionY, resolutionZ);
    
    grid->Build(vec3ui(resolution, resolutionY, resolutionZ), vec3f(dx), 
                bounds.pMin, VertexCentered);
    
    TEST_CHECK(grid->total == expec, "Incorrect amount of nodes");
    
    /* minimal hash tests first */
    for(int i = 0; i < grid->resolution.x; i++){
        for(int j = 0; j < grid->resolution.y; j++){
            for(int k = 0; k < grid->resolution.z; k++){
                vec3ui u(i,j,k);
                vec3f p = grid->GetVertexPosition(u);
                vec3f o = bounds.pMin + vec3f(i, j, k) * grid->spacing;
                
                grid->SetValueAt(i+j+k, u);
                Float f = grid->GetValueAt(u);
                TEST_CHECK((int)f == (i + j + k), "Incorrect hash value");
                for(int d = 0; d < 3; d++){
                    if(!IsZero(p[d] - o[d])){
                        printf("Error at : p = {%g %g %g} o = {%g %g %g}\n", 
                               p.x, p.y, p.z, o.x, o.y, o.z);
                        printf("Index is %d %d %d\n", i, j, k);
                        
                        grid->bounds.PrintSelf();
                        printf("\n");
                        
                    }
                    TEST_CHECK(IsZero(p[d] - o[d]), "Vertex position incorrect");
                }
            }
        }
    }
    
    Float sdMin =  FLT_MAX;
    Float sdMax = -FLT_MAX;
    /* make sdf */
#if 0
    CreateShapeSDFCPU(grid, mesh, shape);
#else
    GPULaunch(grid->total, CreateShapeSDFGPU, grid, mesh, shape);
#endif
    
    for(int i = 0; i < grid->resolution.x; i++){
        for(int j = 0; j < grid->resolution.y; j++){
            for(int k = 0; k < grid->resolution.z; k++){
                vec3ui u(i,j,k);
                Float f = grid->GetValueAt(u);
                if(f < sdMin) sdMin = f;
                if(f > sdMax) sdMax = f;
            }
        }
    }
    
    std::vector<vec3f> particles;
    vec3f p0 = grid->bounds.pMin;
    vec3f p1 = grid->bounds.pMax;
    Float h = 0.05;
    
    for(Float x = p0.x; x < p1.x; x += h){
        for(Float y = p0.y; y < p1.y; y += h){
            for(Float z = p0.z; z < p1.z; z += h){
                vec3f p(x, y, z);
                vec3f pt = p;
                for(int i = 0; i < 5; i++){
                    Float sdf = grid->Sample(pt);
                    if(Absf(sdf) < 0.001){
                        break;
                    }
                    
                    vec3f g = grid->Gradient(pt);
                    pt = pt - sdf * g;
                }
                
                particles.push_back(pt);
            }
        }
    }
    
    int size = particles.size();
    float *pos = new float[size * 3];
    float *col = new float[size * 3];
    
    int itp = 0;
    int itc = 0;
    
    printf("Got %d particles\n", size);
    
    for(vec3f &p : particles){
        pos[itp++] = p.x; pos[itp++] = p.y; pos[itp++] = p.z;
        col[itc++] = 1; col[itc++] = 0; col[itc++] = 0;
    }
    
#if 0
    for(int i = 0; i < grid->resolution.x; i++){
        for(int j = 0; j < grid->resolution.y; j++){
            for(int k = 0; k < grid->resolution.z; k++){
                vec3ui u(i,j,k);
                vec3f p = grid->GetVertexPosition(u);
                Float f = grid->GetValueAt(u);
                if(f > 0) continue;
                Float mp = LinearRemap(f, sdMin, sdMax, 0, 1);
                pos[itp++] = p.x; pos[itp++] = p.y; pos[itp++] = p.z;
                col[itc++] = 1 - mp; col[itc++] = mp; col[itc++] = 0;
            }
        }
    }
#endif
    
    vec3f at(3);
    vec3f to(0);
    
    graphy_set_3d(at.x, at.y, at.z, to.x, to.y, to.z, 45.0, 0.1f, 100.0f);
    graphy_render_points3f(pos, col, itp/3, h/2.0);
    
    getchar();
    
    graphy_close_display();
    printf("===== OK\n");
}

void test_uniform_grid3D(){
    printf("===== Test minimal 3D grid\n");
    Grid3 grid;
    vec3f p0(-1,-1, -1), p1(1,1,1);
    vec3ui res(10, 10, 10);
    Float expLen = 0.2;
    int count = 10 * 10 * 10;
    
    grid.Build(res, p0, p1);
    TEST_CHECK(IsZero(grid.cellsLen[0] - expLen), "Failed grid cell length");
    TEST_CHECK(grid.total == count, "Failed to build correct cell count");
    TEST_CHECK(grid.usizes[0] == 10 && grid.usizes[1] == 10 && grid.usizes[2] == 10, 
               "Invalid cell displacement");
    vec3f dif = grid.minPoint - p0;
    TEST_CHECK(dif.IsZeroVector(), "Invalid minimal point");
    
    /* Grid boundary not necessary is pMin = p0 and pMax = p1, so a inclusion test is better */
    Bounds3f bound = grid.GetBounds();
    TEST_CHECK(Inside(p0, bound), "Invalid grid boundary for minimal point");
    TEST_CHECK(Inside(p1, bound), "Invalid grid boundary for maximal point");
    
    dif = bound.pMin - p0;
    TEST_CHECK(dif.Length() < expLen, "Offset from minimal point to big");
    dif = bound.pMax - p1;
    TEST_CHECK(dif.Length() < expLen, "Offset from maximal point to big");
    
    Float hLen = expLen * 0.5;
    Float hhLen = hLen * 0.5;
    int ci = 0;
    int cj = 0;
    int ck = 0;
    for(Float x = p0.x; x < p1.x; x += expLen){
        if(p1.x - x < hLen) continue; // test does not fix out points
        cj = 0;
        for(Float y = p0.y; y < p1.y; y += expLen){
            if(p1.y - y < hLen) continue; // test does not fix out points
            ck = 0;
            for(Float z = p0.z; z < p1.z; z += expLen){
                if(p1.z - z < hLen) continue; // test does not fix out points
                
                vec3f c(x + hLen, y + hLen, z + hLen);
                vec3f v[] = {
                    /* half points */
                    vec3f(c.x - hhLen, c.y - hhLen, c.z - hhLen),
                    vec3f(c.x - hhLen, c.y + hhLen, c.z - hhLen),
                    vec3f(c.x + hhLen, c.y - hhLen, c.z - hhLen),
                    vec3f(c.x + hhLen, c.y + hhLen, c.z - hhLen),
                    /* center point */
                    vec3f(c.x, c.y, c.z - hhLen),
                };
                
                /* corners can be in other cell as long as it is a valid cell 
                * and neighbor to this cell.
                */
                vec3f cor[] = {
                    vec3f(c.x - hLen, c.y - hLen, c.z - hhLen),
                    vec3f(c.x + hLen, c.y + hLen, c.z - hhLen),
                    vec3f(c.x - hLen, c.y + hLen, c.z - hhLen),
                    vec3f(c.x + hLen, c.y - hLen, c.z - hhLen)
                };
                
                for(int i = 0; i < 5; i++){
                    vec3f p = v[i];
                    vec3ui hashu = grid.GetHashedPosition(p);
                    unsigned int hashid = grid.GetLinearHashedPosition(p);
                    TEST_CHECK(hashu[0] == ci && hashu[1] == cj && hashu[2] == ck,
                               "Failed hash position");
                    TEST_CHECK(hashid == (ci + 10 * cj + ck * 10 * 10),
                               "Failed linear hash position");
                }
                
                for(int i = 0; i < 4; i++){
                    vec3f p = cor[i];
                    vec3ui hashu = grid.GetHashedPosition(p);
                    unsigned int hashid = grid.GetLinearHashedPosition(p);
                    TEST_CHECK(hashid < count, "Corner not in a valid cell");
                    if(hashu[0] != ci || hashu[1] != cj || hashu[2] != ck){
                        int dx = Absf((int)hashu[0] - ci);
                        int dy = Absf((int)hashu[1] - cj);
                        int dz = Absf((int)hashu[2] - ck);
                        bool valid = dx < 2 && dy < 2 && dz < 2;
                        
                        if(!valid){
                            printf("Invalid corners: %d %d %d\n", dx, dy, dz);
                        }
                        
                        TEST_CHECK(valid, "Corner not in neighor cell");
                    }
                }
                
                ck++;
            }
            cj++;
        }
        
        ci++;
    }
    
    printf("===== OK\n");
}
