#include <marching_squares.h>

// This list allows for any edge to find the target vertex pair
// that compose it inside a quad.
static int edgeList[4][2] = {
    {0, 1}, {1, 2}, {2, 3}, {3, 0}
};

static int edgeFlagsTable[16] = {
    0x000, 0x009, 0x003, 0x00a, 0x006, 0x00f, 0x005, 0x00c,
    0x00c, 0x005, 0x00f, 0x006, 0x00a, 0x003, 0x009, 0x000
};

static int triangleList[16][13] = {
    { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
    {  0,  4,  7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
    {  4,  1,  5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
    {  0,  1,  7,  1,  5,  7, -1, -1, -1, -1, -1, -1, -1 },
    {  5,  2,  6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
    {  0,  4,  7,  2,  6,  5, -1, -1, -1, -1, -1, -1, -1 },
    {  4,  1,  6,  1,  2,  6, -1, -1, -1, -1, -1, -1, -1 },
    {  0,  1,  7,  7,  1,  6,  1,  2,  6, -1, -1, -1, -1 },
    {  7,  6,  3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
    {  0,  4,  6,  0,  6,  3, -1, -1, -1, -1, -1, -1, -1 },
    {  3,  7,  6,  6,  7,  4,  6,  4,  5,  1,  5,  4, -1 },
    {  0,  6,  3,  0,  5,  6,  0,  1,  5, -1, -1, -1, -1 },
    {  7,  5,  3,  5,  2,  3, -1, -1, -1, -1, -1, -1, -1 },
    {  3,  0,  4,  3,  4,  5,  3,  5,  2, -1, -1, -1, -1 },
    {  2,  3,  7,  2,  7,  4,  2,  4,  1, -1, -1, -1, -1 },
    {  0,  1,  3,  1,  2,  3, -1, -1, -1, -1, -1, -1, -1 },
};

// Given the first index (0) of a cell, get the other ones.
// Returns the counter-clockwise list from the top-left corner, i.e.: {0, 1, 2, 3}.
//       6
//   3-------2
//   |       |
// 7 |       | 5
//   |       |
//   0-------1
//       4

int Get2DFieldCellIndexes(unsigned int h0, FieldGrid2f *grid, vec2ui *indexes){
    vec2ui u = DimensionalIndex(h0, grid->resolution, 2);
    int rv = -1;
    if(u.x+1 < grid->resolution.x && u.y+1 < grid->resolution.y){
        indexes[0] = u;
        indexes[1] = vec2ui(u.x+1,u.y);
        indexes[2] = vec2ui(u.x+1,u.y+1);
        indexes[3] = vec2ui(u.x,u.y+1);
        rv = 0;
    }
    return rv;
}

void DoSquare(Float *sdfs, vec3f *positions, Float isovalue,
              std::vector<vec3f> *triangles)
{
    // 1 - Find the vertices that are inside the surface
    int idx = 0;
    int edgeFlag = 0;
    int vertices[2];
    vec3f e[4];
    for(int nVert = 0; nVert < 4; nVert++){
        Float value = sdfs[nVert];
        if(value <= isovalue){
            idx |= 1UL << nVert;
        }
    }
    
    // 2 - Get the edges related to the vertices, lookup table.
    edgeFlag = edgeFlagsTable[idx];
    if(edgeFlag == 0) return;
    
    // 3 - Locate points of intersection
    for(int edge = 0; edge < 4; edge++){
        if(edgeFlag & (1UL << edge)){ // if intersectin on this edge
            // Get the vertices
            vertices[0] = edgeList[edge][0];
            vertices[1] = edgeList[edge][1];
            
            // get position
            vec3f p0 = positions[vertices[0]];
            vec3f p1 = positions[vertices[1]];
            
            Float phi0 = sdfs[vertices[0]] - isovalue;
            Float phi1 = sdfs[vertices[1]] - isovalue;
            
            Float alpha = 0;
            // Page 70. Equation 1.62
            if(Absf(phi0) + Absf(phi1) > 1e-12){
                alpha = Absf(phi0) / (Absf(phi0) + Absf(phi1));
            }else{
                alpha = 0.5;
            }
            
            if(alpha < 0.00001f) alpha = 0.00001f;
            if(alpha > 0.99999f) alpha = 0.99999f;
            
            vec3f pe = ((1.0f - alpha) * p0 + alpha * p1);
            e[edge] = pe;
        }
    }
    
    // Build triangles
    for(int tri = 0; tri < 4; tri++){
        if(triangleList[idx][3 * tri] < 0){ // no more triangles
            break;
        }
        
        for(int j = 0; j < 3; j++){
            int vId = triangleList[idx][3 * tri + j];
            // TODO: We need to find if vId is already in the list of triangles
            //       so that we can connect them.
            vec3f p;
            if(vId < 4){ // it is one of the corners of the quad
                p = positions[vId];
            }else{ // it is in a edge
                p = e[vId - 4];
            }
            
            triangles->push_back(p);
        }
    }
}

void MarchingSquares(FieldGrid2f *grid, Float isovalue,
                     std::vector<vec3f> *triangles)
{
    AssureA(grid->type == VertexType::VertexCentered,
            "Marching squares is only supported for VertexCentered grid");
    
    vec2ui indexes[4];
    vec3f positions[4];
    Float sdfs[4];
    for(int i = 0; i < grid->total; i++){
        if(Get2DFieldCellIndexes(i, grid, indexes) == 0){
            for(int k = 0; k < 4; k++){
                vec2f p = grid->GetDataPosition(indexes[k]);
                positions[k] = vec3f(p.x, p.y, 0);
                sdfs[k] = grid->GetValueAt(indexes[k]);
            }
            
            DoSquare(sdfs, positions, isovalue, triangles);
        }
    }
}