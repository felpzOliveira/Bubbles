#include <ply_loader.h>

void writePly(HostTriangleMesh3 *mesh, std::ostream *strm){
    //bool write_normals = mesh->point.size() == mesh->normals.size();
    //bool write_uvs = mesh->point.size() == mesh->uvs.size();
    (*strm) << "ply\n";
    (*strm) << "format ascii 1.0\n";
    (*strm) << "element vertex " << (int) mesh->points.size() << "\n";
    (*strm) << "property double x\n";
    (*strm) << "property double y\n";
    (*strm) << "property double z\n";
/*
    if(write_normals){
        (*strm) << "property double nx\n";
        (*strm) << "property double ny\n";
        (*strm) << "property double nz\n";
    }

    if(write_uvs){
        (*strm) << "property double u\n";
        (*strm) << "property double v\n";
    }
*/
    (*strm) << "element face " << mesh->numberOfTriangles() << "\n";
    (*strm) << "property list uchar int vertex_indices\n";
    (*strm) << "end_header\n";

    for(const auto &pt : mesh->points){
        (*strm) << pt.x << " " << pt.y << " " << pt.z << std::endl;
    }

    for(size_t i = 0; i < mesh->numberOfTriangles(); ++i){
        (*strm) << "3 ";
        for(int j = 0; j < 3; ++j){
            (*strm) << mesh->pointIndices[i][j];
            (*strm) << " ";
        }

        (*strm) << std::endl;
    }
}
