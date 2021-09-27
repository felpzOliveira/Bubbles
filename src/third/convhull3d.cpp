#include <convhull3d.h>
#include <geometry.h>

#if defined(USE_QHULL)
extern "C" {
    #include <libqhull_r/libqhull_r.h>
}
#else
#define CONVHULL_3D_ENABLE
#include <convhull3d_impl.h>
#endif

#if defined(USE_QHULL)
FILE *fp = NULL;
__host__ void ConvexHullPrepare(){
    if(!fp) fp = fopen("/dev/null", "w");
    QHULL_LIB_CHECK
}

__host__ void ConvexHullFinish(){
    if(fp) fclose(fp);
}

__host__ void ConvexHull3D(IndexedParticle<vec3f> *ips, int maxLen,
                           int len, std::function<void(int)> reporter)
{
    coordT *ps = new coordT[3 * maxLen];
    qhT qh_qh;
    qhT *qh= &qh_qh;
    boolT ismalloc = 0;
    qh_init_A(qh, fp, fp, stderr, 0, NULL);
    qh_option(qh, "qhull s FA", NULL, NULL);
    qh->NOerrexit = False;
    qh_initflags(qh, qh->qhull_command);

    for(int j = 0; j < len; j++){
        ps[3 * j + 0] = ips[j].p.x;
        ps[3 * j + 1] = ips[j].p.y;
        ps[3 * j + 2] = ips[j].p.z;
    }

    qh_init_B(qh, ps, len, 3, ismalloc);
    (void)ismalloc;
    qh_qhull(qh);
    qh_check_output(qh);
    qh_produce_output(qh);

    vertexT *vertex;
    for(vertex = qh->vertex_list; vertex && vertex->next; vertex = vertex->next){
        unsigned int id = qh_pointid(qh, vertex->point);
        reporter(id);
    }

    qh_freeqhull(qh, qh_ALL);
    delete[] ps;
}

#else
__host__ void ConvexHullPrepare(){
    static int warned = 0;
    if(!warned){
        printf("[Warning] : ConvexHull is not using QHull implementation\n");
        warned = 1;
    }
}

__host__ void ConvexHullFinish(){}

__host__ void ConvexHull3D(IndexedParticle<vec3f> *ips, int maxLen,
                           int len, std::function<void(int)> reporter)
{
    int *faceIndices = NULL;
    int nFaces = 0;
    ch_vertex *vertex = new ch_vertex[len];
    for(int i = 0; i < len; i++){
        vertex[i].x = ips[i].p.x;
        vertex[i].y = ips[i].p.y;
        vertex[i].z = ips[i].p.z;
    }

    convhull_3d_build(vertex, len, &faceIndices, &nFaces);

    for(int i = 0; i < nFaces; i++){
        reporter(faceIndices[i]);
    }

    delete[] vertex;
    free(faceIndices);
}

#endif
