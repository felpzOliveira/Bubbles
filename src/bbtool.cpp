#include <args.h>
#include <util.h>
#include <map>
#include <iomanip>
#include <gDel3D/GpuDelaunay.h>

int main2(int argc, char **argv){
    vec3f vs[] = {
    #if 1
        vec3f(31.174, -67.3714, 67.002), vec3f(-15.9469, 65.1594, 74.1617),
        vec3f(-69.4628, 70.5513, -14.0509), vec3f(-64.4575, -49.9974, 57.8402),
        vec3f(11.4315, -59.444, -79.5973), vec3f(64.5476, 65.503, -39.2806),
        vec3f(45.1882, 54.7837, 70.4043), vec3f(82.0726, -13.4776, -55.5197),
        vec3f(-38.9737, -43.1381, -81.3643), vec3f(20.6323, -79.6351, 56.8556),
        vec3f(-53.2168, 67.5106, 51.0911), vec3f(-48.0959, -27.2921, 83.3182),
        vec3f(-53.2083, 34.1737, 77.4663), vec3f(26.0799, -79.9647, -54.0878),
        vec3f(19.2022, 81.2582, -55.0308),
    #else
        vec3f(0, 3, 0), vec3f(-3, 0, 3), vec3f(3, 0, 3), vec3f(-3, 0, -3), vec3f(3, 0, -3)
        //vec3f(0, 10, 1), vec3f(-3, 0, 5), vec3f(3, 0, 1), vec3f(-3, 0, -2), vec3f(3, 0, -8)
        //vec3f(0, 10, 0), vec3f(-3, 0, 3), vec3f(3, 0, 3), vec3f(-3, 0, -3), vec3f(3, 0, -3)
    #endif
    };

    int pointNum = sizeof(vs) / sizeof(vs[0]);

    GpuDel triangulator;
    Point3HVec pointVec;
    GDelOutput output;

    for(int i = 0; i < pointNum; i++){
        vec3f vi = vs[i];
        pointVec.push_back({vi.x, vi.y, vi.z});
    }

    triangulator.compute( pointVec, &output );
    UtilGDel3DWritePly(&pointVec, &output, pointNum, "test.ply", false);

    return 0;
}

ARGUMENT_PROCESS(convert_cmd){
    convert_command(argc-1, &argv[1]);
    return 1;
}

ARGUMENT_PROCESS(sdf_cmd){
    sdf_command(argc-1, &argv[1]);
    return 1;
}

ARGUMENT_PROCESS(pbr_cmd){
    pbr_command(argc-1, &argv[1]);
    return 1;
}

ARGUMENT_PROCESS(view_cmd){
    view_command(argc-1, &argv[1]);
    return 1;
}

ARGUMENT_PROCESS(boundary_cmd){
    boundary_command(argc-1, &argv[1]);
    return 1;
}

ARGUMENT_PROCESS(surface_cmd){
    surface_command(argc-1, &argv[1]);
    return 1;
}

std::map<const char *, arg_desc> command_map = {
    {"convert",
        { .processor = convert_cmd,
            .help = "Performs multiple conversions between particles and meshes."}
    },
    {"sdf",
        { .processor = sdf_cmd,
            .help = "Manipulates different SDF."}
    },
    {"pbr",
        { .processor = pbr_cmd,
            .help = "Generates a rendarable PBRT/LIT point cloud geometry from Bubbles output."}
    },
    {"view",
        { .processor = view_cmd,
            .help = "Uses Graphy to display a saved Bubbles simulation."}
    },
    {"boundary",
        { .processor = boundary_cmd,
            .help = "Perform different types of boundary computation." }
    },
    {"surface",
        { .processor = surface_cmd,
            .help = "Perform surface reconstruction." }
    }
};

int main(int argc, char **argv){
    BB_MSG("Bubbles Tool");
    argument_process(command_map, argc, argv, "Bubbles Tool", nullptr);
    return 0;
}
