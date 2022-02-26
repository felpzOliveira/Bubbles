#include <args.h>
#include <util.h>

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
