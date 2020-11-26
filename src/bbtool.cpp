#include <args.h>

ARGUMENT_PROCESS(convert_cmd){
    convert_command(argc-1, &argv[1]);
    return 1;
}

ARGUMENT_PROCESS(sdf_cmd){
    sdf_command(argc-1, &argv[1]);
    return 1;
}

ARGUMENT_PROCESS(pbrt_cmd){
    pbrt_command(argc-1, &argv[1]);
    return 1;
}

ARGUMENT_PROCESS(view_cmd){
    view_command(argc-1, &argv[1]);
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
    {"pbrt",
        { .processor = pbrt_cmd, 
            .help = "Generates a rendarable PBRT point cloud geometry from Bubbles output."} 
    },
    {"view",
        { .processor = view_cmd, 
            .help = "Uses Graphy to display a saved Bubbles simulation."} 
    }
};

int main(int argc, char **argv){
    printf("* BubblesTool - Built %s at %s *\n", __DATE__, __TIME__);
    argument_process(command_map, argc, argv, "BubblesTool", nullptr);
    return 0;
}