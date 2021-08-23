#pragma once
#include <geometry.h>
#include <map>
#include <iostream>
#include <obj_loader.h>
#include <fstream>

#define ARGUMENT_PROCESS(name)\
int name(int argc, char **argv, int &i, const char *arg, void *config)
typedef ARGUMENT_PROCESS(arg_process);

typedef struct{
    arg_process *processor;
    std::string help;
}arg_desc;

inline void print_help_and_quit(const char *caller, 
                                std::map<const char *, arg_desc> argument_map)
{
    std::map<const char *, arg_desc>::iterator it;
    printf("[ %s ] Help:\n", caller);
    for(it = argument_map.begin(); it != argument_map.end(); it++){
        arg_desc desc = it->second;
        printf("  %s : %s\n", it->first, desc.help.c_str());
    }
    exit(0);
}

inline void argument_process(std::map<const char *, arg_desc> argument_map,
                             int argc, char **argv, const char *caller, void *config, 
                             int enforce=1, int start=1)
{
    int argCount = argc - start;
    if(argCount > 0){
        for(int i = start; i < argc; i++){
            int ok = -1;
            std::string arg(argv[i]);
            std::map<const char *, arg_desc>::iterator it;
            if(arg == "--help" || arg == "help"){
                print_help_and_quit(caller, argument_map);
            }
            for(it = argument_map.begin(); it != argument_map.end(); it++){
                if(arg == it->first){
                    arg_desc desc = it->second;
                    ok = desc.processor(argc, argv, i, it->first, config);
                    break;
                }
            }
            
            if(ok < 0){
                std::cout << "Failed processing \'" << arg << "\'" << std::endl;
                exit(0);
            }
            
            /* if return != 0 stop parsing (probably jumped to command) */
            if(ok != 0){ break; }
        }
    }else if(enforce){
        std::cout << "Missing argument." << std::endl;
        print_help_and_quit(caller, argument_map);
    }
}

inline std::string ParseNext(int argc, char **argv, int &i, 
                             const char *arg, int count=1)
{
    int ok = (argc > i+count) ? 1 : 0;
    if(!ok){
        printf("Invalid argument for %s\n", arg);
        exit(0);
    }
    
    std::string res;
    for(int n = 1; n < count+1; n++){
        res += std::string(argv[n+i]);
        if(n < count) res += " ";
    }
    
    i += count;
    return res;
}

template<typename Fn>
inline int ParseNextOrNone(int argc, char **argv, int &i,
                           const char *arg, Fn func)
{
    int ok = (argc > i + 1) ? 1 : 0;
    if(!ok) return 0;
    std::string res(argv[i+1]);
    int r = func(res);
    if(r){
        i += 1;
    }

    return r;
}

inline Float ParseNextFloat(int argc, char **argv, int &i, const char *arg){
    std::string value = ParseNext(argc, argv, i, arg);
    const char *token = value.c_str();
    return ParseFloat(&token);
}

inline bool FileExists(const char *path){
    std::ifstream ifs(path);
    return ifs.is_open();
}

void convert_command(int argc, char **argv);
void sdf_command(int argc, char **argv);
void pbrt_command(int argc, char **argv);
void view_command(int argc, char **argv);
void boundary_command(int argc, char **argv);
