#include <dlfcn.h>
#include <iostream>
#include <graphy.h>
#include <gr_display.hpp>
#include <gr_opengl.hpp>
#define GraphyPath "/home/felipe/Documents/Graphics/build/libgraphy.so"

#define NEW_DISPLAY "_Z14gr_new_displayii"
#define NEW_DISPLAY_EX "_Z14gr_new_displayiiffff"
#define RENDER_PTS "_Z23gr_opengl_render_pointsPfS_iP12gr_display_t"
#define RENDER_PTS3 "_Z24gr_opengl_render_points3PfS_ifP12gr_display_t"
#define RENDER_PTS_SIZE "_Z23gr_opengl_render_pointsPfS_ifP12gr_display_t"
#define SET_VIEW_2D "_Z21gr_display_set_view2dP12gr_display_tffff"
#define SET_VIEW_3D "_Z21gr_display_set_view3dP12gr_display_tfffffffff"
#define CLOSE_DISPLAY "_Z18gr_destroy_displayP12gr_display_t"

static int display_width = 1000;
static int display_height = 1000;

static float *colors;
static float numPt = 0;
static gr_display *display = nullptr;

typedef gr_display*(*GraphyGetDisplayEx)(int, int, float, float, float, float);
typedef gr_display*(*GraphyGetDisplay)(int, int);
typedef void(*GraphyRenderPoints)(float*, float*, int, gr_display *);
typedef void(*GraphyRenderPointsSize)(float*, float*, int, float, gr_display *);
typedef void(*GraphySetView2D)(gr_display *, float, float, float, float);
typedef void(*GraphySetView3D)(gr_display *, float, float, float, float,
                               float, float, float, float, float);
typedef void(*GraphyRenderPoints3D)(float *, float *, int, float, gr_display *);
typedef void(*GraphyCloseDisplay)(gr_display *);

void *GraphyHandle = nullptr;
GraphyGetDisplayEx graphy_get_displayEx;
GraphyGetDisplay graphy_get_display;
GraphyRenderPoints graphy_render_pts;
GraphyRenderPointsSize graphy_render_pts_size;
GraphySetView2D graphy_set_view2D;
GraphySetView3D graphy_set_view3D;
GraphyRenderPoints3D graphy_render_pts_3d;
GraphyCloseDisplay graphy_display_close;

static int graphy_ok = 0;

static void * LoadSymbol(void *handle, const char *name){
    void *ptr = dlsym(handle, name);
    if(!ptr){
        std::cout << "Failed to load symbol " << name << " [ " << 
            dlerror() << " ]" << std::endl;
    }
    
    return ptr;
}

static int LoadFunctions(){
    graphy_get_display = (GraphyGetDisplay) LoadSymbol(GraphyHandle, NEW_DISPLAY);
    graphy_get_displayEx = (GraphyGetDisplayEx) LoadSymbol(GraphyHandle, NEW_DISPLAY_EX);
    graphy_render_pts = (GraphyRenderPoints) LoadSymbol(GraphyHandle, RENDER_PTS);
    graphy_render_pts_size = (GraphyRenderPointsSize) LoadSymbol(GraphyHandle, 
                                                                 RENDER_PTS_SIZE);
    graphy_set_view2D = (GraphySetView2D) LoadSymbol(GraphyHandle, SET_VIEW_2D);
    graphy_set_view3D = (GraphySetView3D) LoadSymbol(GraphyHandle, SET_VIEW_3D);
    graphy_render_pts_3d = (GraphyRenderPoints3D) LoadSymbol(GraphyHandle, RENDER_PTS3);
    graphy_display_close = (GraphyCloseDisplay) LoadSymbol(GraphyHandle, CLOSE_DISPLAY);
    
    return (graphy_get_display && graphy_render_pts && 
            graphy_set_view2D && graphy_render_pts_size &&
            graphy_set_view3D && graphy_render_pts_3d && graphy_display_close) ? 1 : 0;
}

static void graphy_initialize(int width, int height){
    if(!display){
        if(!GraphyHandle){
            GraphyHandle = dlopen(GraphyPath, RTLD_LAZY);
            graphy_ok = -1;
            if(!GraphyHandle){
                std::cout << "Failed to get Graphy library pointer" << std::endl;
            }else{
                if(LoadFunctions()){
                    display = graphy_get_display(width, height);
                    graphy_ok = 1;
                }else{
                    std::cout << "Failed to load Graphy symbols" << std::endl;
                }
            }
        }else if(graphy_get_display){
            display = graphy_get_display(width, height);
            graphy_ok = 1;
        }else{
            std::cout << "Unknown graphy state" << std::endl;
        }
    }
}

void graphy_close_display(){
    graphy_display_close(display);
    display = nullptr;
    graphy_ok = 0;
    if(colors) delete[] colors;
    colors = nullptr;
}

static void set_colors_by_rgb(float rgb[3], int num){
    if(numPt < num){
        if(colors) delete[] colors;
        colors = new float[3 * num];
        numPt = num;
    }
    
    for(int i = 0; i < num; i++){
        colors[3 * i + 0] = rgb[0];
        colors[3 * i + 1] = rgb[1];
        colors[3 * i + 2] = rgb[2];
    }
}

void graphy_render_points_size(float *pos, float *col, float pSize, int num, 
                               float left, float right, float top, float bottom)
{
    if(graphy_ok == 0) graphy_initialize(display_width, display_height);
    if(graphy_ok > 0){
        graphy_set_view2D(display, left, right, top, bottom);
        graphy_render_pts_size(pos, col, num, pSize, display);
    }
}

void graphy_render_pointsEx(float *pos, float *col, int num, float left, 
                            float right, float top, float bottom)
{
    if(graphy_ok == 0) graphy_initialize(display_width, display_height);
    if(graphy_ok > 0){
        graphy_set_view2D(display, left, right, top, bottom);
        graphy_render_pts(pos, col, num, display);
    }
}

void graphy_render_points(float *pos, float rgb[3], int num, float left, 
                          float right, float top, float bottom)
{
    if(graphy_ok == 0) graphy_initialize(display_width, display_height);
    if(graphy_ok > 0){
        set_colors_by_rgb(rgb, num);
        graphy_set_view2D(display, left, right, top, bottom);
        graphy_render_pts(pos, colors, num, display);
    }
}

void graphy_set_3d(float ex, float ey, float ez, float ox, float oy, 
                   float oz, float fov, float near, float far)
{
    if(graphy_ok == 0) graphy_initialize(display_width, display_height);
    if(graphy_ok > 0){
        graphy_set_view3D(display, ex, ey, ez, ox, oy, oz, fov, near, far);
    }
}

void graphy_render_points3f(float *pos, float *col, int num, float radius){
    if(graphy_ok == 0) graphy_initialize(display_width, display_height);
    if(graphy_ok > 0){
        graphy_render_pts_3d(pos, col, num, radius, display);
    }
}

void graphy_render_points3(float *pos, float rgb[3], int num, float radius){
    if(graphy_ok == 0) graphy_initialize(display_width, display_height);
    if(graphy_ok > 0){
        set_colors_by_rgb(rgb, num);
        graphy_render_pts_3d(pos, colors, num, radius, display);
    }
}