#include <dlfcn.h>
#include <iostream>
#include <graphy.h>
#include <gr_display.hpp>
#include <gr_opengl.hpp>
#include <chrono>
#include <thread>
#include <lodepng.h>
#define GraphyPath "/home/felpz/Documents/Graphics/build/libgraphy.so"

#define NEW_DISPLAY "_Z14gr_new_displayii"
#define NEW_DISPLAY_EX "_Z14gr_new_displayiiffff"
#define RENDER_PTS "_Z23gr_opengl_render_pointsPfS_iP12gr_display_t"
#define RENDER_PTS3 "_Z24gr_opengl_render_points3PfS_ifP12gr_display_t"
#define RENDER_PTS_SIZE "_Z23gr_opengl_render_pointsPfS_ifP12gr_display_t"
#define SET_VIEW_2D "_Z21gr_display_set_view2dP12gr_display_tffff"
#define SET_VIEW_3D "_Z21gr_display_set_view3dP12gr_display_tfffffffff"
#define CLOSE_DISPLAY "_Z16gr_close_displayP12gr_display_t"
#define RENDER_LINES "_Z22gr_opengl_render_linesPfS_iP12gr_display_t"
#define RENDER_PIXELS "_Z23gr_opengl_render_pixelsPfiiP12gr_display_t"

#define GMIN(a, b) ((a) < (b) ? (a) : (b))
#define GMAX(a, b) ((a) > (b) ? (a) : (b))

static int display_width = 1000;
static int display_height = 1000;

static float *colors;
static float *tmpColors = nullptr;
static int tmpWidth, tmpHeight;
static float numPt = 0;
static gr_display *display = nullptr;

typedef gr_display*(*GraphyGetDisplayEx)(int, int, float, float, float, float);
typedef gr_display*(*GraphyGetDisplay)(int, int);
typedef void(*GraphyRenderPoints)(float*, float*, int, gr_display *);
typedef void(*GraphyRenderLines)(float*, float*, int, gr_display *);
typedef void(*GraphyRenderPointsSize)(float*, float*, int, float, gr_display *);
typedef void(*GraphySetView2D)(gr_display *, float, float, float, float);
typedef void(*GraphySetView3D)(gr_display *, float, float, float, float,
                               float, float, float, float, float);
typedef void(*GraphyRenderPoints3D)(float *, float *, int, float, gr_display *);
typedef void(*GraphyCloseDisplay)(gr_display *);
typedef void(*GraphyRenderPixels)(float *, int, int, gr_display *);

void *GraphyHandle = nullptr;
GraphyGetDisplayEx graphy_get_displayEx;
GraphyGetDisplay graphy_get_display;
GraphyRenderPoints graphy_render_pts;
GraphyRenderPointsSize graphy_render_pts_size;
GraphySetView2D graphy_set_view2D;
GraphySetView3D graphy_set_view3D;
GraphyRenderPoints3D graphy_render_pts_3d;
GraphyCloseDisplay graphy_display_close;
GraphyRenderLines graphy_lines_render;
GraphyRenderPixels graphy_render_pixels;

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
    graphy_lines_render = (GraphyRenderLines) LoadSymbol(GraphyHandle, RENDER_LINES);
    graphy_render_pixels = (GraphyRenderPixels) LoadSymbol(GraphyHandle, RENDER_PIXELS);

    return (graphy_get_display && graphy_render_pts &&
            graphy_set_view2D && graphy_render_pts_size &&
            graphy_set_view3D && graphy_render_pts_3d &&
            graphy_lines_render && graphy_display_close &&
            graphy_render_pixels) ? 1 : 0;
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
    if(display){
        graphy_display_close(display);
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        display = nullptr;
        graphy_ok = 0;
        if(colors) delete[] colors;
        colors = nullptr;
    }
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

typedef struct RGB{
    float r, g, b;
}RGB;

RGB rgb_mult(RGB rgb, float v){
    RGB _rgb;
    _rgb.r = rgb.r * v;
    _rgb.g = rgb.g * v;
    _rgb.b = rgb.b * v;
    return _rgb;
}

RGB rgb_add(RGB rgb0, RGB rgb1){
    RGB _rgb;
    _rgb.r = rgb0.r + rgb1.r;
    _rgb.g = rgb0.g + rgb1.g;
    _rgb.b = rgb0.b + rgb1.b;
    return _rgb;
}

static RGB graphy_get_tmp_pixel_at(int x, int y){
    int index = x + tmpWidth * y;
    RGB rgb;
    rgb.r = tmpColors[3 * index + 0];
    rgb.g = tmpColors[3 * index + 1];
    rgb.b = tmpColors[3 * index + 2];
    return rgb;
}

static void graphy_upscale_pixel_value(int x, int y){
#if 0
    float ex = (float)x / (float)display_width;
    float ey = (float)y / (float)display_height;
    int p_x = (int)(ex * tmpWidth);
    int p_y = (int)(ey * tmpHeight);
    int index = p_x + tmpWidth * p_y;

    float r = tmpColors[3 * index + 0];
    float g = tmpColors[3 * index + 1];
    float b = tmpColors[3 * index + 2];

    index = x + display_width * y;
    colors[3 * index + 0] = r;
    colors[3 * index + 1] = g;
    colors[3 * index + 2] = b;
#else
    float scale_x = (float)display_width / (float)tmpWidth;
    float scale_y = (float)display_height / (float)tmpHeight;
    float x_ = (float)x / scale_x;
    float y_ = (float)y / scale_y;

    int x1 = GMIN((int)(std::floor(x_)), tmpWidth-1);
    int y1 = GMIN((int)(std::floor(y_)), tmpHeight-1);
    int x2 = GMIN((int)(std::ceil(x_)), tmpWidth-1);
    int y2 = GMIN((int)(std::ceil(y_)), tmpHeight-1);

    RGB e11 = graphy_get_tmp_pixel_at(x1, y1);
    RGB e12 = graphy_get_tmp_pixel_at(x2, y1);
    RGB e21 = graphy_get_tmp_pixel_at(x1, y2);
    RGB e22 = graphy_get_tmp_pixel_at(x2, y2);

    RGB e1 = rgb_add(rgb_mult(e11, (float)x2 - x_),
                     rgb_mult(e12, x_ - (float)x1));
    RGB e2 = rgb_add(rgb_mult(e21, (float)x2 - x_),
                     rgb_mult(e22, x_ - (float)x1));
    if(x1 == x2){
        e1 = e11;
        e2 = e22;
    }

    RGB e = rgb_add(rgb_mult(e1, (float)y2 - y_),
                    rgb_mult(e2, y_ - (float)y1));

    if(y1 == y2){
        e = e1;
    }

    int index = x + display_width * y;
    colors[3 * index + 0] = e.r;
    colors[3 * index + 1] = e.g;
    colors[3 * index + 2] = e.b;
#endif
}

static void graphy_upscale_pixels(){
    for(int x = 0; x < display_width; x++){
        for(int y = 0; y < display_height; y++){
            graphy_upscale_pixel_value(x, y);
        }
    }
}

void graphy_set_image_ptr(float **rgb, int width, int height, int n_width, int n_height){
    if(colors) delete[] colors;
    if(n_width <= 0 || n_height <= 0){
        colors = *rgb;
        display_width = width;
        display_height = height;
    }else{
        tmpWidth = width;
        tmpHeight = height;
        tmpColors = *rgb;
        display_width = n_width;
        display_height = n_height;
        colors = new float[3 * n_width * n_height];
        graphy_upscale_pixels();
    }
}

void graphy_display_pixels(){
    if(graphy_ok == 0) graphy_initialize(display_width, display_height);
    if(graphy_ok > 0){
        if(tmpColors){
            graphy_upscale_pixels();
        }
        graphy_render_pixels(colors, display_width, display_height, display);
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

void graphy_set_orthographic(float left, float right, float top, float bottom){
    if(graphy_ok == 0) graphy_initialize(display_width, display_height);
    if(graphy_ok > 0){
        graphy_set_view2D(display, left, right, top, bottom);
    }
}

void graphy_render_lines(float *pos, float rgb[3], int num){
    if(graphy_ok == 0) graphy_initialize(display_width, display_height);
    if(graphy_ok > 0){
        set_colors_by_rgb(rgb, num);
        graphy_lines_render(pos, colors, num, display);
    }
}

void graphy_render_points3(float *pos, float rgb[3], int num, float radius){
    if(graphy_ok == 0) graphy_initialize(display_width, display_height);
    if(graphy_ok > 0){
        set_colors_by_rgb(rgb, num);
        graphy_render_pts_3d(pos, colors, num, radius, display);
    }
}

void graphy_write_image(float *rgbs, int channels, int _width,
                        int _height, const char *path)
{
    float *cols = rgbs ? rgbs : colors;
    int width = rgbs ? _width : display_width;
    int height = rgbs ? _height : display_height;
    channels = rgbs ? channels : 3;

    int imChannels = channels;

    int length = width * height;
    if(channels <= 0){
        printf("No image channels given\n");
        exit(0);
    }

    if(channels > 4){
        printf("Unsupported image channel count ( %d )\n", channels);
        imChannels = 4;
    }

    unsigned char *ptr = new unsigned char[4 * length];
    for(int x = 0; x < width; x++){
        for(int y = 0; y < height; y++){
            int i = x + y * width;
            int inv = x + (height - 1 - y) * width;
            ptr[4 * i + 0] = 0;
            ptr[4 * i + 1] = 0;
            ptr[4 * i + 2] = 0;
            ptr[4 * i + 3] = 255;
            for(int n = 0; n < imChannels; n++){
                float fval = cols[channels * inv + n];
                fval = fval < 0 ? 0 : (fval > 1 ? 1 : fval);
                ptr[4 * i + n] = (unsigned char)(fval * 255.f);
            }
        }
    }
    lodepng_encode32_file(path, ptr, width, height);

    delete[] ptr;
}
