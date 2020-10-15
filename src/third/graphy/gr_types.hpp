#if !defined(GRAPHY_TYPES_HPP)
#define GRAPHY_TYPES_HPP

#include <vector>

typedef struct gr_display_t gr_display;
typedef void (*render_call)(gr_display *display);

typedef enum{
    GL_OP_WINDOW_REQUEST,
    GL_OP_SHADER_COMPILE,
    GL_OP_SET_POINTS,
    GL_OP_SET_POINTS3,
    GL_OP_SET_LINES,
    GL_OP_SET_PIXELS,
    GL_OP_NONE
}gr_opengl_op;

typedef struct gr_point_t{
    float pos[3];
    float color[3];
}gr_point;

typedef struct gr_shader_t{
    unsigned int id;
    int valid;
    char vertex_path[128];
    char fragment_path[128];
}gr_shader;

typedef struct gr_opengl_buffer_t{
    unsigned int vao;
    unsigned int vbo;
    unsigned int vbo_size;
    gr_shader shader;
    int is_valid;
}gr_opengl_buffer;

typedef struct gr_opengl_point_set_t{
    gr_opengl_buffer point_buffer;
    gr_point *points;
    float pointSize;
    int points_size;
    int points_len;
}gr_opengl_point_set;

typedef gr_opengl_point_set gr_opengl_point_set3;

typedef struct gr_opengl_pixels_texture_t{
    gr_opengl_buffer pixels_buffer;
    int res_x, res_y;
    unsigned int id;
}gr_opengl_pixels_texture;

typedef gr_opengl_point_set gr_opengl_line_set;

typedef struct gr_opengl_stack_t{
    render_call render_function;
    gr_opengl_point_set point_set;
    gr_opengl_point_set3 point_set3;
    gr_opengl_line_set line_set;
    gr_opengl_pixels_texture pixels_texture;
}gr_opengl_stack;

typedef struct gr_resolution_t{
    int x;
    int y;
}gr_resolution;

typedef struct gr_display_t{
    void *handle;
    int number;
    int valid;
    int changed;
    gr_resolution resolution;
    gr_opengl_stack opengl_stack;
    float r,g,b,a;
    
    float right, left;
    float bottom, top;
    float near, far;
    float fromx, fromy, fromz;
    float tox, toy, toz;
    float fov;
    int viewchange;
    float ortho[4][4];
    float perspective[4][4];
    float view[4][4];
}gr_display;

typedef struct gr_opengl_request_t{
    void *data;
    gr_opengl_op op;
    int has_request;
}gr_opengl_request;

typedef struct{
    int res_x;
    int res_y;
    float r,g,b,a;
    gr_display *display;
}gr_window_request;

typedef struct{
    gr_shader *shader;
}gr_shader_compile_request;

typedef struct{
    float *rgb;
    int res_x, res_y;
    gr_display *display;
}gr_pixels_update_request;

typedef struct{
    float *positions;
    float *colors;
    float pointSize;
    int point_count;
    gr_display *display;
}gr_point_update_request;

typedef struct{
    float *positions;
    float *colors;
    float radius;
    int point_count;
    gr_display *display;
}gr_point3_update_request;

typedef struct{
    float *positions;
    float *colors;
    int line_count;
    gr_display *display;
}gr_line_update_request;

#endif
