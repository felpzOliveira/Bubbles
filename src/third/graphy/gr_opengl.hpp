#if !defined(GRAPHY_OPENGL_HPP)
#define GRAPHY_OPENGL_HPP

#include "gr_types.hpp"

void gr_opengl_window_request(gr_window_request *request);
void gr_opengl_close_window_request(gr_destroy_window_request *request);
gr_shader gr_opengl_shader(const char *vertexFile, const char *fragmentFile);

void gr_shader_use(gr_shader *shader);

void gr_opengl_render_points(float *pos, float *colors, 
                             int num, gr_display *display);

void gr_opengl_render_points(float *pos, float *colors,
                             int num, float pointSize, gr_display *display);

void gr_opengl_render_points3(float *pos, float *colors, int num,
                              float radius, gr_display *display);

void gr_opengl_render_lines(float *pos, float *colors,
                            int num, gr_display *display);

void gr_opengl_render_pixels(float *rgb, int res_x, int res_y,
                             gr_display *display);

void gr_eternal_loop();

#endif
