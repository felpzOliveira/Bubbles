#if !defined(GRAPHY_DISPLAY_HPP)
#define GRAPHY_DISPLAY_HPP
#include <gr_types.hpp>

gr_display *gr_new_display(int res_x, int res_y);
gr_display *gr_new_display(int res_x, int res_y, float r, float g, float b, float a);

void gr_display_set_view2d(gr_display *display,
                           float left, float right,
                           float top,  float bottom);

void gr_display_set_view2d(gr_display *display,
                           float left, float right,
                           float top,  float bottom,
                           float near, float far);

void gr_display_set_view3d(gr_display *display, float fromx, float fromy, float fromz, 
                           float tox, float toy, float toz, float fov, float near, float far);

gr_display * gr_display_new_gl(int res_x, int res_y);
void gr_display_set_title_gl(gr_display *display);
void gr_destroy_display(gr_display *display);
#endif
