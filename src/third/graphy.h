#pragma once

void graphy_render_points(float *pos, float rgb[3], int num, float left, 
                          float right, float top, float bottom);

void graphy_render_pointsEx(float *pos, float *colors, int num, float left, 
                            float right, float top, float bottom);

void graphy_render_points_size(float *pos, float *col, float pSize, int num, 
                               float left, float right, float top, float bottom);

void graphy_set_3d(float ex, float ey, float ez, float ox, float oy, 
                   float oz, float fov, float near, float far);

void graphy_render_points3(float *pos, float rgb[3], int num, float radius);

void graphy_render_points3f(float *pos, float *col, int num, float radius);

void graphy_render_lines(float *pos, float rgb[3], int num);

void graphy_set_orthographic(float left, float right, float top, float bottom);

void graphy_close_display();