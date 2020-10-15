#if !defined(GR_MATRIX_HPP)
#define GR_MATRIX_HPP

void gr_math_orthographic(float matrix[4][4],
                          float left, float right,
                          float bottom, float top,
                          float near, float far);

void gr_math_lookat(float fromx, float fromy, float fromz,
                    float tox, float toy, float toz,
                    float matrix[4][4]);

void gr_math_perspective(float fov, float aspect, float near,
                         float far, float matrix[4][4]);

template<typename T> inline void gr_math_lookat(T from, T to, 
                                                float matrix[4][4])
{
    gr_math_lookat(from[0], from[1], from[2], to[0], to[1], to[2], matrix);
}

#endif
