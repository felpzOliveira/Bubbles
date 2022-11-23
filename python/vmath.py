import taichi as ti

# linearly interpolates between two values:
#(left) - - - - - > dx   (right)
#  |                       |
#  |_______________________|
# dx must be in [0,1] range
@ti.func
def lerp(left, right, dx):
    return left + dx * (right - left)

# bilinearly interpolate between 4 values:
#(f01) - - - - - - > dx  dy (f11)
#  |                     ^  |
#  |                     |  |
#  |________________________|
#(f00)                      (f10)
# dx and dy must be in [0, 1] range
@ti.func
def bilerp(f00, f10, f01, f11, dx, dy):
    return lerp(lerp(f00, f10, dx), lerp(f01, f11, dx), dy)

