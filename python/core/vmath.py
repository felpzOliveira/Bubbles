import taichi as ti

# linearly interpolates between two values:
#(left) - - - - - > dx   (right)
#  |                       |
#  |_______________________|
# dx must be in [0,1] range
@ti.func
def lerp(left, right, dx):
    return left + dx * (right - left)

# interpolate between 4 values:
#(f01) - - - - - - > dx  dy (f11)
#  |                     ^  |
#  |                     |  |
#  |________________________|
#(f00)                      (f10)
# dx and dy must be in [0, 1] range
@ti.func
def bilerp(f00, f10, f01, f11, dx, dy):
    return lerp(lerp(f00, f10, dx), lerp(f01, f11, dx), dy)

@ti.func
def trilerp(f000, f100, f010, f110, f001, f101, f011, f111, dx, dy, dz):
    return lerp(bilerp(f000, f100, f010, f110, dx, dy),
                bilerp(f001, f101, f011, f111, dx, dy), dz)

@ti.func
def clamp(val, low, high):
    res = val
    if val < low:
        res = low
    elif val > high:
        res = high
    return res

@ti.func
def smoothstep(edge0, edge1, x):
    t = clamp((x - edge0) / (edge1 - edge0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)
