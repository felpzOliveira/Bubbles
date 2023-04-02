import taichi as ti
import core.vmath
import struct

def write_n_floats(file_fd, n, values):
    s = struct.pack('f'*len(values), *values)
    file_fd.write(s)

def write_n_ints(file_fd, n, values):
    s = struct.pack('i'*len(values), *values)
    file_fd.write(s)

@ti.kernel
def smooth_field(density: ti.template(), smooth: ti.template()):
    for i, j, k in density:
        edge = 3
        edgef = 3.0
        nx = density.shape[0]
        ny = density.shape[1]
        nz = density.shape[2]
        d_val = density[i,j,k]
        if i < edge:
            d_val *= core.vmath.smoothstep(0.0, edgef, float(i))
        if i > nx-1-edge:
            d_val *= core.vmath.smoothstep(0.0, edgef, float(nx-1-i))
        if j < edge:
            d_val *= core.vmath.smoothstep(0.0, edgef, float(j))
        if j > ny-1-edge:
            d_val *= core.vmath.smoothstep(0.0, edgef, float(ny-1-j))
        if k < edge:
            d_val *= core.vmath.smoothstep(0.0, edgef, float(k))
        if k > nz-1-edge:
            d_val *= core.vmath.smoothstep(0.0, edgef, float(nz-1-k))

        smooth[i,j,k] = d_val

def as_vol(densities, file_path, ds):
    # TODO: move this allocation?
    smooth = ti.field(float, shape=densities.shape)
    smooth_field(densities, smooth)

    with open(file_path, 'wb') as fd:
        data_vals = []
        data = smooth.to_numpy()
        nx = densities.shape[0]
        ny = densities.shape[1]
        nz = densities.shape[2]

        dx = ds[0]
        dy = ds[1]
        dz = ds[2]
        len_x = nx * dx
        len_y = ny * dy
        len_z = nz * dz
        h_lenx = 0.5 * len_x
        h_leny = 0.5 * len_y
        h_lenz = 0.5 * len_z
        p0x, p1x = -h_lenx, h_lenx
        p0y, p1y = -h_leny, h_leny
        p0z, p1z = -h_lenz, h_lenz
        bounds = [p0x, p0y, p0z, p1x, p1y, p1z]

        # 1 - 'VOL' + version = 3
        header = bytearray([ord('V'), ord('O'), ord('L'), 3])
        fd.write(header)
        # 2 - encoding = float 32 bits (1)
        write_n_ints(fd, 1, [1])
        # 3 - resolution = 3 ints
        write_n_ints(fd, 3, [nx, ny, nz])
        # 4 - channels = 1 int
        write_n_ints(fd, 1, [1])
        # 5 - bounds = 6 floats
        write_n_floats(fd, 6, bounds)
        # 6 - write data
        counter = 0
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    # TODO: Why does this needs to be (k, j, i) to match C?
                    data_vals.append(data[k][j][i])
                    counter += 1
        write_n_floats(fd, counter, data_vals)
