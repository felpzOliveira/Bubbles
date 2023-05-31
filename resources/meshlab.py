import pymeshlab
import sys
ms = pymeshlab.MeshSet()

path = "build/delaunay2.ply"
if len(sys.argv) > 1:
    path = sys.argv[1]

print("Loading mesh...")
ms.load_new_mesh(path)
print("Finished loading\nFiltering...")
ms.apply_coord_taubin_smoothing()
#ms.meshing_re_orient_faces_coherentely()
#for _ in range(0, 4):
 #   ms.apply_coord_taubin_smoothing()

print("Done filtering\nSaving...")

ms.save_current_mesh("test.ply")
print("Finished")
