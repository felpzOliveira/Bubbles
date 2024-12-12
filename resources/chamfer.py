import sys
import numpy as np
from scipy.spatial import KDTree

def get_point_cloud(path):
    result = []
    with open(path, 'r') as file:
        for line in file:
            coords = line.strip().split()
            result.append([float(coord) for coord in coords])
    if len(result) == 0:
        print("Empty point set?")
        sys.exit(0)

    return result

if len(sys.argv) != 3:
    print("Incorrect arguments, use <cloud_0> <cloud_1>")
    sys.exit(0)

cloud_0_path = sys.argv[1]
cloud_1_path = sys.argv[2]

point_cloud0 = get_point_cloud(cloud_0_path)
point_cloud1 = get_point_cloud(cloud_1_path)

print("Read Cloud0= " + str(len(point_cloud0)) +
      " points, Cloud1= " + str(len(point_cloud1)) + " points")

print("Building KDTrees...")
tree_0 = KDTree(point_cloud0)
tree_1 = KDTree(point_cloud1)

print("Querying distance for forward tree...")
dist_forward, _ = tree_0.query(point_cloud1)
print("Querying distance for backward tree...")
dist_backward, _ = tree_1.query(point_cloud0)
chamfer = np.mean(dist_forward) + np.mean(dist_backward)

print(" -- Forward")
print(dist_forward)
print(" -- Backward")
print(dist_backward)

print(" -- Chamfer")
print(chamfer)
