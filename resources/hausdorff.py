import pymeshlab
import sys

if len(sys.argv) < 3:
    print("Invalid usage, call: <base_mesh> <reconstructed_mesh> [samples]")
    sys.exit(0)

base_path = sys.argv[1]
rec_path  = sys.argv[2]
samples = 1000

if len(sys.argv) > 3:
    samples = int(sys.argv[3])

ms = pymeshlab.MeshSet()
ms.load_new_mesh(base_path)
ms.load_new_mesh(rec_path)

print("[Running for " + str(samples) + " samples]")
hausdorf = ms.get_hausdorff_distance(sampledmesh=1, targetmesh=0, samplenum=samples)

print("- Hausdorff:\n > RMS: " + str(hausdorf['RMS']) + "\n > Max: " + str(hausdorf['max']))
