import bpy
import sys

path = "/home/felpz/Documents/Bubbles/test.ply"
if len(sys.argv) > 4:
    path = sys.argv[4]

bpy.ops.object.select_all(action='DESELECT')
bpy.data.objects['Cube'].select_set(True)
bpy.ops.object.delete()

bpy.ops.import_mesh.ply(filepath=path)
obj_objects = bpy.context.selected_objects[:]

obj = obj_objects[0]
obj.select_set(True)

bpy.context.view_layer.objects.active = obj
bpy.ops.object.mode_set(mode='EDIT')
bpy.ops.mesh.select_all(action='SELECT')
bpy.ops.mesh.normals_make_consistent(inside=False)

bpy.ops.mesh.select_all(action='DESELECT')
bpy.ops.mesh.select_non_manifold()
bpy.ops.mesh.edge_collapse()

bpy.ops.object.editmode_toggle()
bpy.ops.export_mesh.ply(filepath="export.ply")
