import math
import numpy as np
import bmesh
import bpy
import sys
#from rendering import python_octree
#from python_octree import *

import math
import numpy as np


def morton3d(x, y, z):
    x = (x | (x << 16)) & 0x030000FF
    x = (x | (x << 8)) & 0x0300F00F
    x = (x | (x << 4)) & 0x030C30C3
    x = (x | (x << 2)) & 0x09249249

    y = (y | (y << 16)) & 0x030000FF
    y = (y | (y << 8)) & 0x0300F00F
    y = (y | (y << 4)) & 0x030C30C3
    y = (y | (y << 2)) & 0x09249249

    z = (z | (z << 16)) & 0x030000FF
    z = (z | (z << 8)) & 0x0300F00F
    z = (z | (z << 4)) & 0x030C30C3
    z = (z | (z << 2)) & 0x09249249

    return np.uint32(x | (y << 1) | (z << 2))


def inverse_morton3d(z):
    x = z & 0x09249249
    y = (z >> 1) & 0x09249249
    z = (z >> 2) & 0x09249249

    x = ((x >> 2) | x) & 0x030C30C3
    x = ((x >> 4) | x) & 0x0300F00F
    x = ((x >> 8) | x) & 0x030000FF
    x = ((x >> 16) | x) & 0x000003FF

    y = ((y >> 2) | y) & 0x030C30C3
    y = ((y >> 4) | y) & 0x0300F00F
    y = ((y >> 8) | y) & 0x030000FF
    y = ((y >> 16) | y) & 0x000003FF

    z = ((z >> 2) | z) & 0x030C30C3
    z = ((z >> 4) | z) & 0x0300F00F
    z = ((z >> 8) | z) & 0x030000FF
    z = ((z >> 16) | z) & 0x000003FF

    return x, y, z


def max_level():
    return 4 * 8 // 3


def min_level():
    return 0


def clz(x):
    return (bin(x)[2:].zfill(32)+'1').index('1')


def compute_level(key):
    return (max_level() * 3 - clz(key) + 1) // 3


def compute_key(x, y, z, l):
    return morton3d(np.uint32(x), np.uint32(y), np.uint32(z)) \
        | (np.uint32(1) << 3 * l)


def compute_coord(key):
    l = compute_level(key)
    x, y, z = inverse_morton3d(key & ~(np.uint32(1) << l * 3))
    return x, y, z, l


def import_ot(file_name):
    output_model = {}
    max_level = 0
    with open(file_name) as f:
        content = f.readlines()
    parts = content[0].split(" ")
    for i in range(9, len(parts), 2):
        output_model[parts[i]] = parts[i+1]
        lev = compute_level(np.uint32(parts[i]))
        if lev > max_level:
            max_level = lev
    return output_model, pow(2, max_level)


def octree_to_voxel_grid(ot, resolution):
    output = np.zeros((resolution, resolution, resolution))
    max_level = np.uint32(math.log(resolution, 2))

    for key in ot:
        l = compute_level(np.uint32(key))
        code = (np.uint32(key) & ~(np.uint32(1) << l * 3)) \
            << (max_level - l) * 3
        x, y, z = inverse_morton3d(code)
        cube_len = np.uint32(pow(2, max_level - l))
        for i in range(0, cube_len):
            for j in range(0, cube_len):
                for k in range(0, cube_len):
                    if ot[key] == "1":
                        output[x+i, y+j, z+k] = 1

    return output


def get_cube_params(key, resolution):
    l = compute_level(np.uint32(key))
    max_level = np.uint32(math.log(resolution, 2))
    code = (np.uint32(key) & ~(np.uint32(1) << np.uint32(l * 3))) \
        << np.uint32((max_level - l) * 3)

    x, y, z = inverse_morton3d(code)
    side_len = pow(2, max_level - l)

    x = x + float(side_len) / 2.0
    y = y + float(side_len) / 2.0
    z = z + float(side_len) / 2.0
    return x, y, z, side_len


def intersection_over_union(gt, pr):
    arr_int = np.multiply(gt, pr)
    arr_uni = np.add(gt, pr)
    return float(np.count_nonzero(arr_int)) / float(np.count_nonzero(arr_uni))

MAX_NUM_LEVELS = 5

CLASS_EMPTY = 0
CLASS_FILLED = 1

cube_materials = []
proto_cubes = []

def clear_selection():
  bpy.context.scene.objects.active = None
  for o in bpy.data.objects:
    o.select = False

def select_object(obj):
  clear_selection()
  bpy.context.selected_objects.clear()
  bpy.context.scene.objects.active = obj
  obj.select = True
  return obj

def setup_camera():
    # set position
    scene = bpy.data.scenes["Scene"]
    scene.camera.location.x = 0
    scene.camera.location.y = 0
    scene.camera.location.z = 0
    bpy.data.cameras["Camera"].clip_end = 10000.

    # track invisible cube at (0, 0, 0)
    bpy.data.objects['Cube'].hide_render = True
    ttc = scene.camera.constraints.new(type='TRACK_TO')
    ttc.target = bpy.data.objects['Cube']
    ttc.track_axis = 'TRACK_NEGATIVE_Z'
    ttc.up_axis = 'UP_Y'

def setup_proto_cubes():
    for level in range(0, MAX_NUM_LEVELS):
      mat = bpy.data.materials.new('cube_material_%02d'%(level))
      col = float(level + 1) / MAX_NUM_LEVELS / 3
      mat.diffuse_color = (col,col,col)
      cube_materials.append(mat)
      bpy.ops.mesh.primitive_cube_add(location=(-1000,-1000,-1000),
                                      rotation=(0,0,0),
                                      radius=1-.00625)
      cube = bpy.context.object
      select_object(cube)
      cube.name = 'proto_cube_level_%01d'%(level)
      bpy.ops.object.material_slot_add()
      cube.material_slots[0].material = mat
      bpy.ops.object.modifier_add(type='BEVEL')
      cube.modifiers['Bevel'].width = 0.2
      cube.modifiers['Bevel'].segments = 3
      proto_cubes.append(cube)

def setup_general():
    bpy.data.worlds["World"].horizon_color = (1, 1, 1)
    bpy.data.scenes["Scene"].render.engine = "CYCLES"
    bpy.data.scenes["Scene"].cycles.samples = 72
    bpy.data.scenes["Scene"].use_nodes = True
    bpy.data.scenes["Scene"].render.filepath = "animation/"
    bpy.data.scenes["Scene"].render.use_compositing = False
    bpy.data.scenes["Scene"].render.layers["RenderLayer"].use_pass_z = False
    bpy.data.screens['Default'].scene = bpy.data.scenes['Scene']

def load_data(file_name):
    vg, resolution = import_ot(file_name)
    global_scale = 128.0 / resolution

    ply_vertices = {}
    for key in vg:
        if np.uint32(vg[key]) == CLASS_FILLED:
            x, y, z, side_len = get_cube_params(key, resolution)
            x -= resolution//2
            y -= resolution//2
            z -= resolution//2
            if side_len not in ply_vertices:
              ply_vertices[side_len] = []
            ply_vertices[side_len].append('%f %f %f'%(x*global_scale,y*global_scale,z*global_scale))

    cnt = 1

    ply_template = """ply
format ascii 1.0
element vertex %d
property float x
property float y
property float z
end_header
%s"""
    print(len(ply_vertices))
    for side_len in ply_vertices:
      print(side_len)
      path = 'tmp-ply-%d.ply'%(side_len)
      with open(path, 'w') as f:
        f.write(ply_template%(len(ply_vertices[side_len]),
                              '\n'.join(ply_vertices[side_len])))
      bpy.ops.import_mesh.ply(filepath=path)
      obj = bpy.context.selected_objects[0]
      bpy.ops.object.particle_system_add()
      ps = obj.particle_systems[0]
      ps = bpy.data.particles[-1]
      ps.count = len(ply_vertices[side_len])
      ps.emit_from = 'VERT'
      ps.use_emit_random = False
      ps.normal_factor = 0.0
      ps.physics_type = 'NO'
      ps.use_render_emitter = False
      ps.show_unborn = True
      ps.use_dead = False
      ps.lifetime = 250
      ps.render_type = 'OBJECT'
      ps.dupli_object = proto_cubes[-cnt]
      ps.particle_size = 0.5*side_len*global_scale

      cnt += 1

def save_occ_data(file_name):
    vg, resolution = import_ot(file_name)
    global_scale = 128.0 / resolution

    ply_vertices = {}
    for key in vg:
        if np.uint32(vg[key]) == CLASS_FILLED:
            x, y, z, side_len = get_cube_params(key, resolution)
            x -= resolution//2
            y -= resolution//2
            z -= resolution//2
            if side_len not in ply_vertices:
              ply_vertices[side_len] = []
            ply_vertices[side_len].append('%f %f %f'%(x*global_scale,y*global_scale,z*global_scale))

    cnt = 1

    ply_template = """ply
format ascii 1.0
element vertex %d
property float x
property float y
property float z
end_header
%s"""
    print(len(ply_vertices))
    for side_len in ply_vertices:
      print(side_len)
      path = 'tmp-ply-%d.ply'%(side_len)
      with open(path, 'w') as f:
        f.write(ply_template%(len(ply_vertices[side_len]),
                              '\n'.join(ply_vertices[side_len])))

def save_colored_ply(file_name):
    vg, resolution = import_ot(file_name)
    global_scale = 128.0 / resolution

    ply_vertices = {}
    for key in vg:
        value = np.uint32(vg[key])
        div = []
        for divIter in range(4):
            div.append(value & ((1 << 8) - 1))
            value = value >> 8

        #flag = 0
        #for divIter in range(4):
        #    if (div[divIter] > 0):
        #        flag = 1

        if np.uint32(div[3]) == CLASS_FILLED:
        #if flag == 1:
            x, y, z, side_len = get_cube_params(key, resolution)
            x -= resolution//2
            y -= resolution//2
            z -= resolution//2

            if side_len not in ply_vertices:
              ply_vertices[side_len] = []
            ply_vertices[side_len].append('%f %f %f %d %d %d'%(x*global_scale,y*global_scale,z*global_scale,div[0],div[1],div[2]))

    cnt = 1

    ply_template = """ply
format ascii 1.0
element vertex %d
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
end_header
%s"""
    print(len(ply_vertices))
    for side_len in ply_vertices:
      print(side_len)
      path = 'tmp-ply-%d.ply'%(side_len)
      with open(path, 'w') as f:
        f.write(ply_template%(len(ply_vertices[side_len]),
                              '\n'.join(ply_vertices[side_len])))


def parse_arguments(argv):
    cnt = 0
    for arg in argv:
        if arg == "-P":
            return argv[cnt+2]
        cnt += 1

def main():
  file_name = parse_arguments(sys.argv)
  setup_camera()
  setup_proto_cubes()
  save_colored_ply(file_name)
  #save_occ_data(file_name)
  #load_data(file_name)
  setup_general()

if __name__ == '__main__':
  main()
