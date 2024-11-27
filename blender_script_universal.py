import argparse
import json
import math
import os
import random
import sys
from typing import Any, Callable, Dict, Generator, List, Literal, Optional, Set, Tuple

import glob
import time
import bpy
import numpy as np
from mathutils import Matrix, Vector

### FOR BLENDER 4.2 +
#bpy.ops.extensions.package_install(repo_index=0,pkg_id="io_scene_max")
#bpy.ops.wm.save_userpref()


IMPORT_FUNCTIONS: Dict[str, Callable] = {
    "obj": bpy.ops.wm.obj_import,
    "glb": bpy.ops.import_scene.gltf,
    "gltf": bpy.ops.import_scene.gltf,
    "usd": bpy.ops.import_scene.usd,
    "fbx": bpy.ops.import_scene.fbx,
    "stl": bpy.ops.import_mesh.stl,
    "usda": bpy.ops.import_scene.usda,
    "dae": bpy.ops.wm.collada_import,
    "ply": bpy.ops.import_mesh.ply,
    "abc": bpy.ops.wm.alembic_import,
    "blend": bpy.ops.wm.append,
    "max":bpy.ops.import_scene.max
}

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def reset_cameras() -> None:
    """Resets the cameras in the scene to a single default camera."""
    # Delete all existing cameras
    bpy.ops.object.select_all(action="DESELECT")
    bpy.ops.object.select_by_type(type="CAMERA")
    bpy.ops.object.delete()
    
    # Create a new camera with default properties
    bpy.ops.object.camera_add()
    
    # Rename the new camera to 'NewDefaultCamera'
    new_camera = bpy.context.active_object
    new_camera.name = "Camera"
    
    # Set the new camera as the active camera for the scene
    scene.camera = new_camera


def sample_point_on_sphere(radius: float) -> Tuple[float, float, float]:
    """Samples a point on a sphere with the given radius.

    Args:
        radius (float): Radius of the sphere.

    Returns:
        Tuple[float, float, float]: A point on the sphere.
    """
    theta = random.random() * 2 * math.pi
    phi = math.acos(2 * random.random() - 1)
    return (
        radius * math.sin(phi) * math.cos(theta),
        radius * math.sin(phi) * math.sin(theta),
        radius * math.cos(phi),
    )


def _sample_spherical(
        radius_min: float = 1.5,
        radius_max: float = 2.0,
        maxz: float = 1.6,
        minz: float = -0.75,
) -> np.ndarray:
    """Sample a random point in a spherical shell.

    Args:
        radius_min (float): Minimum radius of the spherical shell.
        radius_max (float): Maximum radius of the spherical shell.
        maxz (float): Maximum z value of the spherical shell.
        minz (float): Minimum z value of the spherical shell.

    Returns:
        np.ndarray: A random (x, y, z) point in the spherical shell.
    """
    correct = False
    vec = np.array([0, 0, 0])
    while not correct:
        vec = np.random.uniform(-1, 1, 3)
        #         vec[2] = np.abs(vec[2])
        radius = np.random.uniform(radius_min, radius_max, 1)
        vec = vec / np.linalg.norm(vec, axis=0) * radius[0]
        if maxz > vec[2] > minz:
            correct = True
    return vec


def randomize_camera_orig(
        radius_min: float = 1.5,
        radius_max: float =1.5,
        maxz: float = 2.2,
        minz: float = -2.2,
        only_northern_hemisphere: bool = False,
) -> bpy.types.Object:
    """Randomizes the camera location and rotation inside of a spherical shell.

    Args:
        radius_min (float, optional): Minimum radius of the spherical shell. Defaults to
            1.5.
        radius_max (float, optional): Maximum radius of the spherical shell. Defaults to
            2.0.
        maxz (float, optional): Maximum z value of the spherical shell. Defaults to 1.6.
        minz (float, optional): Minimum z value of the spherical shell. Defaults to
            -0.75.
        only_northern_hemisphere (bool, optional): Whether to only sample points in the
            northern hemisphere. Defaults to False.

    Returns:
        bpy.types.Object: The camera object.
    """
    
    x, y, z = _sample_spherical(
        radius_min=radius_min, radius_max=radius_max, maxz=maxz, minz=minz
    )
    camera = bpy.data.objects["Camera"]
    
    # only positive z
    if only_northern_hemisphere:
        z = abs(z)
    
    camera.location = Vector(np.array([x, y, z]))
    
    direction = -camera.location
    rot_quat = direction.to_track_quat("-Z", "Y")
    camera.rotation_euler = rot_quat.to_euler()
    
    return camera

def randomize_camera(x,y,z,
                     radius_min: float = 1.5,
                     radius_max: float = 2.2,
                     maxz: float = 2.2,
                     minz: float = -2.2,
                     only_northern_hemisphere: bool = False,
                     ) -> bpy.types.Object:
    """Randomizes the camera location and rotation inside of a spherical shell.

    Args:
        radius_min (float, optional): Minimum radius of the spherical shell. Defaults to
            1.5.
        radius_max (float, optional): Maximum radius of the spherical shell. Defaults to
            2.0.
        maxz (float, optional): Maximum z value of the spherical shell. Defaults to 1.6.
        minz (float, optional): Minimum z value of the spherical shell. Defaults to
            -0.75.
        only_northern_hemisphere (bool, optional): Whether to only sample points in the
            northern hemisphere. Defaults to False.

    Returns:
        bpy.types.Object: The camera object.
    """
    
    #x, y, z = _sample_spherical(
    #    radius_min=radius_min, radius_max=radius_max, maxz=maxz, minz=minz
    #)
    camera = bpy.data.objects["Camera"]
    
    # only positive z
    if only_northern_hemisphere:
        z = abs(z)
    
    camera.location = Vector(np.array([x, y, z]))
    
    print("!!!!!")
    print(camera.location)
    print("!!!!!")
    direction = -camera.location
    rot_quat = direction.to_track_quat("-Z", "Y")
    camera.rotation_euler = rot_quat.to_euler()
    
    return camera


def _set_camera_at_size(i: int, scale: float = 1.5) -> bpy.types.Object:
    """Debugging function to set the camera on the 6 faces of a cube.

    Args:
        i (int): Index of the face of the cube.
        scale (float, optional): Scale of the cube. Defaults to 1.5.

    Returns:
        bpy.types.Object: The camera object.
    """
    if i == 0:
        x, y, z = scale, 0, 0
    elif i == 1:
        x, y, z = -scale, 0, 0
    elif i == 2:
        x, y, z = 0, scale, 0
    elif i == 3:
        x, y, z = 0, -scale, 0
    elif i == 4:
        x, y, z = 0, 0, scale
    elif i == 5:
        x, y, z = 0, 0, -scale
    else:
        raise ValueError(f"Invalid index: i={i}, must be int in range [0, 5].")
    camera = bpy.data.objects["Camera"]
    camera.location = Vector(np.array([x, y, z]))
    direction = -camera.location
    rot_quat = direction.to_track_quat("-Z", "Y")
    camera.rotation_euler = rot_quat.to_euler()
    return camera


def _create_light(
        name: str,
        light_type: Literal["POINT", "SUN", "SPOT", "AREA"],
        location: Tuple[float, float, float],
        rotation: Tuple[float, float, float],
        energy: float,
        use_shadow: bool = False,
        specular_factor: float = 1.0,
):
    """Creates a light object.

    Args:
        name (str): Name of the light object.
        light_type (Literal["POINT", "SUN", "SPOT", "AREA"]): Type of the light.
        location (Tuple[float, float, float]): Location of the light.
        rotation (Tuple[float, float, float]): Rotation of the light.
        energy (float): Energy of the light.
        use_shadow (bool, optional): Whether to use shadows. Defaults to False.
        specular_factor (float, optional): Specular factor of the light. Defaults to 1.0.

    Returns:
        bpy.types.Object: The light object.
    """
    
    
    
    
    light_data = bpy.data.lights.new(name=name, type=light_type)
    light_object = bpy.data.objects.new(name, light_data)
    bpy.context.collection.objects.link(light_object)
    light_object.location = location
    light_object.rotation_euler = rotation
    #light_data.use_shadow = use_shadow
    light_data.use_shadow = False
    light_data.specular_factor = specular_factor
    light_data.energy = energy
    return light_object


def hadmare_lighting() -> Dict[str, bpy.types.Object]:
    """Randomizes the lighting in the scene.

    Returns:
        Dict[str, bpy.types.Object]: Dictionary of the lights in the scene. The keys are
            "key_light", "fill_light", "rim_light", and "bottom_light".
    """
    
    # Clear existing lights
    bpy.ops.object.select_all(action="DESELECT")
    bpy.ops.object.select_by_type(type="LIGHT")
    bpy.ops.object.delete()
    
    
    kf = 0.6
    
    # Create key light
    key_light = _create_light(
        name="Key_Light",
        light_type="SUN",
        location=(0, 0, 0),
        rotation=(0.785398, 0, -0.785398),
        energy=random.choice([kf * 4]),
    )
    #return
    
    # Create fill light
    fill_light = _create_light(
        name="Fill_Light",
        light_type="SUN",
        location=(0, 0, 0),
        rotation=(0.785398, 0, 2.35619),
        energy=random.choice([ kf *3]),
    )
    
    # Create rim light
    rim_light = _create_light(
        name="Rim_Light",
        light_type="SUN",
        location=(0, 0, 0),
        rotation=(-0.785398, 0, -3.92699),
        energy=random.choice([ kf *4]),
    )
    
    # Create bottom light
    bottom_light = _create_light(
        name="Bottom_Light",
        light_type="SUN",
        location=(0, 0, 0),
        rotation=(3.14159, 0, 0),
        energy=random.choice([kf*2]),
    )
    
    #for light in lights:
    #    light.data.use_shadow = False
    return dict(
        key_light=key_light,
        fill_light=fill_light,
        rim_light=rim_light,
        bottom_light=bottom_light,
    )


def randomize_lighting(random_kf=False) -> Dict[str, bpy.types.Object]:
    """Randomizes the lighting in the scene.

    Returns:
        Dict[str, bpy.types.Object]: Dictionary of the lights in the scene. The keys are
            "key_light", "fill_light", "rim_light", and "bottom_light".
    """
    
    # Clear existing lights
    bpy.ops.object.select_all(action="DESELECT")
    bpy.ops.object.select_by_type(type="LIGHT")
    bpy.ops.object.delete()
    
    
    kf = 1.0
    
    # Create key light
    key_light = _create_light(
        name="Key_Light",
        light_type="SUN",
        location=(0, 0, 0),
        rotation=(0.785398, 0, -0.785398),
        energy=random.choice([kf * 3, kf * 4, kf *5]),
    )
    #return
    
    # Create fill light
    fill_light = _create_light(
        name="Fill_Light",
        light_type="SUN",
        location=(0, 0, 0),
        rotation=(0.785398, 0, 2.35619),
        energy=random.choice([kf *2, kf *3, kf *4]),
    )
    
    # Create rim light
    rim_light = _create_light(
        name="Rim_Light",
        light_type="SUN",
        location=(0, 0, 0),
        rotation=(-0.785398, 0, -3.92699),
        energy=random.choice([kf *3, kf *4, kf *5]),
    )
    
    # Create bottom light
    bottom_light = _create_light(
        name="Bottom_Light",
        light_type="SUN",
        location=(0, 0, 0),
        rotation=(3.14159, 0, 0),
        energy=random.choice([kf *1, kf*2, kf *3]),
    )
    
    #for light in lights:
    #    light.data.use_shadow = False
    return dict(
        key_light=key_light,
        fill_light=fill_light,
        rim_light=rim_light,
        bottom_light=bottom_light,
    )


def reset_scene() -> None:
    """Resets the scene to a clean state.

    Returns:
        None
    """
    # delete everything that isn't part of a camera or a light
    for obj in bpy.data.objects:
        if obj.type not in {"CAMERA", "LIGHT"}:
            bpy.data.objects.remove(obj, do_unlink=True)
    
    # delete all the materials
    for material in bpy.data.materials:
        bpy.data.materials.remove(material, do_unlink=True)
    
    # delete all the textures
    for texture in bpy.data.textures:
        bpy.data.textures.remove(texture, do_unlink=True)
    
    # delete all the images
    for image in bpy.data.images:
        bpy.data.images.remove(image, do_unlink=True)


def load_object(object_path: str) -> None:
    """Loads a model with a supported file extension into the scene.

    Args:
        object_path (str): Path to the model file.

    Raises:
        ValueError: If the file extension is not supported.

    Returns:
        None
    """
    file_extension = object_path.split(".")[-1].lower()
    if file_extension is None:
        raise ValueError(f"Unsupported file type: {object_path}")
    
    if file_extension == "usdz":
        # install usdz io package
        dirname = os.path.dirname(os.path.realpath(__file__))
        usdz_package = os.path.join(dirname, "io_scene_usdz.zip")
        bpy.ops.preferences.addon_install(filepath=usdz_package)
        # enable it
        addon_name = "io_scene_usdz"
        bpy.ops.preferences.addon_enable(module=addon_name)
        # import the usdz
        from io_scene_usdz.import_usdz import import_usdz
        
        import_usdz(context, filepath=object_path, materials=True, animations=True)
        return None
    
    # load from existing import functions
    import_function = IMPORT_FUNCTIONS[file_extension]
    
    if file_extension == "blend":
        import_function(directory=object_path, link=False)
    elif file_extension in {"glb", "gltf"}:
        import_function(filepath=object_path, merge_vertices=True)
    elif  file_extension == "max":
        import_function(files=[{"name":object_path}])    
    else:
        import_function(filepath=object_path)


def scene_bbox(
        single_obj: Optional[bpy.types.Object] = None, ignore_matrix: bool = False
) -> Tuple[Vector, Vector]:
    """Returns the bounding box of the scene.

    Taken from Shap-E rendering script
    (https://github.com/openai/shap-e/blob/main/shap_e/rendering/blender/blender_script.py#L68-L82)

    Args:
        single_obj (Optional[bpy.types.Object], optional): If not None, only computes
            the bounding box for the given object. Defaults to None.
        ignore_matrix (bool, optional): Whether to ignore the object's matrix. Defaults
            to False.

    Raises:
        RuntimeError: If there are no objects in the scene.

    Returns:
        Tuple[Vector, Vector]: The minimum and maximum coordinates of the bounding box.
    """
    bbox_min = (math.inf,) * 3
    bbox_max = (-math.inf,) * 3
    found = False
    for obj in get_scene_meshes() if single_obj is None else [single_obj]:
        found = True
        for coord in obj.bound_box:
            coord = Vector(coord)
            if not ignore_matrix:
                coord = obj.matrix_world @ coord
            bbox_min = tuple(min(x, y) for x, y in zip(bbox_min, coord))
            bbox_max = tuple(max(x, y) for x, y in zip(bbox_max, coord))
    
    if not found:
        raise RuntimeError("no objects in scene to compute bounding box for")
    
    return Vector(bbox_min), Vector(bbox_max)


def get_scene_root_objects() -> Generator[bpy.types.Object, None, None]:
    """Returns all root objects in the scene.

    Yields:
        Generator[bpy.types.Object, None, None]: Generator of all root objects in the
            scene.
    """
    for obj in bpy.context.scene.objects.values():
        if not obj.parent:
            yield obj


def get_scene_meshes() -> Generator[bpy.types.Object, None, None]:
    """Returns all meshes in the scene.

    Yields:
        Generator[bpy.types.Object, None, None]: Generator of all meshes in the scene.
    """
    for obj in bpy.context.scene.objects.values():
        if isinstance(obj.data, (bpy.types.Mesh)):
            yield obj


def get_3x4_RT_matrix_from_blender(cam: bpy.types.Object) -> Matrix:
    """Returns the 3x4 RT matrix from the given camera.

    Taken from Zero123, which in turn was taken from
    https://github.com/panmari/stanford-shapenet-renderer/blob/master/render_blender.py

    Args:
        cam (bpy.types.Object): The camera object.

    Returns:
        Matrix: The 3x4 RT matrix from the given camera.
    """
    # Use matrix_world instead to account for all constraints
    location, rotation = cam.matrix_world.decompose()[0:2]
    R_world2bcam = rotation.to_matrix().transposed()
    
    # Use location from matrix_world to account for constraints:
    T_world2bcam = -1 * R_world2bcam @ location
    
    # put into 3x4 matrix
    RT = Matrix(
        (
            R_world2bcam[0][:] + (T_world2bcam[0],),
            R_world2bcam[1][:] + (T_world2bcam[1],),
            R_world2bcam[2][:] + (T_world2bcam[2],),
        )
    )
    return RT

def get_blender_matrix_from_3x4RT(cam_matrix, cam):
    R_world2bcam = cam_matrix[:3,:3]
    T_world2bcam = cam_matrix[:3,3]
    
    location = -1* R_world2bcam.T @ T_world2bcam
    rotation_mat = R_world2bcam.T
    
    cam.location = Vector(location)
    
    mat = Matrix((
        rotation_mat[0][:],
        rotation_mat[1][:],
        rotation_mat[2][:],))
    print(mat)
    
    cam.rotation_euler = mat.to_euler()
    
    return cam


def delete_invisible_objects() -> None:
    """Deletes all invisible objects in the scene.

    Returns:
        None
    """
    bpy.ops.object.select_all(action="DESELECT")
    for obj in scene.objects:
        if obj.hide_viewport or obj.hide_render:
            obj.hide_viewport = False
            obj.hide_render = False
            obj.hide_select = False
            obj.select_set(True)
    bpy.ops.object.delete()
    
    # Delete invisible collections
    invisible_collections = [col for col in bpy.data.collections if col.hide_viewport]
    for col in invisible_collections:
        bpy.data.collections.remove(col)


def normalize_scene(gso=False) -> None:
    """Normalizes the scene by scaling and translating it to fit in a unit cube centered
    at the origin.

    Mostly taken from the Point-E / Shap-E rendering script
    (https://github.com/openai/point-e/blob/main/point_e/evals/scripts/blender_script.py#L97-L112),
    but fix for multiple root objects: (see bug report here:
    https://github.com/openai/shap-e/pull/60).

    Returns:
        None
    """
    if len(list(get_scene_root_objects())) > 1:
        # create an empty object to be used as a parent for all root objects
        parent_empty = bpy.data.objects.new("ParentEmpty", None)
        bpy.context.scene.collection.objects.link(parent_empty)
        
        # parent all root objects to the empty object
        for obj in get_scene_root_objects():
            if obj != parent_empty:
                obj.parent = parent_empty
    #print(bbox_min, bbox_max)
    bbox_min, bbox_max = scene_bbox()
    scale = 1 / max(bbox_max - bbox_min)
    for obj in get_scene_root_objects():
        obj.scale = obj.scale * scale
    
    if gso:
        obj.rotation_euler = (math.radians(270), math.radians(0), math.radians(0))
    
    # Apply scale to matrix_world.
    bpy.context.view_layer.update()
    bbox_min, bbox_max = scene_bbox()
    print(bbox_min, bbox_max)
    offset = -(bbox_min + bbox_max) / 2
    for obj in get_scene_root_objects():
        obj.matrix_world.translation += offset
    
    print(obj.matrix_world.translation)
    bpy.ops.object.select_all(action="DESELECT")
    
    # unparent the camera
    bpy.data.objects["Camera"].parent = None


def delete_missing_textures() -> Dict[str, Any]:
    """Deletes all missing textures in the scene.

    Returns:
        Dict[str, Any]: Dictionary with keys "count", "files", and "file_path_to_color".
            "count" is the number of missing textures, "files" is a list of the missing
            texture file paths, and "file_path_to_color" is a dictionary mapping the
            missing texture file paths to a random color.
    """
    missing_file_count = 0
    out_files = []
    file_path_to_color = {}
    
    # Check all materials in the scene
    for material in bpy.data.materials:
        if material.use_nodes:
            for node in material.node_tree.nodes:
                if node.type == "TEX_IMAGE":
                    image = node.image
                    if image is not None:
                        file_path = bpy.path.abspath(image.filepath)
                        if file_path == "":
                            # means it's embedded
                            continue
                        
                        if not os.path.exists(file_path):
                            # Find the connected Principled BSDF node
                            connected_node = node.outputs[0].links[0].to_node
                            
                            if connected_node.type == "BSDF_PRINCIPLED":
                                if file_path not in file_path_to_color:
                                    # Set a random color for the unique missing file path
                                    random_color = [random.random() for _ in range(3)]
                                    file_path_to_color[file_path] = random_color + [1]
                                
                                connected_node.inputs[
                                    "Base Color"
                                ].default_value = file_path_to_color[file_path]
                            
                            # Delete the TEX_IMAGE node
                            material.node_tree.nodes.remove(node)
                            missing_file_count += 1
                            out_files.append(image.filepath)
    return {
        "count": missing_file_count,
        "files": out_files,
        "file_path_to_color": file_path_to_color,
    }


def _get_random_color() -> Tuple[float, float, float, float]:
    """Generates a random RGB-A color.

    The alpha value is always 1.

    Returns:
        Tuple[float, float, float, float]: A random RGB-A color. Each value is in the
        range [0, 1].
    """
    return (random.random(), random.random(), random.random(), 1)


def _apply_color_to_object(
        obj: bpy.types.Object, color: Tuple[float, float, float, float]
) -> None:
    """Applies the given color to the object.

    Args:
        obj (bpy.types.Object): The object to apply the color to.
        color (Tuple[float, float, float, float]): The color to apply to the object.

    Returns:
        None
    """
    
    mat = bpy.data.materials.new(name=f"RandomMaterial_{obj.name}")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    principled_bsdf = nodes.get("Principled BSDF")
    if principled_bsdf:
        principled_bsdf.inputs["Base Color"].default_value = color
    obj.data.materials.append(mat)




def apply_single_random_color_to_all_objects() -> Tuple[float, float, float, float]:
    """Applies a single random color to all objects in the scene.

    Returns:
        Tuple[float, float, float, float]: The random color that was applied to all
        objects.
    """
    print("!!!!1111!!!")
    rand_color = _get_random_color()
    for obj in bpy.context.scene.objects:
        if obj.type == "MESH":
            _apply_color_to_object(obj, rand_color)
    return rand_color


class MetadataExtractor:
    """Class to extract metadata from a Blender scene."""
    
    def __init__(
            self, object_path: str, scene: bpy.types.Scene, bdata: bpy.types.BlendData
    ) -> None:
        """Initializes the MetadataExtractor.

        Args:
            object_path (str): Path to the object file.
            scene (bpy.types.Scene): The current scene object from `bpy.context.scene`.
            bdata (bpy.types.BlendData): The current blender data from `bpy.data`.

        Returns:
            None
        """
        self.object_path = object_path
        self.scene = scene
        self.bdata = bdata
    
    def get_poly_count(self) -> int:
        """Returns the total number of polygons in the scene."""
        total_poly_count = 0
        for obj in self.scene.objects:
            if obj.type == "MESH":
                total_poly_count += len(obj.data.polygons)
        return total_poly_count
    
    def get_vertex_count(self) -> int:
        """Returns the total number of vertices in the scene."""
        total_vertex_count = 0
        for obj in self.scene.objects:
            if obj.type == "MESH":
                total_vertex_count += len(obj.data.vertices)
        return total_vertex_count
    
    def get_edge_count(self) -> int:
        """Returns the total number of edges in the scene."""
        total_edge_count = 0
        for obj in self.scene.objects:
            if obj.type == "MESH":
                total_edge_count += len(obj.data.edges)
        return total_edge_count
    
    def get_lamp_count(self) -> int:
        """Returns the number of lamps in the scene."""
        return sum(1 for obj in self.scene.objects if obj.type == "LIGHT")
    
    def get_mesh_count(self) -> int:
        """Returns the number of meshes in the scene."""
        return sum(1 for obj in self.scene.objects if obj.type == "MESH")
    
    def get_material_count(self) -> int:
        """Returns the number of materials in the scene."""
        return len(self.bdata.materials)
    
    def get_object_count(self) -> int:
        """Returns the number of objects in the scene."""
        return len(self.bdata.objects)
    
    def get_animation_count(self) -> int:
        """Returns the number of animations in the scene."""
        return len(self.bdata.actions)
    
    def get_linked_files(self) -> List[str]:
        """Returns the filepaths of all linked files."""
        image_filepaths = self._get_image_filepaths()
        material_filepaths = self._get_material_filepaths()
        linked_libraries_filepaths = self._get_linked_libraries_filepaths()
        
        all_filepaths = (
                image_filepaths | material_filepaths | linked_libraries_filepaths
        )
        if "" in all_filepaths:
            all_filepaths.remove("")
        return list(all_filepaths)
    
    def _get_image_filepaths(self) -> Set[str]:
        """Returns the filepaths of all images used in the scene."""
        filepaths = set()
        for image in self.bdata.images:
            if image.source == "FILE":
                filepaths.add(bpy.path.abspath(image.filepath))
        return filepaths
    
    def _get_material_filepaths(self) -> Set[str]:
        """Returns the filepaths of all images used in materials."""
        filepaths = set()
        for material in self.bdata.materials:
            if material.use_nodes:
                for node in material.node_tree.nodes:
                    if node.type == "TEX_IMAGE":
                        image = node.image
                        if image is not None:
                            filepaths.add(bpy.path.abspath(image.filepath))
        return filepaths
    
    def _get_linked_libraries_filepaths(self) -> Set[str]:
        """Returns the filepaths of all linked libraries."""
        filepaths = set()
        for library in self.bdata.libraries:
            filepaths.add(bpy.path.abspath(library.filepath))
        return filepaths
    
    def get_scene_size(self) -> Dict[str, list]:
        """Returns the size of the scene bounds in meters."""
        bbox_min, bbox_max = scene_bbox()
        return {"bbox_max": list(bbox_max), "bbox_min": list(bbox_min)}
    
    def get_shape_key_count(self) -> int:
        """Returns the number of shape keys in the scene."""
        total_shape_key_count = 0
        for obj in self.scene.objects:
            if obj.type == "MESH":
                shape_keys = obj.data.shape_keys
                if shape_keys is not None:
                    total_shape_key_count += (
                            len(shape_keys.key_blocks) - 1
                    )  # Subtract 1 to exclude the Basis shape key
        return total_shape_key_count
    
    def get_armature_count(self) -> int:
        """Returns the number of armatures in the scene."""
        total_armature_count = 0
        for obj in self.scene.objects:
            if obj.type == "ARMATURE":
                total_armature_count += 1
        return total_armature_count
    
    def read_file_size(self) -> int:
        """Returns the size of the file in bytes."""
        return os.path.getsize(self.object_path)
    
    def get_transparent(self):
        for mat in self.bdata.materials:
            if mat.blend_method != 'OPAQUE':
                return True
        return False
    
    def get_metadata(self) -> Dict[str, Any]:
        """Returns the metadata of the scene.

        Returns:
            Dict[str, Any]: Dictionary of the metadata with keys for "file_size",
            "poly_count", "vert_count", "edge_count", "material_count", "object_count",
            "lamp_count", "mesh_count", "animation_count", "linked_files", "scene_size",
            "shape_key_count", and "armature_count".
        """
        return {
            "file_size": self.read_file_size(),
            "poly_count": self.get_poly_count(),
            "vert_count": self.get_vertex_count(),
            "edge_count": self.get_edge_count(),
            "material_count": self.get_material_count(),
            "object_count": self.get_object_count(),
            "lamp_count": self.get_lamp_count(),
            "mesh_count": self.get_mesh_count(),
            "animation_count": self.get_animation_count(),
            "linked_files": self.get_linked_files(),
            "scene_size": self.get_scene_size(),
            "shape_key_count": self.get_shape_key_count(),
            "armature_count": self.get_armature_count(),
            "transparent": self.get_transparent(),
        }


def setting_color():
    # Create a new material
    material = bpy.data.materials.new(name="CustomMaterial")
    material.use_nodes = True
    nodes = material.node_tree.nodes
    
    # Clear all the nodes to start clean
    for node in nodes:
        nodes.remove(node)
    
    # Create Principled BSDF material
    principled_node = nodes.new(type='ShaderNodeBsdfPrincipled')
    principled_node.location = 300, 0
    
    # Set the values for the Principled BSDF
    principled_node.inputs['Base Color'].default_value = (1, 1, 1, 1) # RGB + Alpha
    principled_node.inputs['Metallic'].default_value = 0.0
    principled_node.inputs['Roughness'].default_value = 1.0
    principled_node.inputs['Specular'].default_value = 0.5
    
    # Create Hue/Saturation node
    hue_sat_node = nodes.new(type='ShaderNodeHueSaturation')
    hue_sat_node.location = 100, 0
    
    # Set Hue/Saturation values
    hue_sat_node.inputs['Hue'].default_value = 0.5
    hue_sat_node.inputs['Saturation'].default_value = 1.45
    hue_sat_node.inputs['Value'].default_value = 1.0
    
    # Create Image Texture node
    texture_node = nodes.new(type='ShaderNodeTexImage')
    texture_node.location = -100, 0
    # Load an image (replace 'path_to_image.png' with your image path)
    texture_node.image = bpy.data.images.load('path_to_image.png')
    
    # Link nodes together
    links = material.node_tree.links
    links.new(texture_node.outputs['Color'], hue_sat_node.inputs['Color'])
    links.new(hue_sat_node.outputs['Color'], principled_node.inputs['Base Color'])
    
    # Link the Principled BSDF to Material Output
    output_node = nodes.new(type='ShaderNodeOutputMaterial')
    output_node.location = 500, 0
    links.new(principled_node.outputs['BSDF'], output_node.inputs['Surface'])
    
    # Assign it to the active object
    bpy.context.active_object.data.materials.append(material)
    
    # Render the scene
    bpy.ops.render.render(write_still=True)



def set_sky_texture():
    
    print("set_sky_texture")
    #bpy.context.space_data.context = 'WORLD'
    bpy.data.worlds["World"].node_tree.nodes["Background"].inputs[0].default_value = (0.0508761, 0.0508761, 0.0508761, 1)
    
    
    world = bpy.data.worlds["World"]
    
    # Use nodes for the world
    world.use_nodes = True
    
    
    
    nodes = bpy.data.worlds["World"].node_tree.nodes
    # Add Environment Texture node
    env_texture_node = nodes.new(type='ShaderNodeTexSky')
    env_texture_node.name = "Sky Texture"
    
    bpy.data.worlds["World"].node_tree.nodes["Sky Texture"].sky_type = 'NISHITA'
    bpy.data.worlds["World"].node_tree.nodes["Sky Texture"].sun_elevation = 1.5708
    bpy.data.worlds["World"].node_tree.nodes["Sky Texture"].sun_disc = False
    #bpy.data.worlds["World"].node_tree.nodes["Sky Texture"].sun_size = 15
    
    
    bpy.ops.object.select_all(action='DESELECT')
    bpy.ops.object.select_by_type(type='LIGHT')
    #bpy.context.space_data.shading.type = 'RENDERED'
    for area in bpy.context.screen.areas:
        if area.type == 'VIEW_3D':
            for space in area.spaces:
                if space.type == 'VIEW_3D':
                    print("shading RENDERED: ok")
                    space.shading.type = 'RENDERED'
    
    bg = bpy.context.scene.world.node_tree.nodes["Background"]
    bpy.context.scene.world.node_tree.links.new(bg.inputs["Color"], env_texture_node.outputs["Color"])



def set_up_world_sun_light(sun_config=None, strength=1.0):
    world_node_tree = bpy.context.scene.world.node_tree
    world_node_tree.nodes.clear()
    
    node_location_x_step = 300
    node_location_x = 0
    
    node_sky = world_node_tree.nodes.new(type="ShaderNodeTexSky")
    node_location_x += node_location_x_step
    
    world_background_node = world_node_tree.nodes.new(type="ShaderNodeBackground")
    world_background_node.inputs["Strength"].default_value = strength
    world_background_node.location.x = node_location_x
    node_location_x += node_location_x_step
    
    world_output_node = world_node_tree.nodes.new(type="ShaderNodeOutputWorld")
    world_output_node.location.x = node_location_x
    
    if sun_config:
        print("Updating ShaderNodeTexSky params:")
        for attr, value in sun_config.items():
            if hasattr(node_sky, attr):
                print("\t %s set to %s", attr, str(value))
                setattr(node_sky, attr, value)
            else:
                print("\t warning: %s is not an attribute of ShaderNodeTexSky node", attr)
    
    world_node_tree.links.new(node_sky.outputs["Color"], world_background_node.inputs["Color"])
    world_node_tree.links.new(world_background_node.outputs["Background"], world_output_node.inputs["Surface"])


def random_color_hdrmap(mode="all"):
    # Select random hdri
    # light_ind = random.choice(["circus_arena_2k", "golden_bay_2k", "hangar_interior_2k", "modern_buildings_2_2k", 
    #                            "overcast_soil_puresky_2k", "studio_small_09_2k", "laufenurg_church_2k", "resting_place_2_2k",
    #                            "thatch_chapel_2k", "sunset_in_the_chalk_quarry_2k"])
    # light_path = "/home/jovyan/3dgen/data/irrmaps/color_maps/" + light_ind + ".hdr"
    light_inds = []
    
    if mode == "bw":
        light_inds = ["one_light_0", "one_light_1", "one_light_2", "one_light_3",
                      "one_light_4", "one_light_5", "one_light_6", "one_light_7",
                      "one_light_8", "one_light_9",
                      "three_lights_0", "three_lights_1", "three_lights_2", "three_lights_3", 
                      "three_lights_4", "three_lights_5", "three_lights_6", "three_lights_7",
                      "three_lights_8", "three_lights_9",
                      "new_lightAroundMap"]
    elif mode == "color":
        light_inds = ["circus_arena_2k", "golden_bay_2k", "hangar_interior_2k", 
                     "modern_buildings_2_2k", "overcast_soil_puresky_2k", 
                     "studio_small_09_2k", "laufenurg_church_2k", 
                     "resting_place_2_2k", "thatch_chapel_2k",
                     "sunset_in_the_chalk_quarry_2k",
                      "new_lightAroundMap"]

    elif mode == "all":
        light_inds = ["circus_arena_2k", "golden_bay_2k", "hangar_interior_2k", 
                     "modern_buildings_2_2k", "overcast_soil_puresky_2k", 
                     "studio_small_09_2k", "laufenurg_church_2k", 
                     "resting_place_2_2k", "thatch_chapel_2k",
                     "sunset_in_the_chalk_quarry_2k",
                     "one_light_0", "one_light_1", "one_light_2", "one_light_3",
                      "one_light_4", "one_light_5", "one_light_6", "one_light_7",
                      "one_light_8", "one_light_9",
                      "three_lights_0", "three_lights_1", "three_lights_2", "three_lights_3", 
                      "three_lights_4", "three_lights_5", "three_lights_6", "three_lights_7",
                      "three_lights_8", "three_lights_9",
                      "new_lightAroundMap"]

    light_ind = random.choice(light_inds)
    light_path = "/home/jovyan/3dgen/data/irrmaps/" + light_ind + ".hdr"
    
    # Ensure we are using the correct context
    bpy.context.scene.use_nodes = True
    nodes = bpy.context.scene.world.node_tree.nodes
    links = bpy.context.scene.world.node_tree.links

    # Clear existing nodes
    for node in nodes:
        nodes.remove(node)

    # Add Background node
    node_background = nodes.new(type='ShaderNodeBackground')
    
    # Add Environment Texture node
    node_environment = nodes.new('ShaderNodeTexEnvironment')
    # Load and assign the image to the node property
    node_environment.image = bpy.data.images.load(light_path) # Relative path
    
    # Add Output node
    node_output = nodes.new(type='ShaderNodeOutputWorld')
    
    # Link all nodes
    link = links.new(node_environment.outputs["Color"], node_background.inputs["Color"])
    link = links.new(node_background.outputs["Background"], node_output.inputs["Surface"])

    return light_ind


def setting_nodes_hsv_default():
    #print("setting_nodes")
    # Проходим по всем объектам сцены
    for obj in bpy.data.objects:
        # Проверяем, есть ли у объекта материалы
        if obj.material_slots:
            #print("Материалы для объекта", obj.name)
            # Проходим по всем материалам в слотах объекта
            for slot in obj.material_slots:
                material = slot.material
                if material:
                    #print("Материал:", material.name)
                    # Проверяем, использует ли материал узлы
                    if material.use_nodes:
                        # Получаем узловое дерево материала
                        tree = material.node_tree
                        # Добавляем узел Hue/Saturation/Value
                        hsv_node = tree.nodes.new('ShaderNodeHueSaturation')
                        hsv_node.location = (-600, -400)  # Позиционируем узел на панели
                        hsv_node.inputs[1].default_value = 1.25
                        
                        # Пройдемся по всем узлам, чтобы найти BSDF узел
                        for node in tree.nodes:
                            #print(node.type )
                            #print(node.name )
                            if node.type == 'BSDF_PRINCIPLED':
                                for input in node.inputs:
                                    # Проверяем, соответствует ли текущая входная связь входу Base Color
                                    if input.name == 'Base Color':
                                        # Проверяем, есть ли связь на этом входе
                                        if input.links:
                                            # Получаем входной узел
                                            connected_node = input.links[0].from_node
                                            # Выводим информацию о входной ноде BSDF_PRINCIPLED
                                            print("Входная нода BSDF_PRINCIPLED для Base Color:", connected_node.name)
                                            tree.links.new(connected_node.outputs['Color'], hsv_node.inputs['Color'])
                                            
                                            tree.links.new(hsv_node.outputs['Color'], node.inputs['Base Color'])
                                            print("Link with bsdf ok!")
                                            break
                                        else:
                                            print("Для входа Base Color нет связанной ноды.")
                                        break  # Найден вход Base Color, выходим из цикла
                                
                                
                                #tree.links.new(node.outputs, hsv_node.inputs['Color'])
                            
                            if node.type == "TEX_IMAGE" and False:
                                tree.links.new(node.outputs['Color'], hsv_node.inputs['Color'])
                                print("Link with BASE COLOR ok!")
#
#                        # Пройдемся по всем узлам, чтобы найти BASE COLOR узел
#                        for node in tree.nodes:
#                            print(node.type )
#                            print(dir(node.outputs))
#                            print(node.outputs.values)

def setting_nodes(kf_met, kf_rough):
    print("setting_nodes")
    # Проходим по всем объектам сцены
    for obj in bpy.data.objects:
        # Проверяем, есть ли у объекта материалы
        if obj.material_slots:
            print("Материалы для объекта", obj.name)
            # Проходим по всем материалам в слотах объекта
            for slot in obj.material_slots:
                material = slot.material
                if material:
                    print("Материал:", material.name)
                    # Проверяем, использует ли материал узлы
                    if material.use_nodes:
                        # Получаем узловое дерево материала
                        tree = material.node_tree
                        # Добавляем узел Hue/Saturation/Value
                        #hsv_node = tree.nodes.new('ShaderNodeHueSaturation')
                        #hsv_node.location = (-600, -400)  # Позиционируем узел на панели
                        #hsv_node.inputs[1].default_value = 1.25
                        
                        # Пройдемся по всем узлам, чтобы найти BSDF узел
                        for node in tree.nodes:
                            print(node.type )
                            print(node.name )
                            if node.type == 'BSDF_PRINCIPLED':
                                print("BSDF_PRINCIPLED NODES::::")
                                
                                #                                for input_link in bsdf_node.inputs["Metallic"].links:
                                #                                    node.links.remove(input_link)
                                for input in node.inputs:
                                    
                                    print(input.name)
                                    
                                    if input.name == "Metallic":
                                        print("Metallic!!!")
                                        if input.links:
                                            print("INPUT NODES")
                                            
                                            #for l in input.links:
                                            #    shader_node_tree_links.remove(l)
                                            for l in  input.links:
                                                tree.links.remove(l)
                                        
                                        
                                        else:
                                            print("NO INPUT NODES")
                                        input.default_value = kf_met
                                    
                                    if input.name == "Roughness":
                                        print("Roughness!!!")
                                        if input.links:
                                            print("INPUT NODES")
                                            for l in  input.links:
                                                tree.links.remove(l)
                                        else:
                                            print("NO INPUT NODES")
                                        input.default_value = kf_rough


def multiply_nodes(kf_met, kf_rough):
    for obj in bpy.data.objects:
        if obj.material_slots:
            for slot in obj.material_slots:
                material = slot.material
                default_albedo    = None
                default_roughness = None
                default_metallic  = None
                if material:
                    if material.use_nodes:
                        tree = material.node_tree
                        for node in tree.nodes:
                            if node.type == 'BSDF_PRINCIPLED':
                                if "Metallic" in node.inputs:
                                    if node.inputs['Metallic'].links:
                                        inputnode_metalloc = node.inputs['Metallic'].links[0].from_socket
                                        math_node_a = tree.nodes.new('ShaderNodeMath')
                                        math_node_a.operation = 'ADD'
                                        default_value_m = random.choices([0.1, 0.2, 0.3, 0.4, 0.5],
                                                                         [0.6, 0.5, 0.4, 0.3, 0.2],
                                                                          k=1)[0]
                                        math_node_a.inputs[1].default_value = default_value_m
                                        tree.links.new(inputnode_metalloc, math_node_a.inputs[0])
                                        
                                        math_node_m = tree.nodes.new('ShaderNodeMath')
                                        math_node_m.operation = 'MULTIPLY'
                                        math_node_m.use_clamp = True
                                        math_node_m.inputs[1].default_value = kf_met
                                        tree.links.new(math_node_a.outputs[0], math_node_m.inputs[0])
                                        tree.links.new(math_node_m.outputs[0], node.inputs['Metallic'])
                                    else:
                                        inputnode_metalloc = "No input color node at all"
                                        default_metallic = node.inputs['Metallic'].default_value
                                        math_node_a = tree.nodes.new('ShaderNodeMath')
                                        math_node_a.operation = 'ADD'
                                        math_node_a.inputs[0].default_value = default_metallic
                                        math_node_a.inputs[1].default_value = 0.1
                                        
                                        math_node_m = tree.nodes.new('ShaderNodeMath')
                                        math_node_m.operation = 'MULTIPLY'
                                        math_node_m.use_clamp = True
                                        math_node_m.inputs[1].default_value = kf_met
                                        tree.links.new(math_node_a.outputs[0], math_node_m.inputs[0])
                                        tree.links.new(math_node_m.outputs[0], node.inputs['Metallic'])
                                else:
                                    inputnode_metalloc = None

                                if "Roughness" in node.inputs:
                                    if node.inputs['Roughness'].links:
                                        inputnode_roughness = node.inputs['Roughness'].links[0].from_socket                                  
                                        math_node_r = tree.nodes.new('ShaderNodeMath')
                                        math_node_r.operation = 'MULTIPLY'
                                        math_node_r.use_clamp = True
                                        math_node_r.inputs[1].default_value = kf_rough
                                        tree.links.new(inputnode_roughness, math_node_r.inputs[0])
                                        tree.links.new(math_node_r.outputs[0], node.inputs['Roughness'])
                                    else:
                                        inputnode_roughness = "No input color node at all"
                                        default_roughness =  node.inputs['Roughness'].default_value
                                        math_node_r = tree.nodes.new('ShaderNodeMath')
                                        math_node_r.operation = 'MULTIPLY'
                                        math_node_r.use_clamp = True
                                        math_node_r.inputs[0].default_value = default_roughness
                                        math_node_r.inputs[1].default_value = kf_rough
                                        tree.links.new(math_node_r.outputs[0], node.inputs['Roughness'])
                                else:
                                    inputnode_roughness = None
                                    

def get_normal_prefix(tp):
    return "" if tp == "smooth" else f"{tp}_"

def setup_normal_aov(normal_type=("face",)):
    
    for obj in bpy.data.objects:
        if obj.material_slots:
            for slot in obj.material_slots:
                material = slot.material
                if material:
                    if material.use_nodes:
                        tree = material.node_tree
                        
                        ### SETUP NORMAL AOV
                        
                        node_types = {node.type: node for node in tree.nodes}
                        node_names = {node.label: node for node in tree.nodes}
                        
                        for tp in normal_type:
                            prefix = get_normal_prefix(tp)
                            geometry_node_key = 'NORMAL_MAP' if tp == "bump" else 'NEW_GEOMETRY'
                            if geometry_node_key in node_types:
                                geometry_node = node_types[geometry_node_key]
                            elif normal_type == "bump":
                                geometry_node = tree.nodes.new('ShaderNodeNormalMap')
                                geometry_node.space = 'TANGENT'
                            else:
                                geometry_node = tree.nodes.new('ShaderNodeNewGeometry')
                            node_types[geometry_node_key] = geometry_node
                            
                            range_node_key = f'{prefix}NORMAL_RANGE_NODE'.upper()
                            if range_node_key in node_names:
                                range_node = node_names[range_node_key]
                            else:
                                range_node = tree.nodes.new('ShaderNodeMapRange')
                                
                                range_node.data_type = 'FLOAT_VECTOR'
                                range_node.clamp = True
                                for idx in range(3):
                                    range_node.inputs['From Min'].default_value[idx] = -1
                                    range_node.inputs['From Max'].default_value[idx] = 1
                                    
                                    range_node.inputs['To Min'].default_value[idx] = 0
                                    range_node.inputs['To Max'].default_value[idx] = 1
                                range_node.label = range_node_key
                                node_names[range_node_key] = range_node
                                
                                output_key = 'True Normal' if tp == "face" else 'Normal'
                                tree.links.new(geometry_node.outputs[output_key], range_node.inputs['Vector'])
                            
                            aov_node_key = f'{prefix}NORMAL_AOV'.upper()
                            if aov_node_key in node_names:
                                true_normal_aov = node_names[aov_node_key]
                            else:
                                true_normal_aov = tree.nodes.new('ShaderNodeOutputAOV')
                                true_normal_aov.name = f"{prefix}normal_aov"
                                true_normal_aov.label = aov_node_key
                                node_names[aov_node_key] = true_normal_aov
                                tree.links.new(range_node.outputs['Vector'], true_normal_aov.inputs['Color'])
    
    for tp in normal_type:
        bpy.ops.scene.view_layer_add_aov()
        bpy.context.scene.view_layers["ViewLayer"].active_aov.name = f"{get_normal_prefix(tp)}normal_aov"


def setup_pbr_aov():
    
    if len(bpy.data.objects) > 50:
        print("Skip because of too big object")
        exit(0)

    for obj in bpy.data.objects:
        if obj.material_slots:
            for slot in obj.material_slots:
                material = slot.material
                default_albedo    = None
                default_roughness = None
                default_metallic  = None
                if material:
                    if material.use_nodes:
                        tree = material.node_tree
                        for node in tree.nodes:
                            if node.type == 'BSDF_PRINCIPLED':

                                if "Base Color" in node.inputs:
                                    if node.inputs['Base Color'].links:
                                        inputnode = node.inputs['Base Color'].links[0]
                                    else:
                                        default_albedo = node.inputs['Base Color'].default_value#)
                                        inputnode = "No input color node at all"
                                else:
                                    inputnode = None

                                if "Metallic" in node.inputs:
                                    if node.inputs['Metallic'].links:
                                        inputnode_metalloc = node.inputs['Metallic'].links[0].from_socket
                                    else:
                                        inputnode_metalloc = "No input color node at all"
                                        default_metallic =  node.inputs['Metallic'].default_value
                                else:
                                    inputnode_metalloc = None

                                if "Roughness" in node.inputs:
                                    if node.inputs['Roughness'].links:
                                        inputnode_roughness = node.inputs['Roughness'].links[0].from_socket
                                    else:
                                        inputnode_roughness = "No input color node at all"
                                        default_roughness =  node.inputs['Roughness'].default_value
                                else:
                                    inputnode_roughness = None

                            if node.type == 'OUTPUT_MATERIAL':
                                try:
                                    if inputnode is not None:
                                        true_albedo_aov = tree.nodes.new('ShaderNodeOutputAOV')
                                        true_albedo_aov.name = "TrueAlbedo"
                                        if inputnode == "No input color node at all":
                                            value_node = tree.nodes.new('ShaderNodeRGB')
                                            value_node.outputs["Color"].default_value = default_albedo 
                                            tree.links.new(value_node.outputs[0], true_albedo_aov.inputs['Color'])
                                        else:
                                            tree.links.new(inputnode.from_node.outputs[inputnode.from_socket.name], true_albedo_aov.inputs['Color']) 
                                except:
                                    print(exit)
                                    exit(0)
                                    
                                if inputnode_metalloc is not None:
                                    met_aov = tree.nodes.new('ShaderNodeOutputAOV')
                                    met_aov.name = "AOVMetallic"
                                    if inputnode_metalloc == "No input color node at all":
                                        value_node = tree.nodes.new('ShaderNodeValue')
                                        value_node.outputs["Value"].default_value = default_metallic
                                        tree.links.new(value_node.outputs[0], met_aov.inputs[0])
                                    else:
                                        tree.links.new(inputnode_metalloc, met_aov.inputs[0])

                                if inputnode_roughness is not None:
                                    roughness_aov = tree.nodes.new('ShaderNodeOutputAOV')
                                    roughness_aov.name = "AOVRoughness"
                                    if inputnode_roughness == "No input color node at all":
                                        value_node = tree.nodes.new('ShaderNodeValue')
                                        value_node.outputs["Value"].default_value = default_roughness
                                        tree.links.new(value_node.outputs[0], roughness_aov.inputs[0])
                                    else:
                                        tree.links.new(inputnode_roughness, roughness_aov.inputs[0])
    
    bpy.ops.scene.view_layer_add_aov()
    bpy.context.scene.view_layers["ViewLayer"].active_aov.name = "TrueAlbedo"

    bpy.ops.scene.view_layer_add_aov()
    bpy.context.scene.view_layers["ViewLayer"].active_aov.name = "AOVMetallic"

    bpy.ops.scene.view_layer_add_aov()
    bpy.context.scene.view_layers["ViewLayer"].active_aov.name = "AOVRoughness"
        

def wath_nodes():
    # Проходим по всем объектам сцены
    for obj in bpy.data.objects:
        # Проверяем, есть ли у объекта материалы
        if obj.material_slots:
            print("Материалы для объекта", obj.name)
            # Проходим по всем материалам в слотах объекта
            for slot in obj.material_slots:
                material = slot.material
                if material:
                    print("Материал:", material.name)
                    if material.use_nodes:
                        # Получаем узловое дерево материала
                        tree = material.node_tree
                        # Проходим по всем узлам в дереве
                        for node in tree.nodes:
                            print(" - Узел:", node.name, "Тип:", node.type)


def setup_default_material():
    
    base_material = bpy.data.materials.new(name="BaseCustomMaterial")
    
    base_material.use_nodes = True
    nodes = base_material.node_tree.nodes
    principled_bsdf = nodes.get("Principled BSDF")
    if principled_bsdf:
        principled_bsdf.inputs["Base Color"].default_value = (1, 1, 1, 1)
        principled_bsdf.inputs["Metallic"].default_value = 0
        principled_bsdf.inputs["Roughness"].default_value = 1
    
    for obj in bpy.context.scene.objects:
        if hasattr(obj, 'type') and obj.type == 'MESH':
            if len(obj.data.materials.items()) == 0:
                obj.data.materials.clear()
                obj.data.materials.append(base_material)



def rename_materials_to_format(output_dir, render_index,  fine_index, **kwargs):

    if kwargs.get("render_normals", False):
        path = glob.glob(f"{output_dir}/tmp*normal*{render_index:04d}*")[0]
        file_ext = path.split(".")[-1]
        basename = os.path.basename(path)
        basename = basename.replace("tmp_", "")
        basename = basename.replace(f"_{render_index:04d}", f"_{fine_index:04d}")
        right_path = os.path.join(output_dir, basename)
        os.rename(path, right_path)

    if kwargs.get("render_depth", False):
        path = glob.glob(f"{output_dir}/tmp*depth*{render_index:04d}*")[0]
        file_ext = path.split(".")[-1]
        basename = os.path.basename(path)
        basename = basename.replace("tmp_", "")
        basename = basename.replace(f"_{render_index:04d}", f"_{fine_index:04d}")
        right_path = os.path.join(output_dir, basename)
        os.rename(path, right_path)

    if kwargs.get("render_albedo", False):
        path = glob.glob(f"{output_dir}/tmp*albedo*{render_index:04d}*")[0]
        file_ext = path.split(".")[-1]
        basename = os.path.basename(path)
        basename = basename.replace("tmp_", "")
        basename = basename.replace(f"_{render_index:04d}", f"_{fine_index:04d}")
        right_path = os.path.join(output_dir, basename)
        # right_path = os.path.join(output_dir, f"albedo_{fine_index:04d}.{file_ext}")
        os.rename(path, right_path)

    if kwargs.get("render_roughness", False):
        path = glob.glob(f"{output_dir}/tmp*roughness*{render_index:04d}*")[0]
        file_ext = path.split(".")[-1]
        basename = os.path.basename(path)
        basename = basename.replace("tmp_", "")
        basename = basename.replace(f"_{render_index:04d}", f"_{fine_index:04d}")
        right_path = os.path.join(output_dir, basename)
        # right_path = os.path.join(output_dir, f"roughness_{fine_index:04d}.{file_ext}")
        os.rename(path, right_path)

    if kwargs.get("render_metallic", False):
        path = glob.glob(f"{output_dir}/tmp*metallic*{render_index:04d}*")[0]
        file_ext = path.split(".")[-1]
        basename = os.path.basename(path)
        basename = basename.replace("tmp_", "")
        basename = basename.replace(f"_{render_index:04d}", f"_{fine_index:04d}")
        right_path = os.path.join(output_dir, basename)
        # right_path = os.path.join(output_dir, f"metallic_{fine_index:04d}.{file_ext}")
        os.rename(path, right_path)
        

def render4view(output_dir, 
                render_normals , normal_output_node, 
                render_depth, depth_output_node, 
                render_albedo, albedo_file_output, 
                render_roughness, roughness_file_output, 
                render_metallic, metallic_file_output,
                name_prefix="", ortho=False,):

    start = time.time()
    # Устанавливаем угол зенита в радианах (20 градусов)
    elevation_angle = math.radians(20)
    # Четыре равноудаленных азимута
    if ortho:
        azimuth_angles = [
            math.radians(0),
            math.radians(90),
            math.radians(180),
            math.radians(270),
        ]
    else:
        azimuth_angles = [
            math.radians(45),
            math.radians(135),
            math.radians(225),
            math.radians(315),
        ]
    
    
    bpy.context.scene.render.engine = 'CYCLES'
    bpy.context.scene.render.use_persistent_data = True
    
    if ortho:
        radius = 1.5
    else:
        radius = random.uniform(1.4,1.7 )
    print("Render 4 view")

    # render the images
    for i in range(4):
        
        if render_normals:
            for node in normal_output_node:
                node.file_slots[0].path = f"tmp_{node.label}_{name_prefix}"
        
        if render_depth:
            depth_output_node.file_slots[0].path = f"tmp_depth_{name_prefix}"

        if render_albedo:
            albedo_file_output.file_slots[0].path = f"tmp_albedo_{name_prefix}"

        if render_roughness:
            roughness_file_output.file_slots[0].path = f"tmp_roughness_{name_prefix}"

        if render_metallic:
            metallic_file_output.file_slots[0].path = f"tmp_metallic_{name_prefix}"
        
        
        
        # set camera
        x = radius * math.cos(azimuth_angles[i]) * math.cos(elevation_angle)
        y = radius * math.sin(azimuth_angles[i]) * math.cos(elevation_angle)
        z = radius * math.sin(elevation_angle)
        camera = randomize_camera(x,y,z,
                                  only_northern_hemisphere=False
                                  )
        
        
        
        # render the image
        render_path = os.path.join(output_dir, f"{name_prefix}{i:03d}.png")
        scene.render.filepath = render_path
        bpy.ops.render.render(write_still=True)
        
        
        # save camera RT matrix
        rt_matrix = get_3x4_RT_matrix_from_blender(camera)
        rt_matrix_path = os.path.join(output_dir, f"{name_prefix}{i:03d}.npy")
        np.save(rt_matrix_path, rt_matrix)
        
        
        # save camera RT matrix
        camera_angle_x = camera.data.angle_x
        camera_angle_path = os.path.join(output_dir, f"{name_prefix}{i:03d}_angle_x.npy")
        np.save(camera_angle_path, camera_angle_x)

        rename_materials_to_format(output_dir=output_dir,
                                  render_index=bpy.context.scene.frame_current,
                                  fine_index=i,
                                  render_normals=render_normals,
                                  render_depth=render_depth,
                                  render_albedo=render_albedo,
                                  render_roughness=render_roughness,
                                  render_metallic=render_metallic)

    print(f"Total time for 32 view rendering", time.time()-start)

def render32view(output_dir, 
                 render_normals , normal_output_node, 
                 render_depth, depth_output_node, 
                 render_albedo, albedo_file_output, 
                 render_roughness, roughness_file_output, 
                 render_metallic, metallic_file_output):
    
    start = time.time()
    # render the images
    bpy.context.scene.render.use_persistent_data = True
    for i in range(32):
        start_iteration = time.time()
        if render_normals:
            for node in normal_output_node:
                node.file_slots[0].path = f"tmp_{node.label}_"
        
        if render_depth:
            depth_output_node.file_slots[0].path =  f"tmp_depth_"

        if render_albedo:
            albedo_file_output.file_slots[0].path = f"tmp_albedo_"

        if render_roughness:
            roughness_file_output.file_slots[0].path = f"tmp_roughness_"

        if render_metallic:
            metallic_file_output.file_slots[0].path = f"tmp_metallic_"
        
        
        # set camera
        camera = randomize_camera_orig(
            only_northern_hemisphere=False, radius_min = 1.4, radius_max=1.7
        )
        
        # render the image
        render_path = os.path.join(output_dir, f"{i:03d}.png")
        scene.render.filepath = render_path
        bpy.ops.render.render(write_still=True)
        
        # save camera RT matrix
        rt_matrix = get_3x4_RT_matrix_from_blender(camera)
        rt_matrix_path = os.path.join(output_dir, f"{i:03d}.npy")
        np.save(rt_matrix_path, rt_matrix)
        
        # save camera RT matrix
        camera_angle_x = camera.data.angle_x
        camera_angle_path = os.path.join(output_dir, f"{i:03d}_angle_x.npy")
        np.save(camera_angle_path, camera_angle_x)

        rename_materials_to_format(output_dir=output_dir,
                                  render_index=bpy.context.scene.frame_current,
                                  fine_index=i,
                                  render_normals=render_normals,
                                  render_depth=render_depth,
                                  render_albedo=render_albedo,
                                  render_roughness=render_roughness,
                                  render_metallic=render_metallic)
        print(f"Total time for frame {i} view rendering", time.time() - start_iteration)
        
    print(f"Total time for i view rendering", time.time()-start)
    #exit(0)

def render36view(output_dir,
                 render_normals, normal_output_node,
                 render_depth,depth_output_node,
                 render_albedo, albedo_file_output, 
                 render_roughness, roughness_file_output, 
                 render_metallic, metallic_file_output, ortho):
    render4view(output_dir,
                render_normals, normal_output_node, render_depth,depth_output_node,
                render_albedo, albedo_file_output, render_roughness, roughness_file_output, 
                render_metallic, metallic_file_output, "4view_", ortho)
    
    render32view(output_dir, 
                 render_normals, normal_output_node, render_depth,depth_output_node,
                 render_albedo, albedo_file_output, render_roughness, roughness_file_output, 
                 render_metallic, metallic_file_output)

def render_gif(output_dir):
    
    # Устанавливаем угол зенита в радианах (20 градусов)
    elevation_angle = math.radians(20)
    # Четыре равноудаленных азимута
    azimuth_angles = [math.radians(45), math.radians(135), math.radians(225), math.radians(315)]
    
    
    bpy.context.scene.render.engine = 'CYCLES'
    bpy.context.scene.render.use_persistent_data = True
    
    
    
    radius = 1.7#random.uniform(1.4,1.7 )
    # render the images
    for i in range(40):
        
        
        # set camera
        x = radius * math.cos(math.radians(i * 9)) * math.cos(elevation_angle)
        y = radius * math.sin(math.radians(i * 9)) * math.cos(elevation_angle)
        z = radius * math.sin(elevation_angle)
        camera = randomize_camera(x,y,z,
                                  only_northern_hemisphere=False
                                  )
        
        
        
        # render the image
        render_path = os.path.join(output_dir, f"{i:03d}.png")
        scene.render.filepath = render_path
        bpy.ops.render.render(write_still=True)
        bpy.context.scene.render.use_persistent_data = True

def run_convert_soyuzscene2blend(object_path):

    print("FOLDER OBJECT: ", object_path)

    model_dir = os.path.join(object_path, "mdl", "publish", "caches", "fbx")
    models = os.listdir(model_dir)
    models = [mdl for mdl in models if mdl.endswith(".fbx")]
    model = models[0]
    model_path = os.path.join(model_dir, model)
    print("MODEL PATH: ", model)

    textures_paths = glob.glob(os.path.join(object_path, "texturing", "publish", "textures")+"/*/*.tif")

    texture_dict = {}

    for texture_path in textures_paths:
        texture_basename = texture_path.split("/")[-1]
        texture_groupname = texture_basename.split(".")[0]
        texture_materialname = "_".join(texture_groupname.split("_")[:-1])
        texture_component = texture_groupname.split("_")[-1]

        if texture_materialname not in texture_dict:
            texture_dict[texture_materialname] = {}
        if texture_component not in texture_dict[texture_materialname]:
            texture_dict[texture_materialname][texture_component] = {"full_path": [], "basenane": []}

        texture_dict[texture_materialname][texture_component]["full_path"].append(texture_path)
        texture_dict[texture_materialname][texture_component]["basenane"].append(texture_basename)



    reset_scene()
    load_object(model_path)

    for im in bpy.data.images:
        bpy.data.images.remove(im)


    for material in bpy.data.materials:

        if material.name not in texture_dict:
            continue

        # Проверяем, использует ли материал узлы
        if material.use_nodes:
            # Получаем узловое дерево материала
            tree = material.node_tree
            links = tree.links

            print("ЕСТЬ НОДЫ")

            # Пройдемся по всем узлам, чтобы найти BSDF узел
            for node in tree.nodes:

                print(node.type)

                if node.type == 'BSDF_PRINCIPLED':
                    print("BSDF_PRINCIPLED NODES")

                    bsdf_node = node

                if node.type == "NORMAL_MAP":
                    normal_node = node

                if node.type == "TEX_IMAGE":
                    tree.nodes.remove(node)


            for texture_component in texture_dict[material.name]:
                if texture_component not in ["dif", "rgh", "nrm", "met", "opc"]:
                    continue
                print("ДЕЛАЕМ КОМПОНЕНТ:", texture_component)
                tex_node = tree.nodes.new("ShaderNodeTexImage")
                tex_node.name = material.name+"_"+texture_component

                pre_images = bpy.data.images.keys()

                bpy.ops.image.open(filepath=os.path.join(os.getcwd(), texture_dict[material.name][texture_component]["full_path"][0]),
                                   relative_path=True, show_multiview=False)

                after_images = bpy.data.images.keys()
                image = set(after_images).difference(set(pre_images))
                image = bpy.data.images[list(image)[0]]
                tex_node.image = image

                if texture_component == "dif":
                    links.new(tex_node.outputs["Color"], bsdf_node.inputs["Base Color"])
                elif texture_component == "opc":
                    image.colorspace_settings.name = "Non-Color"
                    links.new(tex_node.outputs["Color"], bsdf_node.inputs["Alpha"])
                elif texture_component == "rgh":
                    image.colorspace_settings.name = "Non-Color"
                    links.new(tex_node.outputs["Color"], bsdf_node.inputs["Roughness"])
                elif texture_component == "nrm":
                    image.colorspace_settings.name = "Non-Color"
                    links.new(tex_node.outputs["Color"], normal_node.inputs["Color"])
                elif texture_component == "met":
                    image.colorspace_settings.name = "Non-Color"
                    links.new(tex_node.outputs["Color"], bsdf_node.inputs["Metallic"])


    # bpy.ops.export_scene.gltf(filepath=os.path.join(object_path, "output.glb"))
    # bpy.ops.wm.save_as_mainfile(filepath=os.path.join(object_path, "output_tmp.blend"))



def render_object(
        object_file: str,
        num_renders: int,
        only_northern_hemisphere: bool,
        output_dir: str,
        path_cameras:str,
        num_pbr:int,
        light:str,
        number_view:int,
        fov: float,
        camera_lens:int,
        render_normals:bool,
        normal_type:str,
        render_depth:bool,
        render_albedo:bool,
        render_roughness:bool,
        render_metallic:bool,
        gso:bool = False,
        soyuz_data: bool = False,
        fix_transparent=False,
        rand_fov=False,
        ortho=False,
) -> None:
    """Saves rendered images with its camera matrix and metadata of the object.

    Args:
        object_file (str): Path to the object file.
        num_renders (int): Number of renders to save of the object.
        only_northern_hemisphere (bool): Whether to only render sides of the object that
            are in the northern hemisphere. This is useful for rendering objects that
            are photogrammetrically scanned, as the bottom of the object often has
            holes.
        output_dir (str): Path to the directory where the rendered images and metadata
            will be saved.

    Returns:
        None
    """
    os.makedirs(output_dir, exist_ok=True)

    if soyuz_data:
        reset_scene()
        reset_cameras()
        run_convert_soyuzscene2blend(object_path=object_file)
    else:
        # load the object
        if object_file.endswith(".blend"):
            bpy.ops.object.mode_set(mode="OBJECT")
            reset_cameras()
            delete_invisible_objects()
            load_object(object_file)
        else:
            reset_scene()
            load_object(object_file)

    
    # Set up cameras
    cam = scene.objects["Camera"]

    cam.data.sensor_width = 32
    
    if fov == 0.0:
        if rand_fov:
            cam.data.lens = random.uniform(35,39)
        else:
            cam.data.lens = 35
    else:
        cam.data.lens_unit = "FOV"
        cam.data.angle_x = fov/180*math.pi
    

    # Set up camera constraints
    cam_constraint = cam.constraints.new(type="TRACK_TO")
    cam_constraint.track_axis = "TRACK_NEGATIVE_Z"
    cam_constraint.up_axis = "UP_Y"
    # empty = bpy.data.objects.new("Empty", None)
    # scene.collection.objects.link(empty)
    # cam_constraint.target = empty
    

    # Extract the metadata. This must be done before normalizing the scene to get
    # accurate bounding box information.
    metadata_extractor = MetadataExtractor(
        object_path=object_file, scene=scene, bdata=bpy.data
    )
    metadata = metadata_extractor.get_metadata()

    # delete all objects that are not meshes
    if object_file.lower().endswith(".usdz") or soyuz_data:
        # don't delete missing textures on usdz files, lots of them are embedded
        missing_textures = None
    else:
        missing_textures = delete_missing_textures()
    metadata["missing_textures"] = missing_textures


    
    # possibly apply a random color to all objects
    if object_file.endswith(".stl") or object_file.endswith(".ply"):
        assert len(bpy.context.selected_objects) == 1
        rand_color = apply_single_random_color_to_all_objects()
        metadata["random_color"] = rand_color
    else:
        metadata["random_color"] = None

    # save metadata
    metadata_path = os.path.join(output_dir, "metadata.json")
    os.makedirs(os.path.dirname(metadata_path), exist_ok=True)
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, sort_keys=True, indent=2)
        
    if fix_transparent and metadata["transparent"]:
        for mat in bpy.data.materials:
            mat.blend_method = 'OPAQUE'

    # normalize the scene
    normalize_scene(gso=gso)
    
    # Удаляем все источники освещения из сцены
    for obj in bpy.data.objects:
        if obj.type == 'LIGHT':
            bpy.data.objects.remove(obj)

    setup_default_material()
    
    if light=="skybox":
        set_sky_texture()
    elif light=="hadmare":
        hadmare_lighting()
    elif light=="hadmare_random":
        randomize_lighting()
    elif light=="hdr_maps_all":
        light = random_color_hdrmap("all")
    elif light=="hdr_maps_bw":
        light = random_color_hdrmap("bw")
    elif light=="hdr_maps_color":
        light = random_color_hdrmap("color")
    
    
    kf_met = None
    kf_rough = None
    if num_pbr <= 0:
        kf_met = -1
        kf_rough = -1
    else:
        # kf_met = random.choice([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        # kf_rough = random.choice([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        # kf_met = random.choice([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        # kf_rough = random.choice([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        kf_met = random.choices([0.1, 0.2, 0.3, 0.4, 0.5],
                                [0.2, 0.3, 0.4, 0.5, 0.6],
                                k=1)[0]
        kf_rough = random.choices([0.4, 0.5, 0.6, 0.7, 0.8],
                                  [0.6, 0.5, 0.4, 0.3, 0.2],
                                  k=1)[0]
        

    print(f"kf_met: {kf_met}")
    print(f"kf_rough: {kf_rough}")
    
    
    #wath_nodes()
    if num_pbr > 0:
        #setting_nodes(1.0, 0)
        # setting_nodes(kf_met, kf_rough)
        multiply_nodes(1 + kf_met, kf_rough)

    if num_pbr == -1:
        setting_nodes_hsv_default() ###In very beautiful :))))
    
    #wath_nodes()
    
    scene.use_nodes = True
    do_extra_passes = False
    
    if render_depth:
        scene.view_layers["ViewLayer"].use_pass_z              = True
    if render_normals:
        scene.view_layers["ViewLayer"].use_pass_normal         = True
    if do_extra_passes:
        scene.view_layers["ViewLayer"].use_pass_diffuse_color  = True
        scene.view_layers["ViewLayer"].use_pass_glossy_color   = True
        scene.view_layers["ViewLayer"].use_pass_diffuse_direct = True

    if render_normals:
        setup_normal_aov(normal_type=normal_type)

    if render_albedo or render_roughness or render_metallic:
        setup_pbr_aov()

        print("setting up pbr aov")
    
    tree = scene.node_tree
    links = tree.links
    render_layers_node = tree.nodes.new(type="CompositorNodeRLayers")
    
    normal_output_nodes = []
    if render_normals:
        for tp in normal_type:
            normal_output_node = tree.nodes.new(type="CompositorNodeOutputFile")
            normal_output_node.label = f"{get_normal_prefix(tp)}normal"
            normal_output_node.base_path = output_dir
            normal_output_node.file_slots[0].use_node_format = True
            normal_output_node.format.file_format = "OPEN_EXR"
            # normal_output_node.format.file_format = "PNG"
            normal_output_node.format.color_management = "OVERRIDE"
            normal_output_node.format.color_mode = "RGBA"
            # normal_output_node.format.color_depth = "32"
            normal_output_node.format.color_depth = "16"
            normal_output_node.format.view_settings.view_transform = 'Raw'
            links.new(render_layers_node.outputs[f"{get_normal_prefix(tp)}normal_aov"], normal_output_node.inputs[0])
            normal_output_nodes.append(normal_output_node)

    if render_depth:
        depth_output_node = tree.nodes.new(type="CompositorNodeOutputFile")
        depth_output_node.label = "Depth_Output"
        depth_output_node.base_path =  output_dir #os.path.join(output_dir, "depth") #output_dir
        depth_output_node.file_slots[0].use_node_format = True
        depth_output_node.format.file_format = "OPEN_EXR"
        depth_output_node.format.color_depth = "16"
        links.new(render_layers_node.outputs['Depth'], depth_output_node.inputs[0])
    else:
        depth_output_node = None


    bpy.context.scene.display_settings.display_device = 'sRGB'            
    bpy.context.scene.view_settings.view_transform = 'Standard'
    bpy.context.scene.view_settings.exposure = 0
    bpy.context.scene.view_settings.gamma = 1.0
    bpy.context.scene.view_settings.look = 'None'
    bpy.context.scene.view_settings.use_curve_mapping = False
    bpy.context.scene.sequencer_colorspace_settings.name = 'sRGB'
    

    if render_albedo:
        
        true_albedo = tree.nodes.new(type="CompositorNodeSetAlpha")
        links.new(render_layers_node.outputs["TrueAlbedo"], true_albedo.inputs["Image"])
        links.new(render_layers_node.outputs["Alpha"], true_albedo.inputs["Alpha"])
    
        albedo_file_output = tree.nodes.new(type="CompositorNodeOutputFile")
        albedo_file_output.label = "Albedo"
        albedo_file_output.base_path = output_dir
        albedo_file_output.file_slots[0].use_node_format = True
        albedo_file_output.format.file_format = "PNG"
        albedo_file_output.format.color_mode = "RGBA"
        links.new(true_albedo.outputs["Image"], albedo_file_output.inputs[0])

    else:
        true_albedo = albedo_file_output = None

    if render_roughness:
        
        true_roughness = tree.nodes.new(type="CompositorNodeSetAlpha")
        links.new(render_layers_node.outputs["AOVRoughness"], true_roughness.inputs["Image"])
        links.new(render_layers_node.outputs["Alpha"], true_roughness.inputs["Alpha"])
    
        roughness_file_output = tree.nodes.new(type="CompositorNodeOutputFile")
        roughness_file_output.label = "Roughness Output"
        roughness_file_output.base_path = output_dir #os.path.join(output_dir, "albedo")
        roughness_file_output.file_slots[0].use_node_format = True
        roughness_file_output.format.color_mode = "RGBA"
        roughness_file_output.format.file_format = "PNG"
        # roughness_file_output.format.file_format = "OPEN_EXR"
        # roughness_file_output.format.color_depth = "16"
        links.new(true_roughness.outputs["Image"], roughness_file_output.inputs[0])

    else:
        true_roughness = roughness_file_output = None

    if render_metallic:
        
        true_metallic = tree.nodes.new(type="CompositorNodeSetAlpha")
        links.new(render_layers_node.outputs["AOVMetallic"], true_metallic.inputs["Image"])
        links.new(render_layers_node.outputs["Alpha"], true_metallic.inputs["Alpha"])
    
        metallic_file_output = tree.nodes.new(type="CompositorNodeOutputFile")
        metallic_file_output.label = "Metallic Output"
        metallic_file_output.base_path = output_dir #os.path.join(output_dir, "albedo")
        metallic_file_output.file_slots[0].use_node_format = True
        metallic_file_output.format.color_mode = "RGBA"
        metallic_file_output.format.file_format = "PNG"
        # metallic_file_output.format.file_format = "OPEN_EXR"
        # metallic_file_output.format.color_depth = "16"
        links.new(true_metallic.outputs["Image"], metallic_file_output.inputs[0])

    else:
        true_metallic = metallic_file_output = None
        


        # depth_output_node.file_slots[0].path = "4view_depth_"


    
    print(f"Do rendering for {number_view}")

    if number_view == 4:
        render4view(output_dir,render_normals, normal_output_nodes,render_depth,depth_output_node,
                    render_albedo, albedo_file_output, 
                    render_roughness, roughness_file_output, 
                    render_metallic, metallic_file_output, ortho=ortho)
    if number_view == 32:
        render32view(output_dir,render_normals, normal_output_nodes,render_depth,depth_output_node,
                     render_albedo, albedo_file_output, 
                     render_roughness, roughness_file_output, 
                     render_metallic, metallic_file_output)
    if number_view == 36:
        render36view(output_dir,render_normals, normal_output_nodes,render_depth,depth_output_node,
                     render_albedo, albedo_file_output, 
                     render_roughness, roughness_file_output, 
                     render_metallic, metallic_file_output, ortho)
    if number_view == 40:
        render_gif(output_dir)


    # Data to be written
    dictionary = {
        "kf_met": kf_met,
        "kf_rough": kf_rough,
        "light":light,
        "num_pbr": num_pbr,
        "camera_lens":cam.data.lens
        
    }

    # Serializing json
    json_object = json.dumps(dictionary, indent=4)

    # Writing to sample.json
    with open( os.path.join(output_dir,"kf.json"), "w") as outfile:
        outfile.write(json_object)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--object_path",
        type=str,
        required=True,
        help="Path to the object file",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Path to the directory where the rendered images and metadata will be saved.",
    )
    parser.add_argument(
        "--engine",
        type=str,
        default="CYCLES",
        choices=["CYCLES", "BLENDER_EEVEE", "BLENDER_EEVEE_NEXT"],
    )
    parser.add_argument(
        "--only_northern_hemisphere",
        action="store_true",
        help="Only render the northern hemisphere of the object.",
        default=False,
    )
    parser.add_argument(
        "--num_renders",
        type=int,
        default=32,
        help="Number of renders to save of the object.",
    )
    parser.add_argument(
        "--path_cameras",
        type=str,
        help="Path with old cameras.",
    )
    parser.add_argument(
        "--light",
        type=str,
        help="light",
        default="skybox",
    )
    parser.add_argument(
        "--num_pbr",
        type=int,
        help="Path with old cameras.",
    )

    parser.add_argument(
        "--number_view",
        type=int,
        help="number_view in [4, 32, 36]",
    )

    parser.add_argument(
        "--resolution",
        type=int,
        help="resolution rendering",
        default=512,
    )
    parser.add_argument(
        "--format",
        type=str,
        help="format save image",
        default="PNG",
    )

    parser.add_argument(
        "--camera_lens",
        type=float,
        help="camera_lens",
        default=35,
    )
    parser.add_argument(
        "--fov_angle",
        type=float,
        help="field of view in degrees(0-180)",
        default=0.0,
    )
    parser.add_argument(
        "--normals",
        type=str2bool,
        help="render normals",
        default="False",
    )
    parser.add_argument(
        "--normal_type",
        type=str,
        nargs='+',
        help="format normals",
        choices=["face", "smooth", "bump"],
        default=["face"],
    )
    
    parser.add_argument(
        "--depth",
        type=str2bool,
        help="render depth",
        default="False",
    )

    parser.add_argument(
        "--albedo",
        type=str2bool,
        help="render albedo",
        default="False",
    )

    parser.add_argument(
        "--roughness",
        type=str2bool,
        help="render roughness",
        default="False",
    )

    parser.add_argument(
        "--metallic",
        type=str2bool,
        help="render metallic",
        default="False",
    )
    
    parser.add_argument(
        "--gso",
        type=str2bool,
        help="render gso",
        default="False",
    )
    parser.add_argument(
        "--soyuz_data",
        type=str2bool,
        help="soyuz_data",
        default="False",
    )
    parser.add_argument(
        "--fix_transparent",
        type=str2bool,
        help="disable transparency",
        default="False",
    )
    parser.add_argument(
        "--rand_fov",
        type=str2bool,
        help="use random fov",
        default="False",
    )
    parser.add_argument(
        "--ortho",
        type=str2bool,
        help="render first 4 view with fixed radius for crm",
        default="False",
    )
    
    
    
    
    argv = sys.argv[sys.argv.index("--") + 1 :]
    args = parser.parse_args(argv)

    context = bpy.context
    scene = context.scene
    render = scene.render

    # Set render settings
    render.engine = args.engine
    render.image_settings.file_format = args.format
    render.image_settings.color_mode = "RGBA"

    render.resolution_x = args.resolution
    render.resolution_y = args.resolution
    render.resolution_percentage = 100

    # Set cycles settings
    scene.cycles.device = "GPU"
    scene.cycles.samples = 128
    scene.cycles.diffuse_bounces = 1
    scene.cycles.glossy_bounces = 1
    scene.cycles.transparent_max_bounces = 3
    scene.cycles.transmission_bounces = 3
    scene.cycles.filter_width = 0.5
    #bpy.context.view_layer.cycles.denoising_store_passes = True
    #bpy.context.scene.render.denoising_data = 'COMBINED'

    render.compositor_device = "GPU"

    scene.cycles.use_denoising = True
    scene.cycles.denoiser = 'OPENIMAGEDENOISE'#'OPTIX'
    #scene.cycles.denoiser = 'OPTIX'
    scene.cycles.denoising_use_gpu = True
    scene.render.film_transparent = True
    bpy.context.preferences.addons["cycles"].preferences.get_devices()
    bpy.context.preferences.addons[
        "cycles"
    ].preferences.compute_device_type = "CUDA"#"HIP"  # or "OPENCL"

    # Render the images
    
    print(args.normals)
    print(type(args.normals))
    
    print(args.depth)
    print(type(args.depth))
    print(args.normal_type)
    

    print(f"Render args albedo {args.albedo} roughness {args.roughness} metallic {args.metallic}, num_view {args.number_view}")
    render_object(
        object_file=args.object_path,
        num_renders=args.num_renders,
        only_northern_hemisphere=args.only_northern_hemisphere,
        output_dir=args.output_dir,
        path_cameras=args.path_cameras,
        num_pbr=args.num_pbr,
        light = args.light,
        number_view = args.number_view,
        fov = args.fov_angle,
        camera_lens = args.camera_lens,
        render_normals = args.normals,
        normal_type=args.normal_type,
        render_depth = args.depth,
        render_albedo = args.albedo,
        render_roughness = args.roughness,
        render_metallic = args.metallic,
        gso=args.gso,
        soyuz_data=args.soyuz_data,
        fix_transparent=args.fix_transparent,
        rand_fov=args.rand_fov,
        ortho=args.ortho
    )
