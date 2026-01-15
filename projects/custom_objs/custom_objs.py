import numpy as np

from robosuite.models.objects import MujocoXMLObject
from robosuite.utils.mjcf_utils import array_to_string, find_elements, xml_path_completion
from robosuite.models.objects import PotWithHandlesObject

from robosuite.models.objects import CompositeObject
from robosuite.utils.mjcf_utils import BLUE, GREEN, RED, CustomMaterial, add_to_dict, array_to_string
import robosuite.utils.transform_utils as T
from pathlib import Path

_HERE = Path(__file__).parent
_TO_OBJ = _HERE / "assets" / "objects"

class CenteredDoorObject(MujocoXMLObject):
    """
    Door with handle (used in Door)

    Args:
        friction (3-tuple of float): friction parameters to override the ones specified in the XML
        damping (float): damping parameter to override the ones specified in the XML
        lock (bool): Whether to use the locked door variation object or not
    """

    def __init__(self, name, friction=None, damping=None, lock=False):
        xml_path = _TO_OBJ / "centered_door.xml"
        if lock:
            xml_path = _TO_OBJ / "centered_door_lock.xml"
        super().__init__(
            xml_path, name=name, joints=None, obj_type="all", duplicate_collision_geoms=True
        )

        # Set relevant body names
        self.door_body = self.naming_prefix + "door"
        self.frame_body = self.naming_prefix + "frame"
        self.latch_body = self.naming_prefix + "latch"
        self.hinge_joint = self.naming_prefix + "hinge"

        self.lock = lock
        self.friction = friction
        self.damping = damping
        if self.friction is not None:
            self._set_door_friction(self.friction)
        if self.damping is not None:
            self._set_door_damping(self.damping)

    def _set_door_friction(self, friction):
        """
        Helper function to override the door friction directly in the XML

        Args:
            friction (3-tuple of float): friction parameters to override the ones specified in the XML
        """
        hinge = find_elements(root=self.worldbody, tags="joint", attribs={"name": self.hinge_joint}, return_first=True)
        hinge.set("frictionloss", array_to_string(np.array([friction])))

    def _set_door_damping(self, damping):
        """
        Helper function to override the door friction directly in the XML

        Args:
            damping (float): damping parameter to override the ones specified in the XML
        """
        hinge = find_elements(root=self.worldbody, tags="joint", attribs={"name": self.hinge_joint}, return_first=True)
        hinge.set("damping", array_to_string(np.array([damping])))

    @property
    def important_sites(self):
        """
        Returns:
            dict: In addition to any default sites for this object, also provides the following entries

                :`'handle'`: Name of door handle location site
        """
        # Get dict from super call and add to it
        dic = super().important_sites
        dic.update({"handle": self.naming_prefix + "handle"})
        return dic

class BigCubeObject(MujocoXMLObject):
    """
    Big cube object - wide, tall, but shallow depth
    
    Args:
        name (str): Name of this object instance
        friction (3-tuple of float): friction parameters to override the ones specified in the XML
        mass (float): mass parameter to override the one specified in the XML
    """
    
    def __init__(self, name, friction=None, mass=None):
        xml_path = _TO_OBJ / "big_cube.xml"
        super().__init__(
            xml_path, 
            name=name, 
            joints=[dict(type="free", damping="0.0005")], 
            obj_type="all", 
            duplicate_collision_geoms=True
        )
        
        # Set relevant body names
        self.cube_body = self.naming_prefix + "object"
        
        # Store parameters
        self.friction = friction
        self.mass = mass
        
        # Apply custom parameters if provided
        if self.friction is not None:
            self._set_cube_friction(self.friction)
        if self.mass is not None:
            self._set_cube_mass(self.mass)
    
    def _set_cube_friction(self, friction):
        """
        Helper function to override the cube friction directly in the XML
        
        Args:
            friction (3-tuple of float): friction parameters to override the ones specified in the XML
        """
        collision_geom = find_elements(
            root=self.worldbody, 
            tags="geom", 
            attribs={"name": self.naming_prefix + "cube_collision"}, 
            return_first=True
        )
        if collision_geom is not None:
            collision_geom.set("friction", array_to_string(np.array(friction)))
    
    def _set_cube_mass(self, mass):
        """
        Helper function to override the cube mass directly in the XML
        
        Args:
            mass (float): mass parameter to override the one specified in the XML
        """
        inertial = find_elements(
            root=self.worldbody, 
            tags="inertial", 
            return_first=True
        )
        if inertial is not None:
            inertial.set("mass", array_to_string(np.array([mass])))
    
    @property
    def important_sites(self):
        """
        Returns:
            dict: In addition to any default sites for this object, also provides the following entries
            
                :`'center'`: Name of cube center site
                :`'top'`: Name of cube top site  
                :`'front_face'`: Name of cube front face site
                :`'left_face'`: Name of cube left face site
                :`'right_face'`: Name of cube right face site
        """
        # Get dict from super call and add to it
        dic = super().important_sites
        dic.update({
            "center": self.naming_prefix + "center",
            "top": self.naming_prefix + "top", 
            "front_face": self.naming_prefix + "front_face",
            "left_face": self.naming_prefix + "left_face",
            "right_face": self.naming_prefix + "right_face"
        })
        return dic
    
class PotVisualObject(MujocoXMLObject):
    """
    Visual-only pot with handles for target placement visualization.
    Semi-transparent with no collision detection.
    """
    
    def __init__(self, name):
        xml_path = _TO_OBJ / "visual_potwhandle.xml"
        super().__init__(
            xml_path, 
            name=name, 
            joints=None,  # No joints - static object
            obj_type="visual", 
            duplicate_collision_geoms=False
        )
    
    @property
    def important_sites(self):
        """
        Returns:
            dict: Important sites for manipulation
            
                :`'handle0'`: Name of handle0 site (green)
                :`'handle1'`: Name of handle1 site (blue) 
                :`'center'`: Name of pot center site
        """
        return {
            "handle0": self.naming_prefix + "handle0",
            "handle1": self.naming_prefix + "handle1", 
            "center": self.naming_prefix + "center"
        }
    
    @property
    def handle_geoms(self):
        """
        Returns:
            list: All handle geom names
        """
        return [
            self.naming_prefix + "handle0_c",
            self.naming_prefix + "handle0_-", 
            self.naming_prefix + "handle0_+",
            self.naming_prefix + "handle1_c",
            self.naming_prefix + "handle1_-",
            self.naming_prefix + "handle1_+"
        ]
    
    @property
    def pot_geoms(self):
        """
        Returns:
            list: All pot body geom names
        """
        return [
            self.naming_prefix + "base",
            self.naming_prefix + "body0",
            self.naming_prefix + "body1", 
            self.naming_prefix + "body2",
            self.naming_prefix + "body3"
        ]
    
class WineGlassObject(MujocoXMLObject):
    """
    Wine glass object

    Args:
        name (str): Name of this object instance
    """

    def __init__(self, name, friction=None, mass=None):
        xml_path = _TO_OBJ / "wine_glass.xml"
        super().__init__(
            xml_path,
            name=name,
            joints=[dict(type="free", damping="0.0005")],
            obj_type="all",
            duplicate_collision_geoms=True,
        )

class ShelfObject(MujocoXMLObject):
    """
    Shelf object

    Args:
        name (str): Name of this object instance
    """

    def __init__(self, name):
        xml_path = _TO_OBJ / "shelf.xml"
        super().__init__(
            xml_path,
            name=name,
            joints=None,
            obj_type="all",
            duplicate_collision_geoms=True,
        )
    
    @property
    def important_sites(self):
        """
        Returns:
            dict: Important sites for the shelf
            
                :`'top_site'`: Name of shelf top site
                :`'bottom_site'`: Name of shelf bottom site
        """
        return {
            "top": self.naming_prefix + "top_site",
            "bottom": self.naming_prefix + "bottom_site"
        }


class BatteryWithHandlesObject(CompositeObject):
    """
    Generates the Battery object with side handles (used in ExpNarrowGap)

    Args:
        name (str): Name of this Battery object

        body_half_size (3-array of float): If specified, defines the (x,y,z) half-dimensions of the main battery
            body. Otherwise, defaults to [0.07, 0.07, 0.07]

        handle_radius (float): Determines the battery handle radius

        handle_length (float): Determines the battery handle length

        handle_width (float): Determines the battery handle width

        handle_friction (float): Friction value to use for battery handles. Defaults to 1.0

        density (float): Density value to use for all geoms. Defaults to 1000

        use_texture (bool): If true, geoms will be defined by realistic textures and rgba values will be ignored

        rgba_body (4-array or None): If specified, sets pot body rgba values

        rgba_handle_0 (4-array or None): If specified, sets handle 0 rgba values

        rgba_handle_1 (4-array or None): If specified, sets handle 1 rgba values

        solid_handle (bool): If true, uses a single geom to represent the handle

        thickness (float): How thick to make the pot body walls
    """

    def __init__(
        self,
        name,
        body_half_size=(0.07, 0.10, 0.10),
        handle_radius=0.01,
        handle_length=0.09,
        handle_width=0.09,
        handle_friction=1.0,
        density=400,
        use_texture=True,
        rgba_body=None,
        rgba_handle_0=None,
        rgba_handle_1=None,
        solid_handle=False,
        thickness=0.08,  # For body
    ):
        # Set name
        self._name = name

        # Set object attributes
        self.body_half_size = np.array(body_half_size)
        self.thickness = thickness
        self.handle_radius = handle_radius
        self.handle_length = handle_length
        self.handle_width = handle_width
        self.handle_friction = handle_friction
        self.density = density
        self.use_texture = use_texture
        self.rgba_body = np.array(rgba_body) if rgba_body else RED
        self.rgba_handle_0 = np.array(rgba_handle_0) if rgba_handle_0 else GREEN
        self.rgba_handle_1 = np.array(rgba_handle_1) if rgba_handle_1 else BLUE
        self.solid_handle = solid_handle

        # Element references to be filled when generated
        self._handle0_geoms = None
        self._handle1_geoms = None
        self.pot_base = None

        # Other private attributes
        self._important_sites = {}

        # Create dictionary of values to create geoms for composite object and run super init
        super().__init__(**self._get_geom_attrs())

        # Define materials we want to use for this object
        tex_attrib = {
            "type": "cube",
        }
        mat_attrib = {
            "texrepeat": "1 1",
            "specular": "0.4",
            "shininess": "0.1",
        }
        steelbrushed = CustomMaterial(
            texture="SteelBrushed",
            tex_name="steelBrushed",
            mat_name="pot_mat",
            tex_attrib=tex_attrib,
            mat_attrib=mat_attrib,
        )
        greenwood = CustomMaterial(
            texture="WoodGreen",
            tex_name="greenwood",
            mat_name="handle0_mat",
            tex_attrib=tex_attrib,
            mat_attrib=mat_attrib,
        )
        bluewood = CustomMaterial(
            texture="WoodBlue",
            tex_name="bluewood",
            mat_name="handle1_mat",
            tex_attrib=tex_attrib,
            mat_attrib=mat_attrib,
        )
        self.append_material(steelbrushed)
        self.append_material(greenwood)
        self.append_material(bluewood)

    def _get_geom_attrs(self):
        """
        Creates geom elements that will be passed to superclass CompositeObject constructor

        Returns:
            dict: args to be used by CompositeObject to generate geoms
        """
        full_size = np.array(
            (
                self.body_half_size,
                self.body_half_size + self.handle_length * 2,
                self.body_half_size,
            )
        )
        # Initialize dict of obj args that we'll pass to the CompositeObject constructor
        base_args = {
            "total_size": full_size / 2.0,
            "name": self.name,
            "locations_relative_to_center": True,
            "obj_types": "all",
        }
        site_attrs = []
        obj_args = {}

        # Initialize geom lists
        self._handle0_geoms = []
        self._handle1_geoms = []

        # Add main pot body
        # Base geom
        name = f"base"
        self.pot_base = [name]
        add_to_dict(
            dic=obj_args,
            geom_types="box",
            geom_locations=(0, 0, -self.body_half_size[2] + self.thickness / 2),
            geom_quats=(1, 0, 0, 0),
            geom_sizes=np.array([self.body_half_size[0], self.body_half_size[1], self.thickness / 2]),
            geom_names=name,
            geom_rgbas=None if self.use_texture else self.rgba_body,
            geom_materials="pot_mat" if self.use_texture else None,
            geom_frictions=None,
            density=self.density,
        )

        # Walls
        x_off = np.array(
            [0, -(self.body_half_size[0] - self.thickness / 2), 0, self.body_half_size[0] - self.thickness / 2]
        )
        y_off = np.array(
            [-(self.body_half_size[1] - self.thickness / 2), 0, self.body_half_size[1] - self.thickness / 2, 0]
        )
        w_vals = np.array(
            [self.body_half_size[0], self.body_half_size[1], self.body_half_size[0], self.body_half_size[1]]
        )
        r_vals = np.array([np.pi / 2, 0, -np.pi / 2, np.pi])
        for i, (x, y, w, r) in enumerate(zip(x_off, y_off, w_vals, r_vals)):
            add_to_dict(
                dic=obj_args,
                geom_types="box",
                geom_locations=(x, y, 0),
                geom_quats=T.convert_quat(T.axisangle2quat(np.array([0, 0, r])), to="wxyz"),
                geom_sizes=np.array([self.thickness / 2, w, self.body_half_size[2]]),
                geom_names=f"body{i}",
                geom_rgbas=None if self.use_texture else self.rgba_body,
                geom_materials="pot_mat" if self.use_texture else None,
                geom_frictions=None,
                density=self.density,
            )

        # Add handles
        main_bar_size = np.array(
            [
                self.handle_width / 2 + self.handle_radius,
                self.handle_radius,
                self.handle_radius,
            ]
        )
        side_bar_size = np.array([self.handle_radius, self.handle_length / 2, self.handle_radius])
        handle_z = self.body_half_size[2] - self.handle_radius
        for i, (g_list, handle_side, rgba) in enumerate(
            zip([self._handle0_geoms, self._handle1_geoms], [1.0, -1.0], [self.rgba_handle_0, self.rgba_handle_1])
        ):
            handle_center = np.array((0, handle_side * (self.body_half_size[1] + self.handle_length), handle_z))
            # Solid handle case
            if self.solid_handle:
                name = f"handle{i}"
                g_list.append(name)
                add_to_dict(
                    dic=obj_args,
                    geom_types="box",
                    geom_locations=handle_center,
                    geom_quats=(1, 0, 0, 0),
                    geom_sizes=np.array([self.handle_width / 2, self.handle_length / 2, self.handle_radius]),
                    geom_names=name,
                    geom_rgbas=None if self.use_texture else rgba,
                    geom_materials=f"handle{i}_mat" if self.use_texture else None,
                    geom_frictions=(self.handle_friction, 0.005, 0.0001),
                    density=self.density,
                )
            # Hollow handle case
            else:
                # Center bar
                name = f"handle{i}_c"
                g_list.append(name)
                add_to_dict(
                    dic=obj_args,
                    geom_types="box",
                    geom_locations=handle_center,
                    geom_quats=(1, 0, 0, 0),
                    geom_sizes=main_bar_size,
                    geom_names=name,
                    geom_rgbas=None if self.use_texture else rgba,
                    geom_materials=f"handle{i}_mat" if self.use_texture else None,
                    geom_frictions=(self.handle_friction, 0.005, 0.0001),
                    density=self.density,
                )
                # Side bars
                for bar_side, suffix in zip([-1.0, 1.0], ["-", "+"]):
                    name = f"handle{i}_{suffix}"
                    g_list.append(name)
                    add_to_dict(
                        dic=obj_args,
                        geom_types="box",
                        geom_locations=(
                            bar_side * self.handle_width / 2,
                            handle_side * (self.body_half_size[1] + self.handle_length / 2),
                            handle_z,
                        ),
                        geom_quats=(1, 0, 0, 0),
                        geom_sizes=side_bar_size,
                        geom_names=name,
                        geom_rgbas=None if self.use_texture else rgba,
                        geom_materials=f"handle{i}_mat" if self.use_texture else None,
                        geom_frictions=(self.handle_friction, 0.005, 0.0001),
                        density=self.density,
                    )
            # Add relevant site
            handle_site = self.get_site_attrib_template()
            handle_name = f"handle{i}"
            handle_site.update(
                {
                    "name": handle_name,
                    "pos": array_to_string(handle_center - handle_side * np.array([0, 0.005, 0])),
                    "size": "0.005",
                    "rgba": rgba,
                }
            )
            site_attrs.append(handle_site)
            # Add to important sites
            self._important_sites[f"handle{i}"] = self.naming_prefix + handle_name

        # Add pot body site
        pot_site = self.get_site_attrib_template()
        center_name = "center"
        pot_site.update(
            {
                "name": center_name,
                "size": "0.005",
            }
        )
        site_attrs.append(pot_site)
        # Add to important sites
        self._important_sites["center"] = self.naming_prefix + center_name

        # Add back in base args and site args
        obj_args.update(base_args)
        obj_args["sites"] = site_attrs  # All sites are part of main (top) body

        # Return this dict
        return obj_args

    @property
    def handle_distance(self):

        """
        Calculates how far apart the handles are

        Returns:
            float: handle distance
        """
        return self.body_half_size[1] * 2 + self.handle_length * 2

    @property
    def handle0_geoms(self):
        """
        Returns:
            list of str: geom names corresponding to handle0 (green handle)
        """
        return self.correct_naming(self._handle0_geoms)

    @property
    def handle1_geoms(self):
        """
        Returns:
            list of str: geom names corresponding to handle1 (blue handle)
        """
        return self.correct_naming(self._handle1_geoms)

    @property
    def handle_geoms(self):
        """
        Returns:
            list of str: geom names corresponding to both handles
        """
        return self.handle0_geoms + self.handle1_geoms

    @property
    def important_sites(self):
        """
        Returns:
            dict: In addition to any default sites for this object, also provides the following entries

                :`'handle0'`: Name of handle0 location site
                :`'handle1'`: Name of handle1 location site
        """
        # Get dict from super call and add to it
        dic = super().important_sites
        dic.update(self._important_sites)
        return dic

    @property
    def bottom_offset(self):
        return np.array([0, 0, -1 * self.body_half_size[2]])

    @property
    def top_offset(self):
        return np.array([0, 0, self.body_half_size[2]])

    @property
    def horizontal_radius(self):
        return np.sqrt(2) * (max(self.body_half_size) + self.handle_length)