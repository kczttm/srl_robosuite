import mujoco
from scipy.spatial.transform import Rotation

R_std_mjcam = Rotation.from_euler("ZYX", [-90, 0, 65], degrees=True).as_matrix()
# looking down 30 degrees by default

def list_available_cameras(model: mujoco.MjModel) -> list[str]:
    """
    Lists the names of all cameras defined in the MuJoCo model.

    Args:
        model (mujoco.MjModel): The MuJoCo model.

    Returns:
        list[str]: A list containing the names of all available cameras.
                   Returns an empty list if no cameras are found or named.
    """
    camera_names = []
    # Iterate through all cameras in the model by their ID
    for i in range(model.ncam):
        # mj_id2name returns the name of the object given its type and ID
        camera_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_CAMERA, i)
        # Add the name to the list if it exists
        if camera_name:
            camera_names.append(camera_name)
    return camera_names


def set_viewer_camera(viewer: mujoco.viewer, model: mujoco.MjModel, camera_name: str):
    """
    Attaches the viewer camera to a specific named camera in the model.

    Args:
        viewer (mujoco.viewer): The passive viewer instance.
        model (mujoco.MjModel): The MuJoCo model.
        camera_name (str): The name of the camera to attach to, as defined in the XML.
    """
    try:
        camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
        viewer.cam.fixedcamid = camera_id
        print(f"Successfully switched viewer to camera: '{camera_name}'")
    except ValueError as e:
        print(
            f"Error: Could not find a camera named '{camera_name}'. Please check your XML model."
        )
        print(f"Details: {e}")


