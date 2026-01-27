import os
import sys
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")   # headless safe
from PIL import Image
from typing import Any, List, Dict, Optional, Union, Tuple
import os
import yaml
import cv2
import numpy as np
from PIL import Image
from transformers import SamModel, AutoProcessor, pipeline
import torch
import torch.multiprocessing
from dataclasses import dataclass
from autoencoder_training import Autoencoder_V2
from autoencoder_training import AutoencoderKL
from pathlib import Path
import math
import torch.nn.functional as F
from pdb import set_trace
from cotracker.predictor import CoTrackerPredictor
from sam3.model_builder import build_sam3_image_model  # importing sam3 leaves no space for sapien rendering
from sam3.model.sam3_image_processor import Sam3Processor
try:
    import imageio.v2 as imageio
    HAS_IMAGEIO = True
except:
    HAS_IMAGEIO = False
from tqdm import tqdm

@dataclass
class BoundingBox:
    xmin: int
    ymin: int
    xmax: int
    ymax: int

    @property
    def xyxy(self) -> List[float]:
        return [self.xmin, self.ymin, self.xmax, self.ymax]

@dataclass
class DetectionResult:
    score: float
    label: str
    box: BoundingBox
    mask: Optional[np.array] = None

    @classmethod
    def from_dict(cls, detection_dict: Dict) -> 'DetectionResult':
        return cls(score=detection_dict['score'],
                   label=detection_dict['label'],
                   box=BoundingBox(xmin=detection_dict['box']['xmin'],
                                   ymin=detection_dict['box']['ymin'],
                                   xmax=detection_dict['box']['xmax'],
                                   ymax=detection_dict['box']['ymax']))
     
def mask_to_polygon(mask: np.ndarray) -> List[List[int]]:
    # Find contours in the binary mask
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find the contour with the largest area
    largest_contour = max(contours, key=cv2.contourArea)

    # Extract the vertices of the contour
    polygon = largest_contour.reshape(-1, 2).tolist()

    return polygon

def polygon_to_mask(polygon: List[Tuple[int, int]], image_shape: Tuple[int, int]) -> np.ndarray:
    """
    Convert a polygon to a segmentation mask.

    Args:
    - polygon (list): List of (x, y) coordinates representing the vertices of the polygon.
    - image_shape (tuple): Shape of the image (height, width) for the mask.

    Returns:
    - np.ndarray: Segmentation mask with the polygon filled.
    """
    # Create an empty mask
    mask = np.zeros(image_shape, dtype=np.uint8)

    # Convert polygon to an array of points
    pts = np.array(polygon, dtype=np.int32)

    # Fill the polygon with white color (255)
    cv2.fillPoly(mask, [pts], color=(255,))

    return mask

def get_boxes(results: DetectionResult) -> List[List[List[float]]]:
    boxes = []
    for result in results:
        xyxy = result.box.xyxy
        boxes.append(xyxy)

    return [boxes]

def refine_masks(masks: torch.BoolTensor, polygon_refinement: bool = False) -> List[np.ndarray]:
    masks = masks.cpu().float()
    masks = masks.permute(0, 2, 3, 1)
    masks = masks.mean(axis=-1)
    masks = (masks > 0).int()
    masks = masks.numpy().astype(np.uint8)
    masks = list(masks)

    if polygon_refinement:
        for idx, mask in enumerate(masks):
            shape = mask.shape
            polygon = mask_to_polygon(mask)
            mask = polygon_to_mask(polygon, shape)
            masks[idx] = mask
            
    return masks

def detect(
    image: Image.Image,
    labels: List[str],
    threshold: float = 0.3,
    detector_id: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Use Grounding DINO to detect a set of labels in an image in a zero-shot fashion.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    detector_id = detector_id if detector_id is not None else "IDEA-Research/grounding-dino-tiny"
    object_detector = pipeline(model=detector_id, task="zero-shot-object-detection", device=device)
    
    labels = [label if label.endswith(".") else label+"." for label in labels]
    results = object_detector(image,  candidate_labels=labels, threshold=threshold)
    results = [DetectionResult.from_dict(result) for result in results]

    return results

def segment(
    image: Image.Image,
    detection_results: List[Dict[str, Any]],
    polygon_refinement: bool = False,
    segmenter_id: Optional[str] = None
) -> List[DetectionResult]:
    """
    Use Segment Anything (SAM) to generate masks given an image + a set of bounding boxes.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    segmenter_id = segmenter_id if segmenter_id is not None else "facebook/sam-vit-base"

    segmentator = SamModel.from_pretrained(segmenter_id).to(device)
    processor = AutoProcessor.from_pretrained(segmenter_id)

    boxes = get_boxes(detection_results)
    inputs = processor(images=image, input_boxes=boxes, return_tensors="pt").to(device)

    outputs = segmentator(**inputs)
    masks = processor.post_process_masks(
        masks=outputs.pred_masks,
        original_sizes=inputs.original_sizes,
        reshaped_input_sizes=inputs.reshaped_input_sizes
    )[0]

    masks = refine_masks(masks, polygon_refinement)

    for detection_result, mask in zip(detection_results, masks):
        detection_result.mask = mask

    return detection_results, masks

def grounded_segmentation(
    image: Union[Image.Image, str],
    labels: List[str],
    threshold: float = 0.3,
    polygon_refinement: bool = False,
    detector_id: Optional[str] = None,
    segmenter_id: Optional[str] = None
) -> Tuple[np.ndarray, List[DetectionResult]]:

    detections = detect(image, labels, threshold, detector_id)
    detections, mask = segment(image, detections, polygon_refinement, segmenter_id)

    return np.array(image), detections, mask

def grounded_segmentation_SAM3(
    image: Union[Image.Image, str],
    labels: List[str],
    polygon_refinement: bool = False,
) -> Tuple[np.ndarray, List[DetectionResult]]:

    model = build_sam3_image_model()
    processor = Sam3Processor(model)

    inference_state = processor.set_image(image)
    output = processor.set_text_prompt(state=inference_state, prompt=labels[0])
    masks = output["masks"]

    ####################################################################################
    # mask = masks[0]
    # mask = (mask > 0).cpu().numpy().astype(np.uint8)
    # img_np = np.array(image).astype(np.uint8)
    # overlay = img_np.copy()
    # red = np.zeros_like(img_np)
    # red[:, :, 1] = mask * 255
    # overlay = (img_np * (1 - 0.5) + red * 0.5).astype(np.uint8)
    # timestamp = time()
    # Image.fromarray(overlay).save(f"/home/zxiao93/Documents/flow_koopman/tmp/masked_overlay_{timestamp}.png")
    ####################################################################################


    masks = refine_masks(masks, polygon_refinement)

    return np.array(image), masks

def resize_tensor(tensor: torch.Tensor, M: int) -> torch.Tensor:
    """
    Uniformly resizes the 3rd dimension (N) to size M.
    - If N > M → downsample (uniform subsampling)
    - If N < M → upsample (linear interpolation)

    Supports:
      - (1, T, N, 2)
      - (1, T, N)
      - bool tensors (auto-converts to float for interpolation)
    """
    device = tensor.device
    original_dtype = tensor.dtype
    is_bool = tensor.dtype == torch.bool

    if is_bool:
        tensor = tensor.float()

    dim = tensor.dim()
    if dim not in [3, 4]:
        raise ValueError(f"Unsupported tensor shape {tensor.shape}. Must be (1, T, N) or (1, T, N, 2)")

    if dim == 3:
        B, T, N = tensor.shape
        C = 1
    else:
        B, T, N, C = tensor.shape

    # ✅ Downsample
    if N > M:
        indices = torch.linspace(0, N - 1, steps=M).long().to(device)
        out = tensor.index_select(2, indices)

    # ✅ Upsample
    elif N < M:
        if dim == 3:
            tensor_3d = tensor.reshape(-1, 1, N)
            upsampled = F.interpolate(tensor_3d, size=M, mode="linear", align_corners=True)
            out = upsampled.reshape(B, T, M)
        else:
            tensor_3d = tensor.permute(0, 1, 3, 2).reshape(-1, C, N)
            upsampled = F.interpolate(tensor_3d, size=M, mode="linear", align_corners=True)
            out = upsampled.reshape(B, T, C, M).permute(0, 1, 3, 2)

    else:
        out = tensor  # no resize

    # ✅ Convert back to bool if original was bool
    if is_bool:
        out = out > 0.5
    else:
        out = out.to(original_dtype)

    return out

def get_obj_flow_SAM(image, num_flow, object_label, object_grid_size, random_mask=False):
    '''
    Output the flow video generated by co-tracker and also the flow trajectories.
    '''
    torch.cuda.empty_cache()

    detector_id = "IDEA-Research/grounding-dino-tiny"
    segmenter_id = "facebook/sam-vit-base"
    labels = [object_label]
    
    threshold = 0.3

    image_array, detections, masks = grounded_segmentation(
        image=image,
        labels=labels,
        threshold=threshold,
        polygon_refinement=True,
        detector_id=detector_id,
        segmenter_id=segmenter_id
    )  # obtain the initial mask

    # generate flow points
    # Get all coordinates where mask == 1
    if len(masks) == 0:
        return None, None 
    
    mask = masks[0] > 0  # the correct mask

    # create a new empty mask
    if random_mask:
        num_foreground = np.count_nonzero(mask)   # number of True pixels

        random_mask = np.zeros_like(mask, dtype=bool)

        # randomly choose the same number of pixels
        h, w = mask.shape
        rand_indices = np.random.choice(h * w, num_foreground, replace=False)

        # set those pixels to True
        random_mask[np.unravel_index(rand_indices, (h, w))] = True  # the random mask
        mask = random_mask

    video_np = np.stack([image_array], axis=0)  # (1, H, W, 3)
    video = torch.from_numpy(video_np).permute(0,3,1,2)[None].float()  # (1, 3, H, W)
    model = CoTrackerPredictor(
        checkpoint='/home/yhan389/Desktop/srl_robosuite/projects/visual_koopman/co-tracker/checkpoints/scaled_offline.pth')
    if torch.cuda.is_available():
        model = model.cuda()
        video = video.cuda()
    pred_tracks, pred_visibility = model(video, grid_size = object_grid_size, segm_mask = torch.from_numpy(masks[0])[None, None])

    # Randomly sample num_flow points
    resized_pred_tracks = resize_tensor(pred_tracks, num_flow)

    return resized_pred_tracks[0][0].detach().cpu().numpy(), mask

def get_obj_flow_SAM3(image, num_flow, object_label, object_grid_size):
    '''
    Output the flow video generated by co-tracker and also the flow trajectories.
    '''
    torch.cuda.empty_cache()

    labels = [object_label]

    image_array, masks = grounded_segmentation_SAM3(
        image=image,
        labels=labels,
        polygon_refinement=True,
    )  # obtain the initial mask

    # generate flow points
    # Get all coordinates where mask == 1
    if len(masks) == 0:
        return None, None 
    
    mask = masks[0] > 0  # the correct mask

    video_np = np.stack([image_array], axis=0)  # (1, H, W, 3)

    video = torch.from_numpy(video_np).permute(0,3,1,2)[None].float()  # (1, 3, H, W)
    model = CoTrackerPredictor(
        checkpoint='/home/yhan389/Desktop/Visual_Koopman/CoTracker_Experiments/co-tracker/checkpoints/scaled_offline.pth')
    if torch.cuda.is_available():
        model = model.cuda()
        video = video.cuda()
    pred_tracks, pred_visibility = model(video, grid_size = object_grid_size, segm_mask = torch.from_numpy(masks[0])[None, None])
    resized_pred_tracks = resize_tensor(pred_tracks, num_flow)
    
    # return sampled_coords, mask
    return resized_pred_tracks[0][0].detach().cpu().numpy(), mask

## Above is for the flow estimation
## Below is for the simulation tasks

# laod the pre-trained flow autoencoders
def load_auencoders(config, autoencoder_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    vae_v2 = config.get('model', {}).get('vae_v2', False)
    
    if vae_v2:
        flow_autoencoder = Autoencoder_V2(config=config)  # 
    else:
        flow_autoencoder = AutoencoderKL()

    flow_autoencoder.load_state_dict(torch.load(autoencoder_path, weights_only=True))  # using the same flower encoder

    flow_autoencoder.to(device)
    flow_autoencoder.eval()

    return flow_autoencoder
    
# obtain the flow features given the flow points
def get_features(config, flow_autoencoder, flow_point, initial_flow_latent = None, feature_type = "only_cur_features"):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    vae_v2 = config.get('model', {}).get('vae_v2', False)
    num_flow_points = config.get('model', {}).get('num_flows', 256)
    reshape_size = int(np.sqrt(num_flow_points))

    if not vae_v2:
        flow_point[..., 0] /= config['image_shape']['image_width']  # Normalize x | image height = 640
        flow_point[..., 1] /= config['image_shape']['image_height']  # Normalize y | image width = 360

    with torch.no_grad():
        cur_pos = flow_point
        cur_pos = cur_pos.reshape(reshape_size,reshape_size,2)
        cur_pos = cur_pos.transpose(2, 0, 1)
        cur_pos_tensor = torch.from_numpy(cur_pos).float().to(device)  
        cur_pos_tensor = cur_pos_tensor.unsqueeze(0)    
        
        if vae_v2:
            object_flow_latent = flow_autoencoder.encoder(cur_pos_tensor)
        else:
            object_flow_latent, _, _ = flow_autoencoder.encoder(cur_pos_tensor)

        object_flow_latent = object_flow_latent.view(object_flow_latent.shape[0], -1)

        if initial_flow_latent is None:
            initial_flow_latent = object_flow_latent.clone()

    if feature_type == "only_cur_features":
        flow_feature = object_flow_latent.cpu().numpy().squeeze()
        return flow_feature, initial_flow_latent
    
    elif feature_type == "init_cur_features":
        flow_feature = torch.cat((initial_flow_latent, object_flow_latent), dim=-1).cpu().numpy().squeeze()
        return flow_feature, initial_flow_latent

def load_config(path: str) -> dict:
    """Load a YAML config file and return as a dictionary."""
    with open(path, "r") as f:
        config = yaml.safe_load(f)
    return config

def image_vis(img):
    # Assuming arr is your numpy array
    # shape: (480, 640, 3)
    plt.imshow(img.astype(np.uint8))  # cast to uint8 if pixel values are [0-255]
    plt.axis("off")
    plt.show()

def process_camera_image(image, autoencoder_config, flow_autoencoder, num_flow, object_label, object_grid_size, save_path):
    raw = np.array(image.copy())
    flow_points, mask = get_obj_flow_SAM3(image = image, num_flow = num_flow, object_label = object_label, object_grid_size = object_grid_size)

    # mask: H×W or H×W×1 array 
    save_path = os.path.join(save_path, "mask.png")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    alpha = 0.5 # transparency 
    mask_rgb = np.zeros_like(raw)
    mask_rgb[mask] = [255, 0, 0] # red mask 
    overlay = (raw * (1 - alpha) + mask_rgb * alpha).astype(np.uint8)

    plt.imshow(overlay)
    plt.axis("off")
    plt.savefig(save_path, bbox_inches="tight", dpi=300)
    
    # Initial flow featue can just be initial flow latent, if feature_type is "only_cur_features"
    unscaled_initial_flow_feature, _ = get_features(autoencoder_config, flow_autoencoder, flow_points)
    
    return unscaled_initial_flow_feature


