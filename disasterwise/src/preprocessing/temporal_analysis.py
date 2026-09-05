import cv2
import numpy as np
from typing import Tuple, Dict

def load_pre_post_images(pre_image_path: str, post_image_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """ Loads pre- and post-disaster images. """
    pre_image = cv2.imread(pre_image_path)
    post_image = cv2.imread(post_image_path)
    return pre_image, post_image

def align_images(pre_image: np.ndarray, post_image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Aligns pre- and post-disaster images. For now, we assume they are co-registered.
    A more advanced implementation would use feature matching (e.g., SIFT, ORB).
    """
    # Placeholder for alignment
    return pre_image, post_image

def compute_change_features(pre_image: np.ndarray, post_image: np.ndarray, detections: Dict) -> Dict:
    """
    Computes change features between pre- and post-disaster images.
    """
    # 1. Debris Increase (simple image differencing)
    diff = cv2.absdiff(pre_image, post_image)
    gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    _, threshold_diff = cv2.threshold(gray_diff, 30, 255, cv2.THRESH_BINARY)
    debris_increase = np.sum(threshold_diff > 0) / (diff.shape[0] * diff.shape[1])

    # 2. Vegetation Loss (difference in green channel)
    pre_green = pre_image[:,:,1]
    post_green = post_image[:,:,1]
    vegetation_diff = np.mean(pre_green) - np.mean(post_green)
    vegetation_loss = max(0, vegetation_diff) # Only consider loss

    # 3. Building Damage Ratio & Infrastructure Change (from detections)
    pre_buildings = [d for d in detections.get('pre', []) if 'building' in d['label']]
    post_buildings = [d for d in detections.get('post', []) if 'damaged' in d['label']]
    building_damage_ratio = len(post_buildings) / len(pre_buildings) if len(pre_buildings) > 0 else 0

    pre_infra = [d for d in detections.get('pre', []) if 'infrastructure' in d['label']]
    post_infra = [d for d in detections.get('post', []) if 'infrastructure' in d['label']]
    infra_change = len(pre_infra) - len(post_infra)

    return {
        "building_damage_ratio": building_damage_ratio,
        "infrastructure_change": infra_change,
        "debris_increase": debris_increase,
        "vegetation_loss": vegetation_loss
    }
