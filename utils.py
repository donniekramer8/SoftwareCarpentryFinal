# image_processing_utils.py

import numpy as np
import cv2
from PIL import Image
from typing import Dict, Tuple, Union, Optional
from scipy.ndimage import gaussian_filter
from skimage.morphology import remove_small_objects
from skimage.measure import label


def scale_up_mask(mask_im: np.ndarray, scale: int, ref: np.ndarray) -> np.ndarray:
    arr = np.asarray(mask_im)
    if not (isinstance(scale, int) and scale > 0):
        raise ValueError(f"scale must be a positive integer, got {scale!r}")
    up = cv2.resize(arr, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
    rh, rw = np.asarray(ref).shape[:2]
    return cv2.resize(up, (rw, rh), interpolation=cv2.INTER_NEAREST)


def get_bounding_boxes(mask: np.ndarray, margin: int = 0) -> Dict[int, Tuple[int, int, int, int]]:
    h, w = mask.shape
    bboxes = {}
    for obj_id in np.unique(mask)[1:]:
        rows, cols = np.where(mask == obj_id)
        if rows.size == 0:
            continue
        x_min, x_max = cols.min(), cols.max()
        y_min, y_max = rows.min(), rows.max()
        x_min = max(0, x_min - margin)
        y_min = max(0, y_min - margin)
        x_max = min(w - 1, x_max + margin)
        y_max = min(h - 1, y_max + margin)
        bboxes[obj_id] = (x_min, x_max, y_min, y_max)
    sorted_items = sorted(bboxes.items(), key=lambda item: item[1][0])
    return {obj_id: box for obj_id, box in sorted_items}


def crop_image_with_boxes(
    image: Union[np.ndarray, Image.Image],
    boxes: Dict[int, Tuple[int, int, int, int]],
    mask: np.ndarray
) -> Dict[int, np.ndarray]:
    img = np.asarray(image)
    if img.ndim < 2:
        raise ValueError(f"Expected at least 2D image, got ndim={img.ndim}")
    h, w = img.shape[:2]
    if mask.shape != (h, w):
        raise ValueError(f"mask shape {mask.shape} != image shape {(h,w)}")

    crops = {}
    for obj_id, (x_min, x_max, y_min, y_max) in boxes.items():
        if not (0 <= x_min <= x_max < w and 0 <= y_min <= y_max < h):
            raise ValueError(f"Box {obj_id} {(x_min,x_max,y_min,y_max)} out of bounds for image {h}x{w}")

        patch = img[y_min:y_max + 1, x_min:x_max + 1].copy()
        patch_mask = (mask[y_min:y_max + 1, x_min:x_max + 1] == obj_id)

        if patch.ndim == 3:
            mode_color = []
            for c in range(patch.shape[2]):
                chan = patch[..., c]
                bg = chan[~patch_mask].ravel().astype(int)
                if bg.size:
                    counts = np.bincount(bg)
                    mode_color.append(int(counts.argmax()))
                else:
                    mode_color.append(0)
            patch[~patch_mask] = tuple(mode_color)
        else:
            bg = patch[~patch_mask].ravel().astype(int)
            mode_val = int(np.bincount(bg).argmax()) if bg.size else 0
            patch[~patch_mask] = mode_val

        crops[obj_id] = patch
    return crops


def get_mask(im: np.ndarray) -> np.ndarray:
    threshold = 200
    mask = (im[:, :, 1] > threshold).astype(float)
    blurred = gaussian_filter(mask, sigma=2)
    mask = blurred < 0.70
    mask = remove_small_objects(mask, min_size=100000)
    labeled_mask = label(mask, connectivity=2)
    return labeled_mask


def downsample_mask(mask: np.ndarray, ds: int) -> np.ndarray:
    if not (isinstance(ds, int) and ds > 0):
        raise ValueError(f"ds must be a positive integer, got {ds!r}")
    return mask[::ds, ::ds]


def estimate_affine_transform(
    mask_ref_ds: np.ndarray,
    mask_mov_ds: np.ndarray,
    criteria: Tuple[int, int, float],
    warp_init: Optional[np.ndarray] = None
) -> np.ndarray:
    if warp_init is None:
        warp_init = np.eye(2, 3, dtype=np.float32)

    cc, warp_matrix = cv2.findTransformECC(
        templateImage=mask_ref_ds.astype(np.float32),
        inputImage=mask_mov_ds.astype(np.float32),
        warpMatrix=warp_init,
        motionType=cv2.MOTION_AFFINE,
        criteria=criteria,
        inputMask=None,
        gaussFiltSize=1
    )
    return warp_matrix


def upsample_warp_matrix(warp_ds: np.ndarray, ds: int) -> np.ndarray:
    M_full = warp_ds.copy()
    M_full[0, 2] *= ds
    M_full[1, 2] *= ds
    return M_full


def compute_border_color(img: np.ndarray) -> Tuple[int, int, int]:
    modes = []
    for c in range(3):
        channel = img[..., c].ravel()
        counts = np.bincount(channel, minlength=256)
        modes.append(int(np.argmax(counts)))
    return tuple(modes)


def warp_image_with_affine(
    img: np.ndarray,
    M: np.ndarray,
    output_shape: Tuple[int, int],
    border_color: Tuple[int, int, int],
    interp: int = cv2.INTER_NEAREST
) -> np.ndarray:
    flags = interp | cv2.WARP_INVERSE_MAP
    return cv2.warpAffine(
        src=img,
        M=M,
        dsize=output_shape,
        flags=flags,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=border_color
    )
