# image_processing_utils.py

import numpy as np
import cv2
from PIL import Image
from typing import Dict, Tuple, Union, Optional
from scipy.ndimage import gaussian_filter
from skimage.morphology import remove_small_objects
from skimage.measure import label


def scale_up_mask(mask_im: np.ndarray, scale: int, ref: np.ndarray) -> np.ndarray:
    """
    Scale a labeled mask up by an integer factor, then resize it exactly to match a reference image's dimensions.

    Parameters
    ----------
    mask_im : array-like or PIL.Image
        Input 2D mask (H0 x W0) where pixel values are integer labels.
    scale : int
        Integer factor by which to initially upscale the mask using nearest-neighbor interpolation.
    ref : array-like or PIL.Image
        Reference image whose height and width (H_ref, W_ref) will be used for the final output size.

    Returns
    -------
    numpy.ndarray
        A 2D array of shape (H_ref, W_ref) containing the scaled mask. The upscale is done in two steps:
          1. Nearest-neighbor upscale by exactly 'scale' in each dimension (no label mixing).
          2. Final nearest-neighbor resize to match 'ref''s dimensions, correcting any ±1 pixel mismatches.

    Raises
    ------
    ValueError
        If 'scale' is not a positive integer, or if 'mask_im' or 'ref' cannot be converted to a NumPy array.
    """
    arr = np.asarray(mask_im)

    if not (isinstance(scale, int) and scale > 0):
        raise ValueError(f"scale must be a positive integer, got {scale!r}")
    
    # 1) quick integer upscale
    up = cv2.resize(arr, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
    # 2) final resize to guarantee exact match
    rh, rw = np.asarray(ref).shape[:2]
    return cv2.resize(up, (rw, rh), interpolation=cv2.INTER_NEAREST)


def get_bounding_boxes(mask, margin=0):
    """
    Extract bounding boxes for each labeled object in a 2D mask.

    Parameters
    ----------
    mask : array-like of int, shape (H, W)
        2D array where each pixel value is an integer label (0 = background,
        >0 = object ID).
    margin : int, optional
        Number of pixels to expand each bounding box in all directions.
        Defaults to 0. Expanded boxes are clipped to the image bounds.

    Returns
    -------
    dict[int, tuple[int, int, int, int]]
        Dictionary mapping each object ID to its bounding box, given as a tuple:
        (x_min, x_max, y_min, y_max). Coordinates are inclusive, and boxes are
        sorted in left-to-right order by x_min.

    Notes
    -----
    - Labels with no pixels in the mask are skipped.
    - Bounding boxes are clipped to lie within [0, W-1] for x and [0, H-1] for y.
    """
    h, w = mask.shape
    bboxes = {}
    for obj_id in np.unique(mask)[1:]:
        rows, cols = np.where(mask == obj_id)
        if rows.size == 0:
            continue # skip if not valid
        # get min/max of borders of the mask
        x_min, x_max = cols.min(), cols.max()
        y_min, y_max = rows.min(), rows.max()
        # extend with margin and clip
        x_min = max(0, x_min - margin)
        y_min = max(0, y_min - margin)
        x_max = min(w - 1, x_max + margin)
        y_max = min(h - 1, y_max + margin)
        bboxes[obj_id] = (x_min, x_max, y_min, y_max)

    # sort items by x_min (first element of box tuple)
    sorted_items = sorted(bboxes.items(), key=lambda item: item[1][0])
    # output the sorted dict
    return {obj_id: box for obj_id, box in sorted_items}

def crop_image_with_boxes(
    image: Union[np.ndarray, Image.Image],
    boxes: Dict[int, Tuple[int, int, int, int]],
    mask: np.ndarray
) -> Dict[int, np.ndarray]:
    """
    Crop an image to mask-specific patches and fill background pixels with 
    the mode color (background color).

    Parameters
    ----------
    image : array-like or PIL.Image
        Input image of shape (H, W, C) (RGB) or convertible via 'np.asarray'.
    boxes : dict[int, tuple[int, int, int, int]]
        Mapping from object ID to bounding box coordinates:
        (x_min, x_max, y_min, y_max), inclusive pixel indices.
    mask : array-like of int, shape (H, W)
        Integer mask labeling each pixel with an object ID (0 for background).

    Returns
    -------
    dict[int, ndarray]
        Mapping from each object ID to its cropped patch as a np array.
        In each patch:
          - Pixels where 'mask == obj_id' retain their original color.
          - Pixels where 'mask != obj_id' are filled with the channel-wise mode
            color computed from the patch's background pixels.

    Raises
    ------
    ValueError
        If 'image' has fewer than 2 dimensions, if 'mask' shape does not match
        the image height and width, or if any box coordinates fall outside the
        image bounds.
    """
    img = np.asarray(image)
    if img.ndim < 2:
        raise ValueError(f"Expected at lease 2D image, got ndim={img.ndim}")
    h, w = img.shape[:2]
    if mask.shape != (h, w):
        raise ValueError(f"mask shape {mask.shape} != image shape {(h,w)}")
    
    crops = {}
    for obj_id, (x_min, x_max, y_min, y_max) in boxes.items():
        # sanity check
        if not (0 <= x_min <= x_max < w and 0 <= y_min <= y_max < h):
            raise ValueError(f"Box {obj_id} {(x_min,x_max,y_min,y_max)} out of bounds for image {h}x{w}")
        
        patch = img[y_min:y_max+1, x_min:x_max+1].copy()
        patch_mask = (mask[y_min:y_max+1, x_min:x_max+1] == obj_id)
        
        # For each channel, compute mode of background pixels
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
            mode_color = tuple(mode_color)
            # fill background
            patch[~patch_mask] = mode_color
        
        else:
            # grayscale fallback
            bg = patch[~patch_mask].ravel().astype(int)
            mode_val = int(np.bincount(bg).argmax()) if bg.size else 0
            patch[~patch_mask] = mode_val
        
        crops[obj_id] = patch

    return crops


def get_mask(im: np.ndarray) -> np.ndarray:
    """
    Generate a labeled mask by thresholding the green channel and extracting
    connected components after smoothing and size filtering.

    Parameters
    ----------
    im : array_like, shape (H, W, C)
        Input RGB image as a NumPy array (or array-like). Only the green channel
        (channel index 1) is used for mask generation.

    Returns
    -------
    labeled_mask : ndarray, shape (H, W)
        Integer array where each remaining connected component is assigned a
        unique label (1..N). Background pixels are 0.

    Notes
    -----
    The function performs the following steps:
      1. Threshold the green channel at > 200 to create an initial binary mask.
         (this works well for H&E-stained tissue sections)
      2. Smooth the binary mask with a Gaussian filter (sigma=2).
      3. Re-binarize the smoothed mask at < 0.70.
      4. Remove objects smaller than 100000 pixels.
      5. Label the connected components with connectivity=2.
    """
    threshold = 200  # threshold for green channel
    mask = (im[:, :, 1] > threshold).astype(float)

    blurred = gaussian_filter(mask, sigma=2)
    mask = blurred < 0.70

    cc_threshold = 100000  # minimum object size (in pixels) to keep
    mask = remove_small_objects(mask, min_size=cc_threshold)

    labeled_mask = label(mask, connectivity=2)
    return labeled_mask


def downsample_mask(mask: np.ndarray, ds: int) -> np.ndarray:
    """
    Downsample a 2D mask by an integer factor using strided slicing.

    Parameters
    ----------
    mask : np.ndarray, shape (H, W)
        2D array of integer labels (e.g. an instance mask).
    ds : int
        Downsampling factor (stride). Must be a positive integer.

    Returns
    -------
    np.ndarray, shape (ceil(H/ds), ceil(W/ds))
        The downsampled mask containing every ds-th row and ds-th column
        from the input mask.

    Raises
    ------
    ValueError
        If ds is not a positive integer.
    """
    if not (isinstance(ds, int) and ds > 0):
        raise ValueError(f"ds must be a positive integer, got {ds!r}")
    return mask[::ds, ::ds]


def estimate_affine_transform(
    mask_ref_ds: np.ndarray,
    mask_mov_ds: np.ndarray,
    criteria: Tuple[int, int, float],
    warp_init: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Estimate a 2x3 affine warp mapping mask_mov_ds onto mask_ref_ds using ECC.

    Parameters
    ----------
    mask_ref_ds : np.ndarray, shape (H, W)
        Reference mask (downsampled) as float32 or convertible to float32.
    mask_mov_ds : np.ndarray, shape (H, W)
        Moving mask (downsampled) as float32 or convertible to float32.
    criteria : tuple of (type, max_iter, epsilon)
        Termination criteria for the ECC algorithm:
        - type : cv2.TermCriteria type flag (e.g. cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT)
        - max_iter : int, maximum number of iterations
        - epsilon : float, threshold of increment in correlation coefficient
    warp_init : np.ndarray, shape (2, 3), optional
        Initial affine warp matrix. If None, uses the identity matrix.

    Returns
    -------
    warp_matrix : np.ndarray, shape (2, 3)
        Affine transformation matrix mapping mask_mov_ds -> mask_ref_ds.
    """
    if warp_init is None:
        warp_init = np.eye(2, 3, dtype=np.float32)

    cc, warp_matrix = cv2.findTransformECC(
        templateImage = mask_ref_ds.astype(np.float32),
        inputImage    = mask_mov_ds.astype(np.float32),
        warpMatrix    = warp_init,
        motionType    = cv2.MOTION_AFFINE,
        criteria      = criteria,
        inputMask     = None,
        gaussFiltSize = 1
    )
    return warp_matrix


def compute_border_color(img: np.ndarray) -> Tuple[int, int, int]:
    """
    Compute the mode of each RGB channel and return as a RGB tuple for borders.

    Parameters
    ----------
    img : np.ndarray, shape (H, W, 3)
        Input RGB image with 8-bit per channel.

    Returns
    -------
    border_color : tuple of three ints
        (B, G, R) mode values for each channel, suitable for cv2.warpAffine
        borderValue or cv2.copyMakeBorder.
    """
    modes = []
    for c in range(3):
        channel = img[..., c].ravel()
        counts = np.bincount(channel, minlength=256)
        modes.append(int(np.argmax(counts)))
    return (modes[0], modes[1], modes[2])


def upsample_warp_matrix(warp_ds: np.ndarray, ds: int) -> np.ndarray:
    """
    Scale the translation terms of a downsampled warp matrix back to full resolution.

    Parameters
    ----------
    warp_ds : np.ndarray, shape (2, 3)
        Affine warp matrix estimated on images downsampled by factor ds.
    ds : int
        Downsampling factor used to generate warp_ds.

    Returns
    -------
    M_full : np.ndarray, shape (2, 3)
        Affine warp matrix with translation terms multiplied by ds.
    """
    M_full = warp_ds.copy()
    M_full[0, 2] *= ds
    M_full[1, 2] *= ds
    return M_full


def warp_image_with_affine(
    img: np.ndarray,
    M: np.ndarray,
    output_shape: Tuple[int, int],
    border_color: Tuple[int, int, int],
    interp: int = cv2.INTER_NEAREST
) -> np.ndarray:
    """
    Apply an affine transformation to an image, filling outside pixels with a constant color.

    Parameters
    ----------
    img : np.ndarray
        Source image (H, W) or (H, W, C).
    M : np.ndarray, shape (2, 3)
        Affine transformation matrix.
    output_shape : tuple of (width, height)
        Size of the output image (dsize) passed to cv2.warpAffine.
    border_color : tuple of three ints
        BGR color used to fill pixels outside the source image.
    interp : int, optional
        Interpolation flag (e.g. cv2.INTER_NEAREST, cv2.INTER_LINEAR).

    Returns
    -------
    warped : np.ndarray
        The warped image of shape (height, width) or (height, width, C).
    """
    flags = interp | cv2.WARP_INVERSE_MAP
    return cv2.warpAffine(
        src         = img,
        M           = M,
        dsize       = output_shape,
        flags       = flags,
        borderMode  = cv2.BORDER_CONSTANT,
        borderValue = border_color
    )
