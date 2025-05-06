# Automatic Cropping and Alignment of H&E tissue sections

## Notebook Workflow

### 0) images_to_downscaled_masks.ipynb

- **Purpose**: 
- **Inputs**:
  - Folder of raw `.tif` or `.png` images.
  - Threshold and size-filter parameters defined in the notebook.
- **Outputs**:
  - `downscaled_masks/` folder containing downsampled mask `.tif` files.
- **Key Steps**:
  1. 

### 1) get_bounding_box_crop.ipynb

- **Purpose**: From downsampled masks, compute bounding boxes for each object and crop corresponding patches from the original full-resolution images.
- **Inputs**:
  - Original images folder.
  - Corresponding downscaled masks folder.
  - Margin parameter for bounding box extension.
- **Outputs**:
  - `crops/` folder with one image patch per object (as `.png` or `.npy`).
- **Key Steps**:
  1. Read each mask and compute `x_min, x_max, y_min, y_max` for every object, with optional margin.
  2. Sort bounding boxes left-to-right.
  3. Crop the original image per bounding box and fill background pixels with the channel-wise mode color.

### 2) align_all_images_in_folder.ipynb

- **Purpose**: Align a series of image crops (e.g. serial sections) to a reference using an affine ECC-based registration.
- **Inputs**:
  - Folder of cropped image patches (from step 1).
  - Downsample factor `ds` used previously.
- **Outputs**:
  - `aligned_steps/` folder with `.png` or `.npy` aligned at full resolution.
- **Key Steps**:
  1. Load each pair of downsampled masks and estimate affine 2×3 warp via ECC.
  2. Upscale the warp translation back to full resolution.
  3. Compute border color (mode of RGB channels) and apply `cv2.warpAffine` to the full-res image.

---

## Key Functions

- `get_mask(im: np.ndarray) -> np.ndarray`  
  Generate a labeled mask from an RGB image.
- `downsample_mask(mask: np.ndarray, ds: int) -> np.ndarray`  
  Strided downsampling of a mask.
- `get_bounding_boxes(mask: np.ndarray, margin: int=0) -> Dict[int, Tuple[int,int,int,int]]`  
  Extract and sort bounding boxes.
- `crop_image_with_boxes(image: np.ndarray, boxes: Dict[int,Tuple[int,int,int,int]], mask: np.ndarray) -> Dict[int,np.ndarray]`  
  Crop patches and fill backgrounds.
- `estimate_affine_transform(mask_ref_ds: np.ndarray, mask_mov_ds: np.ndarray, criteria: Tuple[int,int,float], warp_init: Optional[np.ndarray]=None) -> Tuple[np.ndarray,float]`  
  Compute ECC-based affine warp.
- `upsample_warp_matrix(warp_ds: np.ndarray, ds: int) -> np.ndarray`  
  Rescale translation terms of the warp matrix.
- `compute_border_color(img: np.ndarray) -> Tuple[int,int,int]`  
  Mode-based border fill color.
- `warp_image_with_affine(img: np.ndarray, M: np.ndarray, output_shape: Tuple[int,int], border_color: Tuple[int,int,int], interp: int) -> np.ndarray`  
  Apply affine transform with constant border.

---