import unittest
import numpy as np
from PIL import Image
import cv2

from utils import (
    scale_up_mask,
    get_bounding_boxes,
    crop_image_with_boxes,
    downsample_mask,
    estimate_affine_transform,
    compute_border_color,
    upsample_warp_matrix,
    warp_image_with_affine
)

class TestImageProcessing(unittest.TestCase):
    def setUp(self):
        # Small dummy data for most tests
        self.mask = np.array([
            [0, 0, 1, 1],
            [0, 0, 1, 1],
            [2, 2, 0, 0],
            [2, 2, 0, 0]
        ])
        self.image = np.zeros((4, 4, 3), dtype=np.uint8)
        self.image[0:2, 2:4] = [255, 0, 0]  # Red object
        self.image[2:4, 0:2] = [0, 255, 0]  # Green object

    def test_scale_up_mask(self):
        scaled = scale_up_mask(self.mask, scale=2, ref=np.zeros((8, 8)))
        self.assertEqual(scaled.shape, (8, 8))
        self.assertTrue((np.unique(scaled) == [0, 1, 2]).all())

    def test_get_bounding_boxes(self):
        boxes = get_bounding_boxes(self.mask, margin=0)
        self.assertEqual(len(boxes), 2)
        self.assertEqual(boxes[1], (2, 3, 0, 1))  # box for label 1
        self.assertEqual(boxes[2], (0, 1, 2, 3))  # box for label 2

    def test_crop_image_with_boxes(self):
        boxes = get_bounding_boxes(self.mask)
        crops = crop_image_with_boxes(self.image, boxes, self.mask)
        self.assertEqual(len(crops), 2)
        for crop in crops.values():
            self.assertEqual(crop.ndim, 3)
            self.assertEqual(crop.shape[2], 3)  # RGB

    def test_downsample_mask(self):
        ds = downsample_mask(self.mask, ds=2)
        self.assertEqual(ds.shape, (2, 2))

    def test_upsample_warp_matrix(self):
        warp = np.array([[1.0, 0.0, 2.0], [0.0, 1.0, 3.0]], dtype=np.float32)
        up = upsample_warp_matrix(warp, ds=4)
        self.assertEqual(up[0, 2], 8.0)
        self.assertEqual(up[1, 2], 12.0)

    def test_compute_border_color(self):
        color = compute_border_color(self.image)
        self.assertEqual(len(color), 3)
        self.assertTrue(all(isinstance(c, int) for c in color))

    def test_warp_image_with_affine(self):
        warp = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)
        out = warp_image_with_affine(self.image, warp, (4, 4), border_color=(0, 0, 0))
        self.assertEqual(out.shape, self.image.shape)

    def test_estimate_affine_transform_identity(self):
        # Use a mask with structure so ECC can converge
        mask = np.zeros((32, 32), dtype=np.uint8)
        mask[8:24, 8:24] = 1  # square feature
        crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 1e-5)
        warp = estimate_affine_transform(mask, mask, crit)
        np.testing.assert_allclose(warp, np.eye(2, 3), atol=1e-2)

if __name__ == '__main__':
    unittest.main()
