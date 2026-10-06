import unittest
import numpy as np
from siware.motion_lab import frame,evaluate_case


class MotionTruthTests(unittest.TestCase):
    def test_nonwrapping_translation_and_motion_direction(self):
        previous=frame(200,0)[3]
        low,depth,motion,truth,_,_=frame(200,1)
        np.testing.assert_array_equal(truth[:,2:],previous[:,:-2])
        np.testing.assert_array_equal(motion[...,0],-np.ones(depth.shape))

    def test_revealed_background_has_background_motion(self):
        previous=frame(201,0,kind="silhouette")
        current=frame(201,1,kind="silhouette")
        revealed=(previous[4]==3)&(current[4]==10)
        self.assertTrue(revealed.any())
        np.testing.assert_array_equal(current[5][revealed],0)

    def test_clean_translation_stable_history_safe_interior(self):
        row=evaluate_case(202,"translation",0)
        self.assertLess(row["interior_aligned_error_change_mse"]["temporal"],1e-12)

    def test_noisy_history_reduces_error_fluctuation(self):
        row=evaluate_case(200,"translation",.025)
        metric=row["interior_aligned_error_change_mse"]
        self.assertLess(metric["temporal"],metric["spatial"])
