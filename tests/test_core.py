import tempfile
import unittest
from pathlib import Path
import numpy as np
from siware.core import LearnedUpscaler, TemporalReconstructor, resize, sample


class ReconstructionTests(unittest.TestCase):
    def setUp(self):
        self.color = np.random.default_rng(0).random((8,8,3)).astype(np.float32)
        self.depth = np.ones((8,8),np.float32)
        self.motion = np.zeros((8,8,2),np.float32)

    def test_first_frame_and_resolution(self):
        result = TemporalReconstructor(3).process(self.color,self.depth,self.motion)
        self.assertEqual(result.shape,(24,24,3))
        np.testing.assert_allclose(result,resize(self.color,24,24))

    def test_camera_cut_clears_history(self):
        recon = TemporalReconstructor()
        recon.process(1-self.color,self.depth,self.motion)
        result = recon.process(self.color,self.depth,self.motion,reset=True)
        np.testing.assert_allclose(result,resize(self.color,24,24))

    def test_disocclusion_rejects_history(self):
        recon = TemporalReconstructor()
        recon.process(1-self.color,self.depth,self.motion)
        result = recon.process(self.color,self.depth*2,self.motion)
        np.testing.assert_allclose(result,resize(self.color,24,24))

    def test_reactive_mask_bypasses_history(self):
        recon = TemporalReconstructor()
        recon.process(1-self.color,self.depth,self.motion)
        result = recon.process(self.color,self.depth,self.motion,np.ones((8,8)))
        np.testing.assert_allclose(result,resize(self.color,24,24))

    def test_out_of_bounds_history_rejected(self):
        recon = TemporalReconstructor()
        recon.process(1-self.color,self.depth,self.motion)
        result = recon.process(self.color,self.depth,self.motion+100)
        np.testing.assert_allclose(result,resize(self.color,24,24))

    def test_current_to_previous_motion_sign(self):
        recon = TemporalReconstructor(scale=2,history_weight=1)
        h,w = 16,16
        yy,xx = np.indices((h,w),dtype=np.float32)
        recon.history = np.repeat((xx/16)[...,None],3,axis=2)
        recon.depth = np.ones((h,w),np.float32)
        current = np.repeat((np.indices((8,8))[1]/8)[...,None],3,axis=2).astype(np.float32)
        motion = np.zeros((8,8,2),np.float32)
        motion[...,0] = -0.5
        out = recon.process(current,self.depth,motion)
        # x=8 reprojects to x=7; neighborhood clipping doesn't affect this sample.
        np.testing.assert_allclose(out[8,8],np.repeat(7/16,3),atol=1e-6)

    def test_reject_invalid_frame(self):
        recon = TemporalReconstructor()
        with self.assertRaises(ValueError):
            recon.process(self.color,self.depth*0,self.motion)
        with self.assertRaises(ValueError):
            recon.process(self.color,self.depth,self.motion[:4])

    def test_resize_changes_reset_history(self):
        recon = TemporalReconstructor()
        recon.process(self.color,self.depth,self.motion)
        color = self.color[:4,:4]
        np.testing.assert_allclose(recon.process(color,self.depth[:4,:4],self.motion[:4,:4]),resize(color,12,12))

    def test_model_training_and_roundtrip(self):
        model = LearnedUpscaler.fit([(self.color,resize(self.color,24,24))],scale=3)
        np.testing.assert_allclose(model.upscale(self.color),resize(self.color,24,24),atol=1e-5)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/"weights.npz"
            model.save(path)
            restored = LearnedUpscaler.load(path)
            np.testing.assert_allclose(restored.upscale(self.color),model.upscale(self.color))

    def test_fractional_bilinear_sample(self):
        grid = np.array([[0,2],[2,4]],np.float32)
        self.assertEqual(float(sample(grid,np.array(.5),np.array(.5))),2)


if __name__ == "__main__":
    unittest.main()
