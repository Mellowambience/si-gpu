import tempfile
import unittest
from pathlib import Path
import numpy as np
from siware.core import LearnedUpscaler, TemporalReconstructor, features, resize
from siware.__main__ import frame_reset

class RefinementTests(unittest.TestCase):
    def test_configuration_rejects_nonfinite(self):
        for value in (float("nan"),float("inf"),-1):
            with self.assertRaises(ValueError): TemporalReconstructor(depth_threshold=value)
            with self.assertRaises(ValueError): LearnedUpscaler.fit([],ridge=value)

    def test_model_rejects_invalid_scale_and_weights(self):
        with self.assertRaises(ValueError): LearnedUpscaler(3.0,np.zeros((10,9)))
        with self.assertRaises(ValueError): LearnedUpscaler(3,np.full((10,9),np.nan))
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/"bad.npz"
            np.savez(path,scale=3.5,weights=np.zeros((10,9)))
            with self.assertRaises(ValueError): LearnedUpscaler.load(path)

    def test_depth_conversion_rejected(self):
        color=np.ones((2,2,3),np.float32)*.5
        for value in (1e100,1e-100):
            with self.assertRaises(ValueError):
                TemporalReconstructor().process(color,np.full((2,2),value),np.zeros((2,2,2)))

    def test_metadata_preserves_depth_boundary(self):
        recon=TemporalReconstructor(scale=2)
        recon.process(np.full((2,2,3),.5),np.array([[1,10],[1,10]]),np.zeros((2,2,2)))
        self.assertEqual(set(np.unique(recon.depth)),{1,10})

    def test_streaming_inference_matches_reference(self):
        rng=np.random.default_rng(4)
        low=rng.random((7,9,3),dtype=np.float32)
        for scale in (2,3,4):
            model=LearnedUpscaler(scale,rng.normal(0,.01,(10,scale*scale)))
            expected=resize(low,9*scale,7*scale)
            residual=features(low)@model.weights
            for y in range(scale):
                for x in range(scale): expected[y::scale,x::scale]+=residual[...,y*scale+x]
            np.testing.assert_allclose(model.upscale(low),np.clip(expected,0,1),atol=2e-7)

    def test_reset_contract(self):
        self.assertTrue(frame_reset(np.array(True)))
        for value in (np.array(1),np.array([True]),np.array("false")):
            with self.assertRaises(ValueError): frame_reset(value)

    def test_extreme_motion_stays_finite(self):
        recon=TemporalReconstructor()
        color=np.full((2,2,3),.5,np.float32)
        recon.process(color,np.ones((2,2)),np.zeros((2,2,2)))
        result=recon.process(color,np.ones((2,2)),np.full((2,2,2),1e38))
        self.assertTrue(np.isfinite(result).all())
