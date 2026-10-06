from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from siware.core import LearnedUpscaler
from siware.learning import (atomic_json,read_json,digest,model_path,writer_lock,config_default,
    quality_gate,regression_gate,temporal_gate,promote,rollback,gpu_evidence)


class LearningTests(unittest.TestCase):
    def setUp(self):
        self.folder=tempfile.TemporaryDirectory();self.addCleanup(self.folder.cleanup)
        self.root=Path(self.folder.name);(self.root/"models").mkdir()
        self.config=config_default();atomic_json(self.root/"config.json",self.config)
        models={}
        for name,value in (("baseline-0000",0),("cycle-000001-candidate-01",.001)):
            path=model_path(self.root,name);LearnedUpscaler(2,np.full((10,4),value,np.float32)).save(path)
            models[name]={"sha256":digest(path)}
        self.state={"accepted_lab":"baseline-0000","accepted_runtime":None,
            "previous_lab":[],"previous_runtime":[],"models":models,"events":[]}
        atomic_json(self.root/"registry.json",self.state)
        self.candidate="cycle-000001-candidate-01"
        self.report={"cycle":1,"status":"nomination_ready","nominee":self.candidate,
            "reference":"baseline-0000","config_sha256":digest(self.root/"config.json"),
            "candidate_results":[{"id":self.candidate,"sha256":models[self.candidate]["sha256"],"lab_eligible":True,"runtime_eligible":False}]}
        self.report_path=self.root/"experiments/cycle-000001/report.json";atomic_json(self.report_path,self.report)

    def test_promote_rollback_preserves_model_files(self):
        hashes={name:digest(model_path(self.root,name)) for name in self.state["models"]}
        self.assertEqual(promote(self.root,self.candidate)["accepted_lab"],self.candidate)
        self.assertEqual(rollback(self.root)["accepted_lab"],"baseline-0000")
        self.assertEqual(hashes,{name:digest(model_path(self.root,name)) for name in hashes})

    def test_runtime_refusal_leaves_accepted_pointer_unchanged(self):
        before=(self.root/"registry.json").read_bytes()
        with self.assertRaises(RuntimeError):promote(self.root,self.candidate,"runtime")
        self.assertEqual((self.root/"registry.json").read_bytes(),before)

    def test_changed_gate_config_blocks_old_nomination(self):
        c=deepcopy(self.config);c["min_mean_gain_db"]*=2;atomic_json(self.root/"config.json",c)
        with self.assertRaisesRegex(RuntimeError,"configuration changed"):promote(self.root,self.candidate)

    def test_corrupt_candidate_blocks_promotion(self):
        model_path(self.root,self.candidate).write_bytes(b"corrupt")
        with self.assertRaisesRegex(RuntimeError,"checksum"):promote(self.root,self.candidate)

    def test_failed_experiment_cannot_be_promoted(self):
        self.report["status"]="failed";atomic_json(self.report_path,self.report)
        with self.assertRaises(ValueError):promote(self.root,self.candidate)

    def test_writer_lock_excludes_concurrent_mutation(self):
        with writer_lock(self.root):
            with self.assertRaises(RuntimeError):
                with writer_lock(self.root):pass
        self.assertFalse((self.root/"writer.lock").exists())

    def test_mean_gain_cannot_hide_old_scene_regression(self):
        old=[{"seed":1,"psnr_db":20},{"seed":2,"psnr_db":20}]
        new=[{"seed":1,"psnr_db":21},{"seed":2,"psnr_db":19.8}]
        self.assertFalse(quality_gate(old,new,self.config)["pass"])
        self.assertFalse(regression_gate(old,new,self.config)["pass"])

    def test_temporal_gate_rejects_flicker_increase(self):
        row={"seed":300,"kind":"translation","noise_std":0,
            "interior_mse":{"temporal":.1},"interior_aligned_error_change_mse":{"temporal":.001},
            "boundary_aligned_error_change_mse":{"temporal":.002},"disocclusion_mse":{"temporal":None}}
        changed=deepcopy(row);changed["interior_aligned_error_change_mse"]["temporal"]*=2
        self.assertFalse(temporal_gate([row],[changed],self.config)["pass"])

    def test_gpu_busy_snapshot_with_units_defers_before_dispatch(self):
        csv="name, driver_version, memory.used [MiB], memory.free [MiB], utilization.gpu [%]\nGPU, 1, 9000 MiB, 2000 MiB, 87 %"
        with patch("siware.gpu_temporal.gpu_snapshot",return_value=csv),patch("siware.gpu_temporal.run_sequence") as runner:
            evidence=gpu_evidence(self.root,self.root,self.root,None,300,self.config)
        self.assertEqual(evidence["status"],"deferred_busy_or_low_memory")
        runner.assert_not_called()

    def test_model_identifier_cannot_escape_registry(self):
        with self.assertRaises(ValueError):model_path(self.root,"../outside")

    def test_runtime_promotion_after_lab_promotion_uses_same_nomination(self):
        self.report["candidate_results"][0]["runtime_eligible"]=True;atomic_json(self.report_path,self.report)
        promote(self.root,self.candidate,"lab")
        self.assertEqual(promote(self.root,self.candidate,"runtime")["accepted_runtime"],self.candidate)
