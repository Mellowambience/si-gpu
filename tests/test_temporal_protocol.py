from pathlib import Path
import struct
import subprocess
import tempfile
import unittest
import numpy as np
from siware.gpu_temporal import export_sequence


class TemporalProtocolTests(unittest.TestCase):
    def setUp(self):
        self.color=np.full((3,5,3),.5,np.float32)
        self.depth=np.ones((3,5),np.float32)
        self.motion=np.zeros((3,5,2),np.float32)

    def test_protocol_preserves_fields_and_shape(self):
        motion=self.motion.copy();motion[...,0]=-.125
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/"sequence.bin"
            export_sequence(path,[(self.color,self.depth,motion,None,True)],2)
            data=path.read_bytes()
        self.assertEqual(struct.unpack("<6I",data[:24]),(0x53494750,1,5,3,2,1))
        self.assertEqual(struct.unpack("<I",data[24:28]),(1,))
        rgba=np.frombuffer(data[28:28+15*16],dtype="<f4").reshape(3,5,4)
        np.testing.assert_array_equal(rgba[...,:3],self.color)
        np.testing.assert_array_equal(rgba[...,3],1)
        meta=np.frombuffer(data[28+15*16:],dtype="<f4").reshape(3,5,4)
        np.testing.assert_array_equal(meta[...,1],-.125)
        np.testing.assert_array_equal(meta[...,3],0)

    def test_protocol_rejects_resize_without_recreation(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError,"recreate"):
                export_sequence(Path(directory)/"sequence.bin",[
                    (self.color,self.depth,self.motion,None,True),
                    (self.color[:2],self.depth[:2],self.motion[:2],None,True)],2)

    def test_protocol_rejects_nonboolean_reset(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                export_sequence(Path(directory)/"sequence.bin",[(self.color,self.depth,self.motion,None,1)],2)

    def test_protocol_rejects_depth_overflow(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                export_sequence(Path(directory)/"sequence.bin",[(self.color,np.full((3,5),1e100),self.motion,None,True)],2)

    def test_protocol_rejects_invalid_reactive(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                export_sequence(Path(directory)/"sequence.bin",[(self.color,self.depth,self.motion,np.full((3,5),2),True)],2)

    def test_native_rejects_malformed_header_before_dispatch(self):
        executable=Path(__file__).resolve().parents[1]/"gpu/temporal.exe"
        if not executable.is_file():self.skipTest("Native D3D12 runner not built")
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/"bad.bin";path.write_bytes(struct.pack("<6I",0,1,5,3,2,1))
            args=[str(executable),str(path),"unused","unused","unused","unused","unused","1","0","1","0.65","0.02"]
            result=subprocess.run(args,capture_output=True,text=True,timeout=10)
        self.assertNotEqual(result.returncode,0)
        self.assertIn("Invalid sequence header",result.stderr)
