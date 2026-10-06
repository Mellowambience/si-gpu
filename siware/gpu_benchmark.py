"""D3D12 spatial baseline verification. Run from the SI-GPU project folder."""
import argparse
import hashlib
import json
import platform
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
from PIL import Image
from .core import LearnedUpscaler, resize
from .__main__ import pair


def run_case(root, out, low, scale, model, mode, samples, warmups, label, scratch=None):
    scratch = Path(scratch) if scratch is not None else out / "scratch"
    scratch.mkdir(parents=True, exist_ok=True)
    height, width = low.shape[:2]
    rgba = np.concatenate([low, np.ones((height,width,1),np.float32)],axis=2)
    input_file, weight_file = scratch/"input.bin", scratch/"weights.bin"
    output_file = scratch/"output.bin"
    rgba.astype("<f4").tofile(input_file)
    model.weights.astype("<f4").tofile(weight_file)
    metrics_file = out/f"{label}-{mode}.json"
    command = [str(root/"gpu/baseline.exe"),str(width),str(height),str(scale),
               "1" if mode=="learned" else "0",str(input_file),str(weight_file),
               str(output_file),str(metrics_file),str(root/"gpu/upscale.hlsl"),str(samples),str(warmups)]
    subprocess.run(command, check=True, timeout=120)
    result = np.fromfile(output_file,dtype="<f4").reshape(height*scale,width*scale,4)[...,:3]
    started = time.perf_counter()
    reference = model.upscale(low) if mode=="learned" else resize(low,width*scale,height*scale)
    cpu_ms = (time.perf_counter()-started)*1000
    diff = np.abs(reference-result)
    if not np.isfinite(result).all() or float(diff.max())>3e-6:
        raise RuntimeError(f"GPU parity failed for {label}/{mode}: max abs {diff.max()}")
    metrics = json.loads(metrics_file.read_text())
    metrics.update(max_abs_error=float(diff.max()),mean_abs_error=float(diff.mean()),
                   cpu_reference_single_run_ms=cpu_ms,parity_tolerance=3e-6,parity_pass=True,
                   input_sha256=hashlib.sha256(input_file.read_bytes()).hexdigest())
    metrics_file.write_text(json.dumps(metrics,indent=2),encoding="utf-8")
    if label=="1440p":
        Image.fromarray(np.uint8(result*255+.5)).save(out/f"{label}-{mode}.png")
    return metrics


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",default="results/gpu")
    parser.add_argument("--samples",type=int,default=1000)
    parser.add_argument("--warmups",type=int,default=100)
    parser.add_argument("--smoke",action="store_true")
    parser.add_argument("--project",default=".",help="Source checkout containing gpu/baseline.exe")
    parser.add_argument("--work",default="work/gpu-benchmark",help="Directory for intermediate binary buffers")
    args=parser.parse_args()
    if not 1<=args.samples<=10000 or not 0<=args.warmups<=1000:
        parser.error("samples must be 1..10000 and warmups 0..1000")
    root=Path(args.project).resolve()
    if not (root/"gpu/baseline.exe").is_file():
        parser.error("Build gpu/baseline.exe first and run in the source folder or pass --project")
    out=Path(args.output).resolve(); out.mkdir(parents=True,exist_ok=True)
    scratch=Path(args.work).resolve()
    base=LearnedUpscaler.fit([pair(seed,2) for seed in range(12)],2)
    base.save(out/"model-2x.npz")
    rng=np.random.default_rng(301)
    cases=[("tiny",rng.random((13,17,3),dtype=np.float32),2,base),
           ("constant",np.full((7,11,3),.4,np.float32),2,base)]
    for scale in (3,4):
        model=LearnedUpscaler.fit([pair(seed,scale) for seed in range(3)],scale)
        cases.append((f"scale-{scale}",rng.random((9,7,3),dtype=np.float32),scale,model))
    if not args.smoke:
        cases += [("1080p",rng.random((540,960,3),dtype=np.float32),2,base),
                  ("1440p",rng.random((720,1280,3),dtype=np.float32),2,base)]
    rows=[]
    for label,low,scale,model in cases:
        for mode in ("bilinear","learned"):
            count=args.samples if label in ("1080p","1440p") else min(args.samples,20)
            rows.append(run_case(root,out,low,scale,model,mode,count,args.warmups,label,scratch))
    summary={"python":sys.version,"platform":platform.platform(),"numpy":np.__version__,
             "shader_sha256":hashlib.sha256((root/"gpu/upscale.hlsl").read_bytes()).hexdigest(),
             "results":rows,"limitations":["Spatial microbenchmark, not temporal or engine integration",
               "Repeated fixed input; other desktop GPU work may affect timing",
               "CPU reference timing is one run, not a matched speedup measurement",
               "No whole-frame FPS or input-latency claim"]}
    try:
        summary["nvidia_snapshot"]=subprocess.check_output(["nvidia-smi","--query-gpu=name,driver_version,memory.total,memory.used,memory.free,utilization.gpu","--format=csv"],text=True).strip()
    except (OSError,subprocess.CalledProcessError):
        summary["nvidia_snapshot"]=None
    (out/"summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    print(f"All {len(rows)} GPU parity cases passed. Saved {out/'summary.json'}")


if __name__=="__main__":
    main()
