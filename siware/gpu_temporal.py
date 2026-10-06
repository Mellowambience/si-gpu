"""D3D12 temporal sequence parity and separated-pass profiling."""
import argparse
import hashlib
import json
import os
import platform
from pathlib import Path
import struct
import subprocess
import sys

import numpy as np
from PIL import Image
from .core import LearnedUpscaler, TemporalReconstructor, finite32, validate_color
from .__main__ import pair
from .motion_lab import frame


def gpu_snapshot():
    try:
        return subprocess.check_output(["nvidia-smi","--query-gpu=name,driver_version,memory.used,memory.free,utilization.gpu","--format=csv"],text=True).strip()
    except (OSError,subprocess.CalledProcessError):return None


def export_sequence(path, frames, scale):
    if type(scale) is not int or scale not in (2,3,4) or not 1<=len(frames)<=3000:
        raise ValueError("invalid scale or frame count")
    first=validate_color(frames[0][0]); h,w=first.shape[:2]
    with Path(path).open("wb") as output:
        output.write(struct.pack("<6I",0x53494750,1,w,h,scale,len(frames)))
        for color,depth,motion,reactive,reset in frames:
            color=validate_color(color)
            if color.shape!=first.shape:
                raise ValueError("native sequence dimensions must stay fixed; recreate on resize")
            depth=finite32(depth,"depth"); motion=finite32(motion,"motion")
            reactive=np.zeros((h,w),np.float32) if reactive is None else finite32(reactive,"reactive")
            if depth.shape!=(h,w) or motion.shape!=(h,w,2) or reactive.shape!=(h,w):
                raise ValueError("metadata dimensions mismatch")
            if (depth<=0).any() or (reactive<0).any() or (reactive>1).any():
                raise ValueError("invalid depth/reactive values")
            if not isinstance(reset,(bool,np.bool_)):
                raise ValueError("reset must be a boolean scalar")
            rgba=np.concatenate([color,np.ones((h,w,1),np.float32)],axis=2)
            metadata=np.concatenate([depth[...,None],motion,reactive[...,None]],axis=2)
            output.write(struct.pack("<I",int(reset)))
            output.write(rgba.astype("<f4").tobytes())
            output.write(metadata.astype("<f4").tobytes())
    return h,w


def run_sequence(project, out, work, name, frames, scale=2, model=None,
                 repeats=1, warmups=0, capture_all=True, debug=False,
                 weight=.65, threshold=.02, visual_truth=None, timeout_seconds=180):
    if model is not None and model.scale!=scale:
        raise ValueError("model scale mismatch")
    # Reuse public setting validation before dispatch.
    recon=TemporalReconstructor(scale,history_weight=weight,depth_threshold=threshold,model=model)
    scratch=Path(work)/name; scratch.mkdir(parents=True,exist_ok=True)
    out=Path(out); out.mkdir(parents=True,exist_ok=True)
    h,w=export_sequence(scratch/"sequence.bin",frames,scale)
    weights=model.weights if model is not None else np.zeros((10,scale*scale),np.float32)
    weights.astype("<f4").tofile(scratch/"weights.bin")
    metrics=out/f"{name}.json"
    command=[str(Path(project)/"gpu/temporal.exe"),str(scratch/"sequence.bin"),str(scratch/"weights.bin"),
             str(Path(project)/"gpu/upscale.hlsl"),str(Path(project)/"gpu/temporal.hlsl"),
             str(scratch/"output.bin"),str(metrics),str(repeats),str(warmups),
             "1" if capture_all else "0",str(weight),str(threshold)]
    environment=os.environ.copy()
    if debug: environment["SI_GPU_DEBUG"]="1"
    before=gpu_snapshot()
    subprocess.run(command,check=True,timeout=timeout_seconds,env=environment)
    after=gpu_snapshot()
    count=len(frames) if capture_all else 1
    result=np.memmap(scratch/"output.bin",dtype="<f4",mode="r",shape=(count,h*scale,w*scale,4))
    per_frame=[]; animations=[]
    for i,(color,depth,motion,reactive,reset) in enumerate(frames):
        reference=recon.process(color,depth,motion,reactive,reset)
        if not capture_all and i+1<len(frames): continue
        gpu=np.asarray(result[i if capture_all else 0,...,:3])
        difference=np.abs(reference-gpu)
        error=float(difference.max())
        if not np.isfinite(gpu).all() or error>5e-6:
            raise RuntimeError(f"{name} frame {i} parity failed: {error}")
        per_frame.append({"frame":i,"reset":bool(reset),"max_abs_error":error,"mean_abs_error":float(difference.mean())})
        if visual_truth is not None:
            tiles=[Image.fromarray(np.uint8(np.clip(value,0,1)*255+.5)) for value in (visual_truth[i],reference,gpu)]
            image=Image.new("RGB",(w*scale*3,h*scale))
            for j,tile in enumerate(tiles):image.paste(tile,(j*w*scale,0))
            animations.append(image)
    if animations:
        animations[0].save(out/f"{name}.gif",save_all=True,append_images=animations[1:],duration=100,loop=0)
    del result
    report=json.loads(metrics.read_text())
    report.update(parity_pass=True,parity_tolerance=5e-6,frame_parity=per_frame,
                  sequence_sha256=hashlib.sha256((scratch/"sequence.bin").read_bytes()).hexdigest(),
                  python=sys.version,numpy=np.__version__,platform=platform.platform(),
                  nvidia_before=before,nvidia_after=after,
                  shader_sha256={name:hashlib.sha256((Path(project)/"gpu"/name).read_bytes()).hexdigest()
                                 for name in ("upscale.hlsl","temporal.hlsl")})
    # Every frame advances history once. Separate reset/copy pass from steady history.
    for pass_name,timing in report["timings"].items():
        samples=np.asarray(timing["samples_ms"]).reshape(len(frames),repeats)
        steady=np.array([i>0 and not item[4] for i,item in enumerate(frames)])
        if steady.any():
            values=samples[steady].reshape(-1)
            timing["steady_history_p50_ms"]=float(np.median(values))
            timing["steady_history_p95_ms"]=float(np.percentile(values,95))
        reset_values=samples[~steady].reshape(-1)
        if len(reset_values):timing["reset_p50_ms"]=float(np.median(reset_values))
    metrics.write_text(json.dumps(report,indent=2),encoding="utf-8")
    return report


def stress_frames(scale):
    rng=np.random.default_rng(410+scale); h,w=9,13; frames=[]
    for i in range(8):
        color=rng.random((h,w,3),dtype=np.float32)
        depth=np.full((h,w),1 if i!=3 else 2,np.float32)
        motion=np.full((h,w,2),-.125 if i%2 else .375,np.float32)
        if i==6:motion.fill(1e8)
        if i==7:motion.fill(3e38)
        reactive=rng.random((h,w),dtype=np.float32)
        if i==5:reactive.fill(1)
        frames.append((color,depth,motion,reactive,i in (0,4)))
    return frames


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project",default=".")
    parser.add_argument("--output",default="results/gpu-temporal")
    parser.add_argument("--work",default="work/gpu-temporal")
    parser.add_argument("--debug",action="store_true")
    parser.add_argument("--benchmark",action="store_true")
    parser.add_argument("--profile-only",action="store_true",help="Run only the two monitor-size profiles")
    parser.add_argument("--repeats",type=int,default=1000)
    parser.add_argument("--warmups",type=int,default=100)
    args=parser.parse_args()
    root=Path(args.project).resolve(); out=Path(args.output).resolve(); work=Path(args.work).resolve()
    if not (root/"gpu/temporal.exe").is_file():parser.error("Build gpu/temporal.exe first")
    if not 1<=args.repeats<=10000 or not 0<=args.warmups<=1000:parser.error("invalid repeats/warmups")
    reports=[]
    if not args.profile_only:
        for scale in (2,3,4):
            model=LearnedUpscaler.fit([pair(i,scale) for i in range(3)],scale)
            reports.append(run_sequence(root,out,work,f"stress-{scale}",stress_frames(scale),scale,model,debug=args.debug))
        for seed in (200,201,202):
            for kind in ("translation","silhouette"):
                for noisy in (False,True):
                    frames=[]; truth=[]
                    for i in range(16):
                        color,z,motion,high,_,_=frame(seed,i,noise=.025 if noisy else 0,kind=kind)
                        frames.append((color,z,motion,None,i==0)); truth.append(high)
                    reports.append(run_sequence(root,out,work,f"{seed}-{kind}-{'noisy' if noisy else 'clean'}",frames,
                        debug=args.debug,visual_truth=truth if seed==200 else None))
    if args.benchmark or args.profile_only:
        rng=np.random.default_rng(501)
        for label,h,w in (("1080p",540,960),("1440p",720,1280)):
            frames=[]
            for i in range(2):
                frames.append((rng.random((h,w,3),dtype=np.float32),np.ones((h,w),np.float32),
                    np.full((h,w,2),-.125,np.float32),None,i==0))
            reports.append(run_sequence(root,out,work,f"profile-{label}",frames,repeats=args.repeats,
                warmups=args.warmups,capture_all=False,debug=args.debug))
    summary={"sequences":len(reports),"parity_frames":sum(len(r["frame_parity"]) for r in reports),
             "all_parity_pass":all(r["parity_pass"] for r in reports),
             "debug_requested":args.debug,"results":[{k:v for k,v in r.items() if k!="timings"} for r in reports],
             "limitations":["Unjittered display RGB; fixed dimensions per sequence","No camera transform, exposure or HDR",
                "Offline upload/readback and CPU fence per frame; not an engine hot path",
                "Profile repeats fixed current/prior inputs; steady history and reset measurements separated"]}
    (out/"summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    images="".join(f'<h2>{p.stem}</h2><img src="{p.name}" alt="Native truth, CPU temporal, GPU temporal">' for p in sorted(out.glob("*.gif")))
    (out/"index.html").write_text('<!doctype html><meta charset="utf-8"><title>SI-GPU temporal parity</title><style>body{background:#101723;color:#eef;font:16px system-ui;max-width:1000px;margin:30px auto}img{width:100%;image-rendering:pixelated}</style><h1>SI-GPU / GPU temporal parity</h1><p>Left to right: native truth, CPU temporal reference, D3D12 temporal output. Synthetic 2x cases.</p>'+images,encoding="utf-8")
    print(f"All {summary['parity_frames']} compared frames across {len(reports)} sequences passed.")


if __name__=="__main__":main()
