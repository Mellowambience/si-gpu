"""Nonwrapping paired motion truth with explicit boundary and visibility masks."""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
from .__main__ import scene
from .core import TemporalReconstructor, resize, sample, nearest_sample


def frame(seed, index, scale=2, kind="translation", noise=0.0):
    """Deterministic original procedural assets; display RGB, fixed camera depth."""
    if kind not in ("translation","silhouette"):
        raise ValueError("unknown motion case")
    h,w=96,192
    canvas=scene(seed,512)
    offset=160-index*scale if kind=="translation" else 160
    if offset<0 or offset+w>512:
        raise ValueError("frame outside nonwrapping source canvas")
    truth=canvas[160:160+h,offset:offset+w].copy()
    depth=np.full((h,w),10,np.float32)
    motion=np.zeros((h,w,2),np.float32)
    if kind=="translation":
        motion[...,0]=-scale if index else 0
    else:
        x=20+index*scale
        truth[30:66,x:x+36]=[.85,.2,.6]
        truth[30:66:3,x:x+36]=[.2,.8,.9]
        depth[30:66,x:x+36]=3
        motion[30:66,x:x+36,0]=-scale if index else 0
    low=resize(truth,w//scale,h//scale,Image.Resampling.BOX)
    if noise:
        low=np.clip(low+np.random.default_rng(seed*100+index).normal(0,noise,low.shape),0,1).astype(np.float32)
    low_depth=resize(depth,w//scale,h//scale,Image.Resampling.NEAREST)
    low_motion=resize(motion,w//scale,h//scale,Image.Resampling.NEAREST)/scale
    return low,low_depth,low_motion,truth,depth,motion


def evaluate_case(seed,kind,noise,scale=2,frames=16,output=None,model=None):
    recon=TemporalReconstructor(scale,model=model)
    errors={name:[] for name in ("spatial","temporal")}
    changes={name:[] for name in errors}
    boundary_changes={name:[] for name in errors}
    revealed={name:[] for name in errors}
    previous=None
    animations=[]
    for i in range(frames):
        low,z,mv,truth,depth,motion=frame(seed,i,scale,kind,noise)
        spatial=model.upscale(low) if model is not None else resize(low,truth.shape[1],truth.shape[0])
        temporal=recon.process(low,z,mv,reset=(i==0))
        outputs={"spatial":spatial,"temporal":temporal}
        # A fixed interior excludes paths whose history can reach an image boundary.
        interior=np.zeros(depth.shape,bool)
        guard=frames*scale+2*scale
        interior[2*scale:-2*scale,guard:-2*scale]=True
        for name,value in outputs.items():
            errors[name].append(float(np.mean((value[interior]-truth[interior])**2)))
        if previous is not None:
            yy,xx=np.indices(depth.shape,dtype=np.float32)
            px,py=xx+motion[...,0],yy+motion[...,1]
            valid=(px>=0)&(px<=depth.shape[1]-1)&(py>=0)&(py<=depth.shape[0]-1)
            same_surface=np.abs(nearest_sample(previous["depth"],px,py)-depth)<.01
            visible=valid&same_surface
            for name,value in outputs.items():
                old_error=sample(previous[name]-previous["truth"],px,py)
                delta=(value-truth)-old_error
                stable=interior&visible
                if stable.any(): changes[name].append(float(np.mean(delta[stable]**2)))
                boundary=visible&~interior
                if boundary.any(): boundary_changes[name].append(float(np.mean(delta[boundary]**2)))
                disoccluded=interior&valid&~same_surface
                if disoccluded.any(): revealed[name].append(float(np.mean((value[disoccluded]-truth[disoccluded])**2)))
        if output is not None:
            tiles=[Image.fromarray(np.uint8(np.clip(v,0,1)*255+.5)) for v in (truth,spatial,temporal)]
            combined=Image.new("RGB",(truth.shape[1]*3,truth.shape[0]))
            for j,tile in enumerate(tiles): combined.paste(tile,(j*truth.shape[1],0))
            animations.append(combined)
        previous={**outputs,"truth":truth,"depth":depth}
    if animations:
        animations[0].save(output,save_all=True,append_images=animations[1:],duration=100,loop=0)
    avg=lambda values: float(np.mean(values)) if values else None
    return {"seed":seed,"kind":kind,"noise_std":noise,"scale":scale,"frames":frames,
            "interior_mse":{k:avg(v) for k,v in errors.items()},
            "interior_aligned_error_change_mse":{k:avg(v) for k,v in changes.items()},
            "boundary_aligned_error_change_mse":{k:avg(v) for k,v in boundary_changes.items()},
            "disocclusion_mse":{k:avg(v) for k,v in revealed.items()}}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",default="results/motion")
    args=parser.parse_args()
    out=Path(args.output); out.mkdir(parents=True,exist_ok=True)
    cases=[]
    for seed in (200,201,202):
        for kind in ("translation","silhouette"):
            for noise in (0.0,.025):
                path=out/f"{seed}-{kind}-{'noisy' if noise else 'clean'}.gif" if seed==200 else None
                cases.append(evaluate_case(seed,kind,noise,output=path))
    report={"cases":cases,"total_frames":192,
       "contract":"Original procedural display-RGB assets; 2x scale; unjittered; comparable linear depths; no rolling/wrapped image truth",
       "diagnosis":"Separate history-safe interior, boundary region, and newly revealed surfaces. The previous wrapped test used an insufficient crop to exclude boundary-error propagation.",
       "limitations":["Custom aligned-error diagnostic, not a perceptual score","Fixed-camera synthetic translation/silhouette only",
          "No HDR, jitter, exposure or moving-camera validation","Boundary and silhouette regressions remain visible and must not be hidden by interior scores"]}
    (out/"metrics.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    images="".join(f'<h2>{p.stem}</h2><img src="{p.name}" alt="Truth, spatial, temporal motion comparison">' for p in sorted(out.glob("*.gif")))
    (out/"index.html").write_text('<!doctype html><meta charset="utf-8"><title>SI-GPU motion lab</title><style>body{background:#101723;color:#eef;font:16px system-ui;max-width:1000px;margin:30px auto}img{width:100%;image-rendering:pixelated}</style><h1>SI-GPU / Motion lab</h1><p>Panels left to right: native truth, bilinear spatial, temporal reconstruction. Fixed camera; 2x. See metrics.json for separate interior/boundary/disocclusion errors.</p>'+images,encoding="utf-8")
    print(f"Evaluated {report['total_frames']} frames; saved {out/'metrics.json'}")


if __name__=="__main__":
    main()
