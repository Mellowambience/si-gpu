"""Controlled synthetic temporal evaluation; not a gameplay benchmark."""
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image
from .core import TemporalReconstructor, resize, psnr
from .__main__ import pair, scene


def evaluate(scale=3):
    rows=[]
    for seed in (200,201,202):
        for noisy in (False,True):
            recon=TemporalReconstructor(scale)
            errors={"spatial":[],"temporal":[]}
            previous=None
            residuals={"spatial":[],"temporal":[]}
            source=scene(seed,96)
            for i in range(16):
                truth=np.roll(source,i*scale,axis=1)
                low=resize(truth,96//scale,96//scale,Image.Resampling.BOX)
                if noisy:
                    noise=np.random.default_rng(seed*100+i).normal(0,.025,low.shape)
                    low=np.clip(low+noise,0,1).astype(np.float32)
                depth=np.full(low.shape[:2],10,np.float32)
                motion=np.zeros((*low.shape[:2],2),np.float32)
                motion[...,0]=-1 if i else 0
                spatial=resize(low,96,96)
                temporal=recon.process(low,depth,motion,reset=(i==0))
                # Ignore wrap seam and reprojection-invalid strip.
                crop=(slice(6,-6),slice(6,-6))
                for name,result in (("spatial",spatial),("temporal",temporal)):
                    errors[name].append(float(np.mean((result[crop]-truth[crop])**2)))
                    if previous is not None:
                        # Known translation: compare reconstruction error after alignment.
                        delta=(result-truth)-np.roll(previous[name]-previous["truth"],scale,axis=1)
                        residuals[name].append(float(np.mean(delta[crop]**2)))
                previous={"spatial":spatial,"temporal":temporal,"truth":truth}
            rows.append({"seed":seed,"noise_std":.025 if noisy else 0,
                "mean_mse":{k:float(np.mean(v)) for k,v in errors.items()},
                "aligned_error_change_mse":{k:float(np.mean(v)) for k,v in residuals.items()}})
    return {"scale":scale,"frames_per_case":16,"cases":rows,
        "limitations":["Synthetic constant-depth translation only","No camera depth transform, jitter, HDR, silhouettes or gameplay quality claim","Border crop excludes wrap seam","Noise case measures denoising as well as reconstruction"]}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",required=True)
    parser.add_argument("--scale",type=int,choices=(2,3,4),default=3)
    args=parser.parse_args()
    path=Path(args.output)
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(evaluate(args.scale),indent=2),encoding="utf-8")
    print(f"Saved {path}")

if __name__=="__main__": main()
