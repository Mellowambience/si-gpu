import argparse
import json
import platform
import sys
import PIL
from pathlib import Path
from time import perf_counter
import numpy as np
from PIL import Image, ImageDraw
from .core import LearnedUpscaler, TemporalReconstructor, resize, psnr


def load_image(path):
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.float32)/255


def save_image(path, array):
    Image.fromarray(np.uint8(np.clip(array, 0, 1)*255+0.5)).save(path)


def scene(seed, size=192):
    """Procedural geometry only: reproducible, no external training assets."""
    rng = np.random.default_rng(seed)
    image = Image.new("RGB", (size, size), (18, 29, 47))
    draw = ImageDraw.Draw(image)
    for _ in range(65):
        x, y = rng.integers(0, size, 2)
        dx, dy = rng.integers(3, size//3, 2)
        color = tuple(int(v) for v in rng.integers(30, 245, 3))
        if rng.random() < .5:
            draw.rectangle((int(x), int(y), int(x+dx), int(y+dy)), fill=color)
        else:
            draw.ellipse((int(x), int(y), int(x+dx), int(y+dy)), fill=color)
    for x in range(0, size, 7):
        draw.line((x, 0, min(size-1, x+35), size-1), fill=(170, 190, 205), width=1)
    return np.asarray(image, dtype=np.float32)/255


def pair(seed, scale):
    high = scene(seed, size=64*scale)
    low = np.asarray(Image.fromarray(np.uint8(high*255)).resize((64,64), Image.Resampling.BOX), dtype=np.float32)/255
    return low, high


def demo(args):
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    model = LearnedUpscaler.fit([pair(seed, args.scale) for seed in range(12)], args.scale)
    model.save(out/"model.npz")
    rows = []
    for seed in range(100, 106):
        low, high = pair(seed, args.scale)
        baseline = resize(low, high.shape[1], high.shape[0])
        # Warmup then report CPU inference only; includes resize, excludes IO.
        model.upscale(low)
        times = []
        for _ in range(30):
            started = perf_counter()
            learned = model.upscale(low)
            times.append((perf_counter()-started)*1000)
        comparisons = {name: psnr(high, np.clip(resize(low, high.shape[1], high.shape[0], method),0,1)) for name, method in [("bicubic", Image.Resampling.BICUBIC), ("lanczos", Image.Resampling.LANCZOS)]}
        rows.append({"seed": seed, "baselines_psnr_db": comparisons, "cpu_p95_ms": float(np.percentile(times,95)), "bilinear_psnr_db": psnr(high, baseline),
                     "learned_psnr_db": psnr(high, learned), "cpu_p50_ms": float(np.median(times))})
        if seed == 100:
            save_image(out/"bicubic.png", np.clip(resize(low,high.shape[1],high.shape[0],Image.Resampling.BICUBIC),0,1))
            save_image(out/"lanczos.png", np.clip(resize(low,high.shape[1],high.shape[0],Image.Resampling.LANCZOS),0,1))
            for name, image in [("reference",high),("low",low),("baseline",baseline),("learned",learned)]:
                save_image(out/f"{name}.png", image)
    report = {"implementation":"CPU NumPy learned linear residual filter; not neural inference",
              "scale":args.scale,"input_size":[64,64],"output_size":[64*args.scale]*2,
              "training_seeds":list(range(12)),"evaluation_seeds":list(range(100,106)),
              "timing_scope":"CPU inference incl resize; 30 samples after warmup; NOT GPU or end-to-end latency",
              "held_out_mean_gain_db":float(np.mean([r["learned_psnr_db"]-r["bilinear_psnr_db"] for r in rows])),
              "limitations":["Synthetic static evaluation only","No gameplay generalization demonstrated",
                             "No GPU execution","Temporal metrics are separate and limited to synthetic translation"],"frames":rows}
    (out/"metrics.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    (out/"environment.json").write_text(json.dumps({"python":sys.version,"platform":platform.platform(),"numpy":np.__version__,"pillow":PIL.__version__,"device":"CPU; GPU not benchmarked"},indent=2),encoding="utf-8")
    gain = report["held_out_mean_gain_db"]
    (out/"index.html").write_text(f'''<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>SI-WARE Lab</title>
<style>body{{background:#071725;color:#edf6ff;font:16px system-ui;max-width:1100px;margin:40px auto;padding:20px}}h1{{font-size:48px}}strong{{color:#70dcf4}}.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:20px}}figure{{margin:0;padding:15px;border:1px solid #315570}}img{{width:100%;image-rendering:auto}}pre{{white-space:pre-wrap}}a{{color:#70dcf4}}</style>
<h1>SI-WARE / Reconstruction Lab</h1><p>Offline CPU research prototype · v0.1.1</p>
<p><strong>Held-out mean PSNR change: {gain:+.3f} dB versus bilinear.</strong> A negative result is a failed quality gate, not a performance claim.</p>
<div class="grid"><figure><img src="reference.png" alt="Native reference"><figcaption>Native reference</figcaption></figure>
<figure><img src="baseline.png" alt="Bilinear reconstruction"><figcaption>Bilinear baseline</figcaption></figure>
<figure><img src="learned.png" alt="Learned linear reconstruction"><figcaption>Learned linear residual</figcaption></figure></div>
<p>Additional baselines: <a href="bicubic.png">Bicubic</a> · <a href="lanczos.png">Lanczos</a>. PSNR values are included below.</p>
<p>This model was trained on 12 procedural scenes and evaluated on 6 different seeds. It is not a neural network, a GPU benchmark, or a live game integration.</p>
<p><a href="metrics.json">Full measurements</a> · <a href="model.npz">Model weights</a></p>
<h2>Evaluation details</h2><pre>{json.dumps(report,indent=2)}</pre></html>''',encoding="utf-8")
    print(json.dumps(report,indent=2))


def upscale(args):
    low = load_image(args.input)
    model = LearnedUpscaler.load(args.model) if args.model else None
    if model and model.scale != args.scale:
        raise ValueError("--scale must match model scale")
    result = model.upscale(low) if model else resize(low, low.shape[1]*args.scale, low.shape[0]*args.scale)
    Path(args.output).parent.mkdir(parents=True,exist_ok=True)
    save_image(args.output,result)
    print(f"Saved {args.output}; {'learned linear' if model else 'bilinear'} CPU reconstruction")


def frame_reset(value):
    if value.shape != () or value.dtype.kind != "b":
        raise ValueError("reset must be a boolean scalar")
    return bool(value)


def sequence(args):
    paths = sorted(Path(args.input).glob("*.npz"))
    if not paths:
        raise ValueError("no .npz frames found")
    model = LearnedUpscaler.load(args.model) if args.model else None
    recon = TemporalReconstructor(args.scale, model=model)
    output = Path(args.output)
    output.mkdir(parents=True,exist_ok=True)
    manifest = []
    for i, path in enumerate(paths):
        with np.load(path,allow_pickle=False) as data:
            image = recon.process(data["color"],data["depth"],data["motion"],
                                  data["reactive"] if "reactive" in data else None,
                                  frame_reset(data["reset"]) if "reset" in data else False)
        save_image(output/f"{i:05d}.png",image)
        manifest.append({"input":path.name,"output":f"{i:05d}.png"})
    (output/"manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
    print(f"Reconstructed {len(paths)} frames (unjittered CPU reference)")


def main():
    parser = argparse.ArgumentParser(description="SI-WARE offline reconstruction lab")
    sub = parser.add_subparsers(dest="command",required=True)
    for command, handler in [("demo",demo),("upscale",upscale),("sequence",sequence)]:
        p = sub.add_parser(command)
        p.add_argument("--scale",type=int,choices=(2,3,4),default=3)
        p.add_argument("--output",required=True)
        if command != "demo":
            p.add_argument("--input",required=True)
            p.add_argument("--model")
        p.set_defaults(handler=handler)
    args = parser.parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()

