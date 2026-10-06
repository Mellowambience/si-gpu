"""Write a tiny translating-rectangle sequence with known depth and motion."""
from pathlib import Path
import numpy as np

out = Path(__file__).parent/"frames"
out.mkdir(parents=True,exist_ok=True)
for i in range(8):
    h,w = 32,48
    yy,xx = np.indices((h,w))
    color = np.stack([.1+xx/w*.3,.1+yy/h*.3,np.full((h,w),.2)],axis=-1).astype(np.float32)
    depth = np.full((h,w),10,np.float32)
    motion = np.zeros((h,w,2),np.float32)
    x = 6+i*2
    color[8:24,x:x+12] = [.75,.3,.8]
    depth[8:24,x:x+12] = 3
    motion[8:24,x:x+12,0] = -2 if i else 0
    np.savez_compressed(out/f"{i:05d}.npz",color=color,depth=depth,motion=motion,reset=(i==0))
print(f"Wrote 8 procedural moving frames to {out}")
