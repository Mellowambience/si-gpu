"""CPU reference implementation. Display RGB, not HDR or a game-ready SDK."""
from dataclasses import dataclass
import numpy as np
from PIL import Image


def validate_color(color):
    color = np.asarray(color, dtype=np.float32)
    if color.ndim != 3 or color.shape[2] != 3 or min(color.shape[:2]) < 2:
        raise ValueError("color must be H x W x 3, minimum 2 x 2")
    if not np.isfinite(color).all() or color.min() < 0 or color.max() > 1:
        raise ValueError("color must contain finite display RGB values in [0,1]")
    return color


def resize(array, width, height, method=Image.Resampling.BILINEAR):
    """Float channel-wise bilinear resize, preserving depth/motion units."""
    array = np.asarray(array, dtype=np.float32)
    if array.ndim == 2:
        return np.asarray(Image.fromarray(array).resize((width, height), method))
    return np.stack([resize(array[..., i], width, height, method) for i in range(array.shape[2])], axis=-1)


def finite32(array, name):
    """Reject overflow and underflow-sensitive depth after conversion."""
    with np.errstate(over="ignore", invalid="ignore"):
        value = np.asarray(array, dtype=np.float32)
    if not np.isfinite(value).all():
        raise ValueError(f"{name} must be finite and representable as float32")
    return value


def nearest_sample(array, x, y):
    h, w = array.shape[:2]
    return array[np.clip(np.floor(y+0.5), 0, h-1).astype(np.intp),
                 np.clip(np.floor(x+0.5), 0, w-1).astype(np.intp)]


def features(color):
    """Shared 3x3 spatial features per channel; output H x W x C x 10."""
    h, w, _ = color.shape
    pad = np.pad(color, ((1, 1), (1, 1), (0, 0)), mode="edge")
    return np.stack([pad[y:y+h, x:x+w] for y in range(3) for x in range(3)]
                    + [np.ones_like(color)], axis=-1)


@dataclass
class LearnedUpscaler:
    """Learned polyphase residual filter; a small linear model, not a neural net."""
    scale: int
    weights: np.ndarray

    def __post_init__(self):
        if type(self.scale) is not int or self.scale not in (2, 3, 4):
            raise ValueError("scale must be integer 2, 3 or 4")
        self.weights = finite32(self.weights, "weights").copy()
        if self.weights.shape != (10, self.scale*self.scale):
            raise ValueError("invalid model weight shape")

    @classmethod
    def fit(cls, pairs, scale=3, ridge=0.01):
        if type(scale) is not int or scale not in (2, 3, 4) or not np.isfinite(ridge) or ridge <= 0:
            raise ValueError("scale must be 2, 3 or 4 and ridge must be positive")
        gram = np.zeros((10, 10), dtype=np.float64)
        rhs = np.zeros((10, scale*scale), dtype=np.float64)
        count = 0
        for low, high in pairs:
            low, high = validate_color(low), validate_color(high)
            h, w, _ = low.shape
            if high.shape != (h*scale, w*scale, 3):
                raise ValueError("training target dimensions must match scale")
            baseline = resize(low, w*scale, h*scale)
            x = features(low).reshape(-1, 10).astype(np.float64)
            residual = high-baseline
            y = np.stack([residual[dy::scale, dx::scale].reshape(-1)
                          for dy in range(scale) for dx in range(scale)], axis=1)
            gram += x.T @ x
            rhs += x.T @ y
            count += len(x)
        if not count:
            raise ValueError("at least one training pair required")
        weights = np.linalg.solve(gram/count + ridge*np.eye(10), rhs/count)
        return cls(scale, weights.astype(np.float32))

    def upscale(self, low):
        low = validate_color(low)
        h, w, _ = low.shape
        out = resize(low, w*self.scale, h*self.scale)
        # Accumulate one phase at a time: no H x W x C x scale^2 residual.
        pad = np.pad(low, ((1,1),(1,1),(0,0)), mode="edge")
        for dy in range(self.scale):
            for dx in range(self.scale):
                weights = self.weights[:, dy*self.scale+dx]
                residual = np.full_like(low, weights[9])
                for k in range(9):
                    y, x = divmod(k, 3)
                    residual += pad[y:y+h, x:x+w]*weights[k]
                out[dy::self.scale, dx::self.scale] += residual
        return np.clip(out, 0, 1)

    def save(self, path):
        np.savez_compressed(path, scale=self.scale, weights=self.weights)

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as data:
            raw_scale = data["scale"]
            if raw_scale.shape != () or raw_scale.dtype.kind not in "iu":
                raise ValueError("model scale must be an integer scalar")
            scale = int(raw_scale)
            weights = data["weights"].copy()
        if scale not in (2, 3, 4) or weights.shape != (10, scale*scale) or not np.isfinite(weights).all():
            raise ValueError("invalid model shape or values")
        return cls(scale, weights)


def sample(array, x, y):
    """Bilinear sample; caller must reject out-of-bounds samples."""
    h, w = array.shape[:2]
    x, y = np.clip(x, 0, w-1), np.clip(y, 0, h-1)
    x0, y0 = np.floor(x).astype(int), np.floor(y).astype(int)
    x1, y1 = np.minimum(x0+1, w-1), np.minimum(y0+1, h-1)
    tx, ty = (x-x0).astype(np.float32), (y-y0).astype(np.float32)
    if array.ndim == 3:
        tx, ty = tx[..., None], ty[..., None]
    return ((1-ty)*((1-tx)*array[y0, x0]+tx*array[y0, x1])
            + ty*((1-tx)*array[y1, x0]+tx*array[y1, x1]))


class TemporalReconstructor:
    """Motion-aware history blend with depth rejection, reactive masks and clipping.

    Motion is current->previous displacement in LOW-resolution pixels (x,y).
    Depth must be positive linear view depth. Only unjittered frames supported.
    """
    def __init__(self, scale=3, history_weight=0.65, depth_threshold=0.02, model=None):
        if (type(scale) is not int or scale not in (2, 3, 4)
                or not np.isfinite(history_weight) or not 0 <= history_weight <= 1
                or not np.isfinite(depth_threshold) or not 0 <= depth_threshold <= 1):
            raise ValueError("invalid reconstruction settings")
        if model is not None and model.scale != scale:
            raise ValueError("model scale mismatch")
        self.scale, self.weight, self.threshold = scale, history_weight, depth_threshold
        self.model = model
        self.history = self.depth = None

    def reset(self):
        self.history = self.depth = None

    def process(self, color, depth, motion, reactive=None, reset=False):
        color = validate_color(color)
        h, w, _ = color.shape
        depth, motion = finite32(depth, "depth"), finite32(motion, "motion")
        if depth.shape != (h, w) or motion.shape != (h, w, 2):
            raise ValueError("depth/motion dimensions do not match color")
        if not np.isfinite(depth).all() or (depth <= 0).any() or not np.isfinite(motion).all():
            raise ValueError("depth must be positive finite; motion must be finite")
        reactive = np.zeros((h, w), np.float32) if reactive is None else np.asarray(reactive)
        if reactive.shape != (h, w) or not np.isfinite(reactive).all() or (reactive < 0).any() or (reactive > 1).any():
            raise ValueError("reactive mask must be H x W in [0,1]")
        if not isinstance(reset, (bool, np.bool_)):
            raise ValueError("reset must be a boolean scalar")
        if reset:
            self.reset()
        oh, ow = h*self.scale, w*self.scale
        current = self.model.upscale(color) if self.model else resize(color, ow, oh)
        # Preserve surface identity at depth/motion discontinuities.
        z = resize(depth, ow, oh, Image.Resampling.NEAREST)
        if self.history is not None and self.history.shape == current.shape:
            mv = resize(motion, ow, oh, Image.Resampling.NEAREST)
            # Larger displacements cannot yield valid history; bound before scaling.
            mv = np.clip(mv, -max(h,w)*2, max(h,w)*2)*self.scale
            yy, xx = np.indices((oh, ow), dtype=np.float32)
            px, py = xx+mv[..., 0], yy+mv[..., 1]
            valid = (px >= 0) & (px <= ow-1) & (py >= 0) & (py <= oh-1)
            old_z = nearest_sample(self.depth, px, py)
            valid &= np.abs(old_z-z) <= self.threshold*np.maximum(z, 1e-6)
            old = sample(self.history, px, py)
            # Clip history to a current-frame local neighborhood to reduce trails.
            pad = np.pad(current, ((1,1),(1,1),(0,0)), mode="edge")
            lower, upper = current.copy(), current.copy()
            for y in range(3):
                for x in range(3):
                    neighbor = pad[y:y+oh, x:x+ow]
                    np.minimum(lower, neighbor, out=lower)
                    np.maximum(upper, neighbor, out=upper)
            old = np.clip(old, lower, upper)
            alpha = self.weight*valid*(1-resize(reactive, ow, oh))
            current = current*(1-alpha[...,None])+old*alpha[...,None]
        self.history, self.depth = current.astype(np.float32).copy(), z.copy()
        return self.history.copy()


def psnr(reference, result):
    mse = float(np.mean((reference.astype(np.float64)-result.astype(np.float64))**2))
    return None if mse == 0 else float(-10*np.log10(mse))
