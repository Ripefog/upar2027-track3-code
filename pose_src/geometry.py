"""PoseTrack18 keypoints, image crop transforms, targets, decoding."""
from __future__ import annotations
import numpy as np
from PIL import Image

JOINTS = ('nose', 'head_bottom', 'head_top', 'left_ear', 'right_ear',
          'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
          'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
          'left_knee', 'right_knee', 'left_ankle', 'right_ankle')
FLIP_INDEX = np.array([0, 1, 2, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15], dtype=np.int64)
SCORED_INDEX = np.array([0, 1, 2, *range(5, 17)], dtype=np.int64)
MEAN = np.asarray([.485, .456, .406], np.float32).reshape(3, 1, 1)
STD = np.asarray([.229, .224, .225], np.float32).reshape(3, 1, 1)


def crop_params(box, out_hw, pad=1.25, shift=(0., 0.), scale_jitter=1.):
    """Map PIL destination (x,y) to source image (left+x*sx, top+y*sy)."""
    x, y, w, h = [float(v) for v in box[:4]]
    if w <= 0 or h <= 0:
        raise ValueError(f'Invalid bounding box: {box}')
    oh, ow = out_hw
    aspect = ow / oh
    bw, bh = max(w, h * aspect), max(h, w / aspect)
    bw, bh = bw * pad * scale_jitter, bh * pad * scale_jitter
    cx, cy = x + w*.5 + shift[0] * w, y + h*.5 + shift[1] * h
    left, top = cx - bw*.5, cy - bh*.5
    return (left, top, bw / ow, bh / oh)


def crop_image(im: Image.Image, params, out_hw):
    left, top, sx, sy = params
    oh, ow = out_hw
    return im.transform((ow, oh), Image.Transform.AFFINE,
                        (sx, 0., left, 0., sy, top), Image.Resampling.BILINEAR,
                        fillcolor=(0, 0, 0))


def to_crop(points, params):
    pts = np.asarray(points, np.float32).copy()
    left, top, sx, sy = params
    pts[..., 0] = (pts[..., 0] - left) / sx
    pts[..., 1] = (pts[..., 1] - top) / sy
    return pts


def from_crop(points, params):
    pts = np.asarray(points, np.float32).copy()
    left, top, sx, sy = params
    pts[..., 0] = pts[..., 0] * sx + left
    pts[..., 1] = pts[..., 1] * sy + top
    return pts


def as_tensor(crop):
    array = np.asarray(crop, dtype=np.float32).transpose(2, 0, 1) / 255.0
    return (array - MEAN) / STD


def heatmap_targets(keypoints, visible, input_hw, sigma=2.):
    oh, ow = input_hw
    hh, hw = oh // 4, ow // 4
    targets = np.zeros((17, hh, hw), dtype=np.float32)
    weights = np.asarray(visible, np.float32).copy()
    yy, xx = np.mgrid[:hh, :hw]
    for k, (xy, valid) in enumerate(zip(keypoints, weights)):
        if not valid:
            continue
        mx, my = xy[0] / 4., xy[1] / 4.
        if not (0 <= mx < hw and 0 <= my < hh):
            weights[k] = 0
            continue
        targets[k] = np.exp(-((xx-mx)**2 + (yy-my)**2)/(2*sigma*sigma))
    weights[3:5] = 0.   # two ear slots are not scored by the challenge
    return targets, weights


def decode_numpy(heatmap):
    """Argmax + quarter-pixel Taylor sign adjustment. Returns crop pixel coordinates."""
    arr = np.asarray(heatmap, np.float32)
    if arr.ndim == 3:
        arr = arr[None]
    b, k, h, w = arr.shape
    flat = arr.reshape(b, k, -1)
    indices = flat.argmax(-1)
    scores = flat.max(-1)
    xs = (indices % w).astype(np.float32)
    ys = (indices // w).astype(np.float32)
    for bi in range(b):
        for ji in range(k):
            x, y = int(xs[bi, ji]), int(ys[bi, ji])
            if 1 <= x < w-1 and 1 <= y < h-1:
                dx = arr[bi, ji, y, x+1] - arr[bi, ji, y, x-1]
                dy = arr[bi, ji, y+1, x] - arr[bi, ji, y-1, x]
                xs[bi, ji] += np.sign(dx) * 0.25
                ys[bi, ji] += np.sign(dy) * 0.25
    return np.stack([xs, ys], axis=-1) * 4., np.clip(scores, 0., 1.)


def _gaussian_blur_np(arr, kernel=11, sigma=0.8):
    """Separable Gaussian blur on the last two axes (symmetric padding), pure numpy."""
    r = (kernel - 1) // 2
    x = np.arange(-r, r + 1, dtype=np.float32)
    k = np.exp(-0.5 * x * x / (sigma * sigma)); k /= k.sum()
    out = np.pad(arr, ((0, 0), (0, 0), (r, r), (r, r)), mode='symmetric')
    h, w = arr.shape[-2:]
    out = sum(k[i] * out[:, :, :, i:i + w] for i in range(2 * r + 1))
    return sum(k[i] * out[:, :, i:i + h, :] for i in range(2 * r + 1))


def decode_dark(heatmap, kernel=11, sigma=0.8):
    """DARK decoding (Zhang et al., CVPR 2020): blur, take log, 2nd-order Taylor around the argmax.

    Same return convention as decode_numpy: crop-pixel coordinates (heatmap index * 4) and clipped peaks.
    Peaks are taken from the UNblurred heatmap so confidence is unchanged.
    """
    arr = np.asarray(heatmap, np.float32)
    if arr.ndim == 3:
        arr = arr[None]
    b, k, h, w = arr.shape
    flat = arr.reshape(b, k, -1)
    idx = flat.argmax(-1)
    peaks = flat.max(-1)
    xs = (idx % w); ys = (idx // w)
    blur = _gaussian_blur_np(arr, kernel, sigma)
    # keep the original peak height after blurring (DARK "modulation")
    bmax = blur.reshape(b, k, -1).max(-1)
    blur = blur * (peaks / np.maximum(bmax, 1e-9))[..., None, None]
    lg = np.log(np.clip(blur, 1e-3, 50.))
    bi, ki = np.meshgrid(np.arange(b), np.arange(k), indexing='ij')
    inside = (xs >= 1) & (xs < w - 1) & (ys >= 1) & (ys < h - 1)
    x = np.clip(xs, 1, w - 2); y = np.clip(ys, 1, h - 2)
    g = lambda dy, dx: lg[bi, ki, y + dy, x + dx]
    c = g(0, 0)
    dx_ = 0.5 * (g(0, 1) - g(0, -1)); dy_ = 0.5 * (g(1, 0) - g(-1, 0))
    dxx = g(0, 1) - 2 * c + g(0, -1); dyy = g(1, 0) - 2 * c + g(-1, 0)
    dxy = 0.25 * (g(1, 1) - g(1, -1) - g(-1, 1) + g(-1, -1))
    det = dxx * dyy - dxy * dxy
    ok = inside & (det > 1e-9) & (dxx < 0) & (dyy < 0)   # proper maximum only
    det = np.where(ok, det, 1.)
    ox = -(dyy * dx_ - dxy * dy_) / det
    oy = -(-dxy * dx_ + dxx * dy_) / det
    ox = np.where(ok, np.clip(ox, -1., 1.), 0.); oy = np.where(ok, np.clip(oy, -1., 1.), 0.)
    coords = np.stack([xs + ox, ys + oy], axis=-1).astype(np.float32)
    return coords * 4., np.clip(peaks, 0., 1.)


def shift_heatmap_x(hm, n):
    """Shift heatmaps n cells to the right (edge-replicated). Aligns flip-TTA heatmaps: training flips with
    x' = W-1-x, so a plain reversal of the flipped heatmap is ~0.75-1 cell off. Validated on source data."""
    n = int(n)
    if n == 0:
        return hm
    out = np.empty_like(hm)
    if n > 0:
        out[..., n:] = hm[..., :-n]; out[..., :n] = hm[..., :1]
    else:
        out[..., :n] = hm[..., -n:]; out[..., n:] = hm[..., -1:]
    return out


def flip_heatmaps_back(hm):
    """Reverse horizontal input flip and swap left/right joints."""
    return hm[:, FLIP_INDEX, :, ::-1].copy()


def calibrated_joint_confidence(peak_values, bias=0.25, temperature=0.15):
    """Smooth nonconstant joint confidence from raw heatmap maxima.

    Bias and temperature MUST be selected on permitted training-domain
    validation; they are not probabilities calibrated to true correctness.
    """
    if temperature <= 0:
        raise ValueError('confidence temperature must be > 0')
    values = np.asarray(peak_values, np.float32)
    z = np.clip((values - float(bias)) / float(temperature), -20., 20.)
    return 1. / (1. + np.exp(-z))


def person_confidence(peak_values, bias=0.25, temperature=0.15):
    """Instance ranking score computed only from 15 scored joints."""
    conf = calibrated_joint_confidence(peak_values, bias, temperature)
    return np.mean(conf[..., SCORED_INDEX], axis=-1)
