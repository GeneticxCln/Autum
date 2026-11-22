"""Build the 3D stippled face + outer stardust sphere asset (`face_cloud.npz`).

Extracts importance-sampled face points from the reference artwork
(`image_c7fe5c8a.jpg`), lifts them onto an analytical 3D head surface with
mouth/jaw, eye, and crown animation weights, and generates the surrounding
3D spherical stardust mist shell.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np


def _head_mask(nx: np.ndarray, ny: np.ndarray) -> np.ndarray:
    """Soft elliptical mask isolating the sculpted head & neck inside the orb."""
    # Cranium + jawline tapering gently into the neck
    rx = np.where(ny > 0.12, 0.43 - 0.14 * np.clip((ny - 0.12) / 0.50, 0.0, 1.0), 0.43)
    d_head = np.sqrt((nx / np.maximum(rx, 0.18)) ** 2 + ((ny + 0.07) / 0.61) ** 2)
    head = np.clip((1.05 - d_head) / 0.12, 0.0, 1.0)
    # Soft ghostly neck emerging below the chin (ny ~ 0.42..0.72)
    d_neck = np.sqrt((nx / 0.24) ** 2 + ((ny - 0.48) / 0.24) ** 2)
    neck = np.clip((1.05 - d_neck) / 0.20, 0.0, 1.0) * np.clip((0.72 - ny) / 0.22, 0.0, 1.0)
    # Subtle ears at |nx| ~ 0.40..0.48, ny ~ -0.14..0.16
    d_ear = np.sqrt(((np.abs(nx) - 0.41) / 0.085) ** 2 + ((ny + 0.01) / 0.18) ** 2)
    ears = np.clip((1.0 - d_ear) / 0.22, 0.0, 1.0) * 0.85
    return np.clip(np.maximum(np.maximum(head, ears), neck), 0.0, 1.0)


def _head_depth(nx: np.ndarray, ny: np.ndarray) -> np.ndarray:
    """Smooth anatomical 3D z-height for front-facing head points."""
    rx = np.where(ny > 0.12, 0.43 - 0.14 * np.clip((ny - 0.12) / 0.50, 0.0, 1.0), 0.43)
    r2 = (nx / np.maximum(rx, 0.18)) ** 2 + ((ny + 0.07) / 0.62) ** 2
    base = 0.46 * np.sqrt(np.clip(1.0 - 0.82 * r2, 0.02, 1.0))

    # Brow ridge
    brow = 0.055 * np.exp(-((ny + 0.26) / 0.07) ** 2 - (nx / 0.26) ** 2)
    # Recessed eye sockets
    eyes = -0.055 * (
        np.exp(-((nx - 0.16) / 0.075) ** 2 - ((ny + 0.16) / 0.05) ** 2)
        + np.exp(-((nx + 0.16) / 0.075) ** 2 - ((ny + 0.16) / 0.05) ** 2)
    )
    # Nose bridge + tip
    nose_bridge = 0.13 * np.exp(-(nx / 0.055) ** 2 - ((ny + 0.03) / 0.14) ** 2)
    nose_tip = 0.075 * np.exp(-(nx / 0.065) ** 2 - ((ny - 0.07) / 0.055) ** 2)
    # Cheekbones
    cheeks = 0.045 * (
        np.exp(-((nx - 0.20) / 0.10) ** 2 - ((ny - 0.02) / 0.11) ** 2)
        + np.exp(-((nx + 0.20) / 0.10) ** 2 - ((ny - 0.02) / 0.11) ** 2)
    )
    # Lips & chin
    lips = 0.055 * np.exp(-(nx / 0.13) ** 2 - ((ny - 0.29) / 0.055) ** 2)
    chin = 0.065 * np.exp(-(nx / 0.12) ** 2 - ((ny - 0.43) / 0.085) ** 2)
    return (base + brow + eyes + nose_bridge + nose_tip + cheeks + lips + chin).astype(np.float32)


def build_face_cloud(
    image_path: Path,
    out_path: Path,
    *,
    face_step: float = 3.2,
    n_shell: int = 7200,
    n_mist: int = 1400,
) -> None:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtGui import QGuiApplication, QImage

    _ = QGuiApplication.instance() or QGuiApplication([])
    img = QImage(str(image_path)).convertToFormat(QImage.Format.Format_RGBA8888)
    if img.isNull():
        raise FileNotFoundError(f"could not load {image_path}")
    arr = np.frombuffer(img.constBits(), dtype=np.uint8).reshape(img.height(), img.width(), 4)
    lum = arr[:, :, :3].mean(axis=2).astype(np.float32) / 255.0

    # Detect orb center & radius from non-black pixels
    ys_nz, xs_nz = np.where(lum > 0.10)
    cx = float(xs_nz.min() + xs_nz.max()) / 2.0
    cy = float(ys_nz.min() + ys_nz.max()) / 2.0
    r = float(xs_nz.max() - xs_nz.min()) / 2.0

    h, w = lum.shape
    rng = np.random.default_rng(42)

    # Uniform hexagonal lattice across the head silhouette (matches the crisp
    # evenly-spaced point-cloud mesh in the artwork without Monte Carlo holes)
    dy_step = face_step * 0.8660254
    rows = np.arange(int(cy - 0.72 * r), int(cy + 0.74 * r), dy_step, dtype=np.float32)
    cols_base = np.arange(int(cx - 0.52 * r), int(cx + 0.52 * r), face_step, dtype=np.float32)
    pts_x = []
    pts_y = []
    for row_i, y_val in enumerate(rows):
        x_row = cols_base + (0.5 * face_step if (row_i & 1) else 0.0)
        pts_x.append(x_row)
        pts_y.append(np.full_like(x_row, y_val))
    gx = np.concatenate(pts_x)
    gy = np.concatenate(pts_y)
    # Tiny organic jitter so it reads as fine stipple rather than rigid pixels
    gx += rng.uniform(-0.28 * face_step, 0.28 * face_step, gx.size).astype(np.float32)
    gy += rng.uniform(-0.28 * face_step, 0.28 * face_step, gy.size).astype(np.float32)

    px = np.clip(np.rint(gx).astype(np.int32), 0, w - 1)
    py = np.clip(np.rint(gy).astype(np.int32), 0, h - 1)
    nx_all = (gx - cx) / r
    ny_all = (gy - cy) / r
    mask_all = _head_mask(nx_all, ny_all)
    # Sample local 3x3 luminance average to remove JPEG grain while keeping razor-crisp edges
    lum_pad = np.pad(lum, 1, mode="edge")
    lum_smooth = (
        0.40 * lum_pad[py + 1, px + 1]
        + 0.10 * (lum_pad[py, px + 1] + lum_pad[py + 2, px + 1] + lum_pad[py + 1, px] + lum_pad[py + 1, px + 2])
        + 0.05 * (lum_pad[py, px] + lum_pad[py, px + 2] + lum_pad[py + 2, px] + lum_pad[py + 2, px + 2])
    )
    keep = (mask_all > 0.08) & (lum_smooth > 0.045)
    fx = nx_all[keep].astype(np.float32)
    fy = ny_all[keep].astype(np.float32)
    fz = _head_depth(fx, fy)
    # Preserve crisp sculpting: dark eye sockets stay dim, highlights luminous
    raw_l = lum_smooth[keep] * mask_all[keep]
    flum = np.clip(np.power(raw_l, 1.18) * 1.45, 0.03, 1.0).astype(np.float32)

    # Animation region weights on the face
    jaw_w = (
        np.exp(-((fy - 0.37) / 0.11) ** 2 - (fx / 0.18) ** 2)
        * np.clip((fy - 0.27) / 0.05, 0.0, 1.0)
    ).astype(np.float32)
    eye_w = (
        np.exp(-((fx - 0.16) / 0.07) ** 2 - ((fy + 0.16) / 0.045) ** 2)
        + np.exp(-((fx + 0.16) / 0.07) ** 2 - ((fy + 0.16) / 0.045) ** 2)
    ).astype(np.float32)
    crown_w = np.clip((-0.24 - fy) / 0.42, 0.0, 1.0).astype(np.float32)

    # Outer 3D spherical stardust shell (Fibonacci sphere + radial thickness -> natural Fresnel rim!)
    i_s = np.arange(n_shell, dtype=np.float32)
    sy = 1.0 - 2.0 * (i_s + 0.5) / n_shell
    sring = np.sqrt(np.clip(1.0 - sy * sy, 0.0, 1.0))
    slon = i_s * 2.39996323
    # Dense frosty stardust across outer shell [0.78, 1.005], peaking strongly near rim
    srad = (0.78 + 0.225 * np.power(rng.uniform(0.0, 1.0, n_shell), 0.55)).astype(np.float32)
    sseed = rng.uniform(0.0, 1.0, (n_shell, 3)).astype(np.float32)

    # Soft interior dark-mist bokeh particles around the face (between face & outer rim)
    i_m = np.arange(n_mist, dtype=np.float32)
    my = 1.0 - 2.0 * (i_m + 0.5) / n_mist
    mring = np.sqrt(np.clip(1.0 - my * my, 0.0, 1.0))
    mlon = i_m * 2.39996323 + 1.1
    mrad = (0.56 + 0.42 * np.power(rng.uniform(0.0, 1.0, n_mist), 0.72)).astype(np.float32)
    mseed = rng.uniform(0.0, 1.0, (n_mist, 2)).astype(np.float32)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        fx=fx.astype(np.float32),
        fy=fy.astype(np.float32),
        fz=fz.astype(np.float32),
        flum=flum,
        jaw_w=jaw_w,
        eye_w=eye_w,
        crown_w=crown_w,
        sy=sy,
        sring=sring,
        slon=slon,
        srad=srad,
        sseed=sseed,
        my=my,
        mring=mring,
        mlon=mlon,
        mrad=mrad,
        mseed=mseed,
    )


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[3]
    build_face_cloud(root / "image_c7fe5c8a.jpg", Path(__file__).with_name("face_cloud.npz"))
    print("built face_cloud.npz")
