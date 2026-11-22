"""3D stippled face + spherical stardust orb renderer for Autum's presence.

Renders the holographic silver-ice face inside a dark frosted stardust sphere
using a vectorized NumPy pipeline onto a physical-DPI QQuickPaintedItem.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
from PySide6.QtCore import Property, Signal, Slot
from PySide6.QtGui import QColor, QImage, QPainter
from PySide6.QtQml import qmlRegisterType
from PySide6.QtQuick import QQuickPaintedItem

_CLOUD_PATH = Path(__file__).with_name("face_cloud.npz")
_CLOUD_CACHE: dict[str, np.ndarray] | None = None
_QML_REGISTERED = False


def _load_cloud() -> dict[str, np.ndarray]:
    global _CLOUD_CACHE
    if _CLOUD_CACHE is None:
        with np.load(_CLOUD_PATH) as data:
            _CLOUD_CACHE = {k: np.ascontiguousarray(data[k], dtype=np.float32) for k in data.files}
    return _CLOUD_CACHE


def register_qml_types() -> None:
    """Register Autum's custom QML types (idempotent)."""
    global _QML_REGISTERED
    if _QML_REGISTERED:
        return
    qmlRegisterType(FaceCloudItem, "Autum", 1, 0, "FaceCloud")
    _QML_REGISTERED = True


def _splat_bilinear(
    x: np.ndarray,
    y: np.ndarray,
    w: np.ndarray,
    pw: int,
    ph: int,
) -> np.ndarray:
    """Fast vectorized bilinear splat into a (ph, pw) float32 buffer."""
    ix = np.floor(x).astype(np.int32)
    iy = np.floor(y).astype(np.int32)
    valid = (ix >= 0) & (ix < pw - 1) & (iy >= 0) & (iy < ph - 1) & (w > 1e-4)
    if not np.any(valid):
        return np.zeros((ph, pw), dtype=np.float32)
    ix = ix[valid]
    iy = iy[valid]
    fx = (x[valid] - ix).astype(np.float32)
    fy = (y[valid] - iy).astype(np.float32)
    wv = w[valid].astype(np.float32)
    flat = iy * pw + ix
    size = pw * ph
    out = np.bincount(flat, weights=wv * (1.0 - fx) * (1.0 - fy), minlength=size)
    out += np.bincount(flat + 1, weights=wv * fx * (1.0 - fy), minlength=size)
    out += np.bincount(flat + pw, weights=wv * (1.0 - fx) * fy, minlength=size)
    out += np.bincount(flat + pw + 1, weights=wv * fx * fy, minlength=size)
    return out.astype(np.float32, copy=False).reshape(ph, pw)


def _splat_soft(
    x: np.ndarray,
    y: np.ndarray,
    w: np.ndarray,
    pw: int,
    ph: int,
) -> np.ndarray:
    """Fast 5-tap soft bokeh splat for interior dark-mist particles."""
    ix = np.rint(x).astype(np.int32)
    iy = np.rint(y).astype(np.int32)
    valid = (ix >= 2) & (ix < pw - 2) & (iy >= 2) & (iy < ph - 2) & (w > 1e-4)
    if not np.any(valid):
        return np.zeros((ph, pw), dtype=np.float32)
    ix = ix[valid]
    iy = iy[valid]
    wv = w[valid].astype(np.float32)
    flat = iy * pw + ix
    size = pw * ph
    out = np.bincount(flat, weights=wv * 0.32, minlength=size)
    out += np.bincount(flat - 2 * pw, weights=wv * 0.17, minlength=size)
    out += np.bincount(flat + 2 * pw, weights=wv * 0.17, minlength=size)
    out += np.bincount(flat - 2, weights=wv * 0.17, minlength=size)
    out += np.bincount(flat + 2, weights=wv * 0.17, minlength=size)
    return out.astype(np.float32, copy=False).reshape(ph, pw)


def _build_static_masks(pw: int, ph: int, r_px: float) -> dict[str, np.ndarray]:
    """Precompute circular orb geometry, Fresnel rim, head occlusion, and indicator rings."""
    cx = pw * 0.5
    cy = ph * 0.5
    yy, xx = np.mgrid[0:ph, 0:pw].astype(np.float32)
    nx = (xx - cx) / max(r_px, 1.0)
    ny = (yy - cy) / max(r_px, 1.0)
    rr = np.sqrt(nx * nx + ny * ny)
    ang = np.arctan2(ny, nx)

    # Crisp anti-aliased sphere disc + subtle soft outer drop shadow for light windows
    aa = max(2.2 / max(r_px, 1.0), 0.012)
    disc = np.clip((1.005 - rr) / aa, 0.0, 1.0)
    outer_shadow = 0.24 * np.power(np.clip((1.065 - rr) / 0.065, 0.0, 1.0), 2.0) * (1.0 - disc)

    # Deep dark obsidian interior
    orb_alpha = 0.95 * disc + outer_shadow
    void_depth = (0.62 + 0.38 * np.clip(rr, 0.0, 1.0)) * disc

    # Organic frosted Fresnel silver-ice mist glow (particle shell does the crisp sparkle)
    rim_band = np.power(np.clip((rr - 0.52) / 0.48, 0.0, 1.0), 2.15) * disc
    rim_lip = np.exp(-((rr - 0.975) / 0.055) ** 2) * disc
    vertical_shade = 0.92 - 0.16 * np.clip(ny, 0.0, 1.0)
    rim_base = (0.38 * rim_band + 0.15 * rim_lip) * vertical_shade

    # Angular mask for the muted-microphone lower-right gap (+pi/4)
    da = np.abs((ang - 0.785398 + np.pi) % (2.0 * np.pi) - np.pi)
    gap_mask = np.clip(2.0 * np.clip(1.0 - da / 0.62, 0.0, 1.0), 0.0, 1.0)

    # Soft dark head occlusion behind the 3D face so back-hemisphere stardust
    # never washes out the dark eye sockets, nose shadow, or mouth crease
    rx = np.where(ny > 0.12, 0.42 - 0.14 * np.clip((ny - 0.12) / 0.50, 0.0, 1.0), 0.42)
    d_head = np.sqrt((nx / np.maximum(rx, 0.17)) ** 2 + ((ny + 0.07) / 0.59) ** 2)
    head_occ = np.clip((1.02 - d_head) / 0.15, 0.0, 1.0) * np.clip((0.64 - ny) / 0.22, 0.0, 1.0)

    # Indicator outer edge auras (open mic = warm amber; observing = cool cyan-teal)
    warm_ring = np.exp(-((rr - 1.025) / 0.085) ** 2) * np.clip((rr - 0.80) / 0.15, 0.0, 1.0)
    obs_ring = np.exp(-((rr - 1.075) / 0.095) ** 2) * np.clip((rr - 0.86) / 0.14, 0.0, 1.0)

    return {
        "disc": disc.astype(np.float32),
        "orb_alpha": orb_alpha.astype(np.float32),
        "void_depth": void_depth.astype(np.float32),
        "rim_base": rim_base.astype(np.float32),
        "gap_mask": gap_mask.astype(np.float32),
        "head_occ": head_occ.astype(np.float32),
        "warm_ring": warm_ring.astype(np.float32),
        "obs_ring": obs_ring.astype(np.float32),
    }


class FaceCloudItem(QQuickPaintedItem):
    """QQuickPaintedItem drawing the animated 3D stippled face & stardust orb."""

    tintChanged = Signal()
    voiceChanged = Signal()
    spinChanged = Signal()
    swirlChanged = Signal()
    phaseChanged = Signal()
    gatherChanged = Signal()
    looseChanged = Signal()
    brightnessChanged = Signal()
    gapChanged = Signal()
    warmthChanged = Signal()
    observeGlowChanged = Signal()
    orbRadiusChanged = Signal()
    speakingChanged = Signal()
    hearingChanged = Signal()
    busyChanged = Signal()
    observingChanged = Signal()

    def __init__(self, parent: Any = None) -> None:
        super().__init__(parent)
        self.setOpaquePainting(False)
        self.setAntialiasing(False)
        self.setRenderTarget(QQuickPaintedItem.RenderTarget.Image)

        self._cloud = _load_cloud()
        self._mask_key: tuple[int, int, int] | None = None
        self._masks: dict[str, np.ndarray] | None = None

        self._tint = QColor("#dce6f2")
        self._voice = 0.0
        self._spin = 0.0
        self._swirl = 0.0
        self._phase = 0.0
        self._gather = 1.0
        self._loose = 0.12
        self._brightness = 0.90
        self._gap = 0.0
        self._warmth = 0.0
        self._observe_glow = 0.0
        self._orb_radius = 0.0
        self._speaking = False
        self._hearing = False
        self._busy = False
        self._observing = False

    # --- QML properties ------------------------------------------------------

    def _get_tint(self) -> QColor:
        return self._tint

    def _set_tint(self, value: QColor) -> None:
        c = QColor(value)
        if c != self._tint:
            self._tint = c
            self.tintChanged.emit()
            self.update()

    tint = Property(QColor, _get_tint, _set_tint, notify=tintChanged)

    def _get_voice(self) -> float:
        return self._voice

    def _set_voice(self, value: float) -> None:
        v = float(value)
        if v != self._voice:
            self._voice = v
            self.voiceChanged.emit()

    voice = Property(float, _get_voice, _set_voice, notify=voiceChanged)

    def _get_spin(self) -> float:
        return self._spin

    def _set_spin(self, value: float) -> None:
        v = float(value)
        if v != self._spin:
            self._spin = v
            self.spinChanged.emit()

    spin = Property(float, _get_spin, _set_spin, notify=spinChanged)

    def _get_swirl(self) -> float:
        return self._swirl

    def _set_swirl(self, value: float) -> None:
        v = float(value)
        if v != self._swirl:
            self._swirl = v
            self.swirlChanged.emit()

    swirl = Property(float, _get_swirl, _set_swirl, notify=swirlChanged)

    def _get_phase(self) -> float:
        return self._phase

    def _set_phase(self, value: float) -> None:
        v = float(value)
        if v != self._phase:
            self._phase = v
            self.phaseChanged.emit()

    phase = Property(float, _get_phase, _set_phase, notify=phaseChanged)

    def _get_gather(self) -> float:
        return self._gather

    def _set_gather(self, value: float) -> None:
        v = float(value)
        if v != self._gather:
            self._gather = v
            self.gatherChanged.emit()

    gather = Property(float, _get_gather, _set_gather, notify=gatherChanged)

    def _get_loose(self) -> float:
        return self._loose

    def _set_loose(self, value: float) -> None:
        v = float(value)
        if v != self._loose:
            self._loose = v
            self.looseChanged.emit()

    loose = Property(float, _get_loose, _set_loose, notify=looseChanged)

    def _get_brightness(self) -> float:
        return self._brightness

    def _set_brightness(self, value: float) -> None:
        v = float(value)
        if v != self._brightness:
            self._brightness = v
            self.brightnessChanged.emit()
            self.update()

    brightness = Property(float, _get_brightness, _set_brightness, notify=brightnessChanged)

    def _get_gap(self) -> float:
        return self._gap

    def _set_gap(self, value: float) -> None:
        v = float(value)
        if v != self._gap:
            self._gap = v
            self.gapChanged.emit()
            self.update()

    gap = Property(float, _get_gap, _set_gap, notify=gapChanged)

    def _get_warmth(self) -> float:
        return self._warmth

    def _set_warmth(self, value: float) -> None:
        v = float(value)
        if v != self._warmth:
            self._warmth = v
            self.warmthChanged.emit()
            self.update()

    warmth = Property(float, _get_warmth, _set_warmth, notify=warmthChanged)

    def _get_observe_glow(self) -> float:
        return self._observe_glow

    def _set_observe_glow(self, value: float) -> None:
        v = float(value)
        if v != self._observe_glow:
            self._observe_glow = v
            self.observeGlowChanged.emit()
            self.update()

    observeGlow = Property(float, _get_observe_glow, _set_observe_glow, notify=observeGlowChanged)

    def _get_orb_radius(self) -> float:
        return self._orb_radius

    def _set_orb_radius(self, value: float) -> None:
        v = float(value)
        if v != self._orb_radius:
            self._orb_radius = v
            self.orbRadiusChanged.emit()
            self.update()

    orbRadius = Property(float, _get_orb_radius, _set_orb_radius, notify=orbRadiusChanged)

    def _get_speaking(self) -> bool:
        return self._speaking

    def _set_speaking(self, value: bool) -> None:
        v = bool(value)
        if v != self._speaking:
            self._speaking = v
            self.speakingChanged.emit()
            self.update()

    speaking = Property(bool, _get_speaking, _set_speaking, notify=speakingChanged)

    def _get_hearing(self) -> bool:
        return self._hearing

    def _set_hearing(self, value: bool) -> None:
        v = bool(value)
        if v != self._hearing:
            self._hearing = v
            self.hearingChanged.emit()
            self.update()

    hearing = Property(bool, _get_hearing, _set_hearing, notify=hearingChanged)

    def _get_busy(self) -> bool:
        return self._busy

    def _set_busy(self, value: bool) -> None:
        v = bool(value)
        if v != self._busy:
            self._busy = v
            self.busyChanged.emit()
            self.update()

    busy = Property(bool, _get_busy, _set_busy, notify=busyChanged)

    def _get_observing(self) -> bool:
        return self._observing

    def _set_observing(self, value: bool) -> None:
        v = bool(value)
        if v != self._observing:
            self._observing = v
            self.observingChanged.emit()
            self.update()

    observing = Property(bool, _get_observing, _set_observing, notify=observingChanged)

    @Slot()
    def redraw(self) -> None:
        self.update()

    # --- Rendering -----------------------------------------------------------

    def render_rgba(self, pw: int, ph: int, r_px: float) -> np.ndarray:
        """Render the current frame into a premultiplied (ph, pw, 4) uint8 RGBA array."""
        r_key = int(round(r_px * 10))
        if self._mask_key != (pw, ph, r_key) or self._masks is None:
            self._masks = _build_static_masks(pw, ph, r_px)
            self._mask_key = (pw, ph, r_key)
        m = self._masks
        c = self._cloud

        cx = pw * 0.5
        cy = ph * 0.5
        phase = self._phase
        spin = self._spin
        swirl = self._swirl
        voice = float(np.clip(self._voice, 0.0, 1.25))
        gather = float(self._gather)
        loose = float(self._loose)
        bright = float(self._brightness)
        gap = float(np.clip(self._gap, 0.0, 1.0))

        # 1. Outer 3D spherical stardust shell + interior bokeh mist
        tilt = 0.24
        cos_t = math.cos(tilt)
        sin_t = math.sin(tilt)

        sseed = c["sseed"]
        sy = c["sy"]
        sring = c["sring"]
        sa = c["slon"] + spin * 0.75 + swirl * (0.45 + 0.75 * sseed[:, 0]) * (1.0 - np.abs(sy))
        srx = sring * np.cos(sa)
        srz = sring * np.sin(sa)
        sty = sy * cos_t - srz * sin_t
        stz = sy * sin_t + srz * cos_t  # [-1, +1], +1 toward viewer

        # Radial motion: gentle breathing + voice burst outward on speaking
        s_drift = 1.0 + loose * 0.28 * np.sin(phase * (0.65 + sseed[:, 1]) + sseed[:, 2] * 6.28318)
        s_voice = 1.0 + voice * (0.04 + 0.14 * sseed[:, 0])
        s_scale = gather * c["srad"] * s_drift * s_voice
        spx = cx + r_px * s_scale * srx
        spy = cy + r_px * s_scale * sty

        # Fresnel brightness: dense frosty sparkle near rim, delicate in front of face
        r_proj = np.sqrt(srx * srx + sty * sty)
        fresnel = 0.22 + 0.95 * np.power(np.clip(r_proj, 0.0, 1.0), 2.1)
        # Keep the center face area clear of distracting front dust (matches artwork)
        face_clear = np.where(stz > 0.0, np.clip((r_proj - 0.36) / 0.38, 0.05, 1.0), 0.78)
        # Twinkle + depth shading
        twinkle = 0.65 + 0.35 * np.sin(phase * (1.4 + 2.2 * sseed[:, 1]) + sseed[:, 2] * 6.28318)
        sw = fresnel * face_clear * twinkle * (0.45 + 0.55 * sseed[:, 0]) * (0.60 + 0.40 * (stz * 0.5 + 0.5))
        if self._busy:
            sw *= 1.0 + 0.25 * (1.0 - np.abs(sy))
        if voice > 0.01:
            sw *= 1.0 + 0.35 * voice

        # Apply lower-right muted gap to shell points
        if gap > 1e-3:
            s_ang = np.arctan2(sty, srx)
            s_da = np.abs((s_ang - 0.785398 + np.pi) % (2.0 * np.pi) - np.pi)
            sw *= 1.0 - gap * np.clip(2.0 * np.clip(1.0 - s_da / 0.62, 0.0, 1.0), 0.0, 1.0)

        back_mask = stz < 0.0
        front_mask = ~back_mask
        shell_back = _splat_bilinear(spx[back_mask], spy[back_mask], sw[back_mask], pw, ph)
        shell_front = _splat_bilinear(spx[front_mask], spy[front_mask], sw[front_mask], pw, ph)

        # Interior smoky dark-mist bokeh clouds between head and outer rim
        mseed = c["mseed"]
        my = c["my"]
        mring = c["mring"]
        ma = c["mlon"] - spin * 0.35 + swirl * 0.45 * (1.0 - np.abs(my))
        mrx = mring * np.cos(ma)
        mrz = mring * np.sin(ma)
        mty = my * cos_t - mrz * sin_t
        m_scale = gather * c["mrad"] * (1.0 + 0.04 * voice)
        mpx = cx + r_px * m_scale * mrx
        mpy = cy + r_px * m_scale * mty
        mr_proj = np.sqrt(mrx * mrx + mty * mty)
        mw = (0.14 + 0.26 * mseed[:, 0]) * np.clip((mr_proj - 0.38) / 0.42, 0.08, 1.0)
        mist_buf = _splat_soft(mpx, mpy, mw, pw, ph)

        # Attenuate back-hemisphere stardust & mist behind the solid 3D head
        occ = 1.0 - 0.88 * m["head_occ"]
        rim_lum = m["rim_base"] * (1.0 + 0.28 * voice + (0.14 if self._hearing else 0.0))
        if gap > 1e-3:
            rim_lum = rim_lum * (1.0 - gap * m["gap_mask"])

        env_lum = (rim_lum + 0.70 * shell_back + 0.50 * mist_buf) * occ + 0.86 * shell_front

        # 2. Sculpted 3D stippled human face inside the orb
        att = 0.35 if self._hearing else 1.0
        yaw = (0.090 * math.sin(phase * 0.42) + 0.030 * math.sin(phase * 0.89)) * att
        pitch = (0.040 * math.sin(phase * 0.31 + 0.8) - (0.025 if self._hearing else 0.0)) * att

        fx0 = c["fx"]
        fy0 = c["fy"] + 0.05
        fz0 = c["fz"] - 0.22

        # Speaking jaw/lower-lip articulation driven directly by voice level
        if voice > 0.005:
            jw = c["jaw_w"]
            fy0 = fy0 + jw * (0.044 * voice)
            fz0 = fz0 - jw * (0.018 * voice)

        cyaw, syaw = math.cos(yaw), math.sin(yaw)
        cpit, spit = math.cos(pitch), math.sin(pitch)
        rx1 = fx0 * cyaw + fz0 * syaw
        rz1 = -fx0 * syaw + fz0 * cyaw
        ry2 = fy0 * cpit - rz1 * spit - 0.05
        rz2 = fy0 * spit + rz1 * cpit + 0.22

        # Gentle telephoto perspective + subtle living breath
        breath = gather * (1.0 + 0.010 * math.sin(phase * 1.15) + 0.022 * voice)
        persp = 3.4 / (3.4 - rz2)
        fpx = cx + r_px * breath * rx1 * persp
        fpy = cy + r_px * breath * ry2 * persp

        fw = c["flum"]
        # Subtle 3D shading shift with head turn + state-driven facial highlights
        shade = 0.92 + 0.16 * (rz2 - 0.20)
        if self._hearing or self._observing:
            eye_boost = (0.28 if self._hearing else 0.0) + (0.34 if self._observing else 0.0)
            fw = fw + c["eye_w"] * eye_boost
        if self._busy:
            wave = 0.5 + 0.5 * np.sin(phase * 3.8 - c["fy"] * 11.0)
            fw = fw + c["crown_w"] * (0.28 * wave)
        if voice > 0.005:
            fw = fw * (1.0 + 0.16 * voice) + c["jaw_w"] * (0.24 * voice)

        face_raw = _splat_bilinear(fpx, fpy, fw * shade * 0.85, pw, ph)

        # Fast rational tone curve (crisp sculpting, deep obsidian sockets, 4x faster than exp)
        face_lum = face_raw / (face_raw + 0.58)
        env_mapped = env_lum / (env_lum + 0.72)

        # Combine face & environment (face takes crisp visual priority over mist)
        total_lum = np.clip(
            np.maximum(face_lum, env_mapped * (1.0 - 0.50 * face_lum))
            + 0.26 * np.minimum(face_lum, env_mapped),
            0.0,
            1.0,
        )
        total_lum *= bright

        # 3. Colorize & composite into premultiplied RGBA
        tr = self._tint.redF()
        tg = self._tint.greenF()
        tb = self._tint.blueF()

        # Deep obsidian glass interior (very faintly cool-tinted)
        base_r = (0.024 + 0.018 * tr) * m["void_depth"] * bright
        base_g = (0.032 + 0.018 * tg) * m["void_depth"] * bright
        base_b = (0.048 + 0.022 * tb) * m["void_depth"] * bright

        # Highlights push smoothly toward crisp frosted white-silver
        hi = total_lum * total_lum
        r_ch = base_r + total_lum * (0.78 * tr + 0.22) + hi * 0.22
        g_ch = base_g + total_lum * (0.78 * tg + 0.22) + hi * 0.24
        b_ch = base_b + total_lum * (0.78 * tb + 0.22) + hi * 0.26

        # Smooth orb alpha + voice sparkles extending into pad
        out_spark = np.clip(env_mapped * (1.0 - m["disc"]) * bright * 1.15, 0.0, 1.0)
        alpha = np.clip(m["orb_alpha"] * (0.55 + 0.45 * bright) + out_spark, 0.0, 1.0)

        # Warm open-microphone edge aura
        if self._warmth > 1e-3:
            wa = float(self._warmth) * 0.42 * m["warm_ring"]
            r_ch = r_ch + wa * 0.94
            g_ch = g_ch + wa * 0.64
            b_ch = b_ch + wa * 0.46
            alpha = np.clip(alpha + wa, 0.0, 1.0)

        # Cool cyan-teal observation halo
        if self._observe_glow > 1e-3:
            oa = float(self._observe_glow) * 0.44 * m["obs_ring"]
            r_ch = r_ch + oa * 0.40
            g_ch = g_ch + oa * 0.88
            b_ch = b_ch + oa * 0.96
            alpha = np.clip(alpha + oa, 0.0, 1.0)

        # Premultiplied RGBA invariant: rgb <= alpha
        r_ch = np.minimum(np.clip(r_ch, 0.0, 1.0), alpha)
        g_ch = np.minimum(np.clip(g_ch, 0.0, 1.0), alpha)
        b_ch = np.minimum(np.clip(b_ch, 0.0, 1.0), alpha)

        rgba = np.empty((ph, pw, 4), dtype=np.uint8)
        rgba[:, :, 0] = (r_ch * 255.0 + 0.5).astype(np.uint8)
        rgba[:, :, 1] = (g_ch * 255.0 + 0.5).astype(np.uint8)
        rgba[:, :, 2] = (b_ch * 255.0 + 0.5).astype(np.uint8)
        rgba[:, :, 3] = (alpha * 255.0 + 0.5).astype(np.uint8)
        return rgba

    def paint(self, painter: QPainter) -> None:
        w = float(self.width())
        h = float(self.height())
        if w < 2.0 or h < 2.0:
            return
        win = self.window()
        dpr = float(win.effectiveDevicePixelRatio()) if win is not None else 1.0
        pw = max(2, int(round(w * dpr)))
        ph = max(2, int(round(h * dpr)))
        r_logical = self._orb_radius if self._orb_radius > 1.0 else min(w, h) / 3.0
        r_px = r_logical * dpr

        rgba = self.render_rgba(pw, ph, r_px)
        img = QImage(
            rgba.data,
            pw,
            ph,
            pw * 4,
            QImage.Format.Format_RGBA8888_Premultiplied,
        )
        img.setDevicePixelRatio(dpr)
        painter.drawImage(0, 0, img)
