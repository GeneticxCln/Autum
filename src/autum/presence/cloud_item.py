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
            c = {k: np.ascontiguousarray(data[k], dtype=np.float32) for k in data.files}

        rng = np.random.default_rng(1337)

        # Precompute spatial harmonic bases once so every frame evaluates
        # multi-harmonic 3D curl-like fluid currents via pure SIMD multiply-adds
        # (using sin(a+wt) = sin(a)cos(wt) + cos(a)sin(wt)) with near-zero CPU cost.
        fx, fy, fz = c["fx"], c["fy"], c["fz"]
        n_f = fx.size
        f_jit = rng.uniform(0.0, 2.0 * np.pi, n_f).astype(np.float32)
        f_th1 = (fx * 5.8 + fy * 4.4 - fz * 5.2 + 0.45 * f_jit).astype(np.float32)
        f_th2 = (-fx * 4.9 + fy * 6.7 + fz * 4.1 + 0.55 * f_jit).astype(np.float32)
        f_tw_th = (fx * 5.2 + fy * 4.6 + f_jit).astype(np.float32)
        f_rad = np.sqrt(fx * fx + (fy + 0.04) ** 2).astype(np.float32)
        f_ac_th = (f_rad * 12.5).astype(np.float32)

        # Edge-dissolve mask: 0 on core features (eyes/nose/lips), rising smoothly
        # toward 1 along the ears, outer jawline, crown rim, and smoky neck fade.
        fz_edge = np.power(np.clip((0.39 - fz) / 0.29, 0.0, 1.0), 1.35)
        neck_edge = np.clip((fy - 0.34) / 0.28, 0.0, 1.0)
        f_edge = np.clip(np.maximum(fz_edge, neck_edge), 0.0, 1.0).astype(np.float32)

        c["f_edge"] = f_edge
        c["f_s1"] = np.sin(f_th1)
        c["f_c1"] = np.cos(f_th1)
        c["f_s2"] = np.sin(f_th2)
        c["f_c2"] = np.cos(f_th2)
        c["f_tw_s"] = np.sin(f_tw_th)
        c["f_tw_c"] = np.cos(f_tw_th)
        c["f_ac_s"] = np.sin(f_ac_th)
        c["f_ac_c"] = np.cos(f_ac_th)
        c["f_cr_s"] = np.sin(fy * 11.0).astype(np.float32)
        c["f_cr_c"] = np.cos(fy * 11.0).astype(np.float32)

        # Outer stardust shell: individual stream velocities (with gentle counter-flows)
        # + 2-harmonic 3D eddy currents + bioluminescent sparkle basis.
        sy, slon, sseed = c["sy"], c["slon"], c["sseed"]
        s_dir = np.where(sseed[:, 2] < 0.32, -0.65, 1.0).astype(np.float32)
        c["s_vel"] = ((0.48 + 0.85 * sseed[:, 0]) * s_dir).astype(np.float32)
        s_th1 = (slon * 2.0 + sy * 4.8 + sseed[:, 2] * 3.14159).astype(np.float32)
        s_th2 = (-slon * 3.0 + sy * 5.6 + sseed[:, 1] * 6.28318).astype(np.float32)
        s_tw_th = (sseed[:, 2] * 6.28318 + slon * 1.5 + sy * 3.0).astype(np.float32)
        c["s_s1"] = np.sin(s_th1)
        c["s_c1"] = np.cos(s_th1)
        c["s_s2"] = np.sin(s_th2)
        c["s_c2"] = np.cos(s_th2)
        c["s_tw_s"] = np.sin(s_tw_th)
        c["s_tw_c"] = np.cos(s_tw_th)
        c["s_ac_s"] = np.sin(c["srad"] * 14.0).astype(np.float32)
        c["s_ac_c"] = np.cos(c["srad"] * 14.0).astype(np.float32)

        # Interior dark-mist bokeh: lazy convection currents around the head.
        my, mlon, mseed = c["my"], c["mlon"], c["mseed"]
        m_th1 = (mlon * 2.0 + my * 4.0 + mseed[:, 1] * 6.28318).astype(np.float32)
        m_th2 = (-mlon * 1.7 + my * 5.2 + mseed[:, 0] * 6.28318).astype(np.float32)
        c["m_vel"] = (np.where(mseed[:, 1] < 0.45, -1.0, 1.0) * (0.30 + 0.55 * mseed[:, 0])).astype(np.float32)
        c["m_s1"] = np.sin(m_th1)
        c["m_c1"] = np.cos(m_th1)
        c["m_s2"] = np.sin(m_th2)
        c["m_c2"] = np.cos(m_th2)

        _CLOUD_CACHE = c
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
    lookXChanged = Signal()
    lookYChanged = Signal()
    hoveringChanged = Signal()

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
        self._look_x = 0.0
        self._look_y = 0.0
        self._hovering = False

        # Smoothly-eased living 3D head pose state (prevents snapping between AI states)
        self._pose_yaw = 0.0
        self._pose_pitch = 0.0
        self._pose_roll = 0.0
        self._pose_tx = 0.0
        self._pose_ty = 0.0
        self._pose_scale = 1.0
        self._last_phase: float | None = None

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

    def _get_look_x(self) -> float:
        return self._look_x

    def _set_look_x(self, value: float) -> None:
        v = float(np.clip(value, -1.0, 1.0))
        if v != self._look_x:
            self._look_x = v
            self.lookXChanged.emit()

    lookX = Property(float, _get_look_x, _set_look_x, notify=lookXChanged)

    def _get_look_y(self) -> float:
        return self._look_y

    def _set_look_y(self, value: float) -> None:
        v = float(np.clip(value, -1.0, 1.0))
        if v != self._look_y:
            self._look_y = v
            self.lookYChanged.emit()

    lookY = Property(float, _get_look_y, _set_look_y, notify=lookYChanged)

    def _get_hovering(self) -> bool:
        return self._hovering

    def _set_hovering(self, value: bool) -> None:
        v = bool(value)
        if v != self._hovering:
            self._hovering = v
            self.hoveringChanged.emit()

    hovering = Property(bool, _get_hovering, _set_hovering, notify=hoveringChanged)

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

        # 1. Living organic 3D spherical stardust shell + billowing interior mist
        tilt = 0.24 + 0.06 * math.sin(phase * 0.29)
        cos_t = math.cos(tilt)
        sin_t = math.sin(tilt)

        # Scalar time harmonics (exact trig addition on precomputed spatial bases)
        c_e1, s_e1 = math.cos(phase * 0.72), math.sin(phase * 0.72)
        c_e2, s_e2 = math.cos(phase * 1.18), math.sin(phase * 1.18)
        c_tw, s_tw = math.cos(phase * 1.85), math.sin(phase * 1.85)
        c_ac, s_ac = math.cos(phase * 9.5), math.sin(phase * 9.5)

        sseed = c["sseed"]
        sy_base = c["sy"]
        pole_taper = np.clip(1.0 - sy_base * sy_base, 0.0, 1.0)
        # Organic 3D spherical tangent eddy currents (vanishes smoothly at poles -> zero pinch)
        eddy_lon = (c["s_s1"] * c_e1 + c["s_c1"] * s_e1) * 0.24 + (c["s_c2"] * c_e2 - c["s_s2"] * s_e2) * 0.13
        eddy_lat = ((c["s_c1"] * c_e1 - c["s_s1"] * s_e1) * 0.15 - (c["s_s2"] * c_e2 + c["s_c2"] * s_e2) * 0.09) * pole_taper
        eddy_rad = (c["s_s1"] * c_e2 + c["s_c1"] * s_e2) * 0.052

        energy = 1.0 + (0.75 if self._busy else 0.0) + 0.65 * voice
        sy = np.clip(sy_base + eddy_lat * energy, -0.998, 0.998)
        sring = np.sqrt(np.clip(1.0 - sy * sy, 0.0, 1.0))

        sa = (
            c["slon"]
            + spin * c["s_vel"]
            + swirl * (0.45 + 0.85 * sseed[:, 0]) * pole_taper
            + eddy_lon * energy
        )
        srx = sring * np.cos(sa)
        srz = sring * np.sin(sa)
        sty = sy * cos_t - srz * sin_t
        stz = sy * sin_t + srz * cos_t  # [-1, +1], +1 toward viewer

        # Gentle 3D roll precession so currents weave organically across all angles
        roll = 0.16 * math.sin(phase * 0.23) + (0.22 * math.sin(phase * 0.65) if self._busy else 0.0)
        cr, sr = math.cos(roll), math.sin(roll)
        srx_r = srx * cr - sty * sr
        sty_r = srx * sr + sty * cr
        srx, sty = srx_r, sty_r

        # Living organic breathing + undulating acoustic voice ripples
        s_breath = 1.0 + eddy_rad * (0.85 + loose * 4.2) + 0.014 * math.sin(phase * 1.05)
        if voice > 0.005:
            ac_wave = c["s_ac_s"] * c_ac - c["s_ac_c"] * s_ac  # sin(14*r - 9.5*t)
            s_voice = 1.0 + voice * (0.045 + 0.13 * sseed[:, 0] + 0.05 * ac_wave)
        else:
            s_voice = 1.0
        s_scale = gather * c["srad"] * s_breath * s_voice
        spx = cx + r_px * s_scale * srx
        spy = cy + r_px * s_scale * sty

        # Fresnel brightness + bioluminescent living sparkle wave
        r_proj = np.sqrt(srx * srx + sty * sty)
        fresnel = 0.22 + 0.96 * np.power(np.clip(r_proj, 0.0, 1.0), 2.05)
        face_clear = np.where(stz > 0.0, np.clip((r_proj - 0.36) / 0.38, 0.05, 1.0), 0.78)
        tw_wave = 0.5 + 0.5 * (c["s_tw_s"] * c_tw + c["s_tw_c"] * s_tw)
        twinkle = 0.36 + 0.80 * (tw_wave * tw_wave)
        sw = fresnel * face_clear * twinkle * (0.44 + 0.56 * sseed[:, 0]) * (0.58 + 0.42 * (stz * 0.5 + 0.5))
        if self._busy:
            sw *= 1.0 + 0.30 * pole_taper
        if voice > 0.01:
            sw *= 1.0 + 0.38 * voice

        # Apply lower-right muted gap to shell points
        if gap > 1e-3:
            s_ang = np.arctan2(sty, srx)
            s_da = np.abs((s_ang - 0.785398 + np.pi) % (2.0 * np.pi) - np.pi)
            sw *= 1.0 - gap * np.clip(2.0 * np.clip(1.0 - s_da / 0.62, 0.0, 1.0), 0.0, 1.0)

        back_mask = stz < 0.0
        front_mask = ~back_mask
        shell_back = _splat_bilinear(spx[back_mask], spy[back_mask], sw[back_mask], pw, ph)
        shell_front = _splat_bilinear(spx[front_mask], spy[front_mask], sw[front_mask], pw, ph)

        # Interior smoky dark-mist bokeh billowing in lazy convection currents
        mseed = c["mseed"]
        m_pole = np.clip(1.0 - c["my"] * c["my"], 0.0, 1.0)
        m_eddy_a = (c["m_s1"] * c_e1 + c["m_c1"] * s_e1) * 0.24
        m_eddy_y = (c["m_c2"] * c_e1 - c["m_s2"] * s_e1) * 0.14 * m_pole
        my = np.clip(c["my"] + m_eddy_y, -0.995, 0.995)
        mring = np.sqrt(np.clip(1.0 - my * my, 0.0, 1.0))
        ma = c["mlon"] + spin * c["m_vel"] + swirl * 0.55 * m_pole + m_eddy_a
        mrx = mring * np.cos(ma)
        mrz = mring * np.sin(ma)
        mty = my * cos_t - mrz * sin_t
        mrx_r = mrx * cr - mty * sr
        mty_r = mrx * sr + mty * cr
        mrx, mty = mrx_r, mty_r
        m_breath = 1.0 + 0.055 * (c["m_s2"] * c_e2 + c["m_c2"] * s_e2) + 0.045 * voice
        m_scale = gather * c["mrad"] * m_breath
        mpx = cx + r_px * m_scale * mrx
        mpy = cy + r_px * m_scale * mty
        mr_proj = np.sqrt(mrx * mrx + mty * mty)
        mw = (0.14 + 0.28 * mseed[:, 0]) * np.clip((mr_proj - 0.36) / 0.42, 0.08, 1.0)
        mist_buf = _splat_soft(mpx, mpy, mw, pw, ph)

        # Attenuate back-hemisphere stardust & mist behind the moving 3D head
        # Compute target 3D head pose driven by AI state + pointer proximity
        if self._last_phase is None or abs(phase - self._last_phase) > 0.5:
            dt_pose = None  # snap cleanly on first frame or explicit time jump
        else:
            dt_pose = max(0.016, min(0.25, abs(phase - self._last_phase)))
        self._last_phase = phase

        if self._speaking:
            v_act = 0.45 + 0.75 * voice
            t_yaw = 0.14 * math.sin(phase * 1.35) * v_act
            t_pitch = -0.03 + 0.10 * voice * math.sin(phase * 4.6) + 0.04 * math.sin(phase * 1.5)
            t_roll = 0.085 * math.sin(phase * 1.05 + 0.6) * v_act
            t_tx = 0.028 * math.sin(phase * 1.15)
            t_ty = -0.022 * voice + 0.020 * math.sin(phase * 1.9)
            t_scale = 1.04 + 0.042 * voice
        elif self._hearing:
            t_yaw = 0.055 * math.sin(phase * 0.72) + (0.12 * self._look_x if self._hovering else 0.0)
            t_pitch = -0.065 + 0.032 * math.sin(phase * 1.15)
            t_roll = 0.080 * math.sin(phase * 0.48 + 0.5)
            t_tx = 0.018 * math.sin(phase * 0.62)
            t_ty = -0.028 + 0.016 * math.sin(phase * 1.25)
            t_scale = 1.07
        elif self._busy:
            t_yaw = 0.26 * math.sin(phase * 0.68) + 0.06 * math.sin(phase * 1.45)
            t_pitch = -0.13 + 0.065 * math.sin(phase * 0.92)
            t_roll = 0.125 * math.sin(phase * 0.54 + 0.8)
            t_tx = 0.052 * math.sin(phase * 0.74)
            t_ty = -0.036 + 0.032 * math.sin(phase * 1.18)
            t_scale = 1.015
        elif self._observing:
            # Active visual scanning gaze while Eyes are watching the desktop
            t_yaw = 0.27 * math.sin(phase * 0.92)
            t_pitch = 0.10 * math.sin(phase * 0.61 + 0.4)
            t_roll = -0.06 * math.sin(phase * 0.92)
            t_tx = 0.055 * math.sin(phase * 0.92)
            t_ty = 0.035 * math.sin(phase * 0.61 + 0.4)
            t_scale = 1.035
        elif self._hovering:
            # Sentient pointer awareness: head glides to look toward your cursor
            t_yaw = 0.34 * self._look_x
            t_pitch = 0.22 * self._look_y
            t_roll = -0.075 * self._look_x
            t_tx = 0.062 * self._look_x
            t_ty = 0.050 * self._look_y
            t_scale = 1.05
        else:
            # Alive autonomous idle: floating weightlessly and scanning smoothly inside the orb
            t_yaw = 0.27 * math.sin(phase * 0.39) + 0.095 * math.sin(phase * 0.89 + 1.3)
            t_pitch = 0.12 * math.sin(phase * 0.31 + 0.7) + 0.042 * math.sin(phase * 0.77 + 2.1)
            t_roll = 0.095 * math.sin(phase * 0.35 + 1.6)
            t_tx = 0.068 * math.sin(phase * 0.43 + 0.4) + 0.022 * math.sin(phase * 0.97)
            t_ty = 0.056 * math.sin(phase * 0.56) + 0.018 * math.cos(phase * 1.13)
            t_scale = 1.0 + 0.018 * math.sin(phase * 1.05)

        if dt_pose is None:
            self._pose_yaw, self._pose_pitch, self._pose_roll = t_yaw, t_pitch, t_roll
            self._pose_tx, self._pose_ty, self._pose_scale = t_tx, t_ty, t_scale
        else:
            k = 1.0 - math.exp(-5.5 * dt_pose)
            self._pose_yaw += (t_yaw - self._pose_yaw) * k
            self._pose_pitch += (t_pitch - self._pose_pitch) * k
            self._pose_roll += (t_roll - self._pose_roll) * k
            self._pose_tx += (t_tx - self._pose_tx) * k
            self._pose_ty += (t_ty - self._pose_ty) * k
            self._pose_scale += (t_scale - self._pose_scale) * k

        yaw = self._pose_yaw
        pitch = self._pose_pitch
        roll_h = self._pose_roll
        head_tx = self._pose_tx
        head_ty = self._pose_ty
        head_scale = self._pose_scale

        # Shift head occlusion silhouette dynamically so it tracks the floating head
        sx_i = int(round((head_tx + 0.18 * math.sin(yaw)) * r_px))
        sy_i = int(round((head_ty + 0.14 * math.sin(pitch)) * r_px))
        if sx_i == 0 and sy_i == 0:
            head_occ_shifted = m["head_occ"]
        else:
            head_occ_shifted = np.zeros_like(m["head_occ"])
            src_y0 = max(0, -sy_i)
            src_y1 = min(ph, ph - sy_i)
            dst_y0 = max(0, sy_i)
            dst_y1 = dst_y0 + (src_y1 - src_y0)
            src_x0 = max(0, -sx_i)
            src_x1 = min(pw, pw - sx_i)
            dst_x0 = max(0, sx_i)
            dst_x1 = dst_x0 + (src_x1 - src_x0)
            if src_y1 > src_y0 and src_x1 > src_x0:
                head_occ_shifted[dst_y0:dst_y1, dst_x0:dst_x1] = m["head_occ"][src_y0:src_y1, src_x0:src_x1]

        occ = 1.0 - 0.88 * head_occ_shifted
        rim_pulse = 1.0 + 0.08 * math.sin(phase * 1.15) + 0.34 * voice + (0.16 if self._hearing else 0.0)
        rim_lum = m["rim_base"] * rim_pulse
        if gap > 1e-3:
            rim_lum = rim_lum * (1.0 - gap * m["gap_mask"])

        env_lum = (rim_lum + 0.72 * shell_back + 0.52 * mist_buf) * occ + 0.88 * shell_front

        # 2. Living, floating, turning 3D stippled face coalescing out of suspended dust
        c_f1, s_f1 = math.cos(phase * 1.05), math.sin(phase * 1.05)
        c_f2, s_f2 = math.cos(phase * 1.72), math.sin(phase * 1.72)
        flow_x = (c["f_s1"] * c_f1 + c["f_c1"] * s_f1) * 0.62 + (c["f_c2"] * c_f2 - c["f_s2"] * s_f2) * 0.38
        flow_y = (c["f_c1"] * c_f1 - c["f_s1"] * s_f1) * 0.62 - (c["f_s2"] * c_f2 + c["f_c2"] * s_f2) * 0.38
        flow_z = (c["f_s1"] * c_f2 - c["f_c2"] * s_f1) * 0.50

        wisp_amp = 0.0046 + 0.022 * c["f_edge"]
        if self._busy:
            wisp_amp = wisp_amp * (1.20 + 0.42 * c["crown_w"])
        if voice > 0.005:
            wisp_amp = wisp_amp + 0.0085 * voice

        fx0 = c["fx"] + flow_x * wisp_amp
        fy0 = c["fy"] + 0.05 + flow_y * wisp_amp
        fz0 = c["fz"] - 0.18 + flow_z * wisp_amp

        # Speaking jaw/lower-lip articulation + acoustic facial ripple
        if voice > 0.005:
            jw = c["jaw_w"]
            f_ac = c["f_ac_s"] * c_ac - c["f_ac_c"] * s_ac
            fy0 = fy0 + jw * (0.068 * voice) + f_ac * (0.006 * voice)
            fz0 = fz0 - jw * (0.024 * voice) + f_ac * (0.011 * voice)

        # Full 3D rotation: Yaw (Y) -> Pitch (X) -> Roll (Z) around head pivot
        cyaw, syaw = math.cos(yaw), math.sin(yaw)
        cpit, spit = math.cos(pitch), math.sin(pitch)
        crol, srol = math.cos(roll_h), math.sin(roll_h)

        rx1 = fx0 * cyaw + fz0 * syaw
        rz1 = -fx0 * syaw + fz0 * cyaw
        ry2 = fy0 * cpit - rz1 * spit
        rz2 = fy0 * spit + rz1 * cpit + 0.18
        rx3 = rx1 * crol - ry2 * srol
        ry3 = rx1 * srol + ry2 * crol - 0.05

        # Volumetric perspective projection + weightless floating translation inside orb
        breath = gather * head_scale * (1.0 + 0.012 * math.sin(phase * 1.08) + 0.004 * math.sin(phase * 2.16))
        persp = 3.0 / (3.0 - rz2)
        fpx = cx + r_px * (head_tx + breath * rx3 * persp)
        fpy = cy + r_px * (head_ty + breath * ry3 * persp)

        # Bioluminescent AI neural shimmer + natural blink + expressive state lighting
        c_ftw, s_ftw = math.cos(phase * 1.65), math.sin(phase * 1.65)
        f_shimmer = 0.80 + 0.28 * (0.5 + 0.5 * (c["f_tw_s"] * c_ftw + c["f_tw_c"] * s_ftw))
        fw = c["flum"] * f_shimmer

        # Soft natural blink every ~5.2 s while idle (paused during attentive states)
        if not (self._hearing or self._observing or self._speaking):
            b_t = phase % 5.2
            if b_t < 0.18:
                blink = math.sin(math.pi * (b_t / 0.18))
                fw = fw * (1.0 - 0.62 * blink * c["eye_w"])

        # Dynamic 3D directional lighting reacts as the head turns & tilts
        shade = 0.88 + 0.22 * (rz2 - 0.18) + 0.12 * (rx3 * math.sin(yaw))
        if self._hearing or self._observing or self._hovering:
            eye_boost = (
                (0.34 if self._hearing else 0.0)
                + (0.40 if self._observing else 0.0)
                + (0.20 if self._hovering else 0.0)
            )
            fw = fw + c["eye_w"] * eye_boost
        if self._busy:
            c_cr, s_cr = math.cos(phase * 4.2), math.sin(phase * 4.2)
            wave = 0.5 + 0.5 * (s_cr * c["f_cr_c"] - c_cr * c["f_cr_s"])
            fw = fw + c["crown_w"] * (0.38 * wave)
        if voice > 0.005:
            fw = fw * (1.0 + 0.18 * voice) + c["jaw_w"] * (0.36 * voice)

        face_raw = _splat_bilinear(fpx, fpy, fw * shade * 0.88, pw, ph)

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
