"""RF design primitives: a fast patch-antenna model, a higher-fidelity reference, microstrip and matching tools."""
from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np
from scipy.integrate import quad
from scipy.special import j0

C0 = 299_792_458.0
ETA0 = 376.730313668
MU0 = 4e-7 * math.pi


@dataclass(frozen=True)
class Substrate:
    er: float
    h_mm: float
    tand: float = 0.0
    sigma: float = 5.8e7  # copper

    @property
    def h(self) -> float:
        return self.h_mm * 1e-3


SUBSTRATES = {
    "FR4": Substrate(4.4, 1.6, 0.02),
    "FR4_thick": Substrate(4.4, 3.2, 0.02),
    "RO4003C": Substrate(3.55, 0.813, 0.0027),
    "RT5880": Substrate(2.2, 0.787, 0.0009),
    "RT5880_thick": Substrate(2.2, 3.175, 0.0009),
    "Alumina": Substrate(9.8, 0.635, 0.0001),
}


# ---------- microstrip (Hammerstad / Wheeler) ----------

def microstrip_eeff(w: float, h: float, er: float) -> float:
    u = w / h
    e = (er + 1) / 2 + (er - 1) / 2 / math.sqrt(1 + 12 / u)
    if u < 1:
        e += (er - 1) / 2 * 0.04 * (1 - u) ** 2
    return e


def microstrip_z0(w: float, h: float, er: float) -> float:
    u = w / h
    e = microstrip_eeff(w, h, er)
    if u <= 1:
        return 60 / math.sqrt(e) * math.log(8 / u + u / 4)
    return 120 * math.pi / (math.sqrt(e) * (u + 1.393 + 0.667 * math.log(u + 1.444)))


def microstrip_width(z0: float, h: float, er: float) -> float:
    """Synthesis (Pozar 3.197). Returns width in the same unit as h."""
    a = z0 / 60 * math.sqrt((er + 1) / 2) + (er - 1) / (er + 1) * (0.23 + 0.11 / er)
    b = 377 * math.pi / (2 * z0 * math.sqrt(er))
    wh = 8 * math.exp(a) / (math.exp(2 * a) - 2)
    if wh > 2:
        wh = 2 / math.pi * (b - 1 - math.log(2 * b - 1)
                            + (er - 1) / (2 * er) * (math.log(b - 1) + 0.39 - 0.61 / er))
    return wh * h


# ---------- rectangular patch geometry (Balanis transmission-line model) ----------

def patch_eeff(w: float, h: float, er: float) -> float:
    return (er + 1) / 2 + (er - 1) / 2 / math.sqrt(1 + 12 * h / w)


def patch_delta_l(w: float, h: float, eeff: float) -> float:
    return 0.412 * h * (eeff + 0.3) * (w / h + 0.264) / ((eeff - 0.258) * (w / h + 0.8))


def patch_dimensions(f_hz: float, sub: Substrate) -> tuple[float, float]:
    """Classic closed-form W, L (metres) for a target resonance."""
    w = C0 / (2 * f_hz) * math.sqrt(2 / (sub.er + 1))
    e = patch_eeff(w, sub.h, sub.er)
    l = C0 / (2 * f_hz * math.sqrt(e)) - 2 * patch_delta_l(w, sub.h, e)
    return w, l


def _q_total(*qs: float) -> float:
    return 1 / sum(1 / q for q in qs if q and math.isfinite(q))


def _z_parallel_rlc(freqs: np.ndarray, r: float, fr: float, q: float) -> np.ndarray:
    return r / (1 + 1j * q * (freqs / fr - fr / freqs))


def _s11(z: np.ndarray, z0: float) -> np.ndarray:
    return (z - z0) / (z + z0)


@dataclass
class PatchResult:
    fr: float
    r_edge: float
    r_in: float
    q: float
    freqs: np.ndarray
    z: np.ndarray
    s11: np.ndarray
    model: str

    def s11_db_at(self, f_hz: float) -> float:
        return float(20 * np.log10(abs(np.interp(f_hz, self.freqs, self.s11.real)
                                       + 1j * np.interp(f_hz, self.freqs, self.s11.imag)) + 1e-15))

    def summary(self, f_target: float | None = None, z0: float = 50.0) -> dict:
        s_db = 20 * np.log10(np.abs(self.s11) + 1e-15)
        band = self.freqs[s_db < -10]
        out = {
            "model": self.model,
            "resonance_ghz": round(self.fr / 1e9, 5),
            "r_edge_ohm": round(self.r_edge, 2),
            "r_in_ohm": round(self.r_in, 2),
            "q_loaded": round(self.q, 2),
            "min_s11_db": round(float(s_db.min()), 2),
            "f_min_s11_ghz": round(float(self.freqs[s_db.argmin()]) / 1e9, 5),
            "bw_10db_mhz": round(float(band.max() - band.min()) / 1e6, 2) if band.size else 0.0,
        }
        if f_target:
            out["s11_at_target_db"] = round(self.s11_db_at(f_target), 2)
        return out


def _sweep(fr: float, span: float = 0.2, n: int = 801) -> np.ndarray:
    return np.linspace(fr * (1 - span / 2), fr * (1 + span / 2), n)


def fast_patch(w: float, l: float, y0: float, sub: Substrate, z0: float = 50.0,
               freqs: np.ndarray | None = None) -> PatchResult:
    """Surrogate: static eeff, approximate G1, no mutual coupling, no conductor loss."""
    e = patch_eeff(w, sub.h, sub.er)
    dl = patch_delta_l(w, sub.h, e)
    fr = C0 / (2 * (l + 2 * dl) * math.sqrt(e))
    lam0 = C0 / fr
    k0h = 2 * math.pi / lam0 * sub.h
    g1 = w / (120 * lam0) * (1 - k0h ** 2 / 24)
    r_edge = 1 / (2 * g1)
    r_in = r_edge * math.cos(math.pi * y0 / l) ** 2
    q = _q_total(C0 * math.sqrt(e) / (4 * fr * sub.h), 1 / sub.tand if sub.tand else math.inf)
    f = _sweep(fr) if freqs is None else freqs
    z = _z_parallel_rlc(f, r_in, fr, q)
    return PatchResult(fr, r_edge, r_in, q, f, z, _s11(z, z0), "fast")


def _g_integral(w: float, l: float | None, k0: float) -> float:
    a = k0 * w / 2

    def f(t: float) -> float:
        c, s = math.cos(t), math.sin(t)
        val = (a * np.sinc(a * c / math.pi)) ** 2 * s ** 3
        return val * (j0(k0 * l * s) if l is not None else 1.0)

    return quad(f, 0, math.pi, limit=200)[0] / (120 * math.pi ** 2)


def _eeff_dispersive(w: float, h: float, er: float, f: float) -> float:
    """Getsinger dispersion on top of the static patch eeff."""
    e0 = patch_eeff(w, h, er)
    zc = microstrip_z0(w, h, er)
    fp = zc / (2 * MU0 * h)
    g = 0.6 + 0.009 * zc
    return er - (er - e0) / (1 + g * (f / fp) ** 2)


def reference_patch(w: float, l: float, y0: float, sub: Substrate, z0: float = 50.0,
                    freqs: np.ndarray | None = None) -> PatchResult:
    """Higher-fidelity cavity-style model: dispersive eeff, exact G1 + mutual G12,
    dielectric and conductor loss pulling down the resonant resistance."""
    fr = fast_patch(w, l, y0, sub).fr
    for _ in range(8):
        e = _eeff_dispersive(w, sub.h, sub.er, fr)
        dl = patch_delta_l(w, sub.h, e)
        fr = C0 / (2 * (l + 2 * dl) * math.sqrt(e))
    k0 = 2 * math.pi * fr / C0
    g1 = _g_integral(w, None, k0)
    g12 = _g_integral(w, l, k0)
    q_rad = C0 * math.sqrt(e) / (4 * fr * sub.h)
    q_c = sub.h * math.sqrt(math.pi * fr * MU0 * sub.sigma)
    q_d = 1 / sub.tand if sub.tand else math.inf
    q = _q_total(q_rad, q_c, q_d)
    r_edge = 1 / (2 * (g1 + g12)) * (q / q_rad)
    r_in = r_edge * math.cos(math.pi * y0 / l) ** 2
    f = _sweep(fr) if freqs is None else freqs
    z = _z_parallel_rlc(f, r_in, fr, q)
    return PatchResult(fr, r_edge, r_in, q, f, z, _s11(z, z0), "reference")


def inset_for(r_edge: float, l: float, z0: float = 50.0) -> float | None:
    """Inset depth giving R_in = z0, or None when the edge resistance is already below z0."""
    if r_edge <= z0:
        return None
    return l / math.pi * math.acos(math.sqrt(z0 / r_edge))


# ---------- design loops ----------

def open_loop_design(f_hz: float, sub: Substrate, z0: float = 50.0) -> dict:
    """Design using only the fast model (what you'd get trusting the surrogate)."""
    w, l = patch_dimensions(f_hz, sub)
    fast = fast_patch(w, l, 0.0, sub, z0)
    y0 = inset_for(fast.r_edge, l, z0) or 0.0
    return {"w": w, "l": l, "y0": y0, "fast_calls": 2, "ref_calls": 0}


def correct_length(w: float, l: float, fr_seen: float, f_target: float, sub: Substrate) -> float:
    """Resonance scales ~1/(L + 2*dL): rescale the electrical length to move fr_seen onto f_target."""
    dl = patch_delta_l(w, sub.h, _eeff_dispersive(w, sub.h, sub.er, fr_seen))
    return (l + 2 * dl) * (fr_seen / f_target) - 2 * dl


def closed_loop_design(f_hz: float, sub: Substrate, z0: float = 50.0,
                       tol_f: float = 1e-3, tol_r: float = 1.0, max_iter: int = 8) -> dict:
    """Surrogate proposes, reference verifies, correct L and inset, repeat."""
    d = open_loop_design(f_hz, sub, z0)
    w, l, y0 = d["w"], d["l"], d["y0"]
    trace, ref_calls = [], 0
    for i in range(max_iter):
        ref = reference_patch(w, l, y0, sub, z0)
        ref_calls += 1
        f_err = (ref.fr - f_hz) / f_hz
        trace.append({"iter": i, "l_mm": l * 1e3, "y0_mm": y0 * 1e3,
                      "fr_ghz": ref.fr / 1e9, "r_in": ref.r_in})
        converged = abs(f_err) < tol_f and abs(ref.r_in - z0) < tol_r
        if converged or (abs(f_err) < tol_f and inset_for(ref.r_edge, l, z0) is None):
            break
        l = correct_length(w, l, ref.fr, f_hz, sub)
        ref_edge = reference_patch(w, l, 0.0, sub, z0)
        ref_calls += 1
        y0 = inset_for(ref_edge.r_edge, l, z0) or 0.0
    return {"w": w, "l": l, "y0": y0, "fast_calls": d["fast_calls"], "ref_calls": ref_calls,
            "iterations": len(trace), "trace": trace}


# ---------- L-network matching (Pozar 5.1) ----------

def _component(x_or_b: float, omega: float, kind: str) -> dict:
    if kind == "series":
        return ({"type": "inductor", "value_nH": x_or_b / omega * 1e9} if x_or_b > 0
                else {"type": "capacitor", "value_pF": -1 / (omega * x_or_b) * 1e12})
    return ({"type": "capacitor", "value_pF": x_or_b / omega * 1e12} if x_or_b > 0
            else {"type": "inductor", "value_nH": -1 / (omega * x_or_b) * 1e9})


def l_match(zl: complex, z0: float, f_hz: float) -> list[dict]:
    """Both lossless L-section solutions matching zl to z0. Each solution is verified by
    recomputing the input impedance."""
    rl, xl = zl.real, zl.imag
    omega = 2 * math.pi * f_hz
    sols = []
    if rl > z0:
        root = math.sqrt(rl / z0) * math.sqrt(rl ** 2 + xl ** 2 - z0 * rl)
        for sign in (1, -1):
            b = (xl + sign * root) / (rl ** 2 + xl ** 2)
            x = 1 / b + xl * z0 / rl - z0 / (b * rl)
            zin = 1j * x + 1 / (1j * b + 1 / zl)
            sols.append({"topology": "shunt-at-load", "X_ohm": x, "B_S": b,
                         "series": _component(x, omega, "series"),
                         "shunt": _component(b, omega, "shunt"), "zin": zin})
    else:
        for sign in (1, -1):
            x = sign * math.sqrt(rl * (z0 - rl)) - xl
            b = sign * math.sqrt((z0 - rl) / rl) / z0
            zin = 1 / (1j * b + 1 / (zl + 1j * x))
            sols.append({"topology": "series-at-load", "X_ohm": x, "B_S": b,
                         "series": _component(x, omega, "series"),
                         "shunt": _component(b, omega, "shunt"), "zin": zin})
    for s in sols:
        s["gamma_after"] = abs((s["zin"] - z0) / (s["zin"] + z0))
    return sols


def timed(fn, *args, repeat: int = 50, **kw) -> float:
    t = time.perf_counter()
    for _ in range(repeat):
        fn(*args, **kw)
    return (time.perf_counter() - t) / repeat
