"""
7-segment digit classifier using 3×3 zone correlation.

Key insight: each 7-segment digit has a distinctive 3×3 brightness
signature. By comparing observed zone brightness to reference signatures
via normalized cross-correlation, we classify digits robustly even
under significant Gaussian blur.

The 3×3 grid maps to 7-segment geometry:

    TL | TC | TR       a = TC (top horizontal)
    ---+----+---       b = TR/MR (top-right vertical)
    ML | MC | MR       c = MR/BR (bottom-right vertical)
    ---+----+---       d = BC (bottom horizontal)
    BL | BC | BR       e = ML/BL (bottom-left vertical)
                       f = TL/ML (top-left vertical)
                       g = MC (middle horizontal)
"""

from __future__ import annotations

import numpy as np

# ── 3×3 Zone Signatures ──────────────────────────────────────────
# Calibrated from real-world blurry 7-segment photos.
# 1.0 = segment ON (bright), 0.0 = segment OFF (dark).
# Values between account for bloom/bleed from adjacent segments.

SIGNATURES: dict[str, dict[str, float]] = {
    "0": {
        "TL": 0.7,
        "TC": 1.0,
        "TR": 0.7,
        "ML": 0.7,
        "MC": 0.15,
        "MR": 0.7,
        "BL": 0.7,
        "BC": 1.0,
        "BR": 0.7,
    },
    "1": {
        "TL": 0.1,
        "TC": 0.1,
        "TR": 0.7,
        "ML": 0.1,
        "MC": 0.1,
        "MR": 0.7,
        "BL": 0.1,
        "BC": 0.1,
        "BR": 0.7,
    },
    "2": {
        "TL": 0.15,
        "TC": 1.0,
        "TR": 0.7,
        "ML": 0.2,
        "MC": 1.0,
        "MR": 0.5,
        "BL": 0.7,
        "BC": 1.0,
        "BR": 0.15,
    },
    "3": {
        "TL": 0.15,
        "TC": 1.0,
        "TR": 0.7,
        "ML": 0.2,
        "MC": 1.0,
        "MR": 0.7,
        "BL": 0.15,
        "BC": 1.0,
        "BR": 0.7,
    },
    "4": {
        "TL": 0.7,
        "TC": 0.3,
        "TR": 0.7,
        "ML": 0.7,
        "MC": 1.0,
        "MR": 0.7,
        "BL": 0.1,
        "BC": 0.2,
        "BR": 0.7,
    },
    "5": {
        "TL": 0.7,
        "TC": 1.0,
        "TR": 0.15,
        "ML": 0.7,
        "MC": 1.0,
        "MR": 0.7,
        "BL": 0.1,
        "BC": 1.0,
        "BR": 0.7,
    },
    "6": {
        "TL": 0.7,
        "TC": 1.0,
        "TR": 0.15,
        "ML": 0.7,
        "MC": 1.0,
        "MR": 0.5,
        "BL": 0.7,
        "BC": 1.0,
        "BR": 0.7,
    },
    "7": {
        "TL": 0.1,
        "TC": 1.0,
        "TR": 0.7,
        "ML": 0.1,
        "MC": 0.1,
        "MR": 0.7,
        "BL": 0.1,
        "BC": 0.1,
        "BR": 0.7,
    },
    "8": {
        "TL": 0.7,
        "TC": 1.0,
        "TR": 0.7,
        "ML": 0.7,
        "MC": 1.0,
        "MR": 0.7,
        "BL": 0.7,
        "BC": 1.0,
        "BR": 0.7,
    },
    "9": {
        "TL": 0.7,
        "TC": 1.0,
        "TR": 0.7,
        "ML": 0.7,
        "MC": 1.0,
        "MR": 0.7,
        "BL": 0.1,
        "BC": 1.0,
        "BR": 0.7,
    },
    "A": {
        "TL": 0.7,
        "TC": 1.0,
        "TR": 0.7,
        "ML": 0.7,
        "MC": 1.0,
        "MR": 0.7,
        "BL": 0.7,
        "BC": 0.15,
        "BR": 0.7,
    },
    "b": {
        "TL": 0.7,
        "TC": 0.2,
        "TR": 0.2,
        "ML": 0.7,
        "MC": 1.0,
        "MR": 0.7,
        "BL": 0.7,
        "BC": 1.0,
        "BR": 0.7,
    },
    "C": {
        "TL": 0.7,
        "TC": 1.0,
        "TR": 0.1,
        "ML": 0.7,
        "MC": 0.1,
        "MR": 0.1,
        "BL": 0.7,
        "BC": 1.0,
        "BR": 0.1,
    },
    "d": {
        "TL": 0.1,
        "TC": 0.2,
        "TR": 0.7,
        "ML": 0.7,
        "MC": 1.0,
        "MR": 0.7,
        "BL": 0.7,
        "BC": 1.0,
        "BR": 0.7,
    },
    "E": {
        "TL": 0.7,
        "TC": 1.0,
        "TR": 0.1,
        "ML": 0.7,
        "MC": 1.0,
        "MR": 0.1,
        "BL": 0.7,
        "BC": 1.0,
        "BR": 0.1,
    },
    "F": {
        "TL": 0.7,
        "TC": 1.0,
        "TR": 0.1,
        "ML": 0.7,
        "MC": 1.0,
        "MR": 0.1,
        "BL": 0.7,
        "BC": 0.1,
        "BR": 0.1,
    },
    "-": {
        "TL": 0.1,
        "TC": 0.1,
        "TR": 0.1,
        "ML": 0.1,
        "MC": 1.0,
        "MR": 0.1,
        "BL": 0.1,
        "BC": 0.1,
        "BR": 0.1,
    },
}

DIGITS = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
HEX_LETTERS = ["A", "b", "C", "d", "E", "F"]
ZONES = ["TL", "TC", "TR", "ML", "MC", "MR", "BL", "BC", "BR"]


def extract_zones(img_data: np.ndarray, x1: int, y1: int, dw: int, dh: int) -> dict[str, float]:
    """Extract 3×3 zone brightness from a digit rectangle.

    Args:
        img_data: 2D grayscale array (float32, 0..1).
        x1, y1: Top-left corner of digit window.
        dw, dh: Window width and height in pixels.

    Returns:
        Dict mapping zone name (TL, TC, ...) to mean brightness.
    """
    h_img, w_img = img_data.shape
    zone_h = dh // 3
    zone_w = dw // 3
    zones: dict[str, float] = {}

    for idx, name in enumerate(ZONES):
        row, col = divmod(idx, 3)
        cy1 = y1 + row * zone_h
        cy2 = y1 + (row + 1) * zone_h if row < 2 else y1 + dh
        cx1 = x1 + col * zone_w
        cx2 = x1 + (col + 1) * zone_w if col < 2 else x1 + dw

        cy1, cy2 = max(0, cy1), min(h_img, cy2)
        cx1, cx2 = max(0, cx1), min(w_img, cx2)

        cell = img_data[cy1:cy2, cx1:cx2]
        zones[name] = float(np.mean(cell)) if cell.size > 0 else 0.0

    return zones


def classify(
    zones: dict[str, float],
    allowed_chars: list[str] | None = None,
    top_n: int = 3,
) -> tuple[str, float, list[tuple[str, float]]]:
    """Classify a digit from its 3×3 zone brightness values.

    Uses normalized cross-correlation with reference signatures.

    Args:
        zones: Zone brightness dict from ``extract_zones``.
        allowed_chars: Restrict classification to these characters.
                       Defaults to all known signatures.
        top_n: Number of top candidates to return.

    Returns:
        Tuple of (best_char, confidence, top_n_candidates).
        Confidence is the Pearson correlation coefficient [-1, 1].
    """
    vals = np.array([zones[n] for n in ZONES])
    vmin, vmax = vals.min(), vals.max()

    if vmax - vmin < 0.03:
        return "?", 0.0, [("?", 0.0)]

    # Normalize both observed and signature to [0,1] using global range.
    # Per-signature normalization was the original bug: "8" (all 0.7-1.0)
    # collapsed to mostly zeros after (x-min)/(max-min).
    obs = vals  # values already in 0-1 range
    obs_c = obs - obs.mean()
    onorm = np.linalg.norm(obs_c)

    chars_to_try = allowed_chars if allowed_chars else list(SIGNATURES.keys())

    scores: dict[str, float] = {}
    for char in chars_to_try:
        sig = SIGNATURES[char]
        exp_raw = np.array([sig[n] for n in ZONES])
        exp_c = exp_raw - exp_raw.mean()
        enorm = np.linalg.norm(exp_c)

        corr = (
            float(np.dot(obs_c, exp_c) / (onorm * enorm)) if onorm > 1e-6 and enorm > 1e-6 else 0.0
        )

        # ── Rule-based boost/penalty ────────────────────────────────
        boost = 0.0

        # BC = bottom horizontal (d). Strong signal.
        bc_bright = vals[7] > 0.6
        tr_bright = vals[2] > 0.6
        tl_dark = vals[0] < 0.35
        bl_bright = vals[6] > 0.6

        if bc_bright and exp_raw[7] < 0.3:
            boost -= 0.7
        elif bc_bright and exp_raw[7] > 0.7:
            boost += 0.3

        if tr_bright and exp_raw[2] < 0.2:
            boost -= 0.5
        elif tr_bright and exp_raw[2] > 0.6:
            boost += 0.2

        if tl_dark and exp_raw[0] > 0.5:
            boost -= 0.3

        if bl_bright and exp_raw[6] < 0.2:
            boost -= 0.5
        elif bl_bright and exp_raw[6] > 0.5:
            boost += 0.2

        scores[char] = max(-1.0, min(1.0, corr + boost)) + corr * 0.0001  # tiebreak

    ranked = sorted(scores.items(), key=lambda x: -x[1])
    return ranked[0][0], ranked[0][1], ranked[:top_n]
