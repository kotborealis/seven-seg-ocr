"""
Main entry point: ``read_display()`` ties detection and classification together.

Supports two display formats:
- **D.D** — decimal number (e.g., "2.2", "3.5")
- **LD** — hex letter + digit (e.g., "A1", "b3")
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from seven_seg_ocr.classifier import DIGITS, HEX_LETTERS, classify, extract_zones
from seven_seg_ocr.detector import detect_y, find_digits, is_dot


def read_display(
    image_path: str | Path | None = None,
    image_array: np.ndarray | None = None,
    *,
    y: int | None = None,
    dh: int = 35,
    window_w: int = 14,
    positions: list[int] | None = None,
    n_digits: int | None = 2,
) -> dict:
    """Read a 7-segment display from an image.

    Args:
        image_path: Path to an image file (JPEG, PNG, etc.).
        image_array: RGB numpy array (H×W×3, uint8). Takes precedence.
        y: Top of digit band in pixels (auto-detected if None).
        dh: Digit height in pixels.
        window_w: Classification window width per digit.
        positions: Manual left-edge x-coordinates for each digit
                   (bypasses auto-detection).
        n_digits: Expected number of digits (default 2).

    Returns:
        Dict with keys:
        - ``reading`` — the recognized string (e.g., ``"2.2"``)
        - ``details`` — list of ``(char, confidence, top3)`` per position
        - ``confidence`` — mean confidence across all characters
        - ``format`` — ``"D.D"`` or ``"LD"``
    """
    # ── Load image ──────────────────────────────────────────────
    if image_array is not None:
        img_np = image_array
    elif image_path is not None:
        img = Image.open(image_path)
        img_np = np.array(img)
    else:
        return _empty_result()

    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    h_img, w_img = gray.shape

    # ── Vertical position ───────────────────────────────────────
    if y is None:
        y = detect_y(gray)

    # ── Digit positions ─────────────────────────────────────────
    if positions is not None:
        if len(positions) >= 2:
            gaps = [positions[i + 1] - positions[i] for i in range(len(positions) - 1)]
            avg_gap = sum(gaps) / len(gaps)
            dw = max(12, min(20, int(avg_gap * 1.15)))
        else:
            dw = window_w
        pos_list = [(x, dw) for x in positions]
    elif n_digits is not None:
        auto = find_digits(gray, y, dh, window_w)
        pos_list = auto[:n_digits] if auto else []
    else:
        pos_list = find_digits(gray, y, dh, window_w)

    if not pos_list or len(pos_list) < 2:
        return _single_digit_result(gray, pos_list, y, dh)

    # ── Decimal point detection ─────────────────────────────────
    x1_a, dw_a = pos_list[0]
    x1_b, dw_b = pos_list[1]
    dot_x = (x1_a + dw_a + x1_b) // 2
    dot_y = y + dh * 6 // 7

    has_dot = is_dot(gray, dot_x, dot_y, r=6)
    if not has_dot:
        for dy in (-2, 0, 2, 4):
            if is_dot(gray, dot_x, dot_y + dy, r=5):
                has_dot = True
                break

    # ── Classification with format constraints ──────────────────
    chars: list[str] = []
    details: list[tuple] = []

    if has_dot:
        # D.D format — both positions are digits 0-9
        for x1, dw in pos_list[:2]:
            zones = extract_zones(gray, x1, y, dw, dh)
            char, conf, top3 = classify(zones, allowed_chars=DIGITS)
            chars.append(char)
            details.append((char, conf, top3))
        chars.insert(1, ".")
        details.insert(1, (".", 1.0, [(".", 1.0)]))
    else:
        # LD format — first is hex letter, second is digit
        zones_a = extract_zones(gray, x1_a, y, dw_a, dh)
        char_a, conf_a, top3_a = classify(zones_a, allowed_chars=HEX_LETTERS)
        chars.append(char_a)
        details.append((char_a, conf_a, top3_a))

        zones_b = extract_zones(gray, x1_b, y, dw_b, dh)
        char_b, conf_b, top3_b = classify(zones_b, allowed_chars=DIGITS)
        chars.append(char_b)
        details.append((char_b, conf_b, top3_b))

    reading = "".join(chars)
    confs = [d[1] for d in details if d[0] != "."]
    avg_conf = float(np.mean(confs)) if confs else 0.0

    return {
        "reading": reading,
        "details": details,
        "confidence": avg_conf,
        "format": "D.D" if has_dot else "LD",
    }


def _empty_result() -> dict:
    return {"reading": "", "details": [], "confidence": 0.0, "format": "?"}


def _single_digit_result(gray: np.ndarray, pos_list: list, y: int, dh: int) -> dict:
    chars = []
    details = []
    for x1, dw in pos_list:
        zones = extract_zones(gray, x1, y, dw, dh)
        char, conf, top3 = classify(zones)
        chars.append(char)
        details.append((char, conf, top3))

    reading = "".join(chars)
    confs = [d[1] for d in details]
    avg_conf = float(np.mean(confs)) if confs else 0.0

    return {"reading": reading, "details": details, "confidence": avg_conf, "format": "?"}
