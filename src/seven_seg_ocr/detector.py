"""
7-segment digit position detector using gradient edge analysis.

Finds digit boundaries by detecting positive/negative peaks
in the horizontal brightness gradient. No ML, no tuning required
— works on blurry images where brightness-based thresholding fails.
"""

import numpy as np


def find_digits(
    img_data: np.ndarray,
    y: int,
    dh: int,
    window_w: int = 14,
) -> list[tuple[int, int]]:
    """Locate digit windows via gradient edge detection.

    Algorithm:
        1. Compute horizontal brightness projection in the digit band.
        2. Find positive gradient peaks (left edges) and negative
           peaks (right edges).
        3. Pair edges by gradient strength — strongest edges first.
        4. Filter out isolated artifacts (far from other pairs).
        5. Center each window on the brightness-weighted centroid.

    Args:
        img_data: 2D grayscale array (float32, 0..1).
        y: Top of the digit band (pixels).
        dh: Height of the digit band (pixels).
        window_w: Output window width per digit (pixels).

    Returns:
        List of ``(x1, width)`` tuples — left edge and window width
        for each detected digit, sorted left-to-right.
    """
    h_img, w_img = img_data.shape
    y1 = max(0, y)
    y2 = min(h_img, y + dh)
    band = img_data[y1:y2, :]
    hproj = np.mean(band, axis=0)
    hproj_s = np.convolve(hproj, np.ones(3) / 3, mode="same")
    grad = np.gradient(hproj_s)

    # ── Find left edges (positive gradient peaks) ────────────────
    left_edges: list[int] = []
    for x in range(3, len(grad) - 3):
        if (
            grad[x] > 0.01
            and grad[x] >= grad[x - 1]
            and grad[x] >= grad[x + 1]
            and grad[x] >= grad[x - 2]
            and grad[x] >= grad[x + 2]
        ):
            left_edges.append(x)

    # ── Find right edges (negative gradient peaks) ───────────────
    right_edges: list[int] = []
    for x in range(3, len(grad) - 3):
        if (
            grad[x] < -0.01
            and grad[x] <= grad[x - 1]
            and grad[x] <= grad[x + 1]
            and grad[x] <= grad[x - 2]
            and grad[x] <= grad[x + 2]
        ):
            right_edges.append(x)

    # ── Pair edges by gradient strength ─────────────────────────
    left_with_strength = [(lx, float(grad[lx])) for lx in left_edges]
    left_with_strength.sort(key=lambda item: -item[1])

    used_right: set[int] = set()
    pairs: list[tuple[int, int, int]] = []

    for lx, _ in left_with_strength:
        candidates = [(rx, rx - lx) for rx in right_edges if rx not in used_right and rx - lx >= 8]
        if candidates:
            rx, width = min(candidates, key=lambda c: c[1])
            if 8 <= width <= 24:
                pairs.append((lx, rx, width))
                used_right.add(rx)

    # ── Remove isolated artifacts ───────────────────────────────
    if len(pairs) >= 2:
        pairs.sort(key=lambda p: p[0])
        filtered = [pairs[0]]
        for p in pairs[1:]:
            gap = p[0] - (filtered[-1][0] + filtered[-1][2])
            if gap < 20:
                filtered.append(p)
        pairs = filtered

    # ── Center windows on brightness-weighted centroids ──────────
    result: list[tuple[int, int]] = []
    for lx, rx, w in pairs:
        left_val = float(hproj_s[lx])
        right_val = float(hproj_s[rx])
        total = left_val + right_val
        ratio = right_val / total if total > 0 else 0.5
        cx = int(lx + w * ratio)
        x1 = max(0, cx - window_w // 2)
        dw = min(window_w, w_img - x1)
        result.append((x1, dw))

    return result


def detect_y(img_data: np.ndarray) -> int:
    """Auto-detect the top of the digit band.

    Finds where brightness first crosses 60% of its maximum
    when scanning top-to-bottom. This gives the upper edge
    of the digit area, excluding bezel glow.

    Args:
        img_data: 2D grayscale array (float32, 0..1).

    Returns:
        y-coordinate of the digit band top (pixels).
    """
    vproj = np.mean(img_data, axis=1)
    vproj_s = np.convolve(vproj, np.ones(5) / 5, mode="same")
    threshold = float(vproj_s.max()) * 0.6

    for i in range(1, len(vproj_s)):
        if vproj_s[i - 1] < threshold <= vproj_s[i]:
            return max(0, i)

    return 30  # sensible fallback


def is_dot(
    img_data: np.ndarray,
    cx: int,
    cy: int,
    r: int = 4,
) -> bool:
    """Check for a decimal point near (cx, cy).

    A dot is a bright spot significantly above its local background.

    Args:
        img_data: 2D grayscale array (float32, 0..1).
        cx, cy: Approximate center of the dot.
        r: Search radius in pixels.

    Returns:
        True if a bright dot is detected.
    """
    h, w = img_data.shape
    x1, x2 = max(0, int(cx) - r), min(w, int(cx) + r)
    y1, y2 = max(0, int(cy) - r), min(h, int(cy) + r)
    region = img_data[y1:y2, x1:x2]

    if region.size < 4:
        return False

    center = float(np.max(region))
    edges = float(
        np.mean(
            [
                img_data[y1, x1] if y1 < h and x1 < w else 0,
                img_data[min(y2, h - 1), x1] if x1 < w else 0,
                img_data[y1, min(x2, w - 1)] if y1 < h else 0,
                img_data[min(y2, h - 1), min(x2, w - 1)],
            ]
        )
    )
    return center > edges * 1.4
