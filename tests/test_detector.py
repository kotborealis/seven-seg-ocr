"""Tests for the detector module."""

import numpy as np

from seven_seg_ocr.detector import detect_y, find_digits, is_dot


def _make_test_image(width=200, height=100):
    """Create a synthetic 2-digit display image."""
    img = np.zeros((height, width), dtype=np.float32)

    # Two bright digit regions
    # Digit 1: x=50..70, y=30..70
    img[30:71, 50:71] = 0.8
    # Gap
    # Digit 2: x=85..105, y=30..70
    img[30:71, 85:106] = 0.8

    # Add slight gradient for realism
    img += np.random.normal(0, 0.02, img.shape).astype(np.float32)
    img = np.clip(img, 0, 1)

    return img


class TestFindDigits:
    def test_two_clear_digits(self):
        img = _make_test_image()
        positions = find_digits(img, y=25, dh=50)
        assert len(positions) >= 2
        x_values = [p[0] for p in positions[:2]]
        # Should be roughly around 50 and 85
        assert 40 < x_values[0] < 60
        assert 75 < x_values[1] < 95

    def test_returns_tuples(self):
        img = _make_test_image()
        positions = find_digits(img, y=25, dh=50)
        for pos in positions:
            assert isinstance(pos, tuple)
            assert len(pos) == 2
            assert isinstance(pos[0], int)
            assert isinstance(pos[1], int)

    def test_empty_image(self):
        img = np.zeros((100, 100), dtype=np.float32)
        positions = find_digits(img, y=30, dh=35)
        # Should return empty or handle gracefully
        assert isinstance(positions, list)


class TestDetectY:
    def test_returns_int(self):
        img = _make_test_image()
        y = detect_y(img)
        assert isinstance(y, int)
        assert 0 <= y < img.shape[0]

    def test_dark_image(self):
        img = np.zeros((100, 100), dtype=np.float32)
        y = detect_y(img)
        assert y == 30  # fallback value


class TestIsDot:
    def test_bright_dot(self):
        img = np.zeros((100, 100), dtype=np.float32)
        # Place a bright dot at (50, 80)
        img[78:83, 48:53] = 0.9
        assert is_dot(img, 50, 80, r=5)

    def test_no_dot(self):
        img = np.ones((100, 100), dtype=np.float32) * 0.3
        assert not is_dot(img, 50, 80)
