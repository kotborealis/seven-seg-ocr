"""Seven-Seg-OCR: blur-robust 7-segment display recognizer.

Pure image processing — no machine learning, no GPU.
Handles digits 0-9, hex letters A-F, decimal points.

Usage::

    from seven_seg_ocr import read_display

    result = read_display("photo.jpg")
    print(result["reading"])  # "2.2"
"""

from seven_seg_ocr.classifier import classify, extract_zones
from seven_seg_ocr.detector import detect_y, find_digits, is_dot
from seven_seg_ocr.reader import read_display

__version__ = "0.1.0"
__all__ = ["read_display", "find_digits", "classify", "extract_zones", "detect_y", "is_dot"]
