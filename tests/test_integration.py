"""Integration tests with real-world 7-segment photos."""

from pathlib import Path

from seven_seg_ocr import read_display

FIXTURES = Path(__file__).parent / "fixtures"


class TestRealImages:
    def test_2_dot_2(self):
        """Blurry photo of a display showing '2.2'."""
        result = read_display(FIXTURES / "test_2_2.jpg", n_digits=2)
        assert result["reading"] == "2.2"
        assert result["format"] == "D.D"
        assert result["confidence"] > 0.3

    def test_1_dot_8(self):
        """Photo of a display showing '1.8' with uneven brightness."""
        result = read_display(FIXTURES / "test_1_8.jpg", n_digits=2)
        assert result["reading"] == "1.8"
        assert result["format"] == "D.D"
        assert result["confidence"] > 0.5

    def test_2_dot_2_with_positions(self):
        """Same image with manually provided positions."""
        result = read_display(
            FIXTURES / "test_2_2.jpg",
            positions=[32, 46],
        )
        assert result["reading"] == "2.2"

    def test_result_structure(self):
        """Verify the result dict has all expected keys."""
        result = read_display(FIXTURES / "test_2_2.jpg", n_digits=2)
        assert "reading" in result
        assert "details" in result
        assert "confidence" in result
        assert "format" in result
        assert isinstance(result["details"], list)

    def test_details_have_top3(self):
        """Each detail entry should include top-3 candidates."""
        result = read_display(FIXTURES / "test_2_2.jpg", n_digits=2)
        for char, _conf, top3 in result["details"]:
            if char == ".":
                continue
            assert len(top3) == 3
            assert top3[0][0] == char


class TestAPI:
    def test_numpy_array_input(self):
        """Should accept a numpy array directly."""
        import cv2

        img = cv2.imread(str(FIXTURES / "test_2_2.jpg"))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = read_display(image_array=img_rgb, n_digits=2)
        assert result["reading"] == "2.2"

    def test_no_input_returns_empty(self):
        """No image should return empty result."""
        result = read_display()
        assert result["reading"] == ""
        assert result["confidence"] == 0.0
